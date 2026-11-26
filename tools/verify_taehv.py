#!/usr/bin/env python3
"""Generate PyTorch golden reference tensors for TAEHV1.5 encode & decode parity."""
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


def _conv(n_in, n_out, **kw):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kw)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3.0) * 3.0


class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            _conv(n_in * 2, n_out), nn.ReLU(inplace=True),
            _conv(n_out, n_out), nn.ReLU(inplace=True),
            _conv(n_out, n_out),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + x)


class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x):
        _nt, c, h, w = x.shape
        return self.conv(x.reshape(-1, self.stride * c, h, w))


class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        _nt, c, h, w = x.shape
        return self.conv(x).reshape(-1, c, h, w)


class RefTAEHV(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            _conv(12, 64), nn.ReLU(inplace=True),
            TPool(64, 2), _conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 2), _conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 1), _conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            _conv(64, 32),
        )
        self.decoder = nn.Sequential(
            Clamp(), _conv(32, 256), nn.ReLU(inplace=True),
            MemBlock(256, 256), MemBlock(256, 256), MemBlock(256, 256),
            nn.Upsample(scale_factor=2), TGrow(256, 1), _conv(256, 128, bias=False),
            MemBlock(128, 128), MemBlock(128, 128), MemBlock(128, 128),
            nn.Upsample(scale_factor=2), TGrow(128, 2), _conv(128, 64, bias=False),
            MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            nn.Upsample(scale_factor=2), TGrow(64, 2), _conv(64, 64, bias=False),
            nn.ReLU(inplace=True), _conv(64, 12),
        )


def step_seq(model, mem, q):
    while q:
        xt, i = q.pop(0)
        if i == len(model):
            return xt.unsqueeze(1)
        b = model[i]
        if isinstance(b, MemBlock):
            past = torch.zeros_like(xt) if mem[i] is None else mem[i]
            xt_new = b(xt, past)
            mem[i] = xt
            q.insert(0, (xt_new, i + 1))
        elif isinstance(b, TPool):
            if mem[i] is None:
                mem[i] = []
            mem[i].append(xt)
            if len(mem[i]) == b.stride:
                n, c, h, w = xt.shape
                xt = b(torch.cat(mem[i], 1).view(n * b.stride, c, h, w))
                mem[i] = []
                q.insert(0, (xt, i + 1))
        elif isinstance(b, TGrow):
            xt = b(xt)
            nt, c, h, w = xt.shape
            for nxt in reversed(xt.view(nt // b.stride, b.stride * c, h, w).chunk(b.stride, 1)):
                q.insert(0, (nxt, i + 1))
        else:
            q.insert(0, (b(xt), i + 1))
    return None


@torch.no_grad()
def main():
    w_path = "/Users/timi/Developer/spatial/weights/raw/taehv/vae/diffusion_pytorch_model.safetensors"
    out_dir = "/tmp/spatial_taehv_ref"
    os.makedirs(out_dir, exist_ok=True)

    m = RefTAEHV()
    sd = load_file(w_path)
    m.load_state_dict(sd, strict=True)
    m.eval().float()

    # Deterministic colorful synthetic scene [4, 256, 512, 3] uint8
    H, W = 256, 512
    yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
    frames = np.zeros((4, H, W, 3), dtype=np.uint8)
    for t in range(4):
        shift = t * 0.08
        r = np.clip(0.5 + 0.4 * np.sin(6.0 * (xx + shift)), 0, 1)
        g = np.clip(0.5 + 0.4 * np.cos(5.0 * (yy + shift)), 0, 1)
        b = np.clip(0.3 + 0.6 * ((xx + yy) * 0.5), 0, 1)
        frames[t, :, :, 0] = np.round(r * 255).astype(np.uint8)
        frames[t, :, :, 1] = np.round(g * 255).astype(np.uint8)
        frames[t, :, :, 2] = np.round(b * 255).astype(np.uint8)

    # Encode
    enc_mem = [None] * len(m.encoder)
    enc_q = []
    rgb = torch.from_numpy(frames).unsqueeze(0).float().permute(0, 1, 4, 2, 3).contiguous() / 255.0
    rgb_u = F.pixel_unshuffle(rgb, 2)
    enc_q.extend((xt, 0) for xt in rgb_u.unbind(1))
    lat = step_seq(m.encoder, enc_mem, enc_q).squeeze(1)  # [1, 32, 16, 32]

    # Decode (3 warmup passes + 1 output pass)
    dec_mem = [None] * len(m.decoder)
    dec_q = []
    n_dec = 0

    def dec_step(z=None):
        nonlocal n_dec
        if z is not None:
            dec_q.extend((xt, 0) for xt in z.unbind(1))
        while True:
            xt = step_seq(m.decoder, dec_mem, dec_q)
            if xt is None:
                return None
            n_dec += 1
            if n_dec <= 3:
                continue
            return F.pixel_shuffle(xt, 2).clamp(0, 1)

    z = lat.unsqueeze(1)
    for _ in range(3):
        dec_step(z)
        while dec_step() is not None:
            pass
    outs = [dec_step(z)]
    while (f := dec_step()) is not None:
        outs.append(f)
    dec = torch.cat(outs, dim=1)  # [1, 4, 3, H, W]
    dec_u8 = (dec.clamp(0, 1) * 255.0).round().to(torch.uint8).squeeze(0).permute(0, 2, 3, 1).contiguous()

    frames.tofile(f"{out_dir}/in_rgb.bin")
    lat.squeeze(0).numpy().astype(np.float32).tofile(f"{out_dir}/ref_lat_f32.bin")
    dec_u8.numpy().tofile(f"{out_dir}/ref_dec_rgb.bin")
    print(f"Saved golden TAEHV ref to {out_dir} | lat shape={list(lat.shape)} range=[{lat.min():.3f}, {lat.max():.3f}]")


if __name__ == "__main__":
    main()

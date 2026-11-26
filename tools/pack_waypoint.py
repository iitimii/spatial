#!/usr/bin/env python3
"""
Clean-room offline weight packer for the Spatial engine (.spat v1 format).

Supports:
  1) TAEHV1.5 streaming video autoencoder -> taehv1_5.spat (F16, ~23 MB)
  2) Overworld Waypoint-1.5-1B-360P DiT   -> waypoint_1_5_360p_<quant>.spat
     - Losslessly folds 633M noise-conditioning params (denoise_step_emb +
       24x attn_cond_head + 24x mlp_cond_head + out_norm.fc) across the 5
       scheduler sigmas [1.0, 0.9, 0.75, 0.3, 0.0] into tiny F16 LUTs (~3 MB)!
     - Merges per-layer [q_proj; k_proj; v_proj] into one contiguous qkv_proj!
     - Quantizes large 2D DiT linear weights to F16, Q8_0, or Q4_0 (block=32).
     - Page-aligns (16 KB) data section for zero-copy Metal mmap wrapping.
"""

import argparse
import json
import math
import os
import struct
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors import safe_open

PAGE_ALIGN = 16384
TENSOR_ALIGN = 256
QK_BLOCK = 32


def quantize_q8_0(w_f32: np.ndarray) -> bytes:
    """Block-32 symmetric INT8 quantization: [f16 scale (2B), int8 qs[32] (32B)] = 34B/block."""
    flat = np.ascontiguousarray(w_f32, dtype=np.float32).reshape(-1)
    assert flat.size % QK_BLOCK == 0, f"Q8_0 requires numel % 32 == 0, got {flat.size}"
    blocks = flat.reshape(-1, QK_BLOCK)
    amax = np.max(np.abs(blocks), axis=1)
    scales_f32 = amax / 127.0
    scales_f16 = scales_f32.astype(np.float16)
    inv_scales = np.where(scales_f16 != 0, 1.0 / scales_f16.astype(np.float32), 0.0)
    qs = np.clip(np.round(blocks * inv_scales[:, None]), -127, 127).astype(np.int8)

    out = np.empty((blocks.shape[0], 34), dtype=np.uint8)
    out[:, :2] = scales_f16.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = qs.view(np.uint8)
    return out.tobytes()


def quantize_q4_0(w_f32: np.ndarray) -> bytes:
    """Block-32 symmetric INT4 quantization: [f16 scale (2B), uint8 packed[16] (16B)] = 18B/block.
    Low nibble = elements [0..15] + 8; high nibble = elements [16..31] + 8.
    """
    flat = np.ascontiguousarray(w_f32, dtype=np.float32).reshape(-1)
    assert flat.size % QK_BLOCK == 0, f"Q4_0 requires numel % 32 == 0, got {flat.size}"
    blocks = flat.reshape(-1, QK_BLOCK)
    amax = np.max(np.abs(blocks), axis=1)
    scales_f32 = amax / 7.0
    scales_f16 = scales_f32.astype(np.float16)
    inv_scales = np.where(scales_f16 != 0, 1.0 / scales_f16.astype(np.float32), 0.0)
    q = np.clip(np.round(blocks * inv_scales[:, None]), -8, 7).astype(np.int16) + 8  # [0..15]
    q_u8 = q.astype(np.uint8)
    packed = (q_u8[:, :16] & 0x0F) | ((q_u8[:, 16:] & 0x0F) << 4)

    out = np.empty((blocks.shape[0], 18), dtype=np.uint8)
    out[:, :2] = scales_f16.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = packed
    return out.tobytes()


class SpatWriter:
    def __init__(self, metadata: Dict):
        self.metadata = metadata
        self.entries: List[Dict] = []
        self.chunks: List[bytes] = []
        self.cursor = 0

    def add_tensor(self, name: str, arr: np.ndarray, dtype: str = "F16"):
        if dtype == "F32":
            raw = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
        elif dtype == "F16":
            raw = np.ascontiguousarray(arr, dtype=np.float16).tobytes()
        elif dtype == "Q8_0":
            raw = quantize_q8_0(arr)
        elif dtype == "Q4_0":
            raw = quantize_q4_0(arr)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        aligned_off = (self.cursor + TENSOR_ALIGN - 1) & ~(TENSOR_ALIGN - 1)
        if aligned_off > self.cursor:
            self.chunks.append(b"\x00" * (aligned_off - self.cursor))
            self.cursor = aligned_off

        self.entries.append({
            "name": name,
            "dtype": dtype,
            "shape": [int(d) for d in arr.shape],
            "offset": int(self.cursor),
            "nbytes": int(len(raw)),
        })
        self.chunks.append(raw)
        self.cursor += len(raw)

    def write(self, out_path: str):
        header_obj = {"metadata": self.metadata, "tensors": self.entries}
        header_bytes = json.dumps(header_obj, separators=(",", ":")).encode("utf-8")
        min_data_off = 32 + len(header_bytes)
        data_off = (min_data_off + PAGE_ALIGN - 1) & ~(PAGE_ALIGN - 1)

        # Pad payload tail to page boundary for zero-copy Metal mmap wrapping
        tail_pad = ((self.cursor + PAGE_ALIGN - 1) & ~(PAGE_ALIGN - 1)) - self.cursor

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(b"SPAT\x00\x00\x00\x00")
            f.write(struct.pack("<IIQQ", 1, 0, len(header_bytes), data_off))
            f.write(header_bytes)
            f.write(b"\x00" * (data_off - min_data_off))
            for ch in self.chunks:
                f.write(ch)
            if tail_pad > 0:
                f.write(b"\x00" * tail_pad)
        total_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"Wrote {out_path} ({len(self.entries)} tensors, {total_mb:.2f} MB)")


def pack_taehv(safetensors_path: str, config_path: str, out_path: str):
    t0 = time.time()
    with open(config_path) as f:
        cfg = json.load(f)
    meta = {
        "arch": "taehv1_5",
        "latent_channels": int(cfg.get("latent_channels", 32)),
        "patch_size": int(cfg.get("patch_size", 2)),
        "image_channels": int(cfg.get("image_channels", 3)),
        "t_downscale": 4,
        "t_upscale": 4,
    }
    writer = SpatWriter(meta)
    with safe_open(safetensors_path, framework="pt", device="cpu") as sf:
        for k in sorted(sf.keys()):
            t = sf.get_tensor(k).float().numpy()
            writer.add_tensor(k, t, dtype="F16")
    writer.write(out_path)
    print(f"Packed TAEHV1.5 in {time.time() - t0:.2f}s")


@torch.no_grad()
def pack_waypoint(safetensors_path: str, config_path: str, out_path: str, quant: str):
    t0 = time.time()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    n_layers = int(cfg["n_layers"])
    d_model = int(cfg["d_model"])
    n_heads = int(cfg["n_heads"])
    n_kv_heads = int(cfg.get("n_kv_heads", n_heads))
    sigmas = [float(s) for s in cfg["scheduler_sigmas"]]

    meta = {
        "arch": "waypoint_1_5",
        "quant": quant,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "mlp_ratio": int(cfg["mlp_ratio"]),
        "channels": int(cfg["channels"]),
        "height": int(cfg["height"]),
        "width": int(cfg["width"]),
        "patch_h": int(cfg["patch"][0]),
        "patch_w": int(cfg["patch"][1]),
        "tokens_per_frame": int(cfg["tokens_per_frame"]),
        "local_window": int(cfg["local_window"]),
        "global_window": int(cfg["global_window"]),
        "global_pinned_dilation": int(cfg["global_pinned_dilation"]),
        "global_attn_period": int(cfg["global_attn_period"]),
        "global_attn_offset": int(cfg.get("global_attn_offset", 0)),
        "ctrl_conditioning_period": int(cfg.get("ctrl_conditioning_period", 3)),
        "n_buttons": int(cfg.get("n_buttons", 256)),
        "temporal_compression": int(cfg.get("temporal_compression", 4)),
        "base_fps": int(cfg.get("base_fps", 15)),
        "inference_fps": int(cfg.get("inference_fps", 60)),
        "rope_nyquist_frac": float(cfg.get("rope_nyquist_frac", 0.8)),
        "rope_theta": float(cfg.get("rope_theta", 10000.0)),
        "scheduler_sigmas": sigmas,
    }

    writer = SpatWriter(meta)

    with safe_open(safetensors_path, framework="pt", device="cpu") as sf:
        keys = set(sf.keys())

        def get_f32(k: str) -> torch.Tensor:
            return sf.get_tensor(k).to(torch.float32)

        # 1. Losslessly fold denoise_step_emb + all 24 CondHeads + out_norm.fc!
        # Match exact bf16 sigma rounding & FP32 Fourier features of NoiseConditioner
        sig_bf16 = torch.tensor(sigmas, dtype=torch.bfloat16)
        s_f32 = sig_bf16.float() * 1000.0
        half = 256
        freq = torch.logspace(0, -1, steps=half, base=10000.0, dtype=torch.float32)
        phase = s_f32[:, None] * freq[None, :]
        emb = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1) * math.sqrt(2.0)

        w1 = get_f32("denoise_step_emb.mlp.fc1.weight")
        w2 = get_f32("denoise_step_emb.mlp.fc2.weight")
        cond = F.linear(F.silu(F.linear(emb, w1)), w2).to(torch.bfloat16).float()  # [S, D]

        S = len(sigmas)
        # cond_table: [S, n_layers, 6, D] -> (s0, b0, g0, s1, b1, g1)
        cond_table = torch.empty((S, n_layers, 6, d_model), dtype=torch.float32)
        for l in range(n_layers):
            p = f"transformer.blocks.{l}"
            # attn_cond_head
            h_a = F.silu(cond + get_f32(f"{p}.attn_cond_head.bias_in"))
            for j in range(3):
                cond_table[:, l, j, :] = F.linear(h_a, get_f32(f"{p}.attn_cond_head.cond_proj.{j}.weight"))
            # mlp_cond_head
            h_m = F.silu(cond + get_f32(f"{p}.mlp_cond_head.bias_in"))
            for j in range(3):
                cond_table[:, l, 3 + j, :] = F.linear(h_m, get_f32(f"{p}.mlp_cond_head.cond_proj.{j}.weight"))

        # out_norm_table: [S, 2, D] -> (scale, bias)
        out_ab = F.linear(F.silu(cond), get_f32("out_norm.fc.weight")).view(S, 2, d_model)

        writer.add_tensor("folded.cond_table", cond_table.numpy(), dtype="F16")
        writer.add_tensor("folded.out_norm_table", out_ab.numpy(), dtype="F16")
        print(f"Folded 633M noise-conditioning params into {cond_table.numel()*2/1e6:.2f} MB LUT!")

        # 2. Patchify / Unpatchify / Controller embedding (keep F16 for fidelity; tiny!)
        # Patchify Conv2d [D, C, 2, 2] -> reshape [D, C*4] for direct linear GEMM
        w_patch = get_f32("patchify.weight").reshape(d_model, -1).numpy()
        writer.add_tensor("patchify.weight", w_patch, dtype="F16")

        # Unpatchify ConvTranspose2d [D, C, 2, 2] -> transpose to [C*4, D] for direct linear GEMM
        w_unpatch = get_f32("unpatchify.weight").permute(1, 2, 3, 0).reshape(-1, d_model).contiguous().numpy()
        writer.add_tensor("unpatchify.weight", w_unpatch, dtype="F16")
        writer.add_tensor("unpatchify.bias", get_f32("unpatchify.bias").numpy(), dtype="F16")

        # Controller input embedding: [8192, 259] & [2048, 8192]
        writer.add_tensor("ctrl_emb.fc1", get_f32("ctrl_emb.mlp.fc1.weight").numpy(), dtype="F16")
        writer.add_tensor("ctrl_emb.fc2", get_f32("ctrl_emb.mlp.fc2.weight").numpy(), dtype="F16")

        # 3. 24 WorldDiT blocks (merged QKV + quantize large 2D weights)
        w_dtype = quant.upper()
        for l in range(n_layers):
            p = f"transformer.blocks.{l}"
            wq = get_f32(f"{p}.attn.q_proj.weight")
            wk = get_f32(f"{p}.attn.k_proj.weight")
            wv = get_f32(f"{p}.attn.v_proj.weight")
            wqkv = torch.cat([wq, wk, wv], dim=0).numpy()  # [4096, 2048]
            writer.add_tensor(f"blk.{l}.attn.qkv", wqkv, dtype=w_dtype)
            writer.add_tensor(f"blk.{l}.attn.out", get_f32(f"{p}.attn.out_proj.weight").numpy(), dtype=w_dtype)

            vlamb = get_f32(f"{p}.attn.v_lamb").reshape(1).numpy()
            writer.add_tensor(f"blk.{l}.attn.v_lamb", vlamb, dtype="F32")

            if f"{p}.ctrl_mlpfusion.fc1_x.weight" in keys:
                writer.add_tensor(f"blk.{l}.ctrl.fc1_x", get_f32(f"{p}.ctrl_mlpfusion.fc1_x.weight").numpy(), dtype=w_dtype)
                writer.add_tensor(f"blk.{l}.ctrl.fc1_c", get_f32(f"{p}.ctrl_mlpfusion.fc1_c.weight").numpy(), dtype=w_dtype)
                writer.add_tensor(f"blk.{l}.ctrl.fc2", get_f32(f"{p}.ctrl_mlpfusion.fc2.weight").numpy(), dtype=w_dtype)

            writer.add_tensor(f"blk.{l}.mlp.fc1", get_f32(f"{p}.dit_mlp.fc1.weight").numpy(), dtype=w_dtype)
            writer.add_tensor(f"blk.{l}.mlp.fc2", get_f32(f"{p}.dit_mlp.fc2.weight").numpy(), dtype=w_dtype)

            if (l + 1) % 6 == 0:
                print(f"  packed {l + 1}/{n_layers} DiT blocks ({time.time() - t0:.1f}s)")

    writer.write(out_path)
    print(f"Packed Waypoint-1.5 ({w_dtype}) in {time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_vae = sub.add_parser("taehv")
    p_vae.add_argument("--weights", required=True)
    p_vae.add_argument("--config", required=True)
    p_vae.add_argument("--out", required=True)

    p_dit = sub.add_parser("waypoint")
    p_dit.add_argument("--weights", required=True)
    p_dit.add_argument("--config", required=True)
    p_dit.add_argument("--out", required=True)
    p_dit.add_argument("--quant", choices=["f16", "q8_0", "q4_0"], default="q4_0")

    args = ap.parse_args()
    if args.cmd == "taehv":
        pack_taehv(args.weights, args.config, args.out)
    elif args.cmd == "waypoint":
        pack_waypoint(args.weights, args.config, args.out, args.quant)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Exact PyTorch single-pass reference for Waypoint-1.5-1B-360P (24 layers + KV cache)
to verify our clean-room C++20/Metal runtime numerically!
"""
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors import safe_open


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


@torch.no_grad()
def main():
    w_path = "/Users/timi/Developer/spatial/weights/raw/waypoint360/model.safetensors"
    cfg_path = "/Users/timi/Developer/spatial/weights/raw/waypoint360/config.yaml"
    out_dir = "/tmp/spatial_wp_ref"
    os.makedirs(out_dir, exist_ok=True)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    D = int(cfg["d_model"])         # 2048
    H_p = int(cfg["height"])        # 8
    W_p = int(cfg["width"])         # 16
    T = H_p * W_p                   # 128
    C = int(cfg["channels"])        # 32
    n_layers = int(cfg["n_layers"]) # 24
    n_heads = int(cfg["n_heads"])   # 32
    n_kv = int(cfg["n_kv_heads"])   # 16
    dh = D // n_heads               # 64
    sigmas = [float(s) for s in cfg["scheduler_sigmas"]]

    # OrthoRoPE frequency basis
    d_xy, d_t = dh // 8, dh // 4
    max_freq = min(H_p, W_p) * float(cfg.get("rope_nyquist_frac", 0.8))
    xy = (torch.linspace(1.0, max_freq / 2.0, (d_xy + 1) // 2) * math.pi).repeat_interleave(2)[:d_xy]
    inv_t = (1.0 / (float(cfg.get("rope_theta", 10000.0)) ** (torch.arange(0, d_t, 2).float() / d_t))).repeat_interleave(2)

    idx = torch.arange(T)
    y_pos = (idx // W_p).float()
    x_pos = (idx % W_p).float()
    xn = (2.0 * x_pos + 1.0) / W_p - 1.0
    yn = (2.0 * y_pos + 1.0) / H_p - 1.0

    def rope_angles(t_val: float):
        freqs = torch.cat([xn[:, None] * xy, yn[:, None] * xy, torch.full((T, 1), t_val) * inv_t], dim=-1)
        return freqs.cos()[None, None], freqs.sin()[None, None]  # [1, 1, T, 32]

    def apply_rope(x, cos, sin):
        x0, x1 = x.float().unfold(-1, 2, 2).unbind(-1)
        y0 = x0 * cos - x1 * sin
        y1 = x1 * cos + x0 * sin
        return torch.cat([y0, y1], dim=-1).type_as(x)

    # Deterministic inputs: seed frame 0 latent + noisy frame 1 latent + control
    torch.manual_seed(1234)
    seed_lat = torch.randn(1, C, H_p * 2, W_p * 2, dtype=torch.float32) * 0.5
    noisy_lat = torch.randn(1, C, H_p * 2, W_p * 2, dtype=torch.float32)

    mouse = torch.tensor([[[0.25, -0.15]]], dtype=torch.float32)
    button = torch.zeros(1, 1, 256, dtype=torch.float32)
    button[0, 0, 87] = 1.0  # 'W'
    scroll = torch.zeros(1, 1, 1, dtype=torch.float32)

    with safe_open(w_path, framework="pt", device="cpu") as sf:
        keys = set(sf.keys())
        G = lambda k: sf.get_tensor(k).float()

        def noise_emb(sig_val: float):
            s = torch.tensor([sig_val], dtype=torch.bfloat16).float() * 1000.0
            freq = torch.logspace(0, -1, steps=256, base=10000.0, dtype=torch.float32)
            ph = s[:, None] * freq[None, :]
            emb = torch.cat([torch.sin(ph), torch.cos(ph)], dim=-1) * math.sqrt(2.0)
            return F.linear(F.silu(F.linear(emb, G("denoise_step_emb.mlp.fc1.weight"))),
                            G("denoise_step_emb.mlp.fc2.weight")).to(torch.bfloat16).float()

        def ctrl_emb_fn(m, b, sc):
            inp = torch.cat([m, b, sc], dim=-1)
            return F.linear(F.silu(F.linear(inp, G("ctrl_emb.mlp.fc1.weight"))),
                            G("ctrl_emb.mlp.fc2.weight"))

        # Run 2 passes:
        #   Pass 0: cache seed_lat at frame_idx=0, sigma=0.0, empty ctrl -> store K0, V0 per layer
        #   Pass 1: denoise noisy_lat at frame_idx=1, sigma=1.0, active ctrl -> attend to [frame0, frame1]
        k_cache = [None] * n_layers
        v_cache = [None] * n_layers

        def run_pass(lat, sig_val, frame_idx, m, b, sc, write_cache: bool):
            cond = noise_emb(sig_val)              # [1, D]
            c_emb = ctrl_emb_fn(m, b, sc)          # [1, 1, D]
            cos, sin = rope_angles(float(frame_idx))

            x = F.conv2d(lat, G("patchify.weight"), stride=(2, 2))  # [1, D, Hp, Wp]
            x = x.permute(0, 2, 3, 1).reshape(1, T, D)

            v1 = None
            for l in range(n_layers):
                p = f"transformer.blocks.{l}"
                ha = F.silu(cond + G(f"{p}.attn_cond_head.bias_in"))
                s0 = F.linear(ha, G(f"{p}.attn_cond_head.cond_proj.0.weight"))
                b0 = F.linear(ha, G(f"{p}.attn_cond_head.cond_proj.1.weight"))
                g0 = F.linear(ha, G(f"{p}.attn_cond_head.cond_proj.2.weight"))

                hm = F.silu(cond + G(f"{p}.mlp_cond_head.bias_in"))
                s1 = F.linear(hm, G(f"{p}.mlp_cond_head.cond_proj.0.weight"))
                b1 = F.linear(hm, G(f"{p}.mlp_cond_head.cond_proj.1.weight"))
                g1 = F.linear(hm, G(f"{p}.mlp_cond_head.cond_proj.2.weight"))

                resid = x
                h = rms_norm(x) * (1.0 + s0[:, None, :]) + b0[:, None, :]

                q = F.linear(h, G(f"{p}.attn.q_proj.weight")).view(1, T, n_heads, dh).transpose(1, 2)
                k = F.linear(h, G(f"{p}.attn.k_proj.weight")).view(1, T, n_kv, dh).transpose(1, 2)
                v = F.linear(h, G(f"{p}.attn.v_proj.weight")).view(1, T, n_kv, dh).transpose(1, 2)

                if v1 is None:
                    v1 = v
                vlamb = G(f"{p}.attn.v_lamb")
                v = torch.lerp(v, v1, vlamb)

                q = apply_rope(rms_norm(q), cos, sin)
                k = apply_rope(rms_norm(k), cos, sin)

                if k_cache[l] is not None:
                    k_full = torch.cat([k_cache[l], k], dim=2)
                    v_full = torch.cat([v_cache[l], v], dim=2)
                else:
                    k_full, v_full = k, v

                if write_cache:
                    k_cache[l] = k.clone()
                    v_cache[l] = v.clone()

                # GQA attention
                k_rep = k_full.repeat_interleave(n_heads // n_kv, dim=1)
                v_rep = v_full.repeat_interleave(n_heads // n_kv, dim=1)
                y = F.scaled_dot_product_attention(q, k_rep, v_rep)
                y = y.transpose(1, 2).reshape(1, T, D)
                y = F.linear(y, G(f"{p}.attn.out_proj.weight"))
                x = y * g0[:, None, :] + resid

                if f"{p}.ctrl_mlpfusion.fc1_x.weight" in keys:
                    cx = F.linear(rms_norm(x), G(f"{p}.ctrl_mlpfusion.fc1_x.weight"))
                    cc = F.linear(rms_norm(c_emb), G(f"{p}.ctrl_mlpfusion.fc1_c.weight"))
                    x = F.linear(F.silu(cx + cc), G(f"{p}.ctrl_mlpfusion.fc2.weight")) + x

                hm_in = rms_norm(x) * (1.0 + s1[:, None, :]) + b1[:, None, :]
                m_out = F.linear(F.silu(F.linear(hm_in, G(f"{p}.dit_mlp.fc1.weight"))),
                                 G(f"{p}.dit_mlp.fc2.weight"))
                x = m_out * g1[:, None, :] + x

            ab = F.linear(F.silu(cond), G("out_norm.fc.weight")).view(1, 2, D)
            x = F.silu(rms_norm(x) * (1.0 + ab[:, 0:1, :]) + ab[:, 1:2, :])
            x = x.reshape(1, H_p, W_p, D).permute(0, 3, 1, 2).contiguous()
            x = F.conv_transpose2d(x, G("unpatchify.weight"), G("unpatchify.bias"), stride=(2, 2))
            return x

        print("Running PyTorch ref Pass 0 (cache seed frame 0)...")
        run_pass(seed_lat, 0.0, 0, torch.zeros(1, 1, 2), torch.zeros(1, 1, 256), torch.zeros(1, 1, 1), True)
        print("Running PyTorch ref Pass 1 (denoise frame 1, sigma=1.0, W+mouse)...")
        v_ref = run_pass(noisy_lat, 1.0, 1, mouse, button, scroll, False)

    seed_lat.squeeze(0).numpy().astype(np.float32).tofile(f"{out_dir}/seed_lat_f32.bin")
    noisy_lat.squeeze(0).numpy().astype(np.float32).tofile(f"{out_dir}/noisy_lat_f32.bin")
    v_ref.squeeze(0).numpy().astype(np.float32).tofile(f"{out_dir}/ref_v_f32.bin")
    print(f"Saved Waypoint ref to {out_dir} | v_ref range=[{v_ref.min():.4f}, {v_ref.max():.4f}], std={v_ref.std():.4f}")


if __name__ == "__main__":
    main()

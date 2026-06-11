"""
TriMul submission — pure eager PyTorch, zero module overhead.
Fuses projections via stacked GEMM, replaces einsum with torch.bmm.
"""

import torch
import torch.nn.functional as F


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    # Unpack weights (all float32)
    norm_w = weights['norm.weight']
    norm_b = weights['norm.bias']
    lp_w = weights['left_proj.weight']    # [H, D]
    rp_w = weights['right_proj.weight']   # [H, D]
    lg_w = weights['left_gate.weight']    # [H, D]
    rg_w = weights['right_gate.weight']   # [H, D]
    og_w = weights['out_gate.weight']     # [H, D]
    ton_w = weights['to_out_norm.weight']
    ton_b = weights['to_out_norm.bias']
    to_w = weights['to_out.weight']       # [D, H]

    B, N, _, D = input_tensor.shape
    H = config['hidden_dim']

    # LayerNorm input
    x = F.layer_norm(input_tensor, (D,), norm_w, norm_b)  # [B, N, N, D]

    # Flatten spatial dims for batched GEMM: [B*N*N, D]
    x_flat = x.reshape(-1, D)

    # Fuse all 5 projections into a single GEMM: [B*N*N, 5*H]
    # Stack weights: [5*H, D]
    fused_w = torch.cat([lp_w, rp_w, lg_w, rg_w, og_w], dim=0)  # [5H, D]
    fused = x_flat @ fused_w.t()  # [B*N*N, 5H]

    # Split into individual projections
    lp, rp, lg, rg, og = fused.split(H, dim=1)  # each [B*N*N, H]

    # Apply gates
    left  = lp * lg.sigmoid()   # [B*N*N, H]
    right = rp * rg.sigmoid()   # [B*N*N, H]
    out_gate = og.sigmoid()      # [B*N*N, H]

    # Apply mask: [B, N, N] -> [B*N*N, 1]
    mask_flat = mask.reshape(-1, 1)
    left  = left  * mask_flat   # [B*N*N, H]
    right = right * mask_flat   # [B*N*N, H]

    # Reshape for bmm: einsum "b i k h, b j k h -> b i j h"
    # Treat each (b, h) slice: left[b, i, :, h] @ right[b, j, :, h]
    # Reshape: left -> [B, N, N, H] -> [B*H, N, N], right -> [B*H, N, N]
    # Then out[b,h,i,j] = sum_k left[b,h,i,k] * right[b,h,j,k]
    #                    = left[b,h,i,:] @ right[b,h,j,:]^T
    # So: out = bmm(left_r, right_r.transpose(-1,-2))  -> [B*H, N, N]
    left_r  = left.reshape(B, N, N, H).permute(0, 3, 1, 2).reshape(B*H, N, N)   # [B*H, N, N]
    right_r = right.reshape(B, N, N, H).permute(0, 3, 1, 2).reshape(B*H, N, N)  # [B*H, N, N]

    out_bmm = torch.bmm(left_r, right_r.transpose(1, 2))  # [B*H, N, N]

    # Reshape back to [B, N, N, H]
    out = out_bmm.reshape(B, H, N, N).permute(0, 2, 3, 1)  # [B, N, N, H]

    # LayerNorm + out_gate
    out = F.layer_norm(out, (H,), ton_w, ton_b)
    out = out * out_gate.reshape(B, N, N, H)

    # Final projection: [B*N*N, H] @ [H, D] -> [B*N*N, D]
    out_flat = out.reshape(-1, H) @ to_w.t()  # [B*N*N, D]

    return out_flat.reshape(B, N, N, D)

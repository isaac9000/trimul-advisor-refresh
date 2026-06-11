"""
TriMul submission — Triton kernel for the contraction step.
Fuses projections via stacked GEMM, then uses a Triton kernel for the
b i k h, b j k h -> b i j h contraction (fused mask application).
Output LayerNorm and final projection remain as PyTorch ops.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Enable TF32 for projection GEMMs
torch.backends.cuda.matmul.allow_tf32 = True


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['B', 'N', 'H'],
)
@triton.jit
def trimul_contraction_kernel(
    left_ptr,   # [B, N, N, H] float32
    right_ptr,  # [B, N, N, H] float32
    out_ptr,    # [B, N, N, H] float32
    B, N, H,
    stride_lb, stride_li, stride_lk, stride_lh,
    stride_rb, stride_rj, stride_rk, stride_rh,
    stride_ob, stride_oi, stride_oj, stride_oh,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute out[b, i, j, h] = sum_k left[b, i, k, h] * right[b, j, k, h]
    Grid: (ceil(N/BLOCK_M) * ceil(N/BLOCK_N), B*H)
    """
    # Program IDs
    pid_bh = tl.program_id(1)  # combined batch*hidden index
    pid_mn = tl.program_id(0)  # combined i_tile * j_tile index

    b = pid_bh // H
    h = pid_bh % H

    # Number of N-tiles along each dim
    num_tiles_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // num_tiles_n
    pid_n = pid_mn % num_tiles_n

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Base pointers for this (b, h) slice
    # left[b, i, k, h]: stride_lb*b + stride_li*i + stride_lk*k + stride_lh*h
    left_base = left_ptr + b * stride_lb + h * stride_lh
    right_base = right_ptr + b * stride_rb + h * stride_rh

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over k dimension
    for k_start in range(0, N, BLOCK_K):
        k_offs = k_start + offs_k
        k_mask = k_offs < N

        # Load left tile: [BLOCK_M, BLOCK_K] — left[b, offs_m, k_offs, h]
        m_mask = offs_m < N
        left_ptrs = left_base + offs_m[:, None] * stride_li + k_offs[None, :] * stride_lk
        left_tile = tl.load(left_ptrs,
                            mask=m_mask[:, None] & k_mask[None, :],
                            other=0.0)

        # Load right tile: [BLOCK_N, BLOCK_K] — right[b, offs_n, k_offs, h]
        n_mask = offs_n < N
        right_ptrs = right_base + offs_n[:, None] * stride_rj + k_offs[None, :] * stride_rk
        right_tile = tl.load(right_ptrs,
                             mask=n_mask[:, None] & k_mask[None, :],
                             other=0.0)

        # Accumulate dot product: out[i,j] += sum_k left[i,k] * right[j,k]
        # Cast to bf16 for tensor cores
        acc += tl.dot(left_tile.to(tl.bfloat16), right_tile.trans(1, 0).to(tl.bfloat16))

    # Store result: out[b, offs_m, offs_n, h]
    m_mask = offs_m < N
    n_mask = offs_n < N
    out_ptrs = out_ptr + b * stride_ob + offs_m[:, None] * stride_oi + offs_n[None, :] * stride_oj + h * stride_oh
    tl.store(out_ptrs, acc.to(tl.float32), mask=m_mask[:, None] & n_mask[None, :])


def trimul_contraction(left, right):
    """
    left:  [B, N, N, H] float32 — already mask-applied
    right: [B, N, N, H] float32 — already mask-applied
    returns out: [B, N, N, H] float32
    out[b, i, j, h] = sum_k left[b, i, k, h] * right[b, j, k, h]
    """
    B, N, _, H = left.shape
    out = torch.empty(B, N, N, H, device=left.device, dtype=torch.float32)

    # Make contiguous for clean strides
    left = left.contiguous()
    right = right.contiguous()

    def grid(meta):
        return (triton.cdiv(N, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']), B * H)

    trimul_contraction_kernel[grid](
        left, right, out,
        B, N, H,
        left.stride(0), left.stride(1), left.stride(2), left.stride(3),
        right.stride(0), right.stride(1), right.stride(2), right.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )
    return out


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    norm_w = weights['norm.weight']
    norm_b = weights['norm.bias']
    lp_w = weights['left_proj.weight']
    rp_w = weights['right_proj.weight']
    lg_w = weights['left_gate.weight']
    rg_w = weights['right_gate.weight']
    og_w = weights['out_gate.weight']
    ton_w = weights['to_out_norm.weight']
    ton_b = weights['to_out_norm.bias']
    to_w = weights['to_out.weight']

    B, N, _, D = input_tensor.shape
    H = config['hidden_dim']

    # LayerNorm
    x = F.layer_norm(input_tensor, (D,), norm_w, norm_b)

    # Fuse all 5 projections into a single GEMM: [B*N*N, 5*H]
    x_flat = x.reshape(-1, D)
    fused_w = torch.cat([lp_w, rp_w, lg_w, rg_w, og_w], dim=0)  # [5H, D]
    fused = x_flat @ fused_w.t()  # [B*N*N, 5H]

    # Split into individual projections, each [B*N*N, H]
    lp, rp, lg, rg, og = fused.split(H, dim=1)

    # Apply gates
    left  = lp * lg.sigmoid()   # [B*N*N, H]
    right = rp * rg.sigmoid()   # [B*N*N, H]
    out_gate = og.sigmoid()      # [B*N*N, H]

    # Apply mask: [B, N, N] -> [B*N*N, 1]
    mask_flat = mask.reshape(-1, 1)
    left  = left  * mask_flat   # [B*N*N, H]
    right = right * mask_flat   # [B*N*N, H]

    # Reshape to [B, N, N, H] for Triton contraction
    left_4d  = left.reshape(B, N, N, H)
    right_4d = right.reshape(B, N, N, H)

    # Triton contraction: out[b,i,j,h] = sum_k left[b,i,k,h] * right[b,j,k,h]
    out = trimul_contraction(left_4d, right_4d)  # [B, N, N, H]

    # LayerNorm + out_gate
    out = F.layer_norm(out, (H,), ton_w, ton_b)
    out = out * out_gate.reshape(B, N, N, H)

    # Final projection: [B*N*N, H] @ [H, D] -> [B*N*N, D]
    out_flat = out.reshape(-1, H) @ to_w.t()

    return out_flat.reshape(B, N, N, D)

"""
TriMul submission — custom Triton kernel for the N×N contraction.
Eliminates all permute/contiguous copies: kernel reads [B,N,N,H] directly
with native strides (stride_b=N*N*H, stride_i=N*H, stride_k=H, stride_h=1).
Grid: (B*H, cdiv(N,BLOCK_N), cdiv(N,BLOCK_N)), b=bh//H, h=bh%H.
Writes output [B,N,N,H] directly (stride_oj=H, no post-permute needed).
Uses bf16 tl.dot for H100 tensor cores.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Enable TF32 for projection GEMMs
torch.backends.cuda.matmul.allow_tf32 = True


@triton.jit
def _trimul_contraction_kernel(
    left_ptr,   # [B, N, N, H] float32 — left[b, i, k, h]
    right_ptr,  # [B, N, N, H] float32 — right[b, j, k, h]
    out_ptr,    # [B, N, N, H] float32 — out[b, i, j, h]
    N, H,
    # [B,N,N,H] strides for left/right (same layout)
    sl_b, sl_i, sl_k, sl_h,
    # [B,N,N,H] strides for output
    so_b, so_i, so_j, so_h,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B*H, cdiv(N,BLOCK_N), cdiv(N,BLOCK_N))
    bh     = tl.program_id(0)
    i_tile = tl.program_id(1)
    j_tile = tl.program_id(2)

    b = bh // H
    h = bh % H

    i_start = i_tile * BLOCK_N
    j_start = j_tile * BLOCK_N

    i_offs = i_start + tl.arange(0, BLOCK_N)
    j_offs = j_start + tl.arange(0, BLOCK_N)
    i_mask = i_offs < N
    j_mask = j_offs < N

    # Base for this (b, h): left[b, :, :, h] has row stride sl_i, col stride sl_k
    left_base  = left_ptr  + b * sl_b + h * sl_h
    right_base = right_ptr + b * sl_b + h * sl_h
    out_base   = out_ptr   + b * so_b + h * so_h

    # Accumulator [BLOCK_N, BLOCK_N] in fp32
    acc = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)

    # Loop over k tiles — left[b,i,k,h] @ right[b,j,k,h]^T
    for k_start in range(0, N, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < N

        # left_tile[BLOCK_N, BLOCK_K] = left[b, i_offs, k_offs, h]
        left_ptrs  = left_base  + i_offs[:, None] * sl_i + k_offs[None, :] * sl_k
        left_tile  = tl.load(left_ptrs,
                             mask=i_mask[:, None] & k_mask[None, :],
                             other=0.0).to(tl.bfloat16)

        # right_tile[BLOCK_N, BLOCK_K] = right[b, j_offs, k_offs, h]
        right_ptrs = right_base + j_offs[:, None] * sl_i + k_offs[None, :] * sl_k
        right_tile = tl.load(right_ptrs,
                             mask=j_mask[:, None] & k_mask[None, :],
                             other=0.0).to(tl.bfloat16)

        # acc[i,j] += left[i,k] * right[j,k] = left_tile @ right_tile.T
        acc += tl.dot(left_tile, tl.trans(right_tile), allow_tf32=False).to(tl.float32)

    # Write out[b, i_offs, j_offs, h] directly in [B,N,N,H] layout
    out_ptrs = out_base + i_offs[:, None] * so_i + j_offs[None, :] * so_j
    tl.store(out_ptrs, acc, mask=i_mask[:, None] & j_mask[None, :])


def trimul_contraction(left_4d, right_4d, B, N, H):
    """
    left_4d:  [B, N, N, H] float32 (contiguous)
    right_4d: [B, N, N, H] float32 (contiguous)
    out:      [B, N, N, H] float32
    Computes out[b,i,j,h] = sum_k left[b,i,k,h] * right[b,j,k,h]
    No permute/copy — reads/writes native [B,N,N,H] layout.
    """
    out = torch.empty(B, N, N, H, dtype=torch.float32, device=left_4d.device)

    BLOCK_N = 32
    BLOCK_K = 32

    n_tiles = triton.cdiv(N, BLOCK_N)
    grid = (B * H, n_tiles, n_tiles)

    _trimul_contraction_kernel[grid](
        left_4d, right_4d, out,
        N, H,
        left_4d.stride(0), left_4d.stride(1), left_4d.stride(2), left_4d.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


def _trimul_inner(x, mask, lp_w, rp_w, lg_w, rg_w, og_w,
                  ton_w, ton_b, to_w, B, N, H):
    """Core TriMul — compiled projection + Triton contraction."""
    D = x.shape[-1]
    x_flat = x.reshape(-1, D)

    # Fuse all 5 projections into a single GEMM: [B*N*N, 5*H]
    fused_w = torch.cat([lp_w, rp_w, lg_w, rg_w, og_w], dim=0)  # [5H, D]
    fused = x_flat @ fused_w.t()  # [B*N*N, 5H]

    lp, rp, lg, rg, og = fused.split(H, dim=1)

    left  = lp * lg.sigmoid()
    right = rp * rg.sigmoid()
    out_gate = og.sigmoid()

    mask_flat = mask.reshape(-1, 1)
    left  = left  * mask_flat
    right = right * mask_flat

    # Reshape to [B, N, N, H] — contiguous (came from split of contiguous tensor)
    left_4d  = left.reshape(B, N, N, H)
    right_4d = right.reshape(B, N, N, H)

    # Triton contraction: reads [B,N,N,H] natively, no permute copies
    out = trimul_contraction(left_4d, right_4d, B, N, H)  # [B, N, N, H]

    out = F.layer_norm(out, (H,), ton_w, ton_b)
    out = out * out_gate.reshape(B, N, N, H)

    out_flat = out.reshape(-1, H) @ to_w.t()
    return out_flat.reshape(B, N, N, D)


# reduce-overhead for projection + elementwise (Triton kernel escapes graph)
_compiled_trimul = torch.compile(_trimul_inner, mode="reduce-overhead", fullgraph=False)


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

    x = F.layer_norm(input_tensor, (D,), norm_w, norm_b)

    return _compiled_trimul(x, mask, lp_w, rp_w, lg_w, rg_w, og_w,
                            ton_w, ton_b, to_w, B, N, H)

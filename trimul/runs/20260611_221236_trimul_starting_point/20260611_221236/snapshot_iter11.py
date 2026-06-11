"""
TriMul submission — custom Triton kernel for the N×N contraction.
Grid: (B*H, cdiv(N,BLOCK_N), cdiv(N,BLOCK_N)) — each program handles one
(b,h) slice and one (i_tile, j_tile) output tile as a standard GEMM:
  out[b,i_tile,j_tile,h] = left[b,i_tile,:,h] @ right[b,j_tile,:,h]^T
Inputs: left/right as [B,H,N,N] (permuted from [B,N,N,H] once before launch).
Uses bf16 tl.dot for tensor cores. No h-loop in kernel, no BLOCK_H=1 overhead.
Rest of pipeline: fused-5H GEMM projection, reduce-overhead compile.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Enable TF32 for projection GEMMs
torch.backends.cuda.matmul.allow_tf32 = True


@triton.jit
def _trimul_bmm_kernel(
    left_ptr,   # [B*H, N, N] float32 — left[bh, i, k]
    right_ptr,  # [B*H, N, N] float32 — right[bh, j, k]
    out_ptr,    # [B*H, N, N] float32 — out[bh, i, j]
    N,
    stride_bh, stride_i, stride_k,   # left strides (same for right, out)
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B*H, cdiv(N,BLOCK_N), cdiv(N,BLOCK_N))
    bh     = tl.program_id(0)
    i_tile = tl.program_id(1)
    j_tile = tl.program_id(2)

    i_start = i_tile * BLOCK_N
    j_start = j_tile * BLOCK_N

    i_offs = i_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    j_offs = j_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    i_mask = i_offs < N
    j_mask = j_offs < N

    # Base pointers for this (bh) slice
    left_base  = left_ptr  + bh * stride_bh
    right_base = right_ptr + bh * stride_bh
    out_base   = out_ptr   + bh * stride_bh

    # Accumulator [BLOCK_N, BLOCK_N] in fp32
    acc = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)

    # Loop over k tiles — accumulate left[i, k] @ right[j, k]^T
    for k_start in range(0, N, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < N

        # left_tile: [BLOCK_N, BLOCK_K] — left[bh, i_offs, k_offs]
        left_ptrs = left_base + i_offs[:, None] * stride_i + k_offs[None, :] * stride_k
        left_tile = tl.load(left_ptrs,
                            mask=i_mask[:, None] & k_mask[None, :],
                            other=0.0).to(tl.bfloat16)

        # right_tile: [BLOCK_N, BLOCK_K] — right[bh, j_offs, k_offs]
        right_ptrs = right_base + j_offs[:, None] * stride_i + k_offs[None, :] * stride_k
        right_tile = tl.load(right_ptrs,
                             mask=j_mask[:, None] & k_mask[None, :],
                             other=0.0).to(tl.bfloat16)

        # acc[i,j] += sum_k left[i,k] * right[j,k] = left_tile @ right_tile.T
        acc += tl.dot(left_tile, tl.trans(right_tile), allow_tf32=False).to(tl.float32)

    # Write out[bh, i_offs, j_offs]
    out_ptrs = out_base + i_offs[:, None] * stride_i + j_offs[None, :] * stride_k
    tl.store(out_ptrs, acc, mask=i_mask[:, None] & j_mask[None, :])


def trimul_contraction(left_4d, right_4d, B, N, H):
    """
    left_4d:  [B, N, N, H] float32
    right_4d: [B, N, N, H] float32
    out:      [B, N, N, H] float32
    Computes out[b,i,j,h] = sum_k left[b,i,k,h] * right[b,j,k,h]
    """
    # Permute to [B, H, N, N] then reshape to [B*H, N, N] — contiguous for bmm
    left_bh  = left_4d.permute(0, 3, 1, 2).contiguous().reshape(B * H, N, N)
    right_bh = right_4d.permute(0, 3, 1, 2).contiguous().reshape(B * H, N, N)
    out_bh   = torch.empty(B * H, N, N, dtype=torch.float32, device=left_4d.device)

    BLOCK_N = 32
    BLOCK_K = 32

    n_tiles = triton.cdiv(N, BLOCK_N)
    grid = (B * H, n_tiles, n_tiles)

    _trimul_bmm_kernel[grid](
        left_bh, right_bh, out_bh,
        N,
        left_bh.stride(0), left_bh.stride(1), left_bh.stride(2),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    # Reshape back [B*H, N, N] -> [B, H, N, N] -> [B, N, N, H]
    return out_bh.reshape(B, H, N, N).permute(0, 2, 3, 1).contiguous()


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

    # Reshape to [B, N, N, H] for Triton kernel
    left_4d  = left.reshape(B, N, N, H)
    right_4d = right.reshape(B, N, N, H)

    # Triton contraction: no intermediate [B*H,N,N] bmm overhead in Python
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

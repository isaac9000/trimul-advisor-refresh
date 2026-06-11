"""
TriMul submission — Triton fused contraction+LN+gate kernel.
Grid over (B, N_i_tiles, N_j_tiles); all H channels processed in registers
per output (i,j) position, with fused LayerNorm and out_gate application.
Eliminates all permute/reshape overhead for the contraction step.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Enable TF32 for projection GEMMs
torch.backends.cuda.matmul.allow_tf32 = True

# Cache fused weight matrix
_fused_weight_cache = {}


def _get_fused_weights(lp_w, rp_w, lg_w, rg_w, og_w):
    key = (id(lp_w), id(rp_w), id(lg_w), id(rg_w), id(og_w))
    if key not in _fused_weight_cache:
        _fused_weight_cache[key] = torch.cat([lp_w, rp_w, lg_w, rg_w, og_w], dim=0)
    return _fused_weight_cache[key]


@triton.jit
def trimul_fused_kernel(
    left_ptr,    # [B, N, N, H] float32
    right_ptr,   # [B, N, N, H] float32
    og_ptr,      # [B, N, N, H] float32 — out_gate (sigmoid applied)
    ton_w_ptr,   # [H] float32
    ton_b_ptr,   # [H] float32
    out_ptr,     # [B, N, N, H] float32
    N, H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Grid: (B*N*N,) — one CTA per (b, i, j) output position.
    Each CTA computes out[b,i,j,:] = LayerNorm(sum_k left[b,i,k,:]*right[b,j,k,:]) * og[b,i,j,:]
    H is constexpr so the H-vector lives in registers.
    Memory layout [B,N,N,H]: stride_b=N*N*H, stride_row=N*H, stride_col=H, stride_h=1.
    """
    pid = tl.program_id(0)
    # Decode (b, i, j) from flat pid
    j   = pid % N
    tmp = pid // N
    i   = tmp % N
    b   = tmp // N

    h_offs = tl.arange(0, H)   # [H] — H is constexpr

    # Base addresses for left[b,i,:,:] and right[b,j,:,:]
    # left[b,i,k,h] = left_ptr + b*N*N*H + i*N*H + k*H + h
    left_base  = left_ptr  + b * (N * N * H) + i * (N * H)
    right_base = right_ptr + b * (N * N * H) + j * (N * H)

    # Accumulator over H
    acc = tl.zeros((H,), dtype=tl.float32)

    # k-loop: accumulate dot product across k dimension
    for k_start in range(0, N, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_offs < N

        # Load left[b,i,k_offs,h_offs]: [BLOCK_K, H]
        left_tile = tl.load(
            left_base + k_offs[:, None] * H + h_offs[None, :],
            mask=k_mask[:, None],
            other=0.0,
        )  # [BLOCK_K, H]

        # Load right[b,j,k_offs,h_offs]: [BLOCK_K, H]
        right_tile = tl.load(
            right_base + k_offs[:, None] * H + h_offs[None, :],
            mask=k_mask[:, None],
            other=0.0,
        )  # [BLOCK_K, H]

        # Elementwise multiply and sum over k: acc[h] += sum_k left[k,h]*right[k,h]
        acc += tl.sum(left_tile * right_tile, axis=0)  # [H]

    # Inline LayerNorm over H
    mean    = tl.sum(acc, axis=0) / H                    # scalar
    diff    = acc - mean                                   # [H]
    var     = tl.sum(diff * diff, axis=0) / H             # scalar
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    ton_w   = tl.load(ton_w_ptr + h_offs)                 # [H]
    ton_b   = tl.load(ton_b_ptr + h_offs)                 # [H]
    acc_ln  = diff * inv_std * ton_w + ton_b              # [H]

    # Load out_gate[b,i,j,:] and apply
    og_base = og_ptr + b * (N * N * H) + i * (N * H) + j * H
    og_vals = tl.load(og_base + h_offs)                   # [H]
    result  = acc_ln * og_vals                             # [H]

    # Store out[b,i,j,:]
    out_base = out_ptr + b * (N * N * H) + i * (N * H) + j * H
    tl.store(out_base + h_offs, result)


def trimul_fused(left, right, out_gate, ton_w, ton_b, N, H):
    """
    left, right, out_gate: [B, N, N, H] float32 (contiguous)
    Returns out: [B, N, N, H] float32
    """
    B = left.shape[0]
    out = torch.empty_like(left)

    grid = (B * N * N,)

    # Choose BLOCK_K as power of 2 >= min(N, 32) but ≤ N
    BLOCK_K = min(triton.next_power_of_2(N), 64)

    trimul_fused_kernel[grid](
        left, right, out_gate,
        ton_w, ton_b, out,
        N, H,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
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
    to_w  = weights['to_out.weight']

    B, N, _, D = input_tensor.shape
    H = config['hidden_dim']

    # LayerNorm
    x = F.layer_norm(input_tensor, (D,), norm_w, norm_b)

    # Fused 5-projection GEMM (cached weight)
    x_flat  = x.reshape(-1, D)
    fused_w = _get_fused_weights(lp_w, rp_w, lg_w, rg_w, og_w)
    fused   = x_flat @ fused_w.t()   # [B*N*N, 5H]

    lp, rp, lg, rg, og = fused.split(H, dim=1)

    left     = (lp * lg.sigmoid()) * mask.reshape(-1, 1)   # [B*N*N, H]
    right    = (rp * rg.sigmoid()) * mask.reshape(-1, 1)   # [B*N*N, H]
    out_gate = og.sigmoid()                                  # [B*N*N, H]

    # Reshape to [B, N, N, H] for Triton kernel
    left_4d     = left.reshape(B, N, N, H)
    right_4d    = right.reshape(B, N, N, H)
    out_gate_4d = out_gate.reshape(B, N, N, H)

    # Fused Triton kernel: contraction + LayerNorm + out_gate
    out = trimul_fused(left_4d, right_4d, out_gate_4d, ton_w, ton_b, N, H)  # [B, N, N, H]

    # Final projection
    out_flat = out.reshape(-1, H) @ to_w.t()
    return out_flat.reshape(B, N, N, D)

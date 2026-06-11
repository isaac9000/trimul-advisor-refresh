"""
TriMul submission — custom Triton kernel for the N×N contraction.
Accepts left/right in [B*N*N, H] flat layout, tiles over (i, j) output
positions and accumulates over k (seqlen) using bf16 tl.dot for tensor cores.
Eliminates all permute/reshape/cast overhead around the bmm.
Rest of pipeline: fused-5H GEMM projection, reduce-overhead compile.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Enable TF32 for projection GEMMs
torch.backends.cuda.matmul.allow_tf32 = True


@triton.jit
def _trimul_contraction_kernel(
    left_ptr,   # [B, N, N, H] float32, strides: (N*N*H, N*H, H, 1)
    right_ptr,  # [B, N, N, H] float32
    out_ptr,    # [B, N, N, H] float32
    B, N, H,
    stride_lb, stride_li, stride_lk, stride_lh,
    stride_rb, stride_rj, stride_rk, stride_rh,
    stride_ob, stride_oi, stride_oj, stride_oh,
    BLOCK_N: tl.constexpr,  # tile size over i and j
    BLOCK_K: tl.constexpr,  # tile size over k (reduction)
    BLOCK_H: tl.constexpr,  # tile size over h
):
    # Grid: (B * cdiv(N, BLOCK_N) * cdiv(N, BLOCK_N), cdiv(H, BLOCK_H))
    pid_bnn = tl.program_id(0)  # flattened (b, i_tile, j_tile)
    pid_h   = tl.program_id(1)  # h tile

    n_tiles = tl.cdiv(N, BLOCK_N)
    b      = pid_bnn // (n_tiles * n_tiles)
    rem    = pid_bnn % (n_tiles * n_tiles)
    i_tile = rem // n_tiles
    j_tile = rem % n_tiles

    i_start = i_tile * BLOCK_N
    j_start = j_tile * BLOCK_N
    h_start = pid_h * BLOCK_H

    # Offsets for i, j, h dimensions
    i_offs = i_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    j_offs = j_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    h_offs = h_start + tl.arange(0, BLOCK_H)  # [BLOCK_H]

    # Masks
    i_mask = i_offs < N
    j_mask = j_offs < N
    h_mask = h_offs < H

    # Accumulator in float32: [BLOCK_N, BLOCK_N, BLOCK_H] — too large for shared mem
    # Instead: loop over h one at a time inside the i,j tile
    # Actually tile over h separately via pid_h, accumulate [BLOCK_N, BLOCK_N] per h_lane

    # For each h in [h_start, h_start+BLOCK_H):
    # out[b, i, j, h] = sum_k left[b, i, k, h] * right[b, j, k, h]
    # This is a dot product over k for each (i,j,h) triple.
    # We tile over k with BLOCK_K, accumulate in fp32.

    # To use tl.dot (tensor cores), we need 2D tiles: [BLOCK_N, BLOCK_K] x [BLOCK_K, BLOCK_N]
    # For a fixed h: left_slice[i, k] and right_slice[j, k] -> out_slice[i, j]

    # Loop over h in the BLOCK_H tile
    for h_idx in tl.static_range(BLOCK_H):
        h = h_start + h_idx
        h_valid = h < H

        # Accumulator for this h: [BLOCK_N, BLOCK_N]
        acc = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)

        # Loop over k tiles
        for k_start in range(0, N, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < N

            # Load left[b, i_offs, k_offs, h]: [BLOCK_N, BLOCK_K]
            left_ptrs = (left_ptr
                         + b * stride_lb
                         + i_offs[:, None] * stride_li
                         + k_offs[None, :] * stride_lk
                         + h * stride_lh)
            left_tile = tl.load(left_ptrs,
                                mask=i_mask[:, None] & k_mask[None, :] & h_valid,
                                other=0.0).to(tl.bfloat16)

            # Load right[b, j_offs, k_offs, h]: [BLOCK_N, BLOCK_K]
            right_ptrs = (right_ptr
                          + b * stride_rb
                          + j_offs[:, None] * stride_rj
                          + k_offs[None, :] * stride_rk
                          + h * stride_rh)
            right_tile = tl.load(right_ptrs,
                                 mask=j_mask[:, None] & k_mask[None, :] & h_valid,
                                 other=0.0).to(tl.bfloat16)

            # Accumulate: acc[i, j] += sum_k left[i,k] * right[j,k]
            # = left_tile @ right_tile.T
            acc += tl.dot(left_tile, tl.trans(right_tile), allow_tf32=False).to(tl.float32)

        # Write out[b, i_offs, j_offs, h]: [BLOCK_N, BLOCK_N]
        out_ptrs = (out_ptr
                    + b * stride_ob
                    + i_offs[:, None] * stride_oi
                    + j_offs[None, :] * stride_oj
                    + h * stride_oh)
        tl.store(out_ptrs, acc, mask=i_mask[:, None] & j_mask[None, :] & h_valid)


def trimul_contraction(left, right, B, N, H):
    """
    left:  [B, N, N, H] float32
    right: [B, N, N, H] float32
    out:   [B, N, N, H] float32
    Computes out[b,i,j,h] = sum_k left[b,i,k,h] * right[b,j,k,h]
    """
    out = torch.empty(B, N, N, H, dtype=torch.float32, device=left.device)

    BLOCK_N = 32
    BLOCK_K = 32
    BLOCK_H = 1  # process one h at a time in the outer loop

    n_tiles = triton.cdiv(N, BLOCK_N)
    h_tiles = triton.cdiv(H, BLOCK_H)
    grid = (B * n_tiles * n_tiles, h_tiles)

    _trimul_contraction_kernel[grid](
        left, right, out,
        B, N, H,
        left.stride(0), left.stride(1), left.stride(2), left.stride(3),
        right.stride(0), right.stride(1), right.stride(2), right.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_H=BLOCK_H,
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

    left  = lp * lg.sigmoid()   # [B*N*N, H]
    right = rp * rg.sigmoid()
    out_gate = og.sigmoid()

    mask_flat = mask.reshape(-1, 1)
    left  = left  * mask_flat
    right = right * mask_flat

    # Reshape to 4D for Triton kernel
    left_4d  = left.reshape(B, N, N, H).contiguous()
    right_4d = right.reshape(B, N, N, H).contiguous()

    # Custom Triton contraction — no permutes, tensor cores via bf16 tl.dot
    out = trimul_contraction(left_4d, right_4d, B, N, H)  # [B, N, N, H]

    out = F.layer_norm(out, (H,), ton_w, ton_b)
    out = out * out_gate.reshape(B, N, N, H)

    out_flat = out.reshape(-1, H) @ to_w.t()
    return out_flat.reshape(B, N, N, D)


# Keep reduce-overhead for the projection + elementwise parts
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

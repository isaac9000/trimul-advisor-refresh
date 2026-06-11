"""
TriMul submission — pure eager PyTorch, zero module overhead.
Fuses projections via stacked GEMM, bf16 einsum for tensor-core contraction.
TF32 enabled. torch.compile(reduce-overhead) for kernel fusion.
Replaces permute+bmm+permute with torch.einsum on 4D bf16 tensors.
"""

import torch
import torch.nn.functional as F

# Enable TF32 for projection GEMMs (legal in our kernel, not the reference)
torch.backends.cuda.matmul.allow_tf32 = True


def _trimul_inner(x, mask, lp_w, rp_w, lg_w, rg_w, og_w,
                  ton_w, ton_b, to_w, B, N, H):
    """Core TriMul computation — compiled for kernel fusion."""
    D = x.shape[-1]
    x_flat = x.reshape(-1, D)

    # Fuse all 5 projections into a single GEMM: [B*N*N, 5*H]
    fused_w = torch.cat([lp_w, rp_w, lg_w, rg_w, og_w], dim=0)  # [5H, D]
    fused = x_flat @ fused_w.t()  # [B*N*N, 5H]

    # Split into individual projections, each [B*N*N, H]
    lp, rp, lg, rg, og = fused.split(H, dim=1)

    # Apply gates
    left  = lp * lg.sigmoid()   # [B*N*N, H]
    right = rp * rg.sigmoid()   # [B*N*N, H]
    out_gate = og.sigmoid()      # [B*N*N, H]

    # Apply mask
    mask_flat = mask.reshape(-1, 1)
    left  = left  * mask_flat
    right = right * mask_flat

    # Contraction: einsum "b i k h, b j k h -> b i j h" (sum over k=seqlen)
    # Use torch.einsum on 4D bf16 tensors — no explicit permutes, tensor cores.
    left_4d  = left.reshape(B, N, N, H).to(torch.bfloat16)
    right_4d = right.reshape(B, N, N, H).to(torch.bfloat16)
    out = torch.einsum('bikh,bjkh->bijh', left_4d, right_4d).to(torch.float32)

    # LayerNorm + out_gate
    out = F.layer_norm(out, (H,), ton_w, ton_b)
    out = out * out_gate.reshape(B, N, N, H)

    # Final projection: [B*N*N, H] @ [H, D] -> [B*N*N, D]
    out_flat = out.reshape(-1, H) @ to_w.t()

    return out_flat.reshape(B, N, N, D)


# Compile once — PyTorch caches compiled kernels per input shape/dtype
_compiled_trimul = torch.compile(_trimul_inner, mode="reduce-overhead", fullgraph=True)


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

    # LayerNorm outside compile (stable, no torch.cat graph issues)
    x = F.layer_norm(input_tensor, (D,), norm_w, norm_b)

    return _compiled_trimul(x, mask, lp_w, rp_w, lg_w, rg_w, og_w,
                            ton_w, ton_b, to_w, B, N, H)

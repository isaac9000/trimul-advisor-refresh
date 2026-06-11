"""
TriMul submission — pure eager PyTorch, zero module overhead.
Separate F.linear GEMMs, bmm in bf16 for tensor cores.
TF32 enabled. torch.compile(max-autotune) for GEMM autotuning + kernel fusion.
LayerNorm inside compiled region for full-graph fusion.
"""

import torch
import torch.nn.functional as F

# Enable TF32 for projection GEMMs (legal in our kernel, not the reference)
torch.backends.cuda.matmul.allow_tf32 = True


def _trimul_inner(input_tensor, mask,
                  norm_w, norm_b,
                  lp_w, rp_w, lg_w, rg_w, og_w,
                  ton_w, ton_b, to_w, B, N, H):
    """Full TriMul computation inside compiled region for complete graph fusion."""
    D = input_tensor.shape[-1]

    # LayerNorm inside compiled region — fuses with downstream ops
    x = F.layer_norm(input_tensor, (D,), norm_w, norm_b)  # [B, N, N, D]

    # Flatten for GEMMs
    x_flat = x.reshape(-1, D)  # [B*N*N, D]

    # Separate F.linear calls — compiler schedules optimal GEMM tiles per shape
    lp = F.linear(x_flat, lp_w)   # [B*N*N, H]
    rp = F.linear(x_flat, rp_w)
    lg = F.linear(x_flat, lg_w)
    rg = F.linear(x_flat, rg_w)
    og = F.linear(x_flat, og_w)

    # Apply gates
    left  = lp * lg.sigmoid()
    right = rp * rg.sigmoid()
    out_gate = og.sigmoid()

    # Apply mask
    mask_flat = mask.reshape(-1, 1)
    left  = left  * mask_flat
    right = right * mask_flat

    # Contraction: einsum "b i k h, b j k h -> b i j h" via [B*H, N, N] bmm
    left_r  = left.reshape(B, N, N, H).permute(0, 3, 1, 2).reshape(B*H, N, N)
    right_r = right.reshape(B, N, N, H).permute(0, 3, 1, 2).reshape(B*H, N, N)

    # bf16 bmm for tensor cores
    out_bmm = torch.bmm(left_r.to(torch.bfloat16),
                        right_r.transpose(1, 2).to(torch.bfloat16)).to(torch.float32)

    # Reshape back to [B, N, N, H]
    out = out_bmm.reshape(B, H, N, N).permute(0, 2, 3, 1).contiguous()

    # LayerNorm + out_gate
    out = F.layer_norm(out, (H,), ton_w, ton_b)
    out = out * out_gate.reshape(B, N, N, H)

    # Final projection
    out_flat = F.linear(out.reshape(-1, H), to_w)

    return out_flat.reshape(B, N, N, D)


# max-autotune: enables cuBLAS/Triton GEMM autotuning per shape
_compiled_trimul = torch.compile(_trimul_inner, mode="max-autotune", fullgraph=True)


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    B, N, _, D = input_tensor.shape
    H = config['hidden_dim']

    return _compiled_trimul(
        input_tensor, mask,
        weights['norm.weight'], weights['norm.bias'],
        weights['left_proj.weight'], weights['right_proj.weight'],
        weights['left_gate.weight'], weights['right_gate.weight'],
        weights['out_gate.weight'],
        weights['to_out_norm.weight'], weights['to_out_norm.bias'],
        weights['to_out.weight'],
        B, N, H,
    )

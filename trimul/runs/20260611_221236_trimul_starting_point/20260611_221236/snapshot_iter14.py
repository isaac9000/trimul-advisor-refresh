"""
TriMul submission — pure eager PyTorch, zero module overhead.
All GEMMs (projection, bmm, output) in bf16 for full tensor-core acceleration.
TF32 enabled. torch.compile(reduce-overhead) for kernel fusion.
"""

import torch
import torch.nn.functional as F

# Enable TF32 for projection GEMMs (legal in our kernel, not the reference)
torch.backends.cuda.matmul.allow_tf32 = True


def _trimul_inner(x, mask, lp_w, rp_w, lg_w, rg_w, og_w,
                  ton_w, ton_b, to_w, B, N, H):
    """Core TriMul computation — compiled for kernel fusion and autotuning."""
    # Flatten spatial dims for batched GEMM: [B*N*N, D]
    D = x.shape[-1]
    x_flat = x.reshape(-1, D)

    # Fuse all 5 projections in bf16 for tensor cores: [B*N*N, 5*H]
    fused_w = torch.cat([lp_w, rp_w, lg_w, rg_w, og_w], dim=0)  # [5H, D]
    fused = (x_flat.to(torch.bfloat16) @ fused_w.to(torch.bfloat16).t()).to(torch.float32)

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

    # Contraction: einsum "b i k h, b j k h -> b i j h" (sum over k=seqlen)
    # Reshape to [B*H, N, N]: permute H before N dims
    left_4d  = left.reshape(B, N, N, H)
    right_4d = right.reshape(B, N, N, H)
    left_r  = left_4d.permute(0, 3, 1, 2).reshape(B*H, N, N)
    right_r = right_4d.permute(0, 3, 1, 2).reshape(B*H, N, N)

    # bf16 bmm for tensor cores: [B*H, N, N] @ [B*H, N, N]^T -> [B*H, N, N]
    out_bmm = torch.bmm(left_r.to(torch.bfloat16),
                        right_r.transpose(1, 2).to(torch.bfloat16)).to(torch.float32)

    # Reshape back to [B, N, N, H]
    out = out_bmm.reshape(B, H, N, N).permute(0, 2, 3, 1).contiguous()

    # LayerNorm + out_gate
    out = F.layer_norm(out, (H,), ton_w, ton_b)
    out = out * out_gate.reshape(B, N, N, H)

    # Final projection: [B*N*N, H] @ [H, D] -> [B*N*N, D]  (fp32 for accuracy)
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

    # LayerNorm outside compile (norm params are small; could be inside too)
    x = F.layer_norm(input_tensor, (D,), norm_w, norm_b)

    return _compiled_trimul(x, mask, lp_w, rp_w, lg_w, rg_w, og_w,
                            ton_w, ton_b, to_w, B, N, H)

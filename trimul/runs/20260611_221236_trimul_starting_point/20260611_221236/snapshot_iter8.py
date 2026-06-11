"""
TriMul submission — pure eager PyTorch, zero module overhead.
Split projections into 2H + 3H matmuls (eliminates full 5H torch.cat).
LayerNorm inside compiled region. B/N/H derived from tensor shapes inside.
bf16 bmm for tensor cores. TF32 enabled. torch.compile(reduce-overhead).
"""

import torch
import torch.nn.functional as F

# Enable TF32 for projection GEMMs (legal in our kernel, not the reference)
torch.backends.cuda.matmul.allow_tf32 = True


def _trimul_inner(input_tensor, mask,
                  norm_w, norm_b,
                  lp_w, rp_w, lg_w, rg_w, og_w,
                  ton_w, ton_b, to_w):
    """Full TriMul computation — shapes derived internally, no Python int args."""
    B, N, _, D = input_tensor.shape
    H = lp_w.shape[0]  # hidden_dim derived from weight shape

    # LayerNorm inside compiled region
    x = F.layer_norm(input_tensor, (D,), norm_w, norm_b)  # [B, N, N, D]
    x_flat = x.reshape(-1, D)  # [B*N*N, D]

    # Two smaller matmuls instead of one big 5H cat:
    # val_w: [2H, D] for lp, rp — the "value" projections
    val_w = torch.cat([lp_w, rp_w], dim=0)       # [2H, D]
    gate_w = torch.cat([lg_w, rg_w, og_w], dim=0) # [3H, D]

    val  = x_flat @ val_w.t()   # [B*N*N, 2H]
    gate = x_flat @ gate_w.t()  # [B*N*N, 3H]

    lp, rp = val.split(H, dim=1)
    lg, rg, og = gate.split(H, dim=1)

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
    out_flat = out.reshape(-1, H) @ to_w.t()

    return out_flat.reshape(B, N, N, D)


# reduce-overhead: CUDA graphs + kernel fusion, no autotuning bleed
_compiled_trimul = torch.compile(_trimul_inner, mode="reduce-overhead", fullgraph=True)


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    return _compiled_trimul(
        input_tensor, mask,
        weights['norm.weight'], weights['norm.bias'],
        weights['left_proj.weight'], weights['right_proj.weight'],
        weights['left_gate.weight'], weights['right_gate.weight'],
        weights['out_gate.weight'],
        weights['to_out_norm.weight'], weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )

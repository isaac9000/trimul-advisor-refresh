"""
TriMul submission — fp32 projection GEMMs, bf16 bmm contraction, cached weights.
Projection GEMMs stay fp32 (TF32 enabled) for numerical stability.
The bmm contraction uses bf16 for tensor cores.
Caches both fused_w (fp32) and to_w_t (transposed, contiguous fp32) to avoid
repeated torch.cat and .t() view creation.
"""

import torch
import torch.nn.functional as F

# Enable TF32 for projection GEMMs
torch.backends.cuda.matmul.allow_tf32 = True

# Cache: stores (fused_w_fp32, to_w_t_fp32) keyed by weight object ids
_weight_cache = {}


def _get_cached_weights(lp_w, rp_w, lg_w, rg_w, og_w, to_w):
    """Cache fused_w and pre-transposed to_w to avoid allocations per call."""
    key = (id(lp_w), id(rp_w), id(lg_w), id(rg_w), id(og_w), id(to_w))
    if key not in _weight_cache:
        fused_w = torch.cat([lp_w, rp_w, lg_w, rg_w, og_w], dim=0)  # [5H, D] fp32
        to_w_t  = to_w.t().contiguous()                                # [H, D] fp32 contiguous
        _weight_cache[key] = (fused_w, to_w_t)
    return _weight_cache[key]


def _trimul_inner(x, mask, fused_w, to_w_t, ton_w, ton_b, B, N, H):
    """Core TriMul — fp32 GEMMs, bf16 bmm, cached weights."""
    D = x.shape[-1]
    x_flat = x.reshape(-1, D)

    # Fused 5-projection GEMM in fp32 (TF32 enabled): [B*N*N, 5H]
    fused = x_flat @ fused_w.t()

    lp, rp, lg, rg, og = fused.split(H, dim=1)

    left     = lp * lg.sigmoid()   # [B*N*N, H]
    right    = rp * rg.sigmoid()   # [B*N*N, H]
    out_gate = og.sigmoid()         # [B*N*N, H]

    mask_flat = mask.reshape(-1, 1)
    left  = left  * mask_flat
    right = right * mask_flat

    # Contraction: permute to [B*H, N, N] and bf16 bmm
    left_4d  = left.reshape(B, N, N, H)
    right_4d = right.reshape(B, N, N, H)
    left_r   = left_4d.permute(0, 3, 1, 2).reshape(B * H, N, N)
    right_r  = right_4d.permute(0, 3, 1, 2).reshape(B * H, N, N)

    out_bmm = torch.bmm(left_r.to(torch.bfloat16),
                        right_r.transpose(1, 2).to(torch.bfloat16)).to(torch.float32)

    out = out_bmm.reshape(B, H, N, N).permute(0, 2, 3, 1).contiguous()

    # LayerNorm + out_gate
    out = F.layer_norm(out, (H,), ton_w, ton_b)
    out = out * out_gate.reshape(B, N, N, H)

    # Final projection using pre-transposed weight: [B*N*N, H] @ [H, D]
    out_flat = out.reshape(-1, H) @ to_w_t

    return out_flat.reshape(B, N, N, D)


# Compile for kernel fusion
_compiled_trimul = torch.compile(_trimul_inner, mode="reduce-overhead", fullgraph=True)


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    norm_w = weights['norm.weight']
    norm_b = weights['norm.bias']
    lp_w   = weights['left_proj.weight']
    rp_w   = weights['right_proj.weight']
    lg_w   = weights['left_gate.weight']
    rg_w   = weights['right_gate.weight']
    og_w   = weights['out_gate.weight']
    ton_w  = weights['to_out_norm.weight']
    ton_b  = weights['to_out_norm.bias']
    to_w   = weights['to_out.weight']

    B, N, _, D = input_tensor.shape
    H = config['hidden_dim']

    # Input LayerNorm in fp32
    x = F.layer_norm(input_tensor, (D,), norm_w, norm_b)

    # Get cached weights (fused_w fp32, to_w_t pre-transposed contiguous)
    fused_w, to_w_t = _get_cached_weights(lp_w, rp_w, lg_w, rg_w, og_w, to_w)

    return _compiled_trimul(x, mask, fused_w, to_w_t, ton_w, ton_b, B, N, H)

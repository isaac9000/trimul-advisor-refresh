"""
TriMul submission — closure-based compiled functions to minimize CUDA graph arguments.
Weights captured as closure constants; compiled function only sees (x, mask, B, N, H).
Fewer dynamic tensor arguments → fewer guard checks → faster CUDA graph replay.
"""

import torch
import torch.nn.functional as F

# Enable TF32 for projection GEMMs
torch.backends.cuda.matmul.allow_tf32 = True

# Cache: weight tensors → compiled closure function
_weight_cache    = {}   # key → (fused_w, to_w_t)
_compiled_cache  = {}   # key → compiled closure fn


def _get_cached_weights(lp_w, rp_w, lg_w, rg_w, og_w, to_w):
    key = (id(lp_w), id(rp_w), id(lg_w), id(rg_w), id(og_w), id(to_w))
    if key not in _weight_cache:
        fused_w = torch.cat([lp_w, rp_w, lg_w, rg_w, og_w], dim=0)  # [5H, D]
        to_w_t  = to_w.t().contiguous()                                # [H, D]
        _weight_cache[key] = (fused_w, to_w_t)
    return _weight_cache[key]


def _make_compiled_fn(fused_w, to_w_t, ton_w, ton_b):
    """Create a compiled function that closes over constant weight tensors."""

    def _inner(x, mask, B, N, H):
        D = x.shape[-1]
        x_flat = x.reshape(-1, D)

        # Fused 5-projection GEMM (weights are closure constants)
        fused = x_flat @ fused_w.t()

        lp, rp, lg, rg, og = fused.split(H, dim=1)

        left     = lp * lg.sigmoid()
        right    = rp * rg.sigmoid()
        out_gate = og.sigmoid()

        mask_flat = mask.reshape(-1, 1)
        left  = left  * mask_flat
        right = right * mask_flat

        left_4d  = left.reshape(B, N, N, H)
        right_4d = right.reshape(B, N, N, H)
        left_r   = left_4d.permute(0, 3, 1, 2).reshape(B * H, N, N)
        right_r  = right_4d.permute(0, 3, 1, 2).reshape(B * H, N, N)

        out_bmm = torch.bmm(left_r.to(torch.bfloat16),
                            right_r.transpose(1, 2).to(torch.bfloat16)).to(torch.float32)

        out = out_bmm.reshape(B, H, N, N).permute(0, 2, 3, 1).contiguous()

        out = F.layer_norm(out, (H,), ton_w, ton_b)
        out = out * out_gate.reshape(B, N, N, H)

        out_flat = out.reshape(-1, H) @ to_w_t
        return out_flat.reshape(B, N, N, D)

    return torch.compile(_inner, mode="reduce-overhead", fullgraph=True)


def _get_compiled_fn(fused_w, to_w_t, ton_w, ton_b):
    key = (id(fused_w), id(to_w_t), id(ton_w), id(ton_b))
    if key not in _compiled_cache:
        _compiled_cache[key] = _make_compiled_fn(fused_w, to_w_t, ton_w, ton_b)
    return _compiled_cache[key]


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

    # Input LayerNorm outside compiled region (keeps D as a dynamic arg)
    x = F.layer_norm(input_tensor, (D,), norm_w, norm_b)

    # Get cached weights and per-weight compiled function
    fused_w, to_w_t = _get_cached_weights(lp_w, rp_w, lg_w, rg_w, og_w, to_w)
    compiled_fn = _get_compiled_fn(fused_w, to_w_t, ton_w, ton_b)

    return compiled_fn(x, mask, B, N, H)

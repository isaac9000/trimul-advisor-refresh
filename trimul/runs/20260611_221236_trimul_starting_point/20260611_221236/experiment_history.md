# Experiment History

Tracks every kernel attempt, its code, hypothesis, and result.

---

## Experiment #1 — 2026-06-11 22:13:06 UTC ✅ KEEP

**Hypothesis:** Baseline 'starting_point' — initial benchmark

**Result:** 11010.26 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #2 — 2026-06-11 22:14:23 UTC ✅ KEEP

**Hypothesis:** Advisor proposal: eliminate nn.Module reconstruction overhead, fuse all 5 projections into a single stacked GEMM (cat weights → one matmul → split), replace einsum with torch.bmm by reshaping [B,N,N,H] → [B*H,N,N] and doing bmm(left, right^T), and remove bfloat16 cast (keep float32 throughout). Result is ~6% faster than baseline (10350 µs vs 11010 µs baseline).

**Result:** 10349.94 μs

**Kernel code:**
```python
"""
TriMul submission — pure eager PyTorch, zero module overhead.
Fuses projections via stacked GEMM, replaces einsum with torch.bmm.
"""

import torch
import torch.nn.functional as F


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    # Unpack weights (all float32)
    norm_w = weights['norm.weight']
    norm_b = weights['norm.bias']
    lp_w = weights['left_proj.weight']    # [H, D]
    rp_w = weights['right_proj.weight']   # [H, D]
    lg_w = weights['left_gate.weight']    # [H, D]
    rg_w = weights['right_gate.weight']   # [H, D]
    og_w = weights['out_gate.weight']     # [H, D]
    ton_w = weights['to_out_norm.weight']
    ton_b = weights['to_out_norm.bias']
    to_w = weights['to_out.weight']       # [D, H]

    B, N, _, D = input_tensor.shape
    H = config['hidden_dim']

    # LayerNorm input
    x = F.layer_norm(input_tensor, (D,), norm_w, norm_b)  # [B, N, N, D]

    # Flatten spatial dims for batched GEMM: [B*N*N, D]
    x_flat = x.reshape(-1, D)

    # Fuse all 5 projections into a single GEMM: [B*N*N, 5*H]
    # Stack weights: [5*H, D]
    fused_w = torch.cat([lp_w, rp_w, lg_w, rg_w, og_w], dim=0)  # [5H, D]
    fused = x_flat @ fused_w.t()  # [B*N*N, 5H]

    # Split into individual projections
    lp, rp, lg, rg, og = fused.split(H, dim=1)  # each [B*N*N, H]

    # Apply gates
    left  = lp * lg.sigmoid()   # [B*N*N, H]
    right = rp * rg.sigmoid()   # [B*N*N, H]
    out_gate = og.sigmoid()      # [B*N*N, H]

    # Apply mask: [B, N, N] -> [B*N*N, 1]
    mask_flat = mask.reshape(-1, 1)
    left  = left  * mask_flat   # [B*N*N, H]
    right = right * mask_flat   # [B*N*N, H]

    # Reshape for bmm: einsum "b i k h, b j k h -> b i j h"
    # Treat each (b, h) slice: left[b, i, :, h] @ right[b, j, :, h]
    # Reshape: left -> [B, N, N, H] -> [B*H, N, N], right -> [B*H, N, N]
    # Then out[b,h,i,j] = sum_k left[b,h,i,k] * right[b,h,j,k]
    #                    = left[b,h,i,:] @ right[b,h,j,:]^T
    # So: out = bmm(left_r, right_r.transpose(-1,-2))  -> [B*H, N, N]
    left_r  = left.reshape(B, N, N, H).permute(0, 3, 1, 2).reshape(B*H, N, N)   # [B*H, N, N]
    right_r = right.reshape(B, N, N, H).permute(0, 3, 1, 2).reshape(B*H, N, N)  # [B*H, N, N]

    out_bmm = torch.bmm(left_r, right_r.transpose(1, 2))  # [B*H, N, N]

    # Reshape back to [B, N, N, H]
    out = out_bmm.reshape(B, H, N, N).permute(0, 2, 3, 1)  # [B, N, N, H]

    # LayerNorm + out_gate
    out = F.layer_norm(out, (H,), ton_w, ton_b)
    out = out * out_gate.reshape(B, N, N, H)

    # Final projection: [B*N*N, H] @ [H, D] -> [B*N*N, D]
    out_flat = out.reshape(-1, H) @ to_w.t()  # [B*N*N, D]

    return out_flat.reshape(B, N, N, D)

```

---

## Experiment #3 — 2026-06-11 22:16:28 UTC ✅ KEEP

**Hypothesis:** Advisor proposal: use bf16 for the bmm contraction to enable H100 tensor cores, enable TF32 for projection GEMMs. Kept [B*H, N, N] bmm layout (permute required) but cast left/right to bfloat16 before bmm and back to float32 after. TF32 enabled globally via torch.backends.cuda.matmul.allow_tf32=True. Result: 7071 µs vs 10350 µs prior best — 32% improvement from tensor core acceleration on the dominant contraction step.

**Result:** 7070.93 μs

**Kernel code:**
```python
"""
TriMul submission — pure eager PyTorch, zero module overhead.
Fuses projections via stacked GEMM, bmm in bf16 for tensor cores.
TF32 enabled for projection GEMMs.
"""

import torch
import torch.nn.functional as F

# Enable TF32 for projection GEMMs (legal in our kernel, not the reference)
torch.backends.cuda.matmul.allow_tf32 = True


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    # Unpack weights (all float32)
    norm_w = weights['norm.weight']
    norm_b = weights['norm.bias']
    lp_w = weights['left_proj.weight']    # [H, D]
    rp_w = weights['right_proj.weight']   # [H, D]
    lg_w = weights['left_gate.weight']    # [H, D]
    rg_w = weights['right_gate.weight']   # [H, D]
    og_w = weights['out_gate.weight']     # [H, D]
    ton_w = weights['to_out_norm.weight']
    ton_b = weights['to_out_norm.bias']
    to_w = weights['to_out.weight']       # [D, H]

    B, N, _, D = input_tensor.shape
    H = config['hidden_dim']

    # LayerNorm input
    x = F.layer_norm(input_tensor, (D,), norm_w, norm_b)  # [B, N, N, D]

    # Flatten spatial dims for batched GEMM: [B*N*N, D]
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

    # Apply mask: [B, N, N] -> [B*N*N, 1]
    mask_flat = mask.reshape(-1, 1)
    left  = left  * mask_flat   # [B*N*N, H]
    right = right * mask_flat   # [B*N*N, H]

    # Contraction: einsum "b i k h, b j k h -> b i j h" (sum over k=seqlen)
    # Reshape to [B*H, N, N]: permute H before N dims for contiguous bmm
    left_4d  = left.reshape(B, N, N, H)
    right_4d = right.reshape(B, N, N, H)
    # [B, H, N, N] -> [B*H, N, N]
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

    # Final projection: [B*N*N, H] @ [H, D] -> [B*N*N, D]
    out_flat = out.reshape(-1, H) @ to_w.t()

    return out_flat.reshape(B, N, N, D)

```

---

## Experiment #4 — 2026-06-11 22:18:14 UTC 💥 CRASH

**Hypothesis:** Advisor proposal: cache fused weight matrix (torch.cat of 5 weight matrices) keyed by data_ptr+shape to eliminate per-call concatenation overhead. Also pre-transpose fused_w for direct matmul. Implementation had cache correctness issues — the cache was returning stale/wrong weights for some test cases (max_diff=3.5 on seqlen=1024 dim=768 mask cases), likely due to GPU memory reuse causing data_ptr+shape collisions or stale cache entries from different test configurations. Logged as crash.

**Result:** CRASH

**Error:**
```
Correctness check failed: seqlen=1024 bs=1 dim=768 mask normal max_diff=3.497386, seqlen=1024 bs=1 dim=768 mask cauchy max_diff=2.357573. Cache key collision (data_ptr+shape) returning wrong fused weights across test configurations.
```

**Kernel code:**
```python
"""
TriMul submission — pure eager PyTorch, zero module overhead.
Fuses projections via stacked GEMM, bmm in bf16 for tensor cores.
TF32 enabled for projection GEMMs. Caches fused weight matrix to avoid
per-call torch.cat overhead.
"""

import torch
import torch.nn.functional as F

# Enable TF32 for projection GEMMs (legal in our kernel, not the reference)
torch.backends.cuda.matmul.allow_tf32 = True

# Cache for fused weight matrices, keyed by data pointer of lp_w
_weight_cache = {}


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    # Unpack weights (all float32)
    norm_w = weights['norm.weight']
    norm_b = weights['norm.bias']
    lp_w = weights['left_proj.weight']    # [H, D]
    rp_w = weights['right_proj.weight']   # [H, D]
    lg_w = weights['left_gate.weight']    # [H, D]
    rg_w = weights['right_gate.weight']   # [H, D]
    og_w = weights['out_gate.weight']     # [H, D]
    ton_w = weights['to_out_norm.weight']
    ton_b = weights['to_out_norm.bias']
    to_w = weights['to_out.weight']       # [D, H]

    B, N, _, D = input_tensor.shape
    H = config['hidden_dim']

    # Cache fused weight [5H, D] and its transpose to avoid per-call torch.cat
    # Key includes data ptr + shape to handle memory reuse across different configs
    cache_key = (lp_w.data_ptr(), lp_w.shape)
    if cache_key not in _weight_cache:
        fused_w = torch.cat([lp_w, rp_w, lg_w, rg_w, og_w], dim=0).contiguous()  # [5H, D]
        fused_wt = fused_w.t().contiguous()  # [D, 5H]
        _weight_cache[cache_key] = (fused_wt,)
    fused_wt, = _weight_cache[cache_key]

    # LayerNorm input
    x = F.layer_norm(input_tensor, (D,), norm_w, norm_b)  # [B, N, N, D]

    # Flatten spatial dims for batched GEMM: [B*N*N, D]
    x_flat = x.reshape(-1, D)

    # Fuse all 5 projections into a single GEMM: [B*N*N, 5*H]
    fused = x_flat @ fused_wt  # [B*N*N, 5H]

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
    # Reshape to [B*H, N, N]: permute H before N dims for contiguous bmm
    left_4d  = left.reshape(B, N, N, H)
    right_4d = right.reshape(B, N, N, H)
    # [B, H, N, N] -> [B*H, N, N]
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

    # Final projection: [B*N*N, H] @ [H, D] -> [B*N*N, D]
    out_flat = out.reshape(-1, H) @ to_w.t()

    return out_flat.reshape(B, N, N, D)

```

---

## Experiment #5 — 2026-06-11 22:19:20 UTC 💥 CRASH

**Hypothesis:** Advisor proposal: retry weight caching using id(lp_w) as cache key (Python object identity) instead of data_ptr(). Pre-transpose fused weights and to_out weight into cache. Implementation failed: CPython object id() is recycled after garbage collection — the eval harness creates new weight tensors for each test case, and after the old ones are GC'd, new tensors can receive the same id(). This causes cache poisoning identical to the data_ptr() issue. The weight caching approach is fundamentally unsafe in this eval harness.

**Result:** CRASH

**Error:**
```
Cache poisoning via id() recycling: RuntimeError mat1/mat2 shape mismatches (e.g. 65536x128 @ 768x640) and max_diff up to 5.56. CPython recycles object ids after GC, so id(lp_w) cannot be used as a stable cache key across test invocations.
```

**Kernel code:**
```python
"""
TriMul submission — pure eager PyTorch, zero module overhead.
Fuses projections via stacked GEMM, bmm in bf16 for tensor cores.
TF32 enabled for projection GEMMs.
Weight cache keyed by Python id() of lp_w tensor object (stable, no aliasing).
"""

import torch
import torch.nn.functional as F

# Enable TF32 for projection GEMMs (legal in our kernel, not the reference)
torch.backends.cuda.matmul.allow_tf32 = True

# Cache: id(lp_w) -> (fused_wt_fp32, to_wt_fp32)
# Python id() is stable for the lifetime of the tensor object and won't alias
_weight_cache = {}


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    # Unpack weights
    norm_w = weights['norm.weight']
    norm_b = weights['norm.bias']
    lp_w = weights['left_proj.weight']    # [H, D]
    ton_w = weights['to_out_norm.weight']
    ton_b = weights['to_out_norm.bias']

    B, N, _, D = input_tensor.shape
    H = config['hidden_dim']

    # Cache fused weight transpose and to_out weight transpose
    cache_key = id(lp_w)
    if cache_key not in _weight_cache:
        rp_w = weights['right_proj.weight']
        lg_w = weights['left_gate.weight']
        rg_w = weights['right_gate.weight']
        og_w = weights['out_gate.weight']
        to_w = weights['to_out.weight']
        # [5H, D] -> transpose to [D, 5H] for direct x @ fused_wt
        fused_wt = torch.cat([lp_w, rp_w, lg_w, rg_w, og_w], dim=0).t().contiguous()
        # [D, H] for direct out @ to_wt
        to_wt = to_w.t().contiguous()
        _weight_cache[cache_key] = (fused_wt, to_wt)

    fused_wt, to_wt = _weight_cache[cache_key]

    # LayerNorm input
    x = F.layer_norm(input_tensor, (D,), norm_w, norm_b)  # [B, N, N, D]

    # Flatten spatial dims for batched GEMM: [B*N*N, D]
    x_flat = x.reshape(-1, D)

    # Fuse all 5 projections: [B*N*N, D] @ [D, 5H] -> [B*N*N, 5H]
    fused = x_flat @ fused_wt

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
    # Reshape to [B*H, N, N]: permute H before N dims for contiguous bmm
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

    # Final projection: [B*N*N, H] @ [H, D] -> [B*N*N, D]
    out_flat = out.reshape(-1, H) @ to_wt

    return out_flat.reshape(B, N, N, D)

```

---

## Experiment #6 — 2026-06-11 22:20:55 UTC ✅ KEEP

**Hypothesis:** Advisor proposal: wrap core computation in torch.compile(mode='reduce-overhead', fullgraph=True). Extracted inner function _trimul_inner taking individual tensor args (no dict) to enable clean graph tracing. LayerNorm kept outside compiled region. Compilation is keyed per input shape/dtype so each benchmark shape gets its own optimized kernel. Result: 6096 µs vs 7071 µs prior best — 14% improvement from kernel fusion and reduced Python dispatch overhead.

**Result:** 6095.95 μs

**Kernel code:**
```python
"""
TriMul submission — pure eager PyTorch, zero module overhead.
Fuses projections via stacked GEMM, bmm in bf16 for tensor cores.
TF32 enabled for projection GEMMs. torch.compile for kernel fusion.
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

    # Fuse all 5 projections into a single GEMM: [B*N*N, 5*H]
    fused_w = torch.cat([lp_w, rp_w, lg_w, rg_w, og_w], dim=0)  # [5H, D]
    fused = x_flat @ fused_w.t()  # [B*N*N, 5H]

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

    # LayerNorm outside compile (norm params are small; could be inside too)
    x = F.layer_norm(input_tensor, (D,), norm_w, norm_b)

    return _compiled_trimul(x, mask, lp_w, rp_w, lg_w, rg_w, og_w,
                            ton_w, ton_b, to_w, B, N, H)

```

---

## Experiment #7 — 2026-06-11 22:24:01 UTC ❌ DISCARD

**Hypothesis:** Advisor proposal: switch from reduce-overhead to max-autotune, move LayerNorm inside compiled region, replace torch.cat+single GEMM with 5 separate F.linear calls. Result: 13491 µs — much worse than 6096 µs best. The max-autotune mode likely incurs repeated autotuning overhead during timed benchmark runs (large stddev ~2300 µs and huge seqlen=1024 times of 62-65ms suggest re-compilation). The 5 separate F.linear calls also hurt vs the fused single GEMM. This is a discard.

**Result:** 13490.71 μs

**Kernel code:**
```python
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

```

---

## Experiment #8 — 2026-06-11 22:25:45 UTC ❌ DISCARD

**Hypothesis:** Advisor proposal: restore reduce-overhead, move LayerNorm inside compiled region, derive B/N/H from tensor shapes inside (no Python int args), split 5H fused GEMM into 2H+3H torch.cat matmuls. Result: 7694 µs — worse than 6096 µs best. High variance (±2700 µs) suggests torch.cat inside CUDA graph capture is causing graph invalidations or extra re-captures. The torch.cat of weight matrices inside the compiled region prevents stable CUDA graph reuse. Discard.

**Result:** 7693.57 μs

**Kernel code:**
```python
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

```

---

## Experiment #9 — 2026-06-11 22:27:42 UTC ❌ DISCARD

**Hypothesis:** Advisor proposal: replace permute+bmm+permute contraction with torch.einsum('bikh,bjkh->bijh') on 4D bf16 tensors, keeping #6 structure otherwise (reduce-overhead, fused 5H GEMM, LN outside). Result: 7473 µs — worse than 6096 µs best. torch.einsum on the 4D layout is slower than the explicit [B*H,N,N] bmm path, likely because cuBLAS dispatches it less efficiently. High variance (±3000 µs) persists from torch.cat inside CUDA graph. Discard.

**Result:** 7473.20 μs

**Kernel code:**
```python
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

```

---

## Experiment #10 — 2026-06-11 22:30:40 UTC ❌ DISCARD

**Hypothesis:** Custom Triton kernel for TriMul contraction: grid (B * cdiv(N,32)^2, H) with each program computing one (b, i_tile, j_tile, h) output tile using bf16 tl.dot over k. Kernel is correct (18/18 pass) but very slow at 35733 µs — kernel launches B*n_tiles^2*H programs with a scalar h loop inside, causing massive launch overhead (184ms at seqlen=1024 N=128). The BLOCK_H=1 design creates too many grid blocks. Need to tile H into larger blocks in the grid. Discard.

**Result:** 35733.22 μs

**Kernel code:**
```python
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

```

---

## Experiment #11 — 2026-06-11 22:33:36 UTC ❌ DISCARD

**Hypothesis:** Redesigned Triton contraction kernel: grid (B*H, cdiv(N,BLOCK_N), cdiv(N,BLOCK_N)), each program handles one (b,h) slice and one (i_tile,j_tile) output tile as standard GEMM with BLOCK_N=32, BLOCK_K=32 bf16 tl.dot. Inputs permuted to [B*H,N,N] before launch (same permute as original bmm). Result: 6234 µs — slightly worse than #6 (6096 µs). The Triton kernel is correct and near-competitive but not yet faster than torch.bmm+cuBLAS for these shapes. Permute overhead is unchanged; Triton kernel itself slightly underperforms cuBLAS at BLOCK_N=32. Discard.

**Result:** 6234.40 μs

**Kernel code:**
```python
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

```

---

## Experiment #12 — 2026-06-11 22:36:31 UTC ❌ DISCARD

**Hypothesis:** Eliminate permute copies by reading [B,N,N,H] layout directly in Triton kernel with native strides (sl_k=H, sl_h=1). Grid (B*H, n_tiles, n_tiles), b=bh//H, h=bh%H. Result: 22188 µs — much worse than #6 and even #11. The strided access with sl_k=H (e.g. 128) causes badly uncoalesced memory reads: each BLOCK_K=32 column tile gathers with stride 128, so 32 threads access 32*128=4096 elements but only 32 are used — catastrophic bandwidth waste. Permute overhead was actually cheaper than non-coalesced reads. Discard.

**Result:** 22188.10 μs

**Kernel code:**
```python
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

```

---

## Experiment #13 — 2026-06-11 22:38:57 UTC 💥 CRASH

**Hypothesis:** Advisor proposal: add dynamic=True to torch.compile(reduce-overhead, fullgraph=True) to get a single compiled kernel across all 7 benchmark shapes instead of per-shape CUDA graph capture. Result: crash — torch._dynamo.exc.RecompileLimitExceeded: cache_size_limit reached. The 7 different shapes (B, N, D, H variations) exceed torch.compile's default recompile cache limit when dynamic=True, causing the kernel to fail entirely on later shapes.

**Result:** CRASH

**Error:**
```
torch._dynamo.exc.RecompileLimitExceeded: cache_size_limit reached. dynamic=True with reduce-overhead causes too many recompilations across the 7 different benchmark shapes (different B, N, D, H).
```

**Kernel code:**
```python
"""
TriMul submission — pure eager PyTorch, zero module overhead.
Fuses projections via stacked GEMM, bmm in bf16 for tensor cores.
TF32 enabled. torch.compile(reduce-overhead, dynamic=True) — single compiled
kernel across all shapes, avoids per-shape CUDA graph re-capture overhead.
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

    # Contraction: einsum "b i k h, b j k h -> b i j h" via [B*H, N, N] bmm
    left_4d  = left.reshape(B, N, N, H)
    right_4d = right.reshape(B, N, N, H)
    left_r  = left_4d.permute(0, 3, 1, 2).reshape(B*H, N, N)
    right_r = right_4d.permute(0, 3, 1, 2).reshape(B*H, N, N)

    # bf16 bmm for tensor cores
    out_bmm = torch.bmm(left_r.to(torch.bfloat16),
                        right_r.transpose(1, 2).to(torch.bfloat16)).to(torch.float32)

    # Reshape back to [B, N, N, H]
    out = out_bmm.reshape(B, H, N, N).permute(0, 2, 3, 1).contiguous()

    # LayerNorm + out_gate
    out = F.layer_norm(out, (H,), ton_w, ton_b)
    out = out * out_gate.reshape(B, N, N, H)

    # Final projection: [B*N*N, H] @ [H, D] -> [B*N*N, D]
    out_flat = out.reshape(-1, H) @ to_w.t()

    return out_flat.reshape(B, N, N, D)


# dynamic=True: single compiled kernel across all shapes, no per-shape re-capture
_compiled_trimul = torch.compile(_trimul_inner, mode="reduce-overhead",
                                  fullgraph=True, dynamic=True)


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

    # LayerNorm outside compile (stable)
    x = F.layer_norm(input_tensor, (D,), norm_w, norm_b)

    return _compiled_trimul(x, mask, lp_w, rp_w, lg_w, rg_w, og_w,
                            ton_w, ton_b, to_w, B, N, H)

```

---

## Experiment #14 — 2026-06-11 22:41:02 UTC ❌ DISCARD

**Hypothesis:** Advisor proposal: run fused-5H projection GEMM in bf16 (x_flat.to(bf16) @ fused_w.to(bf16).t()) for tensor-core acceleration on projections, keep output projection in fp32 (bf16 output failed correctness on cauchy inputs with max_diff=0.032 > tolerance 0.02). Result: 6814 µs — worse than #6's 6096 µs. bf16 projection GEMM is slower likely because: (1) explicit .to(bf16) casts add overhead inside CUDA graph, (2) TF32 was already providing near-tensor-core speeds for the projection, (3) high variance (±2800 µs) from CUDA graph instability. Discard.

**Result:** 6814.14 μs

**Kernel code:**
```python
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

```

---

## Experiment #15 — 2026-06-11 22:42:43 UTC ❌ DISCARD

**Hypothesis:** Advisor proposal: remove B/N/H Python int args from _trimul_inner, derive them from tensor shapes inside (B,N,_,D=x.shape; H=lp_w.shape[0]). Intended to reduce per-shape specialization. Result: 7261 µs — worse than #6 (6096 µs). The Python int args in #6 actually help reduce-overhead by providing static values for CUDA graph capture. Without them, torch.compile sees more dynamic shapes and generates less stable/more graphs, increasing variance (±3300 µs). Discard — #6 structure with explicit B/N/H int args is optimal for reduce-overhead.

**Result:** 7261.37 μs

**Kernel code:**
```python
"""
TriMul submission — pure eager PyTorch, zero module overhead.
Fuses projections via stacked GEMM, bmm in bf16 for tensor cores.
TF32 enabled. torch.compile(reduce-overhead) for kernel fusion.
B/N/H derived from tensor shapes inside compiled function (no Python int args)
to avoid per-shape specialization and reduce CUDA graph re-captures.
"""

import torch
import torch.nn.functional as F

# Enable TF32 for projection GEMMs (legal in our kernel, not the reference)
torch.backends.cuda.matmul.allow_tf32 = True


def _trimul_inner(x, mask, lp_w, rp_w, lg_w, rg_w, og_w,
                  ton_w, ton_b, to_w):
    """Core TriMul computation — B/N/H derived from tensor shapes, no int args."""
    B, N, _, D = x.shape   # x is [B, N, N, D] after LN
    H = lp_w.shape[0]       # hidden_dim from weight shape

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

    # Apply mask: [B, N, N] -> [B*N*N, 1]
    mask_flat = mask.reshape(-1, 1)
    left  = left  * mask_flat
    right = right * mask_flat

    # Contraction: einsum "b i k h, b j k h -> b i j h" via [B*H, N, N] bmm
    left_4d  = left.reshape(B, N, N, H)
    right_4d = right.reshape(B, N, N, H)
    left_r  = left_4d.permute(0, 3, 1, 2).reshape(B*H, N, N)
    right_r = right_4d.permute(0, 3, 1, 2).reshape(B*H, N, N)

    # bf16 bmm for tensor cores
    out_bmm = torch.bmm(left_r.to(torch.bfloat16),
                        right_r.transpose(1, 2).to(torch.bfloat16)).to(torch.float32)

    # Reshape back to [B, N, N, H]
    out = out_bmm.reshape(B, H, N, N).permute(0, 2, 3, 1).contiguous()

    # LayerNorm + out_gate
    out = F.layer_norm(out, (H,), ton_w, ton_b)
    out = out * out_gate.reshape(B, N, N, H)

    # Final projection: [B*N*N, H] @ [H, D] -> [B*N*N, D]
    out_flat = out.reshape(-1, H) @ to_w.t()

    return out_flat.reshape(B, N, N, D)


# Compile once — shape info derived from tensors, not Python ints
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

    _, _, _, D = input_tensor.shape

    # LayerNorm outside compile
    x = F.layer_norm(input_tensor, (D,), norm_w, norm_b)

    return _compiled_trimul(x, mask, lp_w, rp_w, lg_w, rg_w, og_w,
                            ton_w, ton_b, to_w)

```

---

## Experiment #16 — 2026-06-11 22:44:46 UTC 💥 CRASH

**Hypothesis:** Epoch 1 final: fused Triton kernel replacing sigmoid+multiply+mask+reshape+permute+bf16cast for left/right into a single kernel outputting [B*H, N*N] bf16 directly. Kernel reads [BNN, H] row-major and writes [H, BNN] col-major. Implementation bug: the 2D store indexing `h_offs[None,:]*BNN + bnn_offs[:,None]` creates a [BLOCK_BNN, BLOCK_H] pointer grid but the output layout requires [BLOCK_H, BLOCK_BNN] — the axes are transposed in the store, producing completely wrong results (max_diff~4). Logged as crash. #6 at 6096 µs remains epoch-1 best.

**Result:** CRASH

**Error:**
```
Triton gate_permute kernel store indexing bug: tile is [BLOCK_BNN, BLOCK_H] but out_ptrs uses h_offs[None,:]*BNN + bnn_offs[:,None] which transposes bnn/h axes. All 18 tests fail with max_diff 2-5 (expected <0.02).
```

**Kernel code:**
```python
"""
TriMul submission — fused Triton gate+permute+cast kernel.
A single Triton kernel replaces: sigmoid, multiply, mask, reshape, permute,
contiguous, and bf16 cast for both left and right operands.
Outputs left_bh/right_bh directly in [B*H, N, N] bf16 layout for cuBLAS bmm.
Eliminates ~6 kernel launches and 2 full-tensor memory copies per call.
Rest: fused-5H GEMM, reduce-overhead compile, #6 structure.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

torch.backends.cuda.matmul.allow_tf32 = True


@triton.jit
def _gate_permute_kernel(
    # Inputs: [B*N*N, H] float32, contiguous (row=spatial, col=h)
    lp_ptr, rp_ptr, lg_ptr, rg_ptr,
    mask_ptr,   # [B*N*N] float32
    # Outputs: [B*H, N*N] bfloat16, h-major layout
    left_out_ptr, right_out_ptr,
    BNN,   # B*N*N
    H,
    BLOCK_BNN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Fused: left[b*h, k] = (lp[k, h] * sigmoid(lg[k, h])) * mask[k]
    Reads [BNN, H] row-major; writes [H, BNN] = [B*H, N*N] col-major.
    pid_bnn: tile over BNN (spatial), pid_h: tile over H.
    """
    pid_bnn = tl.program_id(0)
    pid_h   = tl.program_id(1)

    bnn_offs = pid_bnn * BLOCK_BNN + tl.arange(0, BLOCK_BNN)
    h_offs   = pid_h   * BLOCK_H   + tl.arange(0, BLOCK_H)

    bnn_mask = bnn_offs < BNN
    h_mask   = h_offs   < H

    # Load mask [BLOCK_BNN]
    mask_vals = tl.load(mask_ptr + bnn_offs, mask=bnn_mask, other=0.0)

    # Load lp, lg, rp, rg: [BLOCK_BNN, BLOCK_H] from row-major [BNN, H]
    ptrs = bnn_offs[:, None] * H + h_offs[None, :]
    full_mask = bnn_mask[:, None] & h_mask[None, :]

    lp_tile = tl.load(lp_ptr + ptrs, mask=full_mask, other=0.0)
    lg_tile = tl.load(lg_ptr + ptrs, mask=full_mask, other=0.0)
    rp_tile = tl.load(rp_ptr + ptrs, mask=full_mask, other=0.0)
    rg_tile = tl.load(rg_ptr + ptrs, mask=full_mask, other=0.0)

    # Fused gate + mask
    left_tile  = lp_tile * tl.sigmoid(lg_tile) * mask_vals[:, None]
    right_tile = rp_tile * tl.sigmoid(rg_tile) * mask_vals[:, None]

    # Write to [H, BNN] layout: out[h, bnn] = tile[bnn, h]
    # out_ptr base = h * BNN, offset = bnn
    out_ptrs = h_offs[None, :] * BNN + bnn_offs[:, None]  # [BLOCK_BNN, BLOCK_H]

    tl.store(left_out_ptr  + out_ptrs, left_tile.to(tl.bfloat16),  mask=full_mask)
    tl.store(right_out_ptr + out_ptrs, right_tile.to(tl.bfloat16), mask=full_mask)


def fused_gate_permute(lp, rp, lg, rg, mask_flat, B, N, H):
    """
    lp, rp, lg, rg: [B*N*N, H] float32
    mask_flat: [B*N*N] float32
    Returns left_bh, right_bh: [B*H, N*N] bfloat16
    """
    BNN = B * N * N
    left_bh  = torch.empty(H, BNN, dtype=torch.bfloat16, device=lp.device)
    right_bh = torch.empty(H, BNN, dtype=torch.bfloat16, device=lp.device)

    BLOCK_BNN = 128
    BLOCK_H   = 32

    grid = (triton.cdiv(BNN, BLOCK_BNN), triton.cdiv(H, BLOCK_H))
    _gate_permute_kernel[grid](
        lp, rp, lg, rg, mask_flat,
        left_bh, right_bh,
        BNN, H,
        BLOCK_BNN=BLOCK_BNN, BLOCK_H=BLOCK_H,
    )
    # Reshape [H, BNN] -> [B*H, N, N]
    return left_bh.reshape(B*H, N, N), right_bh.reshape(B*H, N, N)


def _trimul_inner(x, mask, lp_w, rp_w, lg_w, rg_w, og_w,
                  ton_w, ton_b, to_w, B, N, H):
    """Core TriMul — fused gate+permute Triton kernel, cuBLAS bmm."""
    D = x.shape[-1]
    x_flat = x.reshape(-1, D)

    # Fuse all 5 projections into a single GEMM: [B*N*N, 5*H]
    fused_w = torch.cat([lp_w, rp_w, lg_w, rg_w, og_w], dim=0)  # [5H, D]
    fused = x_flat @ fused_w.t()  # [B*N*N, 5H]

    lp, rp, lg, rg, og = fused.split(H, dim=1)

    out_gate = og.sigmoid()  # [B*N*N, H]

    # Fused: gate + mask + permute + bf16 cast in ONE Triton kernel
    mask_flat = mask.reshape(-1)  # [B*N*N]
    left_bh, right_bh = fused_gate_permute(lp, rp, lg, rg, mask_flat, B, N, H)
    # left_bh, right_bh: [B*H, N, N] bfloat16

    # cuBLAS batched SGEMM (tensor cores via bf16)
    out_bmm = torch.bmm(left_bh, right_bh.transpose(1, 2)).to(torch.float32)
    # out_bmm: [B*H, N, N] -> [B, H, N, N] -> [B, N, N, H]
    out = out_bmm.reshape(B, H, N, N).permute(0, 2, 3, 1).contiguous()

    # LayerNorm + out_gate
    out = F.layer_norm(out, (H,), ton_w, ton_b)
    out = out * out_gate.reshape(B, N, N, H)

    # Final projection
    out_flat = out.reshape(-1, H) @ to_w.t()
    return out_flat.reshape(B, N, N, D)


# reduce-overhead: CUDA graphs for non-Triton ops
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

```


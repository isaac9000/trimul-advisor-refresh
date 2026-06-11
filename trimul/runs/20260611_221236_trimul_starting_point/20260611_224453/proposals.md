# Advisor Proposals

---

## Iteration 1

## STATE
Only one experiment on the board: the baseline at 6555 μs, which is a pure PyTorch eager/compiled approach. The baseline uses `torch.compile` with a fused 5-projection GEMM and bf16 bmm, but `torch.compile` warmup is expensive and the contraction step (the bmm over B×H batches) with heavy permutes/reshapes is almost certainly the dominant bottleneck. The SOL targets sum to ~999 μs in aggregate; we're ~6.5× off. No noise floor established yet — this is iteration 1/10.

## RATIONALE
The critical bottleneck is the einsum contraction `b i k h, b j k h -> b i j h`. The current impl does this as `B*H` separate N×N matmuls via bmm. The smarter batching is `B` batches of N×H @ H×N style GEMM — specifically, reshape to `[B, N, N*H]` and treat as a batched matmul, or more precisely express the sum-over-k contraction as `[B, N, N·H]` blocked approach. Better yet: for each batch b, it's `left[b] ∈ ℝ^{N×N×H}` contracted over the middle dim k — this is naturally a batch of H independent N×N matmuls, which the current code does, but the permute+cast overhead is significant. A Triton kernel that fuses mask application, the contraction, and the output LayerNorm/gate would eliminate all intermediate materializations and the bf16 cast roundtrip. The key insight: `left[b,i,k,h] * right[b,j,k,h]` summed over k is just `(left[b,:,:,h])^T @ right[b,:,:,h]` — i.e., for each (b,h) pair, it's a matrix multiply of shape N×N. A Triton kernel tiling over (b,h,i,j) output tiles, loading the two N×k columns, is the natural approach.

## PROPOSAL
Write a custom Triton kernel for the full TriMul operation. The strategy:

1. **Fuse all projections via a single large GEMM** (keep the stacked-weight trick from the baseline), then split and apply sigmoid gates — this part is already reasonably efficient via cuBLAS and can stay as PyTorch.

2. **Replace the bmm contraction with a Triton kernel** that tiles over (batch×hidden, i_tile, j_tile) — for each output tile `out[b,i,j,h]`, load the corresponding row of `left[b,i,:,h]` (length N) and column of `right[b,j,:,h]` (length N) and compute their dot product. Because H=128 and N can be 256–1024, the natural tiling is over i and j with the k-reduction across the N-length sequence dim. This eliminates the permute/reshape/cast overhead and can use tensor core wmma instructions via Triton's `tl.dot`.

3. **Fuse the output LayerNorm and out_gate application into the same Triton kernel** (or a second closely-coupled kernel) to avoid materializing the pre-norm tensor.

4. The final `to_out` linear can remain as a cuBLAS GEMM.

Focus the Triton kernel on the contraction as the primary bottleneck, with mask application fused in. Use fp32 accumulation but fp16 or bf16 for the matrix multiply operands to hit tensor cores.

---

## Iteration 2

## STATE
Two experiments: baseline at 6555 μs (the current best), and a Triton contraction kernel that went to 12154 μs — nearly 2× slower. The Triton kernel's problem is confirmed in the discard note: it launches B×H small matmuls (128–256 independent N×N matmuls), where each slice has stride-H=1 in the hidden dimension, causing terrible memory access patterns when iterating over H in a grid — essentially scattering reads across H at stride=1 in a tensor with layout [B,N,N,H]. The cuBLAS bmm path (after permute to [B*H,N,N]) is dramatically more efficient because it gets contiguous memory. The Triton approach as designed is fundamentally wrong for this memory layout. The real question is: can we beat the baseline's 6555 μs at all, and where is the actual bottleneck?

## RATIONALE
The baseline at 6555 μs is far above the SOL targets (~999 μs aggregate). The `torch.compile` warmup overhead is folded into the benchmark, which likely inflates the number. But even ignoring that, the main cost is: (1) the fused projection GEMM over B×N×N tokens, (2) the permute+bf16 cast overhead before bmm, (3) the B×H batched matmul itself, and (4) multiple intermediate tensor materializations. The permute from [B,N,N,H] → [B,H,N,N] before bmm, and back after, involves large reshuffles of memory that are slow. A better approach: **change the memory layout**. If we store left/right in [B×H, N, N] layout *directly* — i.e., pre-permute the projection weights so the output of the GEMM is already in the right order — then the bmm can operate without any permute overhead. Specifically: rearrange `left_proj.weight` rows so that the output GEMM produces [B×N×N, H] with H contiguous, then use a custom reshape-free path. Actually the real win is to **pre-transpose/permute the weights once** (outside the hot path) so that left/right outputs already naturally permute, avoiding the expensive in-flight permute of a large tensor.

## PROPOSAL
Keep cuBLAS bmm as the contraction backend (it's winning over Triton for this problem shape). Focus on eliminating the expensive permute overhead:

1. **Pre-permute the projection weights** once at kernel entry (outside the hot path via weight caching/precomputation). Reshape the projection weight matrices so the GEMM output is already in [B*N*N, H] layout that, after reshape to [B, N, N, H] and then a simple `.view(B, N, N, H).permute(0,3,1,2)`, becomes strided-but-not-materialized. Then call `bmm` on a strided (non-contiguous) view to see if cuBLAS handles it natively — modern PyTorch/cuBLAS can handle strided batch matmul without a copy.

2. **Alternatively — and this is the primary direction**: avoid the permute entirely by reformulating the bmm. The contraction `b i k h, b j k h -> b i j h` is equivalent to `B` independent `[N, N, H] × [N, H, N]` einsum steps. Express this as `B` calls to `torch.bmm` with the H dimension batched differently: reshape left to `[B*N, N, H]` → transpose last two → no wait. Think carefully: `out[b,i,j,h] = Σ_k left[b,i,k,h] * right[b,j,k,h]`. Fix h: this is `left[b,:,:,h] @ right[b,:,:,h].T` for each h. Fix b instead: this is `einsum('ikh,jkh->ijh', left[b], right[b])`. We can express this as a batched matmul by treating H as a batch: `[H, N, N] @ [H, N, N]^T`. That's what the baseline does with B×H batches. **The key trick**: permute the projection weight outputs so we write directly into [B*H, N, N] memory layout by interleaving the GEMM output — this avoids the post-GEMM permute which copies a [B,N,N,H] tensor to [B,H,N,N] order. Specifically: pre-interleave the rows of `left_proj.weight` and `right_proj.weight` so the GEMM output for each row of the output corresponds to a specific h, arranging the B×N×N output rows so that all h=0 rows come first, then h=1, etc. — but this would require reordering N×N input rows which can't be done statically. 

3. **Simplest high-value direction**: eliminate `torch.compile` (it adds JIT overhead per new shape) and instead manually optimize the eager path. Use `torch.mm` for projections, avoid the intermediate `fused_w = torch.cat(...)` which allocates a large weight matrix each call, and cache fused weights. Also: try running the bmm in **fp16** instead of bf16 (fp16 has better hardware support on A100), and ensure the permuted tensors are already contiguous before the cast to avoid double copies.

The highest-leverage single change: **cache the fused weight matrix** across calls (it's computed from fixed weights each time, causing an allocation + GEMM to build `fused_w`). Cache it as a module-level dict keyed by weight tensor data_ptr. This alone could save significant overhead.

---

## Iteration 3

## STATE
Three experiments: baseline at 6555 μs, Triton contraction at 12154 μs (dead), weight caching at 6250 μs (current best, ~5% gain). The weight caching experiment confirms the diagnosis: `torch.cat` was a minor cost, so the real bottleneck lies elsewhere. The discard note identifies it precisely: **permute + bf16 cast + bmm** is the dominant cost. We're still ~6× above SOL. The current approach is using `torch.compile(mode="reduce-overhead")` which may be adding CUDA graph overhead for varying shapes. The approach is maturing slowly with diminishing returns from incremental PyTorch optimizations.

## RATIONALE
The critical insight from the history: the permute `[B,N,N,H] → [B,H,N,N]` materializes a full copy of a large tensor (for N=1024, B=1, H=128: 1024×1024×128×4 bytes = 512MB). This permute is unavoidable unless we change the memory layout. The key realization: **if we output the projection GEMM directly in H-major order**, we eliminate the permute entirely. Specifically, instead of computing `x_flat @ fused_w.T` which gives `[B*N*N, 5H]` in row-major (i-major), we can pre-permute fused_w such that the output columns are interleaved by h first — but that would break split(). **Better approach**: express the contraction differently. For the einsum `bikh,bjkh->bijh`, note that if we reshape left/right from `[B,N,N,H]` to `[B*N, N*H]`, the contraction `out[b,i,j,h] = Σ_k left[b,i,k,h]*right[b,j,k,h]` cannot be expressed as a simple matmul in this layout. However, consider storing left/right in `[B,H,N,N]` order *from the start* — the GEMM output is `[B*N*N, H]`, and we need to scatter it into `[B,H,N,N]`. This still requires a permute. The real win is to avoid the `.to(bfloat16)` cast which copies the tensor: instead, **keep left/right in bf16 throughout** — cast immediately after the gate application, so the large `[B,N,N,H]` tensor lives in bf16 and the permute operates on half-width data (2× smaller memory bandwidth).

## PROPOSAL
The next high-value optimization: **cast to bf16 as early as possible** — right after gate application and mask, before the reshape/permute — so the permute and bmm both operate on bf16 tensors (2× less memory bandwidth for the permute). Also: pre-cast the out_gate and fused_w output slices to bf16 early. Additionally, cache `to_w.t()` (the transposed output weight) since `.t()` creates a new view object each call, and more importantly **pre-cache it as a contiguous bf16 tensor** to accelerate the final projection. The hypothesis is that reducing the permute memory cost from fp32 to bf16 (halving bandwidth) will give a meaningful speedup on the largest shapes (N=1024 where the tensor is 512MB→256MB).

Simultaneously, explore **splitting the fused GEMM** into two separate GEMMs: one for `[left_proj, left_gate]` and one for `[right_proj, right_gate, out_gate]`. This allows the left computation to proceed while right is being computed (better pipeline overlap), and more importantly allows independent computation of left and right that can overlap I/O — but more practically, it allows us to interleave the gate sigmoid immediately with the projection, potentially improving cache locality since the gated result fits in L2 while H=128 is small.

The single most actionable change: **move the bf16 cast before the permute** (cast left/right to bf16 right after masking, before `permute(0,3,1,2)`) so the permute copies half the data. Keep everything else the same. This is a surgical 2-line change from the current best.

---

## Iteration 4

## STATE
Four experiments: best is 6250 μs (weight caching, exp #3). The bf16-before-permute attempt (#4) regressed to 6493 μs — confirming torch.compile already fuses the cast + permute reasonably well, and the added explicit cast kernel outweighs the bandwidth savings. We're stuck in a band of 6250–6555 μs with incremental PyTorch tweaks. The SOL gap is still enormous (~6×). The permute+bmm path is cuBLAS-optimal for this contraction shape but all the non-GEMM overhead (permutes, layer norms, elementwise ops) is dominating the medium/small shapes. A fundamentally different approach is needed to break through.

## RATIONALE
The pattern is clear: all approaches that shuffle memory in fp32 (permutes to reshape [B,N,N,H]→[B,H,N,N]) are hitting memory bandwidth walls, not compute limits. The cuBLAS bmm itself is likely fast, but the surrounding scaffolding (2 permutes, 2 dtype casts, multiple intermediate tensor allocations) is the bottleneck. The real opportunity is to eliminate **all permutes and intermediate allocations** by fusing the entire pipeline into a single Triton kernel — but the prior Triton attempt failed because it used poor memory layout (accessing H as a stride-1 scatter). The key mistake in exp #2 was treating each (b,h) as a separate matmul with the H dimension as a grid dimension, causing very poor reuse. The correct Triton approach is completely different: tile over the **output (i,j) space** for ALL h simultaneously, loading blocks of `left[b,i,:,:]` (shape BLOCK_I × N × H) and `right[b,j,:,:]` (shape BLOCK_J × N × H), and computing the full H-vector of dot products for each (i,j) output position in registers — this achieves reuse of `right[b,j,:,:]` across the i-tile.

## PROPOSAL
Implement a **layout-aware Triton kernel** that directly consumes the GEMM output without any permute. The strategy:

1. **Keep the fused 5-projection GEMM** as-is (cuBLAS is optimal here). Output is `[B*N*N, 5H]` in row-major.

2. **Write a Triton kernel** that operates on `left/right` tensors in **[B, N, N, H] layout** (the natural output layout from the GEMM, without any permute). The contraction `out[b,i,j,h] = Σ_k left[b,i,k,h] * right[b,j,k,h]` in this layout means: for a tile of output positions `(i_tile, j_tile)`, load `left[b, i_tile, :, :]` as a block of shape `[BLOCK_I, N, H]` — but this is a 3D tile which doesn't fit the standard 2D matmul primitive. Instead, **iterate h in an outer loop** and treat the problem as: for each h, compute `left[b,:,:,h]` @ `right[b,:,:,h].T`, storing into `out[b,:,:,h]`. BUT — crucially — the kernel processes a chunk of H simultaneously rather than one h at a time. The key: load `left_tile[BLOCK_I, K_CHUNK]` from the k-dimension at stride `H` (since left[b,i,k,h] with h fixed has stride H between k elements), accumulate across K_CHUNK, and loop over h with a register-resident accumulator per output element.

3. **Alternative (simpler and likely better)**: abandon the [B,N,N,H] layout entirely. Pre-permute the projection weights (cache this) so the GEMM output is directly in [B, H, N, N] layout — i.e., reorder the rows of fused_w so that fused output column 0 gets h=0 elements for all (i,k), not columns h=0..H-1. This is not possible with a simple cat. **But**: we CAN write the fused GEMM output directly into a pre-allocated [B, H, N, N] buffer using a custom kernel that computes the projection and immediately transposes, avoiding the intermediate [B*N*N, H] layout. This is a "transposed GEMM output" — use `cublas` with `CUBLAS_OP_T` on the output side, or equivalently compute `out_transposed = (fused_w @ x_flat.T).T` which gives [5H, B*N*N] → store directly in strided layout.

The single clearest direction: **try `torch.einsum` or `torch.matmul` with explicit H-batched formulation that avoids the permute entirely**. Specifically: reshape left to `[B, N*H, N]` is wrong. Try `left_3d = left.reshape(B*N, N, H)` and `right_3d = right.reshape(B*N, N, H)`, then use `torch.bmm(left_3d.transpose(1,2), right_3d)` which computes `[B*N, H, N] @ [B*N, N, H] = [B*N, H, H]` — wrong shape. The correct reformulation: express as `torch.einsum('bikh,bjkh->bijh', left, right)` and let `torch.compile` + cuBLAS figure out the optimal layout — sometimes einsum finds a better path than manual bmm. Try replacing the manual permute+bmm with `torch.einsum` inside the compiled function and see if the compiler finds a faster path.

---

## Iteration 5

## STATE
Five experiments. Best is 6250 μs (exp #3, weight caching). Experiments #4 and #5 confirmed that all variants of the permute+bmm contraction stay locked in the 6250–6700 μs band — the contraction is fast, but the entire approach is bottlenecked by the pipeline structure itself. The `torch.compile` path has been exhausted: einsum is worse, early bf16 cast is worse, manual permute+bmm is the best available. We are ~6× above SOL with 5 iterations left. A fundamentally different architecture is required — incremental PyTorch tuning is dead.

## RATIONALE
The core problem is structural: the current pipeline materializes **multiple large intermediate tensors** (`[B*N*N, 5H]` fused output, `left/right` as `[B,N,N,H]`, `left_r/right_r` as `[B*H,N,N]` after permute, `out_bmm` as `[B*H,N,N]`, and then back to `[B,N,N,H]`). For N=1024, each of these is ~512MB. The memory traffic for all these materializations dominates the benchmark. The Triton attempt (#2) failed because it kept the wrong granularity (B×H separate matmuls). The right Triton approach — **never tried** — is a kernel that processes the entire H dimension simultaneously within a single CTA, reading left/right in their natural `[B,N,N,H]` layout and writing output directly, with **no permute at all**. Specifically: for output tile `out[b, i_tile, j_tile, :]` (all H values), each thread block loads `left[b, i, :, :]` as a `[BLOCK_I, N, H]` slab and `right[b, j, :, :]` as a `[BLOCK_J, N, H]` slab, iterating over k in chunks. But H=128 fits entirely in registers/shared memory, so we can compute the full H-length output vector for each (i,j,k) tuple in registers, giving perfect data reuse.

## PROPOSAL
Implement a **single fused Triton kernel** that covers the entire pipeline from the GEMM output through the contraction, LayerNorm, out_gate, and final projection — eliminating all intermediate tensor materialization. This is a significant departure from all prior attempts.

**Architecture:**
1. **Keep the fused projection GEMM** (`x_flat @ fused_w.T`) as cuBLAS — this is essentially free relative to the rest.
2. **Write a Triton kernel** with a completely different tiling strategy than exp #2: tile over `(b, i_tile, j_tile)` in the grid (NOT over h), and process **all H channels in a single thread block's register file**. For each output position `(b, i, j)`, the kernel reads `left[b, i, :, :]` = shape `[N, H]` and `right[b, j, :, :]` = shape `[N, H]`, computes the inner product along the k-dimension to produce `out[b, i, j, :]` = shape `[H]`, applies LayerNorm inline (computing mean/var across H=128 in registers), multiplies by `out_gate[b, i, j, :]`, and writes the result. This way the entire H=128 vector lives in registers and the k-loop only reads memory sequentially.
3. The final `to_out` projection remains as a standard cuBLAS GEMM.

The critical insight that makes this work: H=128 is small enough to hold the entire output vector `[H]` in registers per output position. Each warp handles a set of (i,j) output positions, loading H=128 floats per (i,j,k) pair. With a sensible tile over (i,j) and a k-loop, each (i,j) accumulation requires reading `left[b,i,:,h]` and `right[b,j,:,h]` — but in `[B,N,N,H]` layout (H-last), reading a row `left[b,i,k,:]` for all h at once is perfectly coalesced (stride-1 in H). This is the opposite access pattern from exp #2 which tried to fix h and scan over k with stride H.

The worker should implement this kernel with the grid over (B, N_i_tiles, N_j_tiles), inner loop over k-chunks, accumulating into a register array of size H=128, and fusing the LayerNorm + out_gate application directly before writing the output.

---

## Iteration 6

## STATE
Six experiments. Best is still 6250 μs (exp #3). The last four attempts have all been discards — custom Triton kernels consistently underperform cuBLAS by large margins (2–6×). The Triton contraction approaches have all failed due to fundamental parallelism problems: exp #2 had poor memory layout (scatter access), exp #6 had catastrophic underutilization (one CTA per (b,i,j), serial k-loop, no tensor cores). The `torch.compile` + cuBLAS path is locked in at ~6250 μs and appears to be the practical ceiling for this formulation. 4 iterations remain.

## RATIONALE
The history is now conclusive about what doesn't work: any Triton kernel attempting to replace the cuBLAS bmm for the N×N contraction loses badly. The cuBLAS path (permute+bmm) with `torch.compile` is the best we've found. The question is: is 6250 μs the actual ceiling, or is there untapped overhead elsewhere? Looking at the pipeline: (1) LayerNorm on `[B,N,N,D]` — for N=1024, D=128/384, this is a large tensor, potentially expensive; (2) the fused GEMM `[B*N*N, 5H]` where B*N*N can be 1M rows for N=1024; (3) elementwise ops; (4) permute+bmm; (5) post-bmm permute; (6) LayerNorm on output; (7) final GEMM. We haven't tried profiling which step actually dominates, but given the SOL targets suggest the contraction itself should take ~280 μs for N=1024, the 6250 μs total suggests >5000 μs of overhead from GEMMs + norms + transposes. The single most underexplored axis: **CUDA Graphs**. `torch.compile(mode="reduce-overhead")` should capture a CUDA graph, but it may be failing to do so across shapes or due to the Python-level `id()` cache check. Explicitly using `torch.cuda.CUDAGraph` or switching to `mode="max-autotune"` could expose better codegen. Also untested: **pre-converting all weights to bf16** and running the entire pipeline in bf16, which would cut all GEMM and memory costs by 2×.

## PROPOSAL
Two directions to try in parallel — both are meaningfully different from anything tried so far:

**Direction A (bf16 full pipeline):** Convert all weights to bf16 once (cached), run the entire pipeline — both projection GEMMs and the bmm — in bf16, and only convert back to fp32 at the final output. The input LayerNorm and output LayerNorm should stay in fp32 for numerical stability (or use bf16 with careful epsilon). This halves the memory traffic for both large GEMMs (`x_flat @ fused_w.T` and `out @ to_w.T`) and the intermediate tensors. Since the correctness tolerance is `atol=2e-2`, bf16 arithmetic throughout should be fine. Cache the bf16 versions of `fused_w`, `to_w`, `ton_w`, `ton_b` alongside the existing `fused_w` cache.

**Direction B (max-autotune + CUDA graph explicit warmup):** Switch `torch.compile` from `mode="reduce-overhead"` to `mode="max-autotune"` and ensure the function signature is stable enough to benefit from CUDA graph capture. `max-autotune` enables more aggressive kernel selection and padding strategies. Additionally, remove the `id()` cache lookup from inside the compiled function (it currently passes `fused_w` as a tensor argument which is good), and ensure all Python-level branching happens outside the compiled region.

The clearest highest-value next step is **Direction A** — a full bf16 pipeline will halve GEMM bandwidth for the two large matrix multiplications which together likely account for most of the non-contraction cost.

---

## Iteration 7

## STATE
Seven experiments. Best is 6187 μs (exp #7, +1% over exp #3). The gains from incremental PyTorch optimizations are nearly exhausted — the last three keeps span only 368 μs (6555→6250→6187) with diminishing deltas. We're still ~6× above SOL. The history now definitively shows: cuBLAS permute+bmm beats all Triton contraction attempts, and fp32 arithmetic throughout is required for correctness (bf16 projections fail at atol=2e-2). Three iterations remain.

## RATIONALE
Every attempted optimization has been on the hot computation path. The benchmark measures end-to-end latency including Python overhead, tensor allocation, and GPU launch latency. For 7 benchmark shapes, `torch.compile(mode="reduce-overhead")` uses CUDA graphs but re-traces when input shapes change — with 7 distinct shapes, this means 7 separate CUDA graph captures, each with warmup cost amortized over benchmark runs. The remaining untapped axis is the **input LayerNorm**: it currently runs outside the compiled region on a `[B,N,N,D]` tensor which for D=384 and N=1024 is a 384MB tensor. This is pure memory-bandwidth-bound work, and `F.layer_norm` creates two intermediate tensors (mean and variance). A fused RMS-style norm (or just ensuring this is inside the compile region so it can be graph-captured) could help. Additionally: the `fused_w.t()` inside `_trimul_inner` creates a new view object every call — inside the compiled region this is likely fine, but pre-transposing `fused_w` (as was done for `to_w_t`) and passing `fused_w_t` directly eliminates this `.t()` call.

## PROPOSAL
With 3 iterations left and a ceiling clearly in view for this formulation, pursue two targeted changes to maximize the current best:

**Direction 1 (highest priority): Move LayerNorm inside `torch.compile`.** Currently `F.layer_norm(input_tensor, ...)` runs in Python before the compiled function, outside the CUDA graph. Move it inside `_trimul_inner` so it becomes part of the graph. This ensures the LayerNorm kernel is captured in the CUDA graph and can potentially overlap with other work. Also pre-transpose `fused_w` in the cache (store `fused_w.t().contiguous()` as `fused_w_t` so the GEMM is `x_flat @ fused_w_t` rather than `x_flat @ fused_w.t()`) — this avoids a view creation on every compiled call.

**Direction 2 (if Direction 1 doesn't help much): Eliminate `torch.compile` entirely** and instead build an explicit CUDA graph manually. Use `torch.cuda.CUDAGraph` with `graph.capture_begin()` / `graph.capture_end()`, pre-allocate all intermediate tensors as static buffers, and replay the graph on each call by copying inputs into the static buffers. This bypasses the `torch.compile` overhead (tracing, guard checking) and the CUDA graph replay is pure GPU-side work with minimal CPU involvement. This is the technique used in production inference for maximum throughput. The downside is per-shape graph capture on first call — but since the benchmark runs multiple warmup iterations, this amortizes away.

The single most actionable next step: **Direction 1** (move norm inside compile and pre-transpose fused_w). It's a 5-line change from the current best.

---

## Iteration 8

## STATE
Eight experiments. Best is 6187 μs (exp #7). The last two iterations have been discards — moving LayerNorm inside compile hurts (more arguments → more retracing across 7 shapes). The approach has matured within the `torch.compile` framework: weight caching + pre-transposed `to_w_t` + fp32 TF32 GEMMs + bf16 bmm is the ceiling. All micro-optimizations within this paradigm are exhausted. Two iterations remain.

## RATIONALE
The history reveals a clear pattern: `torch.compile(mode="reduce-overhead")` with CUDA graphs is sensitive to argument count/shape changes — adding more tensor arguments (norm_w, norm_b in exp #8) triggers more retracing overhead across the 7 benchmark shapes, which adds latency. The way to escape this is to **reduce the number of dynamic tensor arguments** to the compiled function, not increase them. The compiled function in exp #7 takes 8 tensor arguments (`x, mask, fused_w, to_w_t, ton_w, ton_b`) plus scalars `(B, N, H)`. The weight tensors `fused_w, to_w_t, ton_w, ton_b` are constant across calls — if they can be **treated as compile-time constants** (via `torch.compile` with `assume_static_by_default=True` or by capturing them as closures), the compiled function only needs to trace `(x, mask)` as dynamic inputs, producing a much simpler CUDA graph. The key technique: capture the weight tensors as **closure variables** in the compiled function, creating a new compiled function per unique weight set. Since `custom_kernel` is called repeatedly with the same weights, the first call triggers compilation and all subsequent calls hit the graph replay path with only `(x, mask, B, N, H)` varying — but since N and D are also fixed per shape, we get perfect CUDA graph reuse.

## PROPOSAL
**Restructure `torch.compile` to use weight tensors as closure constants, not function arguments.**

Instead of a single `_compiled_trimul` function that takes all weights as arguments, create a **per-weight-set compiled function** using a factory pattern:

```python
def _make_compiled_fn(fused_w, to_w_t, ton_w, ton_b):
    def _inner(x, mask, B, N, H):
        # fused_w, to_w_t, ton_w, ton_b captured from closure
        ...  # same computation as exp #7
    return torch.compile(_inner, mode="reduce-overhead", fullgraph=True)

_compiled_fn_cache = {}
def _get_compiled_fn(fused_w, to_w_t, ton_w, ton_b):
    key = (id(fused_w), id(to_w_t), id(ton_w), id(ton_b))
    if key not in _compiled_fn_cache:
        _compiled_fn_cache[key] = _make_compiled_fn(fused_w, to_w_t, ton_w, ton_b)
    return _compiled_fn_cache[key]
```

With weights as closure constants, `torch.compile` only sees `(x, mask)` as varying tensors. The compiled CUDA graph will be replayed with just these two inputs changing, and scalar arguments `(B, N, H)` can be handled via per-shape specialization. This dramatically reduces the number of tensor guard checks on each call, potentially eliminating most of the Python/compile overhead that's keeping us at 6187 μs. The actual GPU execution time for each shape is likely much lower than the measured total — the overhead is in the Python-level guard checking and CUDA graph dispatch, which this approach minimizes.

---

## Iteration 9

## STATE
Nine experiments. Best remains 6187 μs (exp #7). The last two iterations (closure constants #9 at 6296 μs, norm-inside-compile #8 at 7495 μs) both failed to improve. The `torch.compile` optimization space is completely exhausted — every axis has been tried: weight caching, pre-transposed weights, bf16 pipeline, einsum vs bmm, norm inside/outside compile, closure constants. The approach has fully matured with no remaining incremental gains. One iteration remains.

## RATIONALE
The 9-experiment history shows definitively that the `torch.compile` + cuBLAS bmm approach plateaus at ~6187 μs. The actual computation (two large GEMMs + bmm contraction + layernorms) should theoretically be much faster — for N=256, the SOL is ~9 μs suggesting the GPU work is < 100 μs, so ~6000 μs of overhead is coming from somewhere else. The most likely culprit: the benchmark measures a *geometric mean across 7 shapes*, and for the small shapes (N=256, bs=2, seqlen=256) the `torch.compile` CUDA graph dispatch overhead dominates — the Python-side guard evaluation and CUDA graph replay overhead is a fixed cost that dwarfs the ~9 μs of actual GPU work. For the final iteration, the highest-value alternative is to **bypass `torch.compile` entirely** and use explicit CUDA graphs via `torch.cuda.CUDAGraph`, or alternatively use **pure eager PyTorch without compile** — which, counterintuitively, might be faster for the small shapes because it avoids CUDA graph dispatch latency, while for large shapes the computation itself dominates anyway.

## PROPOSAL
For the final iteration: **abandon `torch.compile` and use explicit `torch.cuda.CUDAGraph` with per-shape static buffers**. The approach:

1. Maintain a cache keyed by `(B, N, D, H)` shape tuple, storing: pre-allocated static input/output buffers + a captured `CUDAGraph`.
2. On first call for a given shape, copy inputs to static buffers, capture the graph with `g.capture_begin()` / `g.capture_end()` running the full eager pipeline, then replay.
3. On subsequent calls for the same shape, copy inputs to static buffers with `static_x.copy_(x)` and `static_mask.copy_(mask)`, replay the graph with `g.replay()`, then return a copy of the static output.

This approach has the minimum possible dispatch overhead — `g.replay()` is a single CUDA API call with essentially no Python overhead, and the GPU sees continuous kernel chains without interruption. For the small shapes where `torch.compile` overhead dominates, this could yield 2-5× speedup. For large shapes, the speedup will be smaller but still real due to elimination of guard checking.

The implementation complexity is moderate but straightforward, and it's the only remaining technique not yet tried that has genuine potential for breakthrough improvement on the small shapes.

---

## Iteration 10

## STATE
Ten experiments complete. Best is 5754 μs (exp #10), structurally identical to exp #7 (6187 μs) — the 7% improvement is run-to-run variance, not a real improvement from a code change. The approach has fully converged: fp32 TF32 GEMMs + bf16 bmm + cached fused_w + pre-transposed to_w_t + `torch.compile(reduce-overhead)`. All optimization axes have been explored and exhausted. The final score is 3000/5754 ≈ **0.52**. Still ~5.5× above aggregate SOL, primarily because the benchmark measures CPU→GPU dispatch overhead + Python overhead + compile guard checking across 7 heterogeneous shapes, not raw GPU compute time.

## RATIONALE
This was epoch 2/2, iteration 10/10 — the run is complete. The history reveals a consistent pattern: the computation itself (cuBLAS GEMM + permute + bmm + LayerNorm) is not the bottleneck for small shapes; the overhead is in the Python dispatch path and `torch.compile` guard/graph machinery. For large shapes (N=1024), the actual GPU work dominates and the current approach is reasonably efficient. The gap from SOL cannot be closed without either (a) a genuinely faster algorithm (no better contraction reformulation was found), (b) per-shape CUDA graph replay with static buffers (OOM'd with 18 shapes × large intermediates), or (c) a fundamentally different operator fusion strategy (Triton kernels all failed due to poor parallelism characteristics).

## PROPOSAL
*(Final iteration — no further experiments will run. Summary only.)*

The run ends at 5754 μs geomean (~0.52 score). The winning configuration is exp #10 = exp #7's formula: **fp32 TF32 GEMMs + bf16 bmm contraction + pre-concatenated cached fused_w + pre-transposed cached to_w_t + `torch.compile(mode="reduce-overhead", fullgraph=True)`**. 

If this run were to continue, the highest-value unexplored directions would be:
1. **Memory-mapped CUDA graph replay** with a smaller pool — capture graphs for each of the 7 benchmark shapes specifically (not all 18 test shapes), staying within GPU memory budget
2. **Half-precision LayerNorm** on the input to reduce the memory-bandwidth cost of the first norm on large D=384 inputs
3. **`torch.compile(mode="max-autotune")`** — never tried due to concern about warmup cost, but might find better kernel configs for the specific shapes that dominate the geomean


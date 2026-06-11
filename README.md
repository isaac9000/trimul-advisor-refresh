# TriMul Autoresearch — Epoch Refresh

An advisor-worker agent pair that iteratively optimizes a CUDA kernel for the Triangle Multiplicative Update (TriMul) operator on NVIDIA H100. Each iteration the **advisor** reviews experiment history and proposes a strategic direction; the **worker** implements it, evaluates on an H100 via Modal, and logs the result.

After each epoch the run history is committed to git and wiped, and the best kernel from that epoch becomes the baseline for the next — giving the agents a fresh context window to explore without the noise of accumulated dead ends.

## Task

Implement the **outgoing** TriMul operator from AlphaFold3 — a core operation in protein structure prediction models (AlphaFold3, Chai, Protenix):

```
x    = LayerNorm(input)
left = left_proj(x) * sigmoid(left_gate(x))
right= right_proj(x) * sigmoid(right_gate(x))
left, right = left * mask, right * mask
out  = einsum("... i k d, ... j k d -> ... i j d", left, right)
out  = LayerNorm(out) * sigmoid(out_gate(x))
return to_out(out)
```

`custom_kernel` receives a tuple `(input_tensor, mask, weights, config)` and returns the output tensor:

| Argument | Shape | Dtype |
|---|---|---|
| `input_tensor` | `[bs, seqlen, seqlen, dim]` | `float32` |
| `mask` | `[bs, seqlen, seqlen]` | `float32` |
| `weights` | dict of named tensors | `float32` |
| `config` | `{"dim": int, "hidden_dim": int}` | — |
| return | `[bs, seqlen, seqlen, dim]` | `float32` |

**Test cases (correctness) — 18 total:**

| seqlen | bs | dim | nomask | distribution |
|---|---|---|---|---|
| 32 | 1 | 128 | ✓ | normal |
| 32 | 1 | 128 | ✗ | normal |
| 64 | 2 | 256 | ✓ | normal |
| 64 | 2 | 256 | ✗ | normal |
| 128 | 1 | 768 | ✓ | normal |
| 256 | 1 | 128 | ✓ | normal |
| 256 | 1 | 128 | ✗ | normal |
| 768 | 2 | 128 | ✓ | normal |
| 1024 | 1 | 384 | ✗ | normal |
| 1024 | 1 | 768 | ✓ | normal |
| 1024 | 1 | 768 | ✗ | normal |
| 32–1024 | 1–2 | 128–768 | ✓/✗ | cauchy (×7) |

**Benchmark cases (timing) — 7 total:**

| seqlen | bs | dim | nomask | distribution |
|---|---|---|---|---|
| 256 | 2 | 128 | ✓ | normal |
| 768 | 1 | 128 | ✓ | cauchy |
| 256 | 2 | 384 | ✗ | normal |
| 512 | 1 | 128 | ✓ | normal |
| 1024 | 1 | 128 | ✓ | cauchy |
| 768 | 1 | 384 | ✗ | normal |
| 1024 | 1 | 384 | ✓ | normal |

Ranked by geometric mean latency across all seven benchmark cases (lower is better). Score = `3000 / geomean_us` (higher is better). Timing uses adaptive iteration: stops when `stderr/mean < 0.1%`, after 10 s per case, or 120 s wall time. Correctness tolerance: `rtol=2%, atol=2%`.

## Setup

```bash
uv sync
```

Create a `.env` file in the repo root:

```
ANTHROPIC_API_KEY=...
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
AUTORESEARCH_MODEL=claude-sonnet-4-6   # optional, this is the default
```

Deploy the H100 evaluator (once, before any agent runs):

```bash
uv run modal deploy eval_modal_trimul.py
```

## Running the agent

Two epochs of 10 iterations each, starting from scratch:

```bash
uv run trimul/agent.py --epoch-sizes 10 10
```

Start from the provided PyTorch baseline:

```bash
uv run trimul/agent.py --baseline trimul/starting_point.py --epoch-sizes 15 10
```

Use different models for advisor and worker:

```bash
uv run trimul/agent.py --baseline trimul/starting_point.py --advisor-model claude-opus-4-8 --worker-model claude-sonnet-4-6 --epoch-sizes 15 10
```

Or use the provided script (checks for H100 then launches in tmux):

```bash
./run_agent.sh
```

Quick correctness check without a full benchmark:

```bash
cd trimul
python run_eval.py submission.py -o results.json --mode test
```

## Epoch refresh

Each epoch runs for `N` iterations, then:

1. The epoch directory (history, TSV, plots, snapshots) is committed to git.
2. All run artifacts are deleted — the next epoch's agents start with a blank slate.
3. `best_submission.py` from the epoch is copied to `submission.py` as the next epoch's baseline.
4. Advisor and worker agents are rebuilt with fresh memory and new thread IDs.

Epoch directories are named by timestamp (not by epoch number) so agents cannot infer their position in the run from the filesystem.

## Structure

```
eval_modal_trimul.py   — deployable Modal H100 evaluator
run_agent.sh           — H100 check + tmux agent launcher
trimul/
├── agent.py           — advisor-worker agentic loop with epoch refresh
├── advisor_prompt.md  — advisor system prompt: strategy, comparison discipline
├── worker_prompt.md   — worker system prompt: mandatory sequence, rules
├── submission.py      — the kernel file the worker edits each iteration
├── starting_point.py  — baseline PyTorch TriMul kernel to seed each run
├── run_eval.py        — submits submission.py to the deployed Modal evaluator
├── tools.py           — log_experiment and get_experiment_history tools
└── runs/              — one directory per run, containing one directory per epoch
```

Each epoch directory (named by timestamp) contains:
- `experiment_history.md` — full log of every attempt with code and result (deleted after epoch commit)
- `results.tsv` — tab-separated summary for plotting (deleted after epoch commit)
- `progress.png` — latency scatter plot updated each experiment; shows keep/discard/crash points, best-time step line, and cumulative LLM call count (deleted after epoch commit)
- `iterations.png` — best latency per advisor iteration (deleted after epoch commit)
- `best_submission.py` — snapshot of the fastest kernel found in this epoch (kept; promoted to next epoch baseline)
- `proposals.md` — advisor proposals for every iteration (deleted after epoch commit)
- `snapshot_iter{N}.py` — per-iteration snapshots of submission.py before worker edits (deleted after epoch commit)

## LLM Call Counter

The agent tracks how many times the LLM is invoked across both the advisor and worker agents (each tool-calling turn and each plain response counts as one call). This is reported:

- **Per-iteration** in the console: `[advisor]` and `[worker]` call counts accumulated into a running total
- **At each checkpoint** (every `--checkpoint-every` iterations): `LLM calls (total): T`
- **In the final report**: `LLM calls (total): T`
- **On `progress.png`**: displayed as a badge in the bottom-right corner of every plot, updated live as experiments are logged

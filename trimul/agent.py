"""
Advisor-Worker agentic loop for TriMul kernel optimization with epoch refresh.

Architecture:
  Advisor — reviews experiment history, decides direction, outputs a proposal.
             Tools: get_experiment_history + shell (read-only).
  Worker  — receives the proposal, edits submission.py, evaluates, logs.
             Tools: log_experiment, get_experiment_history + shell.

The run is split into epochs. At the end of each epoch the experiment history
is committed to git and cleared; the next epoch starts with fresh context,
seeded from the previous epoch's best kernel.

Usage:
    uv run trimul/agent.py
    uv run trimul/agent.py --epoch-sizes 15 10 --baseline trimul/starting_point.py
    uv run trimul/agent.py --advisor-model claude-opus-4-8 --worker-model claude-sonnet-4-6
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

import anthropic

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

import tools as _tools
from tools import (
    log_experiment,
    get_experiment_history,
    _update_plot,
    _get_next_iteration,
    _log_experiment_direct,
    set_run_directory,
    set_agent_iteration,
    set_llm_call_count,
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PROJECT_DIR)
SUBMISSION_FILE = os.path.join(PROJECT_DIR, "submission.py")
RESULTS_FILE = os.path.join(PROJECT_DIR, "results.json")


def load_prompt(filename: str) -> str:
    with open(os.path.join(PROJECT_DIR, filename)) as f:
        return f.read()


def make_llm(model_name: str):
    if model_name.startswith("claude-"):
        return ChatAnthropic(model=model_name, timeout=180, max_retries=2)
    else:
        return ChatOpenAI(model=model_name, use_responses_api=False, timeout=180, max_retries=2)


def make_env() -> dict:
    venv_path = os.path.join(REPO_ROOT, ".venv", "bin")
    env = {
        "PATH": f"{venv_path}:{os.environ.get('PATH', '')}",
        "VIRTUAL_ENV": os.path.join(REPO_ROOT, ".venv"),
        "PYTHONPATH": PROJECT_DIR,
    }
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"]:
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def build_advisor(model_name: str, env: dict):
    checkpointer = MemorySaver()
    agent = create_deep_agent(
        model=make_llm(model_name),
        tools=[get_experiment_history],
        system_prompt=load_prompt("advisor_prompt.md"),
        backend=LocalShellBackend(root_dir=PROJECT_DIR, virtual_mode=False, env=env),
        checkpointer=checkpointer,
    )
    return agent, checkpointer


def build_worker(model_name: str, env: dict):
    checkpointer = MemorySaver()
    agent = create_deep_agent(
        model=make_llm(model_name),
        tools=[log_experiment, get_experiment_history],
        system_prompt=load_prompt("worker_prompt.md"),
        backend=LocalShellBackend(root_dir=PROJECT_DIR, virtual_mode=False, env=env),
        checkpointer=checkpointer,
    )
    return agent, checkpointer


def stream_agent(agent, config: dict, message: str, label: str) -> tuple[str, int]:
    """Stream an agent to completion. Returns (final_text, llm_call_count)."""
    result = None
    n_llm_calls = 0
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
        stream_mode="values",
    ):
        result = chunk
        last_msg = chunk["messages"][-1]
        msg_type = type(last_msg).__name__
        if msg_type == "AIMessage":
            n_llm_calls += 1
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                print(f"  [{label}] {tc['name']}({str(tc.get('args', ''))[:120]})", flush=True)
        elif hasattr(last_msg, "tool_call_id"):
            print(f"  [{label}] → {str(getattr(last_msg, 'content', ''))[:200]}", flush=True)
        elif msg_type == "AIMessage":
            preview = str(getattr(last_msg, "content", ""))[:200]
            if preview.strip():
                print(f"  [{label}] {preview}", flush=True)

    if result is None:
        return "", 0
    final = result["messages"][-1]
    content = getattr(final, "content", "") or ""
    if isinstance(content, list):
        text = " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    else:
        text = str(content)
    return text, n_llm_calls


def _is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, (anthropic.APITimeoutError, anthropic.APIConnectionError)):
        return True
    msg = str(exc).lower()
    return any(kw in msg for kw in ("timeout", "timed out", "connection reset", "read operation timed out"))


def stream_agent_retrying(
    agent,
    config: dict,
    message: str,
    label: str,
    max_attempts: int = 3,
    base_delay: float = 15.0,
) -> tuple[str, int]:
    """Like stream_agent but retries on transient API errors with fresh thread IDs.
    Returns (final_text, llm_call_count) — counts across all attempts."""
    thread_id = config["configurable"]["thread_id"]
    last_exc: Exception | None = None
    total_calls = 0

    for attempt in range(max_attempts):
        if attempt > 0:
            delay = base_delay * (2 ** (attempt - 1))
            print(f"  [{label}] Retrying (attempt {attempt + 1}/{max_attempts}) in {delay:.0f}s...", flush=True)
            time.sleep(delay)
            cfg = {**config, "configurable": {**config["configurable"],
                                              "thread_id": f"{thread_id}-r{attempt}"}}
        else:
            cfg = config

        try:
            text, n = stream_agent(agent, cfg, message, label)
            return text, total_calls + n
        except Exception as e:
            if _is_transient_error(e) and attempt < max_attempts - 1:
                print(f"  [{label}] Transient error on attempt {attempt + 1}: "
                      f"{type(e).__name__}: {str(e)[:150]}", flush=True)
                last_exc = e
            else:
                raise

    raise last_exc  # type: ignore[misc]


def read_results_summary() -> str:
    if not os.path.exists(_tools.TSV_FILE):
        return "No experiments run yet."
    with open(_tools.TSV_FILE) as f:
        lines = f.readlines()
    if len(lines) < 2:
        return "No experiments run yet."

    total = len(lines) - 1
    keeps, discards, crashes = [], 0, 0
    best_time = float("inf")
    best_desc = ""
    last_5 = []

    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) < 5:
            continue
        it, time_str, status = parts[0], parts[3], parts[4]
        desc = parts[5] if len(parts) > 5 else ""
        try:
            t = float(time_str)
        except ValueError:
            t = 0.0
        if status == "keep" and t > 0:
            keeps.append((int(it) if it.isdigit() else 0, t, desc))
            if t < best_time:
                best_time, best_desc = t, desc
        elif status == "discard":
            discards += 1
        elif status == "crash":
            crashes += 1
        last_5.append(f"  #{it}: {t:.2f}μs ({status}) — {desc[:60]}")

    summary = f"=== EXPERIMENT SUMMARY ({total} total) ===\n"
    if best_time < float("inf"):
        summary += f"Best time: {best_time:.2f} μs — {best_desc[:80]}\n"
    else:
        summary += "Best time: none yet\n"
    summary += f"Keeps: {len(keeps)} | Discards: {discards} | Crashes: {crashes}\n"
    if keeps:
        summary += "Keep history:\n"
        for it, t, d in keeps[-10:]:
            summary += f"  #{it}: {t:.2f}μs — {d[:60]}\n"
    summary += "\nLast 5 experiments:\n" + "\n".join(last_5[-5:]) + "\n"
    return summary


def save_proposals(dir_: str, proposals: list) -> None:
    with open(os.path.join(dir_, "proposals.md"), "w") as f:
        f.write("# Advisor Proposals\n\n")
        for iteration, proposal in proposals:
            f.write(f"---\n\n## Iteration {iteration}\n\n{proposal}\n\n")


def print_checkpoint(iteration: int, total: int, start_time: float, llm_call_count: int = 0) -> None:
    elapsed_min = (time.time() - start_time) / 60
    rate = iteration / elapsed_min if elapsed_min > 0 else 0
    summary = read_results_summary()
    print(f"\n{'#'*60}")
    print(f"  CHECKPOINT — Iteration {iteration}/{total}")
    print(f"  Elapsed: {elapsed_min:.1f} min | Rate: {rate:.1f} iter/min")
    print(f"  LLM calls (total): {llm_call_count}")
    print(f"{'#'*60}")
    print(summary)
    try:
        _update_plot()
    except Exception as e:
        print(f"  Plot update failed: {e}")
    print(f"{'#'*60}\n")


def print_final_report(total_epochs: int, epoch_sizes: list[int], actual_iterations: int, start_time: float, llm_call_count: int = 0):
    elapsed_min = (time.time() - start_time) / 60
    print(f"\n{'='*60}\n  FINAL REPORT\n{'='*60}")
    print(f"  Epochs: {total_epochs} | Sizes: {epoch_sizes} | Total iterations: {actual_iterations}/{sum(epoch_sizes)}")
    print(f"  Time: {elapsed_min:.1f} min")
    print(f"  LLM calls (total): {llm_call_count}")
    summary = read_results_summary()
    if summary != "No experiments run yet.":
        print(summary)
    try:
        _update_plot()
    except Exception:
        pass
    print(f"{'='*60}")


def _ensure_git_repo(repo_root: str) -> None:
    """Initialize git repo if missing."""
    if os.path.isdir(os.path.join(repo_root, ".git")):
        return
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "agent@trimul-refresh"],
                   cwd=repo_root, capture_output=True)
    subprocess.run(["git", "config", "user.name", "TriMul Refresh Agent"],
                   cwd=repo_root, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init: seed repo with source files"],
                   cwd=repo_root, check=True, capture_output=True)
    print("  [git] Initialized repository and created initial commit.", flush=True)


def _commit_and_clear_epoch(repo_root: str, epoch_dir: str, epoch: int, run_name: str) -> None:
    """Git-add + commit epoch_dir, then delete all run artifacts. best_submission.py is preserved."""
    try:
        subprocess.run(["git", "add", "-f", epoch_dir], cwd=repo_root, check=True, capture_output=True)
        result = subprocess.run(
            ["git", "commit", "-m", f"epoch {epoch}: {run_name}"],
            cwd=repo_root, capture_output=True,
        )
        if result.returncode == 0:
            print(f"  [epoch {epoch}] Committed epoch dir to git.", flush=True)
        else:
            stderr = result.stderr.decode()[:200]
            print(f"  [epoch {epoch}] Git commit warning: {stderr}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"  [epoch {epoch}] Git add failed: {e}", flush=True)

    delete_patterns = ["experiment_history.md", "results.tsv", "progress.png",
                       "iterations.png", "proposals.md"]
    deleted = []
    for name in delete_patterns:
        path = os.path.join(epoch_dir, name)
        if os.path.exists(path):
            os.remove(path)
            deleted.append(name)

    for fname in os.listdir(epoch_dir):
        if fname.startswith("snapshot_iter") and fname.endswith(".py"):
            os.remove(os.path.join(epoch_dir, fname))
            deleted.append(fname)

    if deleted:
        print(f"  [epoch {epoch}] Cleared: {', '.join(deleted)}", flush=True)
    print(f"  [epoch {epoch}] best_submission.py preserved.", flush=True)


def _benchmark_baseline(epoch_baseline_name: str) -> tuple[str, float]:
    """Benchmark the current submission.py as the epoch baseline. Returns (kickoff_note, time_us)."""
    venv_python = os.path.join(REPO_ROOT, ".venv", "bin", "python")
    print(f"Benchmarking baseline '{epoch_baseline_name}'...", flush=True)
    ret = os.system(f"cd {PROJECT_DIR} && {venv_python} run_eval.py submission.py -o results.json 2>&1")
    try:
        with open(SUBMISSION_FILE) as f:
            baseline_code = f.read()
        with open(RESULTS_FILE) as f:
            md = json.load(f)
        m = re.search(r"Geometric mean: ⏱ ([\d.]+)", md if isinstance(md, str) else "")
        time_us = float(m.group(1)) if m else 0.0
        status = "keep" if (ret == 0 and time_us > 0) else "crash"
        _log_experiment_direct(
            kernel_code=baseline_code,
            hypothesis=f"Baseline '{epoch_baseline_name}' — initial benchmark",
            time_us=time_us,
            status=status,
            error_message="" if status == "keep" else f"run_eval exited {ret}",
        )
        print(f"Baseline logged: {time_us:.1f} µs ({status})", flush=True)
        kickoff_note = (
            f"The '{epoch_baseline_name}' baseline is already benchmarked and logged as experiment #1 "
            f"({time_us:.1f} µs). Your job is to beat it. "
            if status == "keep" else
            f"The '{epoch_baseline_name}' baseline CRASHED (logged as experiment #1). "
            "Read the crash error in get_experiment_history and fix the kernel. "
        )
        return kickoff_note, time_us
    except Exception as e:
        print(f"Warning: could not log baseline: {e}", flush=True)
        return f"submission.py has been pre-loaded with '{epoch_baseline_name}'. Benchmark it first, then improve. ", 0.0


def main():
    parser = argparse.ArgumentParser(description="Advisor-Worker TriMul Optimization Agent (epoch refresh)")
    parser.add_argument("--epoch-sizes", "-e", type=int, nargs="+", default=[10, 10],
                        help="Number of worker iterations per epoch (e.g. --epoch-sizes 15 10)")
    parser.add_argument("--checkpoint-every", "-c", type=int, default=5)
    parser.add_argument("--baseline", "-b", default=None, help="Path to a baseline file to start from")
    parser.add_argument("--advisor-model", default=None)
    parser.add_argument("--worker-model", default=None)
    args = parser.parse_args()

    load_dotenv(os.path.join(REPO_ROOT, ".env"))

    default_model = os.environ.get("AUTORESEARCH_MODEL", "claude-sonnet-4-6")
    advisor_model = args.advisor_model or default_model
    worker_model = args.worker_model or default_model

    for model in {advisor_model, worker_model}:
        if model.startswith("claude-") and not os.environ.get("ANTHROPIC_API_KEY"):
            print("Error: ANTHROPIC_API_KEY not set"); sys.exit(1)
        elif not model.startswith("claude-") and not os.environ.get("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY not set"); sys.exit(1)

    baseline_path, baseline_name = None, "scratch"
    if args.baseline:
        baseline_path = os.path.abspath(args.baseline)
        if not os.path.isfile(baseline_path):
            print(f"Error: baseline not found: {baseline_path}"); sys.exit(1)
        baseline_name = os.path.splitext(os.path.basename(baseline_path))[0]

    epoch_sizes = args.epoch_sizes
    n_epochs = len(epoch_sizes)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_trimul_{baseline_name}"
    run_dir = os.path.join(PROJECT_DIR, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    env = make_env()

    print(f"Starting advisor-worker optimization loop (epoch refresh)")
    print(f"  Advisor model:  {advisor_model}")
    print(f"  Worker model:   {worker_model}")
    print(f"  Baseline:       {baseline_name}")
    print(f"  Run dir:        {run_dir}")
    print(f"  Epochs:         {n_epochs} × {epoch_sizes}")
    print()

    def _sigterm_handler(signum, frame):
        print("\n--- SIGTERM ---", flush=True)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    _ensure_git_repo(REPO_ROOT)

    start_time = time.time()
    prev_best: str | None = None
    total_llm_calls = 0
    total_iterations = 0

    try:
        for epoch in range(1, n_epochs + 1):
            refresh_every = epoch_sizes[epoch - 1]
            epoch_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            epoch_dir = os.path.join(run_dir, epoch_ts)
            os.makedirs(epoch_dir, exist_ok=True)
            set_run_directory(epoch_dir)

            print(f"\n{'='*60}")
            print(f"  EPOCH {epoch}/{n_epochs} — {refresh_every} iterations")
            print(f"{'='*60}\n", flush=True)

            # Set baseline for this epoch
            if epoch == 1:
                if baseline_path:
                    shutil.copy2(baseline_path, SUBMISSION_FILE)
                    print(f"Copied baseline '{baseline_name}' -> submission.py", flush=True)
                else:
                    print("No baseline — using current submission.py.", flush=True)
                epoch_baseline_name = baseline_name
            else:
                if prev_best and os.path.exists(prev_best):
                    shutil.copy2(prev_best, SUBMISSION_FILE)
                    print(f"Promoted previous best -> submission.py", flush=True)
                else:
                    print("Warning: no previous best found — using current submission.py.", flush=True)
                epoch_baseline_name = "previous_best"

            # Fresh agents per epoch (new MemorySaver, new thread IDs)
            advisor_agent, _ = build_advisor(advisor_model, env)
            worker_agent, _ = build_worker(worker_model, env)
            advisor_config = {"configurable": {"thread_id": f"advisor-trimul-{baseline_name}-{epoch_ts}"}}
            worker_config = {"configurable": {"thread_id": f"worker-trimul-{baseline_name}-{epoch_ts}"}}

            kickoff_note, _ = _benchmark_baseline(epoch_baseline_name)

            epoch_proposals: list = []
            epoch_start_time = time.time()
            iteration = 0

            try:
                while iteration < refresh_every:
                    iteration += 1
                    total_iterations += 1
                    set_agent_iteration(total_iterations)
                    print(f"\n{'='*60}")
                    print(f"  EPOCH {epoch}/{n_epochs} | ITERATION {iteration}/{refresh_every}")
                    print(f"{'='*60}\n", flush=True)

                    summary = read_results_summary()

                    # ── ADVISOR ──────────────────────────────────────────────────
                    print("[advisor] Proposing...", flush=True)
                    advisor_message = (
                        f"Epoch {epoch}/{n_epochs}, iteration {iteration}/{refresh_every}.\n\n"
                        f"{summary}\n\n"
                        "Call get_experiment_history for the full code and results, "
                        "then output your structured proposal."
                    )
                    proposal, advisor_calls = stream_agent_retrying(advisor_agent, advisor_config, advisor_message, label="advisor")
                    total_llm_calls += advisor_calls
                    set_llm_call_count(total_llm_calls)
                    epoch_proposals.append((iteration, proposal))
                    print(f"\n[advisor proposal]\n{'-'*40}\n{proposal[:1000]}\n{'-'*40}\n", flush=True)
                    save_proposals(epoch_dir, epoch_proposals)

                    # ── WORKER ───────────────────────────────────────────────────
                    print("[worker] Implementing...", flush=True)
                    log_count_before = _get_next_iteration() - 1

                    snapshot_path = os.path.join(epoch_dir, f"snapshot_iter{iteration}.py")
                    if os.path.exists(SUBMISSION_FILE):
                        shutil.copy2(SUBMISSION_FILE, snapshot_path)

                    worker_message = (
                        f"Epoch {epoch}/{n_epochs}, iteration {iteration}/{refresh_every}.\n\n"
                        f"## Advisor Proposal\n\n{proposal}\n\n"
                        f"## Your Task\n\n"
                        f"{kickoff_note}"
                        "Implement the advisor's proposal: read submission.py, make ONE targeted change, "
                        "evaluate it with `python run_eval.py submission.py -o results.json`, "
                        "then call log_experiment and stop.\n\n"
                        f"{summary}"
                    )
                    kickoff_note = ""  # only shown on first iteration of each epoch

                    _, worker_calls = stream_agent_retrying(worker_agent, worker_config, worker_message, label="worker")
                    total_llm_calls += worker_calls
                    set_llm_call_count(total_llm_calls)

                    log_count_after = _get_next_iteration() - 1
                    if log_count_after <= log_count_before:
                        print("[WARNING] Worker did not call log_experiment — restoring submission.py from snapshot.", flush=True)
                        if os.path.exists(snapshot_path):
                            shutil.copy2(snapshot_path, SUBMISSION_FILE)
                    else:
                        # Restore from best on crash
                        rows = []
                        if os.path.exists(_tools.TSV_FILE):
                            with open(_tools.TSV_FILE) as f:
                                lines = f.readlines()
                            for line in lines[1:]:
                                parts = line.strip().split("\t")
                                if len(parts) >= 5:
                                    rows.append({
                                        "status": parts[4],
                                        "time_us": float(parts[3]) if parts[3].replace(".", "").isdigit() else 0.0,
                                    })
                        if rows and rows[-1]["status"] == "crash":
                            best_path = os.path.join(epoch_dir, "best_submission.py")
                            restore_src = best_path if os.path.exists(best_path) else snapshot_path
                            if os.path.exists(restore_src):
                                shutil.copy2(restore_src, SUBMISSION_FILE)
                                print(f"  [crash restore] submission.py restored from {os.path.basename(restore_src)}", flush=True)

                    if iteration % args.checkpoint_every == 0:
                        print_checkpoint(iteration, refresh_every, epoch_start_time, total_llm_calls)

            except KeyboardInterrupt:
                save_proposals(epoch_dir, epoch_proposals)
                raise

            save_proposals(epoch_dir, epoch_proposals)

            # Print epoch summary before clearing artifacts
            print(f"\n{'#'*60}")
            print(f"  EPOCH {epoch}/{n_epochs} COMPLETE")
            print(f"  Iterations: {iteration} | Elapsed: {(time.time() - epoch_start_time) / 60:.1f} min")
            print(f"  LLM calls (total): {total_llm_calls}")
            print(f"{'#'*60}")
            print(read_results_summary())
            print(f"{'#'*60}\n")

            # Track best kernel for next epoch
            best_path = os.path.join(epoch_dir, "best_submission.py")
            if os.path.exists(best_path):
                prev_best = best_path
            elif os.path.exists(SUBMISSION_FILE):
                fallback = os.path.join(epoch_dir, "best_submission.py")
                shutil.copy2(SUBMISSION_FILE, fallback)
                prev_best = fallback

            _commit_and_clear_epoch(REPO_ROOT, epoch_dir, epoch, run_name)

    except KeyboardInterrupt:
        print(f"\n--- Interrupted at total iteration {total_iterations} ---")
    except Exception as e:
        print(f"\n--- Error at total iteration {total_iterations}: {e} ---")
        import traceback; traceback.print_exc()
    finally:
        print_final_report(n_epochs, epoch_sizes, total_iterations, start_time, total_llm_calls)


if __name__ == "__main__":
    main()

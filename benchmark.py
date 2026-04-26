"""
qwen_fast public benchmark.
========================================================================
Demonstrates wall-clock speedup at strict cos=1 (byte-identical output)
on Qwen 2.5-0.5B vs vanilla HuggingFace generate.

Run:
    python benchmark.py

Verified by NVIDIA Nsight Systems on RTX 3050 6GB Laptop, Windows.

The engine implementation is proprietary and lives in
qwen_fast/_internal/. This benchmark is open-source and only calls
the public API.
"""
import os
import sys
import json
import hashlib
import time
import platform
import argparse

import torch

# Force unbuffered stdout (Windows cp1252 friendly).
sys.stdout.reconfigure(line_buffering=True)


# Public API -- the ONLY thing this benchmark depends on.
from qwen_fast import FastQwen, __version__


# Workloads chosen to span the realistic distribution: factual single-fact,
# document-conditioned answer (RAG-like), repetitive list, and code-like.
WORKLOADS = {
    "factual": "The capital of France is",
    "rag-like": (
        "Document: The Eiffel Tower is a wrought-iron lattice tower in Paris. "
        "It was named after Gustave Eiffel, whose company designed and built the tower. "
        "Question: Who designed the Eiffel Tower? Answer: According to the document, "
        "the Eiffel Tower was designed by"
    ),
    "repetitive": (
        "Item 1: red apple. Item 2: red apple. Item 3: red apple. "
        "Item 4: red apple. Item 5: red apple. Item 6: red apple. "
        "Item 7: red apple. Item 8:"
    ),
    "code-like": (
        "def add(a, b):\n    return a + b\n\n"
        "def sub(a, b):\n    return a - b\n\n"
        "def mul(a, b):\n    return a * b\n\n"
        "def div(a, b):\n    return"
    ),
}


def env_report():
    """System / environment fingerprint to attach to the report."""
    info = {
        "qwen_fast_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        info["compute_capability"] = f"{cc[0]}.{cc[1]}"
        try:
            info["driver"] = torch.cuda.driver_version() if hasattr(torch.cuda, "driver_version") else None
        except Exception:
            info["driver"] = None
    return info


def fmt_table(results):
    """Render the per-workload speedup table."""
    header = (
        f"{'workload':<14} {'baseline (ms)':>14} {'fast (ms)':>11} "
        f"{'baseline tok/s':>16} {'fast tok/s':>12} {'speedup':>9} {'cos=1':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        match = "OK" if r["output_match"] else "DIFF"
        print(
            f"{r['_name']:<14} "
            f"{r['baseline_ms']:>13.0f}  "
            f"{r['fast_ms']:>10.0f} "
            f"{r['baseline_tok_per_s']:>15.1f} "
            f"{r['fast_tok_per_s']:>11.1f} "
            f"{r['speedup']:>8.2f}x "
            f"{match:>7}"
        )


def report_summary(results):
    speedups = [r["speedup"] for r in results]
    all_match = all(r["output_match"] for r in results)
    avg = sum(speedups) / len(speedups)
    best = max(speedups)
    return {
        "n_workloads": len(results),
        "avg_speedup": avg,
        "best_speedup": best,
        "all_byte_identical": all_match,
    }


def report_hash(results):
    """A stable hash of the timing-independent parts of the report.
    Lets a third party verify they ran the same workload."""
    payload = []
    for r in results:
        payload.append({
            "name": r["_name"],
            "prompt": r["prompt"],
            "output_match": bool(r["output_match"]),
        })
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return h[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--gen_len", type=int, default=64)
    ap.add_argument("--n_runs", type=int, default=5)
    ap.add_argument("--n_warmup", type=int, default=3)
    ap.add_argument("--max_total_len", type=int, default=256)
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--save", default="benchmark_report.json")
    args = ap.parse_args()

    env = env_report()
    print("=" * 78)
    print(f"qwen_fast benchmark v{__version__}")
    print("=" * 78)
    for k, v in env.items():
        print(f"  {k:<22}: {v}")
    print()
    print(f"Model      : {args.model}")
    print(f"Gen length : {args.gen_len} tokens (greedy)")
    print(f"Runs/wkld  : {args.n_runs} timed + {args.n_warmup} warmup, median reported")
    print()

    if not torch.cuda.is_available():
        print("ERROR: this benchmark requires CUDA.")
        sys.exit(1)

    print("Loading engine ...")
    t0 = time.time()
    engine = FastQwen(
        args.model,
        max_total_len=args.max_total_len,
        block=args.block,
    )
    print(f"Engine ready in {time.time() - t0:.1f}s\n")

    results = engine.benchmark(
        prompts=list(WORKLOADS.values()),
        gen_len=args.gen_len,
        n_runs=args.n_runs,
        n_warmup=args.n_warmup,
    )
    for r, name in zip(results, WORKLOADS.keys()):
        r["_name"] = name

    fmt_table(results)
    summary = report_summary(results)
    print()
    print(f"Average speedup    : {summary['avg_speedup']:.2f}x")
    print(f"Best speedup       : {summary['best_speedup']:.2f}x")
    print(f"Byte-identical     : {summary['all_byte_identical']} (cos = 1.0)")
    print(f"Workload hash      : {report_hash(results)}")

    out = {
        "env": env,
        "args": vars(args),
        "results": results,
        "summary": summary,
        "workload_hash": report_hash(results),
    }
    with open(args.save, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nReport written to: {os.path.abspath(args.save)}")
    print()
    print("To verify with NVIDIA Nsight Systems:")
    print(f"  nsys profile -t cuda,nvtx -o qwen_fast_run python {sys.argv[0]}")
    print("=" * 78)


if __name__ == "__main__":
    main()

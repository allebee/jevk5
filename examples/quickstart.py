"""Run one real support-ticket decision and measure warm GPU latency.

    python examples/quickstart.py
    python examples/quickstart.py --record demo-run.json

The recording starts after model loading and three warm-up calls. With --record,
pauses leave time to read the output; those pauses are excluded from timing.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path


STATE = "I was charged twice for order #7120. Please refund the duplicate charge."
QUESTION = {
    "type": "choice",
    "instructions": "Which team should handle this support ticket?",
    "criteria": {
        "billing": "Payments, invoices, charges, and refunds.",
        "technical": "Software bugs, errors, and broken integrations.",
        "sales": "Pricing, product questions, and new purchases.",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="alibiserikbay/JevK5")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--record", type=Path, help="Save results and a paced terminal transcript")
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2")

    import torch
    from jevk5 import JevK5

    if not torch.cuda.is_available():
        parser.exit(1, "This example needs an NVIDIA CUDA GPU. Check nvidia-smi first.\n")
    hardware = torch.cuda.get_device_properties(0)
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 12 * 1024**3:
        parser.exit(1, "Free at least 12 GiB of GPU memory before loading this example.\n")
    print("Loading weights and capturing CUDA graphs; first use may download the model.", flush=True)
    load_start = time.perf_counter()
    model = JevK5(args.model)
    load_seconds = time.perf_counter() - load_start
    for _ in range(3):
        model.decide(STATE, QUESTION)
    torch.cuda.synchronize()

    events = []
    started = time.perf_counter()

    def emit(text: str) -> None:
        events.append({"seconds": round(time.perf_counter() - started, 4), "text": text})
        print(text, flush=True)

    def pause_until(seconds: float) -> None:
        if args.record:
            time.sleep(max(0, seconds - (time.perf_counter() - started)))

    emit("JevK5 | A support ticket becomes a typed decision")
    emit(f"GPU: {hardware.name} ({hardware.total_memory / 1024**3:.1f} GiB)")
    execution = "CUDA graphs" if model.graphs else "eager"
    emit(f"Mode: bf16, batch 1, {execution}, model preloaded")
    pause_until(3)
    emit("\nTicket: " + STATE)
    pause_until(6)
    emit("Question: " + QUESTION["instructions"])
    emit("Options: billing / technical / sales")
    pause_until(9)

    torch.cuda.synchronize()
    decision_start = time.perf_counter()
    answer = model.decide(STATE, QUESTION)
    torch.cuda.synchronize()
    decision_ms = (time.perf_counter() - decision_start) * 1000
    probabilities = answer["probabilities"]
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values()):
        raise RuntimeError("The model returned invalid probabilities")
    if not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-5):
        raise RuntimeError("The model probabilities do not sum to one")
    emit("\nDecision: " + answer["choice"])
    for option, probability in probabilities.items():
        emit(f"  P({option}): {probability:.6f}")
    emit(f"Input: {answer['input_tokens']} tokens | Generated: 0 tokens")
    emit(f"This decision: {decision_ms:.2f} ms")
    pause_until(16)

    timings = []
    for _ in range(args.runs):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        model.decide(STATE, QUESTION)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - tick) * 1000)
    p50 = statistics.median(timings)
    p95 = sorted(timings)[math.ceil(0.95 * len(timings)) - 1]
    emit(f"\n{args.runs} warm calls, same ticket: p50 {p50:.2f} ms | p95 {p95:.2f} ms")
    emit("Timing includes tokenization and GPU synchronization; excludes model loading.")
    pause_until(22)
    emit("\nOpen weights + training code | Apache-2.0")
    emit("github.com/allebee/jevk5")
    pause_until(27)

    if args.record:
        packages = {}
        for name in ("jevk5", "torch", "transformers", "accelerate", "flash-linear-attention"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        payload = {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "model_commit": getattr(model.model.config, "_commit_hash", None),
            "gpu": hardware.name,
            "gpu_memory_gib": hardware.total_memory / 1024**3,
            "python": platform.python_version(),
            "packages": packages,
            "cuda": torch.version.cuda,
            "temperature": model.temperature,
            "graph_lengths": list(model.graphs),
            "load_seconds": load_seconds,
            "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            "warmup_calls": 3,
            "state": STATE,
            "question": QUESTION,
            "answer": answer,
            "decision_ms": decision_ms,
            "warm_timings_ms": timings,
            "p50_ms": p50,
            "p95_ms": p95,
            "duration_seconds": time.perf_counter() - started,
            "events": events,
        }
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Saved actual results and transcript to {args.record}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

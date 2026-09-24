"""Check a GGUF build against the bf16 model on JevBench's public items.

    llama-server --hf-repo alibiserikbay/JevK5-GGUF --hf-file jevk5-4b-v0.2-Q8_0.gguf -c 8192 -ngl 99
    python bench/gguf_check.py --label Q8_0 \
        --tasks <jevbench>/datasets/public --reference results/public231/jevk5-v0.2.jsonl

Quantization moves a calibrated distribution more than it moves the top answer, so this reports
both: how often the quantized build agrees with bf16, and what happens to accuracy, calibration
error and latency. Nothing here is a JevBench run; it is the same public items, scored the same
way, on whatever machine you are on.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from jevk5.gguf import JevK5GGUF


def ece(probs: list[float], hits: list[bool], bins: int = 10) -> float:
    conf, hit = np.array(probs), np.array(hits, dtype=float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return float(
        sum(
            abs(hit[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
            for b in range(bins)
            if (idx == b).any()
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="a running llama-server")
    parser.add_argument("--label", default="gguf")
    parser.add_argument("--tasks", type=Path, required=True, help="JevBench datasets/public")
    parser.add_argument("--reference", type=Path, help="a bf16 results.jsonl to agree with")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--temperature", type=float, help="calibration temperature (default v0.2's)")
    args = parser.parse_args()

    tasks = []
    for split in ("easy", "original", "hard"):
        tasks += [json.loads(line) for line in (args.tasks / f"{split}.jsonl").open()]
    if args.limit:
        tasks = tasks[: args.limit]
    reference = {}
    if args.reference:
        reference = {
            json.loads(line)["task_id"]: json.loads(line) for line in args.reference.open()
        }

    model = JevK5GGUF(args.url, temperature=args.temperature)
    rows, agree = [], 0
    for i, task in enumerate(tasks, 1):
        started = time.perf_counter()
        probs, tokens = model.probabilities(task["state"], task["question"])
        elapsed = time.perf_counter() - started
        top = max(probs, key=probs.get)
        if task["question"]["type"] == "noul":  # JevBench's names, as bench/jevk5_direct.py maps them
            top = {"true": "yes", "false": "no"}[top]
        row = {
            "task_id": task["id"],
            "split": "hard" if task["id"].startswith("hard") else
                     ("easy" if task["id"].startswith("easy") else "standard"),
            "predicted": top,
            "correct": top == str(task["expected"]),  # score answers are stored as ints
            "confidence": max(probs.values()),
            "seconds": round(elapsed, 4),
            "input_tokens": tokens,
        }
        if task["id"] in reference:
            row["reference"] = str(reference[task["id"]]["predicted"])
            row["agrees"] = row["predicted"] == row["reference"]
            agree += row["agrees"]
        rows.append(row)
        if i % 20 == 0:
            print(f"  {i}/{len(tasks)}", flush=True)

    print(f"\n{args.label}  (temperature {model.temperature})")
    for split in ("easy", "standard", "hard"):
        part = [r for r in rows if r["split"] == split]
        if not part:
            continue
        seconds = np.array([r["seconds"] for r in part])
        line = (f"  {split:<9} n {len(part):3d}  acc {np.mean([r['correct'] for r in part]):.3f}"
                f"  ECE {ece([r['confidence'] for r in part], [r['correct'] for r in part]):.3f}"
                f"  p50 {np.percentile(seconds, 50) * 1000:6.0f} ms"
                f"  p95 {np.percentile(seconds, 95) * 1000:6.0f} ms")
        if reference:
            line += f"  agrees with bf16 {np.mean([r['agrees'] for r in part]):.3f}"
        print(line)
    if reference:
        print(f"  overall agreement with bf16: {agree}/{len(rows)}")
    if model.missing:
        print(f"  {model.missing} decisions had an option letter outside the returned top-k")
    if args.out:
        with args.out.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

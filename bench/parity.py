"""Check that a runtime change leaves questions of up to 16 options bit-identical.

    python bench/parity.py dump --model alibiserikbay/JevK5 --tasks <jevbench>/datasets/public \
        --out new.jsonl                       # whichever jevk5 is on the path
    python bench/parity.py compare new.jsonl old.jsonl
    python bench/parity.py compare new.jsonl <jevbench run>.jsonl   # a JevBench results file
    python bench/parity.py dump --url http://127.0.0.1:8080 ...      # the llama.cpp client instead

`dump` writes each item's probabilities as Python floats, whose repr round-trips exactly, so
`compare` can require equality, not closeness. A JevBench results file names noul answers
"yes"/"no", as bench/jevk5_direct.py maps them; they are renamed to "true"/"false" here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def dump(args) -> int:
    import jevk5

    tasks = []
    for split in ("easy", "original", "hard"):
        tasks += [json.loads(line) for line in (args.tasks / f"{split}.jsonl").open()]
    if args.url:
        model = jevk5.JevK5GGUF(args.url, temperature=args.temperature)
    else:
        model = jevk5.JevK5(args.model, temperature=args.temperature)
    print(f"jevk5 {jevk5.__version__} from {Path(jevk5.__file__).parent}, {len(tasks)} items")
    with args.out.open("w") as f:
        for task in tasks:
            started = time.perf_counter()
            probs, tokens = model.probabilities(task["state"], task["question"])
            f.write(
                json.dumps(
                    {
                        "task_id": task["id"],
                        "probs": probs,
                        "tokens": tokens,
                        "seconds": time.perf_counter() - started,
                    }
                )
                + "\n"
            )
    print(f"wrote {args.out}")
    return 0


def load(path: Path) -> dict[str, dict[str, float]]:
    out = {}
    for line in path.open():
        row = json.loads(line)
        probs = row["probs"]
        if set(probs) == {"yes", "no"}:
            probs = {"true": probs["yes"], "false": probs["no"]}
        out[row["task_id"]] = probs
    return out


def compare(args) -> int:
    a, b = load(args.a), load(args.b)
    shared = sorted(set(a) & set(b))
    same = [t for t in shared if a[t] == b[t]]
    diffs = [max(abs(a[t][k] - b[t][k]) for k in a[t]) for t in shared if set(a[t]) == set(b[t])]
    tops = sum(max(a[t], key=a[t].get) == max(b[t], key=b[t].get) for t in shared)
    print(
        f"{args.a.name} vs {args.b.name}: {len(shared)} shared items "
        f"({len(a)} and {len(b)} in the files)"
    )
    print(f"  bit-identical distributions: {len(same)}/{len(shared)}")
    print(f"  same top answer:             {tops}/{len(shared)}")
    print(f"  max |dp|:                    {max(diffs, default=0.0):.3g}")
    return 0 if len(same) == len(shared) == len(a) == len(b) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--model", default="alibiserikbay/JevK5")
    d.add_argument("--tasks", type=Path, required=True, help="JevBench datasets/public")
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--url", help="a running llama-server: use the llama.cpp client")
    d.add_argument("--temperature", type=float, help="default: the model's own (GGUF: 1.532)")
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    args = parser.parse_args()
    return {"dump": dump, "compare": compare}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

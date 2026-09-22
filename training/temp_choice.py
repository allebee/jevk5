"""Compare temperature settings on held-out data only (never on JevBench items).

    cd training && python temp_choice.py ../models/jevk5 ../data/gate4/dev.jsonl \
        --settings '{"raw": {}, "shared": {"all": 1.532}, "kind": {"noul": 2.075, "choice": 1.316}}'

Pooled accuracy, ECE and NLL per held-out group for each setting. JevK5 v0.2 ships the shared
temperature: on the teacher's held-out domains the per-kind fit was slightly better (ECE 0.032 vs
0.034) and on the hand-written hard set clearly worse (0.129 vs 0.091), so the simpler one won.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lora import encode_items, evaluate
from order_avg import ece

from jevk5.runtime import JevK5


def softmax_t(p: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(p, 1e-12, 1.0)) / t
    z = np.exp(z - z.max())
    return z / z.sum()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    parser.add_argument("dev", type=Path)
    parser.add_argument("--settings", required=True, help="JSON: {name: {kind or 'all': T}}")
    parser.add_argument("--max-len", type=int, default=4096)
    args = parser.parse_args()
    settings = json.loads(args.settings)
    model = JevK5(args.model, graphs=False, temperature=1.0)
    rows = encode_items(model, [json.loads(x) for x in args.dev.open()], args.max_len)
    probs = evaluate(model, rows)["probs"]
    groups = {
        "teacher held-out domains": lambda s: s.startswith("teacher_"),
        "hand-written hard set": lambda s: s.startswith("hard_dev_"),
        "replay dev": lambda s: not s.startswith(("teacher_", "hard_dev_")),
    }
    for group, keep in groups.items():
        idx = [i for i, r in enumerate(rows) if keep(r["source"])]
        print(f"{group} (n {len(idx)})")
        gold = [int(rows[i]["target"].argmax()) for i in idx]
        for name, temps in settings.items():
            scaled = [
                softmax_t(probs[i], temps.get("all", temps.get(rows[i]["kind"], 1.0))) for i in idx
            ]
            acc = np.mean([p.argmax() == g for p, g in zip(scaled, gold, strict=True)])
            nll = -np.mean([np.log(p[g] + 1e-12) for p, g in zip(scaled, gold, strict=True)])
            print(f"  {name:<8} acc {acc:.3f}  ECE {ece(scaled, gold):.3f}  NLL {nll:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Does reading each question in two option orders and averaging help? (never on JevBench items)

    cd training && python order_avg.py ../models/jevk5 ../data/gate4/dev.jsonl --prefix teacher_,hard_dev_

Every dev item is scored with its options as given and in reverse; the two distributions are mapped
back to the option ids and averaged. Prints accuracy, top-label ECE and NLL per group.
On JevK5 v0.2's held-out data averaging lowered ECE but cost accuracy (0.799 -> 0.794) and doubles
the work per decision, so v0.2 ships one order. reflex and JevOne average two orders; Jobe and jqv
measured it and kept one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jevk5.runtime import JevK5, decision_options


def ece(probs: list[np.ndarray], gold: list[int], bins: int = 10) -> float:
    conf = np.array([p.max() for p in probs])
    hit = np.array([p.argmax() == g for p, g in zip(probs, gold, strict=True)], dtype=float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return float(
        sum(
            abs(hit[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
            for b in range(bins)
            if (idx == b).any()
        )
    )


def order_probs(model: JevK5, items: list[dict], reverse: bool) -> list[np.ndarray]:
    """Distribution over each item's options, always returned in the original option order."""
    out = []
    for item in items:
        options = decision_options(item["question"])
        used = options[::-1] if reverse else options
        ids = model.encode(item["state"], item["question"]["instructions"], [t for _, t in used])
        z = model.letter_logits(ids, len(used)) / model.temperature
        p = np.exp(z - z.max())
        p /= p.sum()
        out.append(p[::-1].copy() if reverse else p)
    return out


def report(name: str, probs: list[np.ndarray], items: list[dict], keys: list[list[str]]) -> None:
    gold = [k.index(item["expected"]) for item, k in zip(items, keys, strict=True)]
    acc = np.mean([p.argmax() == g for p, g in zip(probs, gold, strict=True)])
    nll = -np.mean([np.log(p[g] + 1e-12) for p, g in zip(probs, gold, strict=True)])
    print(f"  {name:<8} acc {acc:.3f}  ECE {ece(probs, gold):.3f}  NLL {nll:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    parser.add_argument("dev", type=Path)
    parser.add_argument("--prefix", default="teacher_,hard_dev_")
    args = parser.parse_args()
    prefixes = tuple(args.prefix.split(","))
    items = [r for r in map(json.loads, args.dev.open()) if r["source"].startswith(prefixes)]
    model = JevK5(args.model)
    forward, reverse = order_probs(model, items, False), order_probs(model, items, True)
    keys = [[k for k, _ in decision_options(item["question"])] for item in items]
    groups: dict[str, list[int]] = {"ALL": list(range(len(items)))}
    for i, item in enumerate(items):
        groups.setdefault(f"{item['source'].split('_')[0]}_{item['question']['type']}", []).append(i)
    for group, rows in groups.items():
        print(f"{group} (n {len(rows)})")
        pick, ks = [items[i] for i in rows], [keys[i] for i in rows]
        report("forward", [forward[i] for i in rows], pick, ks)
        report("reverse", [reverse[i] for i in rows], pick, ks)
        report("average", [(forward[i] + reverse[i]) / 2 for i in rows], pick, ks)
        disagree = sum(forward[i].argmax() != reverse[i].argmax() for i in rows)
        print(f"  the two orders disagree on {disagree} of {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

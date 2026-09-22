"""Assemble a training folder for laya_turbo.cuda.lora from teacher output plus a general replay.

    python scripts/build_train.py --teacher data/teacher/train.jsonl data/teacher/prob_train.jsonl \
        --test data/teacher/test.jsonl data/teacher/prob_test.jsonl \
        --replay data/decisions/train.jsonl --hard-dev data/decisions/hard_dev.jsonl --out data/gate2

train.jsonl: teacher questions both solutions agreed on (soft target = the teacher's answer
shares, or the exact distribution for the probability family), plus a replay of human-labelled items from openly licensed datasets (no templated
synthetic families) at --replay-share of the total, so easy decisions don't regress.
dev.jsonl: teacher questions from the held-out test domains, the hand-written hard set, and a
slice of the replay sources' held-out items. Nothing in dev is trained on.
"""

import argparse
import json
import random
import re
from pathlib import Path

REPLAY_SOURCES = ("mnli", "wanli", "boolq", "banking77", "arc", "csqa", "mmlu_pro")


def neutral_keys(criteria: dict) -> dict[str, str]:
    """Option keys rebuilt from the option text. Authors sometimes named keys after their role
    ("correct_net_amount", "naive_total"), which gives the answer away; option texts only state
    outcomes, so keys made from them carry no such hint."""
    mapping, used = {}, set()
    for key, text in criteria.items():
        words = re.findall(r"[a-z0-9]+", str(text).lower())[:6] or ["option"]
        new = "_".join(words)
        while new in used:
            new += "_x"
        used.add(new)
        mapping[key] = new
    return mapping


def teacher_items(path: Path) -> list[dict]:
    items = []
    for line in path.open():
        r = json.loads(line)
        if not r.get("agree"):
            continue
        question, expected, probs = dict(r["question"]), r["expected"], r.get("teacher_probs")
        if question["type"] == "choice":
            mapping = neutral_keys(question["criteria"])
            question["criteria"] = {mapping[k]: v for k, v in question["criteria"].items()}
            expected = mapping[expected]
            probs = {mapping[k]: v for k, v in probs.items()} if probs else None
        item = {
            "id": r["id"],
            "source": "teacher_" + r["spec"]["family"],
            "state": r["state"],
            "question": question,
            "expected": expected,
        }
        if probs:
            item["gold_probs"] = probs
        items.append(item)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--teacher", type=Path, nargs="+", required=True)
    parser.add_argument("--test", type=Path, nargs="+", required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--replay-dev", type=Path, default=Path("data/decisions/dev.jsonl"))
    parser.add_argument("--hard-dev", type=Path, required=True)
    parser.add_argument("--replay-share", type=float, default=0.3)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    teacher = [item for path in args.teacher for item in teacher_items(path)]
    replay = [
        r for r in map(json.loads, args.replay.open()) if r["source"].startswith(REPLAY_SOURCES)
    ]
    rng.shuffle(replay)
    n_replay = int(len(teacher) * args.replay_share / (1 - args.replay_share))
    train = teacher + replay[:n_replay]
    rng.shuffle(train)
    replay_dev = [
        r for r in map(json.loads, args.replay_dev.open()) if r["source"].startswith(REPLAY_SOURCES)
    ]
    rng.shuffle(replay_dev)
    dev = (
        [item for path in args.test for item in teacher_items(path)]
        + [json.loads(line) for line in args.hard_dev.open()]
        + replay_dev[:600]
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("dev", dev)):
        with (args.out / f"{name}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"train {len(train)} ({len(teacher)} teacher + {min(n_replay, len(replay))} replay), "
        f"dev {len(dev)}"
    )


if __name__ == "__main__":
    main()

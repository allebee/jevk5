"""Questions with more than 16 options: BANKING77, CLINC150+OOS and MASSIVE, train splits only.

    python bench/many_options.py build --out many/            # items, in the Decision Index's shape
    python bench/many_options.py run --items many/banking77.jsonl --model alibiserikbay/JevK5 \
        --out many/banking77.results.jsonl
    python bench/many_options.py table many/*.results.jsonl
    python bench/many_options.py run --url http://127.0.0.1:8080 ...   # the llama.cpp client
    python bench/many_options.py agree many/gguf.results.jsonl many/banking77.results.jsonl

The BANKING77 and CLINC150 items copy the Decision Index's request shape for those benchmarks
(github.com/apolinario/decision-index, suite/build/normalize_direct.py): an empty state, the
user's text in the instructions, and every intent offered as `option_<i>` in the published label
order. Only train splits are sampled; the test splits, which the index scores, are read solely to
drop train texts that also occur there. Texts JevK5 v0.2 trained on can be excluded with
`--exclude` (its replay held 1,500 BANKING77 train texts). MASSIVE en-US, which is not on the
index, is the set the many-option settings were chosen and fitted on (jevk5.prompt.TEMPERATURES).

`run` times one `probabilities()` call per design and records every pass. `table` adds, from the
knockout's recorded passes, its combination at temperature 1 ("knockout-t1") and the
hierarchical-mass rule ("knockout-mass"), which shares the final's mass on each group's finalists
among the whole group in proportion to the in-group probabilities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

BANKING = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
CLINC = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
CLINC_SHA256 = (
    "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0"  # as the index pins
)
# MASSIVE (not on the index): the >16-option settings are fitted on its en-US train partition.
MASSIVE = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
OOS_SHARE = 1000 / 5500  # CLINC150+OOS on the index: 4,500 in-scope and 1,000 out-of-scope tests


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def order(text: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}:{text}".encode()).hexdigest()


def round_robin(by_label: dict[str, list[str]], count: int, seed: int) -> list[tuple[str, str]]:
    """`count` (text, label) pairs, cycling over labels in a fixed order, each label's texts in
    hash order: the class balance of the index's test splits, whatever the train split's."""
    queues = {
        label: sorted(texts, key=lambda t: order(t, seed)) for label, texts in by_label.items()
    }
    labels = sorted(queues, key=lambda label: order(label, seed))
    picked, depth = [], 0
    while len(picked) < count and any(len(q) > depth for q in queues.values()):
        for label in labels:
            if len(queues[label]) > depth and len(picked) < count:
                picked.append((queues[label][depth], label))
        depth += 1
    return picked


def item(dataset: str, i: int, instructions: str, descriptions: list[str], gold: int) -> dict:
    return {
        "id": f"{dataset}:train:{i}",
        "dataset": dataset,
        "state": {},
        "question": {
            "type": "choice",
            "instructions": instructions,
            "criteria": {f"option_{k}": d for k, d in enumerate(descriptions)},
        },
        "expected": f"option_{gold}",
    }


def build(args) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    excluded = set()
    for path in args.exclude or []:
        for line in path.open():
            row = json.loads(line)
            if isinstance(row.get("state"), str):
                excluded.add(norm(row["state"]))

    labels = json.loads(fetch(BANKING + "categories.json"))
    assert len(labels) == 77
    rows = {
        split: list(csv.DictReader(io.StringIO(fetch(BANKING + f"{split}.csv").decode())))
        for split in ("train", "test")
    }
    test = {norm(r["text"]) for r in rows["test"]}
    by_label, dropped = defaultdict(list), {"test": 0, "trained": 0}
    for r in rows["train"]:
        key = norm(r["text"])
        if key in test:
            dropped["test"] += 1
        elif key in excluded:
            dropped["trained"] += 1
        else:
            by_label[r["category"]].append(r["text"])
    picked = round_robin(by_label, args.count, args.seed)
    items = [
        item(
            "BANKING77",
            i,
            "Classify the banking intent of this user request:\n" + text,
            labels,
            labels.index(label),
        )
        for i, (text, label) in enumerate(picked)
    ]
    write(args.out / "banking77.jsonl", items)
    print(f"BANKING77: {len(rows['train'])} train rows, dropped {dropped}, sampled {len(items)}")

    raw = fetch(CLINC)
    assert hashlib.sha256(raw).hexdigest() == CLINC_SHA256, "data_full.json changed upstream"
    data = json.loads(raw)
    names = sorted({label for _, label in data["train"]} | {"oos"})
    assert len(names) == 151
    descriptions = [
        "out of scope: none of the listed intents" if n == "oos" else n.replace("_", " ")
        for n in names
    ]
    test = {norm(text) for split in ("test", "oos_test") for text, _ in data[split]}
    by_label, dropped = defaultdict(list), {"test": 0, "trained": 0}
    for split in ("train", "oos_train"):
        for text, label in data[split]:
            key = norm(text)
            if key in test:
                dropped["test"] += 1
            elif key in excluded:
                dropped["trained"] += 1
            else:
                by_label[label].append(text)
    oos = round(args.count * OOS_SHARE)
    picked = round_robin({"oos": by_label.pop("oos")}, oos, args.seed)
    picked += round_robin(by_label, args.count - oos, args.seed)
    items = [
        item(
            "CLINC150+OOS",
            i,
            "Classify the intent of this user request, or choose out of scope if none applies:\n"
            + text,
            descriptions,
            names.index(label),
        )
        | {"oos": f"option_{names.index('oos')}"}
        for i, (text, label) in enumerate(picked)
    ]
    write(args.out / "clinc150.jsonl", items)
    print(
        f"CLINC150+OOS: {sum(len(data[s]) for s in ('train', 'oos_train'))} train rows, "
        f"dropped {dropped}, sampled {len(items)} ({oos} out of scope)"
    )

    import tarfile

    if args.massive and args.massive.suffix == ".jsonl":  # en-US.jsonl, already extracted
        rows = [json.loads(line) for line in args.massive.open()]
    else:
        raw = args.massive.read_bytes() if args.massive else fetch(MASSIVE)
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith("/en-US.jsonl"))
            rows = [
                json.loads(line) for line in tar.extractfile(member).read().decode().splitlines()
            ]
    intents = sorted({r["intent"] for r in rows})
    test = {norm(r["utt"]) for r in rows if r["partition"] == "test"}
    by_label = defaultdict(list)
    for r in rows:
        if r["partition"] == "train" and norm(r["utt"]) not in test:
            by_label[r["intent"]].append(r["utt"])
    picked = round_robin(by_label, args.count, args.seed)
    items = [
        item(
            "MASSIVE-en",
            i,
            "Classify the intent of this user request:\n" + text,
            intents,
            intents.index(label),
        )
        for i, (text, label) in enumerate(picked)
    ]
    write(args.out / "massive.jsonl", items)
    print(f"MASSIVE en-US: {len(intents)} intents, sampled {len(items)} train rows")
    return 0


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------------------------
# run


def recorder(model):
    """Wrap the runtime's encode and letter_logits so each pass's options and letter
    distribution are kept, computed exactly as the runtime computes them."""
    passes = []
    encode, letter_logits = model.encode, model.letter_logits

    def rec_encode(state, criterion, options):
        passes.append({"texts": list(options)})
        return encode(state, criterion, options)

    def rec_logits(ids, count):
        z = letter_logits(ids, count)
        logits = z / model.temperature
        p = np.exp(logits - logits.max())
        p /= p.sum()
        passes[-1].update(tokens=len(ids), probs=[float(v) for v in p])
        return z

    model.encode, model.letter_logits = rec_encode, rec_logits
    return passes


def derived(row: dict) -> dict[str, list[float]]:
    """The knockout's recorded passes, combined at temperature 1 by its own rule ("knockout-t1")
    and by the hierarchical-mass rule ("knockout-mass"). Single-final knockouts only."""
    inner, final = row.get("inner"), row.get("final")
    if not inner or final is None:
        return {}
    keys = [f"option_{i}" for i in range(row["options"])]
    parts, at = [], 0
    for p in inner:
        parts.append((keys[at : at + len(p)], [q / sum(p) for q in p]))
        at += len(p)
    masses = [sum(final.get(k, 0.0) for k in part) for part, _ in parts]
    in_final = sum(
        m * sum(q for k, q in zip(part, p) if k in final) for (part, p), m in zip(parts, masses)
    )
    own = [
        final[k] * in_final if k in final else m * q
        for (part, p), m in zip(parts, masses)
        for k, q in zip(part, p)
    ]
    mass = [m * q for (_, p), m in zip(parts, masses) for q in p]
    return {"knockout-t1": [w / sum(own) for w in own], "knockout-mass": mass}


def run(args) -> int:
    from jevk5 import JevK5, JevK5GGUF
    from jevk5.prompt import decision_options

    items = [json.loads(line) for line in args.items.open()]
    if args.limit:
        items = items[: args.limit]
    if args.url:  # llama.cpp: the designs themselves, without the pass-level extras
        model, passes = JevK5GGUF(args.url, method=args.methods[0]), None
    else:
        model = JevK5(args.model, method=args.methods[0])
        passes = recorder(model)
    warm = items[0]
    for method in args.methods:
        model.method = method
        model.probabilities(warm["state"], warm["question"])

    with args.out.open("w") as f:
        for n, it in enumerate(items, 1):
            row = {
                "id": it["id"],
                "dataset": it["dataset"],
                "expected": it["expected"],
                "oos": it.get("oos"),
                "options": len(it["question"]["criteria"]),
                "designs": {},
            }
            texts = [text for _, text in decision_options(it["question"])]
            for method in args.methods:
                model.method = method
                if passes is None:
                    started = time.perf_counter()
                    probs, tokens = model.probabilities(it["state"], it["question"])
                    row["designs"][method] = {
                        "probs": list(probs.values()),
                        "tokens": tokens,
                        "passes": None,
                        "seconds": time.perf_counter() - started,
                    }
                    continue
                passes.clear()
                started = time.perf_counter()
                probs, tokens = model.probabilities(it["state"], it["question"])
                seconds = time.perf_counter() - started
                row["designs"][method] = {
                    "probs": list(probs.values()),
                    "tokens": tokens,
                    "passes": len(passes),
                    "seconds": seconds,
                }
                if method == "knockout" and len(passes) == -(-len(texts) // 16) + 1:
                    row["inner"] = [q["probs"] for q in passes[:-1]]
                    row["final"] = dict(
                        zip([t.split(":", 1)[0] for t in passes[-1]["texts"]], passes[-1]["probs"])
                    )
                if method == "tree":
                    row["tree_root"] = passes[0]["probs"]
            f.write(json.dumps(row) + "\n")
            if n % 50 == 0:
                print(f"  {n}/{len(items)}", flush=True)
    print(f"wrote {args.out}")
    return 0


# ---------------------------------------------------------------------------------------------
# table


def ece(conf: np.ndarray, hit: np.ndarray, bins: int = 10) -> float:
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return float(
        sum(
            abs(hit[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
            for b in range(bins)
            if (idx == b).any()
        )
    )


def macro_f1(gold: list[int], pred: list[int]) -> float:
    """Macro-F1 over the union of observed and predicted labels, as the index scores it."""
    scores = []
    for label in set(gold) | set(pred):
        tp = sum(g == p == label for g, p in zip(gold, pred))
        fp = sum(p == label != g for g, p in zip(gold, pred))
        fn = sum(g == label != p for g, p in zip(gold, pred))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(scores))


def summarize(rows: list[dict], design: str) -> dict:
    gold = [int(r["expected"].split("_")[1]) for r in rows]
    probs = [np.array(r["designs"][design]["probs"]) for r in rows]
    pred = [int(p.argmax()) for p in probs]
    conf = np.array([p.max() for p in probs])
    hit = np.array([p == g for p, g in zip(pred, gold)], dtype=float)
    p_gold = np.array([p[g] for p, g in zip(probs, gold)])
    seconds = np.array([r["designs"][design]["seconds"] for r in rows]) * 1000
    out = {
        "n": len(rows),
        "accuracy": float(hit.mean()),
        "macro_f1": macro_f1(gold, pred),
        "ece": ece(conf, hit),
        "nll": float(-np.log(np.maximum(p_gold, 1e-12)).mean()),
        "brier": float(
            np.mean([((p - np.eye(len(p))[g]) ** 2).sum() for p, g in zip(probs, gold)])
        ),
        "confidence": float(conf.mean()),
        "sum_error": float(max(abs(p.sum() - 1) for p in probs)),
        "passes": float(np.mean([r["designs"][design]["passes"] or np.nan for r in rows])),
        "tokens": float(np.mean([r["designs"][design]["tokens"] for r in rows])),
        "p50_ms": float(np.percentile(seconds, 50)),
        "p95_ms": float(np.percentile(seconds, 95)),
    }
    if rows[0].get("oos"):
        is_oos = np.array([r["expected"] == r["oos"] for r in rows])
        said_oos = np.array([f"option_{p}" == r["oos"] for p, r in zip(pred, rows)])
        out["in_scope_accuracy"] = float(hit[~is_oos].mean())
        out["oos_recall"] = float(said_oos[is_oos].mean())
        out["oos_precision"] = float(is_oos[said_oos].mean()) if said_oos.any() else 0.0
    return out


def table(args) -> int:
    summary = {}
    for path in args.results:
        rows = [json.loads(line) for line in path.open()]
        drift = 0.0
        for row in rows:
            for name, probs in derived(row).items():
                row["designs"][name] = {**row["designs"]["knockout"], "probs": probs}
            if "knockout-t1" in row["designs"]:  # the recorded passes reproduce the shipped rule
                from jevk5.prompt import TEMPERATURES

                t = TEMPERATURES["knockout"]
                q = np.array(row["designs"]["knockout-t1"]["probs"]) ** (1 / t)
                drift = max(
                    drift, float(np.abs(q / q.sum() - row["designs"]["knockout"]["probs"]).max())
                )
        designs = list(rows[0]["designs"])
        print(f"\n{path.name}  ({rows[0]['dataset']}, {rows[0]['options']} options, n {len(rows)})")
        keys = [
            "accuracy",
            "macro_f1",
            "ece",
            "nll",
            "brier",
            "confidence",
            "passes",
            "tokens",
            "p50_ms",
            "p95_ms",
        ]
        extra = ["in_scope_accuracy", "oos_recall", "oos_precision"]
        stats = {d: summarize(rows, d) for d in designs}
        keys += [k for k in extra if k in stats[designs[0]]]
        print(f"  {'design':<14}" + "".join(f"{k:>12}" for k in keys))
        for d in designs:
            print(f"  {d:<14}" + "".join(f"{stats[d][k]:>12.4g}" for k in keys))
        if "final" in rows[0]:
            kept = np.mean([r["expected"] in r["final"] for r in rows])
            print(
                f"  gold reached the knockout final in {kept:.3f} of items; recomputing the "
                f"knockout from its passes differs by at most {drift:.2g}"
            )
        if "tree_root" in rows[0]:
            from jevk5.prompt import LETTERS, groups

            def group_of(r):
                runs = groups(r["options"], min(len(LETTERS), -(-r["options"] // len(LETTERS))))
                gold = int(r["expected"].split("_")[1])
                return next(g for g, run in enumerate(runs) if gold in run)

            first = np.mean([int(np.argmax(r["tree_root"])) == group_of(r) for r in rows])
            print(f"  the tree's first pass chose the gold's group in {first:.3f} of items")
        summary[rows[0]["dataset"]] = stats
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1))
    return 0


def agree(args) -> int:
    """One results file against another on the items and designs they share: how often the top
    answer is the same, and how far apart the distributions are (total variation)."""
    a = {r["id"]: r for r in map(json.loads, args.a.open())}
    b = {r["id"]: r for r in map(json.loads, args.b.open())}
    shared = [i for i in a if i in b]
    print(f"{args.a.name} vs {args.b.name}: {len(shared)} shared items")
    for design in a[shared[0]]["designs"]:
        if design not in b[shared[0]]["designs"]:
            continue
        pa = [np.array(a[i]["designs"][design]["probs"]) for i in shared]
        pb = [np.array(b[i]["designs"][design]["probs"]) for i in shared]
        top = np.mean([x.argmax() == y.argmax() for x, y in zip(pa, pb)])
        tv = np.array([0.5 * np.abs(x - y).sum() for x, y in zip(pa, pb)])
        acc = [
            np.mean(
                [int(p.argmax()) == int(a[i]["expected"].split("_")[1]) for p, i in zip(ps, shared)]
            )
            for ps in (pa, pb)
        ]
        print(
            f"  {design:<14} same top {top:.3f}  TV mean {tv.mean():.4f} max {tv.max():.4f}"
            f"  accuracy {acc[0]:.3f} vs {acc[1]:.3f}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--count", type=int, default=500)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--exclude", type=Path, nargs="*", help="JSONL whose `state` texts to skip")
    b.add_argument(
        "--massive", type=Path, help="a local copy of the MASSIVE tarball or en-US.jsonl"
    )
    r = sub.add_parser("run")
    r.add_argument("--items", type=Path, required=True)
    r.add_argument("--model", default="alibiserikbay/JevK5")
    r.add_argument("--methods", nargs="+", default=["knockout", "tree"])
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--limit", type=int)
    r.add_argument("--url", help="a running llama-server: use the llama.cpp client")
    t = sub.add_parser("table")
    t.add_argument("results", type=Path, nargs="+")
    t.add_argument("--json", type=Path)
    g = sub.add_parser("agree")
    g.add_argument("a", type=Path)
    g.add_argument("b", type=Path)
    args = parser.parse_args()
    return {"build": build, "run": run, "table": table, "agree": agree}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

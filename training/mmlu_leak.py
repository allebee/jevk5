"""Did training on MMLU-Pro test items inflate JevK5's MMLU-Pro accuracy?

The design the Jev Decision Index used for reflex 4B, on our own item formatting. It runs in the
training environment (laya_turbo on the H100 server), next to data/gate4 and data/decisions.

Trained: the 940 MMLU-Pro rows in data/gate4/train.jsonl (JevK5 v0.2's training set).
Unseen: MMLU-Pro test items JevK5 v0.2 never trained on - the rest of the replay pool and the
held-out dev rows. The untrained base model is scored on both sets to control for difficulty.
"""
import json, sys
import numpy as np
sys.modules["fla"] = None
from laya_turbo.cuda.lora import encode_items, evaluate
from laya_turbo.cuda.qwen_direct import DirectModel

def key(r):
    return json.dumps(r["state"])[:400]

gate4 = [json.loads(l) for l in open("data/gate4/train.jsonl")]
trained = [r for r in gate4 if r["source"] == "mmlu_pro"]
seen = {key(r) for r in trained}
pool = [json.loads(l) for l in open("data/decisions/train.jsonl")]
pool += [json.loads(l) for l in open("data/decisions/dev.jsonl")]
unseen = [r for r in pool if r["source"] == "mmlu_pro" and key(r) not in seen]
rng = np.random.default_rng(0)
unseen = [unseen[i] for i in rng.permutation(len(unseen))[:2000]]
print(f"trained items {len(trained)}, unseen items {len(unseen)}", flush=True)

def hits(model):
    dm = DirectModel(model, style="semif")
    out = {}
    for name, rows in (("trained", trained), ("unseen", unseen)):
        rows = [dict(r, source=name) for r in rows]
        enc = encode_items(dm, rows, 4096)
        probs = evaluate(dm, enc)["probs"]
        out[name] = np.array([p.argmax() == e["target"].argmax() for e, p in zip(enc, probs)], dtype=float)
    del dm
    import torch; torch.cuda.empty_cache()
    return out

res = {"JevK5 v0.2": hits("models/gate4"), "untrained Qwen3.5-4B": hits("Qwen/Qwen3.5-4B")}
for name, r in res.items():
    print(f"{name:<22} trained {r['trained'].mean():.3f} (n={len(r['trained'])})  unseen {r['unseen'].mean():.3f} (n={len(r['unseen'])})  gap {r['trained'].mean() - r['unseen'].mean():+.3f}")
# difference in differences, bootstrap over items
a, b = res["JevK5 v0.2"], res["untrained Qwen3.5-4B"]
did = []
for _ in range(2000):
    i = rng.integers(0, len(a["trained"]), len(a["trained"])); j = rng.integers(0, len(a["unseen"]), len(a["unseen"]))
    did.append((a["trained"][i].mean() - a["unseen"][j].mean()) - (b["trained"][i].mean() - b["unseen"][j].mean()))
lo, hi = np.percentile(did, [2.5, 97.5])
point = (a["trained"].mean() - a["unseen"].mean()) - (b["trained"].mean() - b["unseen"].mean())
print(f"memorization effect (JevK5 gap minus base gap): {point:+.3f}  95% CI {lo:+.3f} to {hi:+.3f}")

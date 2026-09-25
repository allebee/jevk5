"""In-process JevBench adapter for JevK5-Lite (a CPU label-conditioned encoder; preview).

Copy to <jevbench>/jevbench/adapters/ and apply jevbench-registration.patch. The endpoint argument is
the model (Hub id or local folder; default alibiserikbay/JevK5-Lite); `revision` pins the Hub commit.

Mapping, one JevK5-Lite head per question:
  * text   = the state (an object is serialized with json.dumps)
  * task   = the question's instructions
  * noul   -> labels "true" / "false"; the probabilities are returned as yes / no
  * choice -> one label per option: its description, or its key when the description is empty
  * score  -> one label per level text, lowest first; the probabilities are returned per level index
  If two options would get the same label, every label of that question becomes "<id>: <text>".
  The answer is JevK5-Lite's calibrated softmax over those labels, mapped back to the option ids.

Length: JevK5-Lite reads at most 512 tokens and puts the task and the labels first. The state is cut from
its end to fit whatever room is left, so a long document is mostly unread; the number of state tokens
dropped is kept in raw["truncated_state_tokens"]. If the task and labels alone do not fit, the runtime
raises and the item is recorded as a failure (none of the 231 public items comes close: at most 165 tokens).
CPU by default: JEVK5_LITE_THREADS (default 16) and JEVK5_LITE_DTYPE (fp32 or bf16; default fp32) set
the runtime.
"""

from __future__ import annotations

import json
import os
import time

from .base import DecisionResult


def lite_head(question: dict) -> tuple[str, list[str], list[str]]:
    """(task, labels, option ids in the same order) for one JevBench question."""
    kind, crit = question["type"], question.get("criteria")
    if kind == "noul":
        return question["instructions"], ["true", "false"], ["true", "false"]
    if kind == "choice":
        if isinstance(crit, list):
            crit = dict.fromkeys(crit)
        pairs = [(str(k), str(v or k)) for k, v in crit.items()]
    else:  # score: ordinal levels, lowest first
        pairs = [(str(i), str(level)) for i, level in enumerate(crit)]
    labels = [text for _, text in pairs]
    if len(set(labels)) < len(labels):
        labels = [f"{key}: {text}" for key, text in pairs]
    return question["instructions"], labels, [key for key, _ in pairs]


def state_text(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


class JevK5LiteAdapter:
    name = "jevk5_lite"
    cost_basis = "self_hosted_cpu"

    def __init__(self, endpoint=None, model=None, key_env="", timeout_s=None,
                 price_input_per_m=None, price_output_per_m=None, revision=None):
        self.endpoint = endpoint or "alibiserikbay/JevK5-Lite"
        self.model = model or self.endpoint
        self.revision = revision
        self.price_input_per_m = price_input_per_m
        self.price_output_per_m = price_output_per_m
        self._loaded = None

    def load(self):
        if self._loaded is None:
            import torch

            from jevk5 import JevK5Lite

            dtypes = {"fp32": torch.float32, "bf16": torch.bfloat16}
            dtype = dtypes[os.environ.get("JEVK5_LITE_DTYPE", "fp32")]
            t0 = time.perf_counter()
            self._loaded = JevK5Lite.from_pretrained(
                self.endpoint, dtype=dtype, threads=int(os.environ.get("JEVK5_LITE_THREADS", "16")),
                revision=self.revision)
            self.load_s = time.perf_counter() - t0
        return self._loaded

    def run(self, task) -> DecisionResult:
        res = DecisionResult(adapter=self.name, ok=False, probs_source="native", model=self.model)
        res.request_body = {"question": task.question}
        try:
            lite = self.load()
        except Exception as e:  # noqa: BLE001
            res.error = f"load failed: {type(e).__name__}: {str(e)[:250]}"
            return res
        t0 = time.perf_counter()
        try:
            name, labels, keys = lite_head(task.question)
            text = state_text(task.state)
            out = lite.classify(text, {name: labels})[name]["probabilities"]
            probs = {key: out[label] for key, label in zip(keys, labels)}
        except Exception as e:  # noqa: BLE001
            res.latency_s = time.perf_counter() - t0
            res.error = f"{type(e).__name__}: {str(e)[:300]}"
            return res
        res.latency_s = time.perf_counter() - t0
        # bookkeeping after the timed call: tokens read, and state tokens cut to fit 512
        ids, _, _, (s, e) = lite.encode(text, [(name, labels, False, None)])
        dropped = len(lite.tok(text, add_special_tokens=False)["input_ids"]) - (e - s)
        res.usage = {"input_tokens": len(ids), "output_tokens": 0}
        res.raw = {"probabilities": probs, "labels": dict(zip(keys, labels)),
                   "truncated_state_tokens": dropped,
                   "temperature_single": lite.temperature_single,
                   "readout": "JevK5-Lite: [LABEL] marker vs pooled text, softmax / temperature"}
        if task.question["type"] == "noul":
            res.probs = {"yes": probs["true"], "no": probs["false"]}
        else:
            res.probs = probs
        res.ok = True
        return res

    def reserve_estimate(self, task) -> float:
        return 0.0

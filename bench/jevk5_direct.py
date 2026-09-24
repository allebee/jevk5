"""In-process JevBench adapter for JevK5 (Qwen3.5-4B + distilled LoRA, one forward pass).

Copy to <jevbench>/jevbench/adapters/ and apply jevbench-registration.patch. The endpoint
argument is the model (Hub id or local folder; default alibiserikbay/JevK5); `revision` pins the
Hub commit. Option mapping is the same as JevBench's semif_direct adapter:
  * noul   -> options "true"/"false", description = criterion text or "The proposition is <id>."
  * choice -> one option per criteria key, "<key>: <description>"
  * score  -> options "0".."k-1", "<i>: <level text>"
The distribution is the softmax over the declared answer letters' logits at the last position,
divided by JevK5's calibration temperature (jevk5_config.json next to the weights). Questions with
more than 16 options take several such passes (jevk5.prompt.spread).
Inputs up to 16,384 tokens per pass are answered (CUDA graphs up to 4,096, eager beyond); longer
ones are refused, never truncated. Usage counts the tokens of every pass.
"""

from __future__ import annotations

import time

from .base import DecisionResult


class JevK5DirectAdapter:
    name = "jevk5_direct"
    cost_basis = "self_hosted_gpu"

    def __init__(self, endpoint=None, model=None, key_env="", timeout_s=None,
                 price_input_per_m=None, price_output_per_m=None, revision=None,
                 max_tokens=16384):
        self.endpoint = endpoint or "alibiserikbay/JevK5"
        self.model = model or self.endpoint
        self.revision = revision
        self.max_tokens = max_tokens
        self.price_input_per_m = price_input_per_m
        self.price_output_per_m = price_output_per_m
        self._loaded = None

    def load(self):
        if self._loaded is None:
            source = self.endpoint
            if self.revision:
                from huggingface_hub import snapshot_download

                source = snapshot_download(self.endpoint, revision=self.revision)
            from jevk5 import JevK5

            t0 = time.perf_counter()
            self._loaded = JevK5(source)
            self.load_s = time.perf_counter() - t0
        return self._loaded

    def run(self, task) -> DecisionResult:
        res = DecisionResult(adapter=self.name, ok=False, probs_source="native", model=self.model)
        res.request_body = {"question": task.question}
        try:
            model = self.load()
        except Exception as e:  # noqa: BLE001
            res.error = f"load failed: {type(e).__name__}: {str(e)[:250]}"
            return res
        t0 = time.perf_counter()
        try:
            probs, tokens = model.probabilities(task.state, task.question)
            longest = getattr(model, "last_pass_tokens", tokens)
            if longest > self.max_tokens:
                raise ValueError(f"{longest} input tokens exceed limit {self.max_tokens}")
        except Exception as e:  # noqa: BLE001
            res.latency_s = time.perf_counter() - t0
            res.error = f"{type(e).__name__}: {str(e)[:300]}"
            return res
        res.latency_s = time.perf_counter() - t0
        res.usage = {"input_tokens": tokens, "output_tokens": 0}
        res.raw = {"probabilities": probs, "temperature": model.temperature,
                   "readout": "last-position answer-letter logits, softmax / temperature"}
        if task.question["type"] == "noul":
            res.probs = {"yes": probs["true"], "no": probs["false"]}
        else:
            res.probs = probs
        res.ok = True
        return res

    def reserve_estimate(self, task) -> float:
        return 0.0

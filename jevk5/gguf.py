"""JevK5 on llama.cpp: the same option-letter readout from a GGUF, on Apple silicon or CPU.

Start llama-server with a GGUF from huggingface.co/alibiserikbay/JevK5-GGUF, then ask it for typed
decisions:

    llama-server --hf-repo alibiserikbay/JevK5-GGUF --hf-file jevk5-4b-v0.3-Q8_0.gguf -c 8192 -ngl 99

    from jevk5 import JevK5GGUF
    model = JevK5GGUF(temperature=1.367)     # each file's temperature is on the JevK5-GGUF card
    model.decide("Delivered to No. 17; the customer lives at No. 71.",
                 {"type": "choice", "instructions": "What happened to the parcel?",
                  "criteria": ["delivered", "misdelivered", "unknown"]})

Nothing is generated: the server evaluates the prompt, returns the log-probabilities at the answer
position, and this module keeps the declared options' letters and renormalises them under the
calibration temperature - the same readout as the CUDA runtime, and mathematically identical,
since a softmax over letter logits and a renormalised slice of the full-vocabulary softmax agree.
Questions with more than 16 options take several such requests, combined exactly as the CUDA
runtime combines its passes (`jevk5.prompt.spread`).

Needs a llama.cpp new enough to load Qwen3.5's hybrid layers (the published files were made and
checked at commit 9575389). To convert your own checkpoint, pass `--no-mtp` to
convert_hf_to_gguf.py, or the speculative block makes a file that fails to load. Only the standard
library is used here, so `pip install --no-deps` is enough for this path.
"""

from __future__ import annotations

import json
import math
import sys
import urllib.request
from pathlib import Path

from .prompt import LETTERS, METHODS, answer, decision_options, prompt_text, spread

MISSING_MARGIN = 2.0  # a letter outside the returned top-k sits at least this far below the last


class JevK5GGUF:
    """JevK5 read through a running llama-server.

    `url` is the server (llama-server's default port); `temperature` defaults to the value in a
    `jevk5_config.json` passed as `config`, or to JevK5 v0.2's 1.532. Pass the file's own value:
    1.367 for JevK5 v0.3 (4B), 1.089 for JevK5-9B, 1.42 for JevK5-2B. `top_k` asks for that many
    token log-probabilities at the answer position, which needs to cover the 16 letters. `method`
    reads questions with more than 16 options, "knockout" or "tree", exactly as the CUDA runtime
    does (`jevk5.prompt.spread`).
    """

    def __init__(
        self,
        url: str = "http://127.0.0.1:8080",
        temperature: float | None = None,
        top_k: int = 40,
        timeout_s: float = 600.0,
        config: str | None = None,
        method: str = "knockout",
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; use one of {METHODS}")
        self.method = method
        self.url = url.rstrip("/")
        self.top_k = top_k
        self.timeout_s = timeout_s
        self.temperature = temperature if temperature is not None else self._temperature(config)
        self.missing = 0  # decisions where an option letter fell outside top_k

    @staticmethod
    def _temperature(config: str | None) -> float:
        if config and Path(config).exists():
            return float(json.loads(Path(config).read_text()).get("temperature", 1.0))
        return 1.532

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read())

    def _logprobs(self, prompt: str) -> tuple[dict[str, float], int, float]:
        # Tokenise first with parse_special, then send ids: /completion treats a prompt string as
        # plain text, which would split <|im_start|> into six characters and wreck the layout.
        tokens = self._post(
            "/tokenize", {"content": prompt, "add_special": False, "parse_special": True}
        )["tokens"]
        out = self._post(
            "/completion",
            {
                "prompt": tokens,
                "n_predict": 1,
                "n_probs": self.top_k,
                "temperature": 0,
                "cache_prompt": False,
            },
        )
        top = out["completion_probabilities"][0]["top_logprobs"]
        timings = out.get("timings") or {}
        return (
            {entry["token"]: entry["logprob"] for entry in top},
            out.get("tokens_evaluated", 0),
            (timings.get("prompt_ms", 0.0) + timings.get("predicted_ms", 0.0)) / 1000,
        )

    def probabilities(self, state, question: dict) -> tuple[dict[str, float], int]:
        """Calibrated probability per option id, and the input token count (summed over passes
        when a question has more than 16 options; the longest pass is `last_pass_tokens`)."""
        options = decision_options(question)
        tokens, self.last_pass_tokens, self.last_seconds = 0, 0, 0.0

        def read(texts: list[str]) -> list[float]:
            nonlocal tokens
            prompt = prompt_text(state, question["instructions"], texts)
            seen, count, seconds = self._logprobs(prompt)
            tokens += count
            self.last_pass_tokens = max(self.last_pass_tokens, count)
            self.last_seconds += seconds
            floor = min(seen.values(), default=0.0) - MISSING_MARGIN
            logprobs = [seen.get(LETTERS[i], floor) for i in range(len(texts))]
            if any(LETTERS[i] not in seen for i in range(len(texts))):
                self.missing += 1
            top = max(logprobs)
            weights = [math.exp((z - top) / self.temperature) for z in logprobs]
            total = sum(weights)
            return [w / total for w in weights]

        probs = spread(read, [text for _, text in options], self.method)
        return {key: v for (key, _), v in zip(options, probs, strict=True)}, tokens

    def decide(self, state, question: dict) -> dict:
        probs, tokens = self.probabilities(state, question)
        return answer(question, probs, tokens)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    model = JevK5GGUF(argv[0] if argv else "http://127.0.0.1:8080")
    print(
        model.decide(
            "Refunds need a receipt and a purchase within 30 days. The customer bought 12 days "
            "ago and has no receipt.",
            {"type": "noul", "instructions": "Is a refund permitted under the policy?"},
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""JevK5: typed decisions from Qwen3.5-4B (+ a distilled LoRA, merged) in one forward pass.

The prompt and readout follow SemIf (TheoLeeCJ/SemIf, MIT): a fixed system instruction, the
decision as JSON (evidence, criterion, lettered options), the chat template with thinking off,
and a softmax over the answer letters' next-token logits. JevK5 adds weights distilled from a
thinking teacher, one calibration temperature, and a CUDA-graph runtime: one graph is recorded
per padded input length and replayed, so a decision costs ~13 ms on an H100 instead of ~70 ms.

    from jevk5 import JevK5
    model = JevK5("alibiserikbay/JevK5")
    model.decide("I was billed twice, please refund the duplicate.",
                 {"type": "noul", "instructions": "Does the customer ask for money back?"})
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

LETTERS = "ABCDEFGHIJKLMNOP"
SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
GRAPH_LENGTHS = (128, 192, 256, 320, 384, 512, 640, 768, 1024, 1536, 2048, 3072, 4096)


def messages(state, criterion: str, options: list[str]) -> list[dict]:
    payload = {
        "evidence": state,
        "criterion": criterion,
        "options": [{"letter": LETTERS[i], "description": d} for i, d in enumerate(options)],
    }
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def decision_options(question: dict) -> list[tuple[str, str]]:
    """(option id, option text) for a typed question: noul -> true/false, choice -> its
    criteria, score -> level indices. Texts are "id: description", as SemIf's JevBench mapping."""
    crit = question.get("criteria")
    if question["type"] == "noul":
        pairs = [(k, (crit or {}).get(k) or f"The proposition is {k}.") for k in ("true", "false")]
    elif question["type"] == "choice":
        if isinstance(crit, list):
            crit = dict.fromkeys(crit)
        pairs = [(k, v or k) for k, v in crit.items()]
    else:
        pairs = [(str(i), level) for i, level in enumerate(crit)]
    return [(k, f"{k}: {d}") for k, d in pairs]


def _load_temperature(source: str) -> float:
    """JevK5's calibration temperature, stored next to the weights in jevk5_config.json."""
    path = Path(source) / "jevk5_config.json"
    if not path.exists():
        try:
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(source, "jevk5_config.json"))
        except Exception:  # noqa: BLE001 - base models have no config; use 1.0
            return 1.0
    return float(json.loads(path.read_text()).get("temperature", 1.0))


class JevK5:
    def __init__(
        self,
        source: str = "alibiserikbay/JevK5",
        device: str = "cuda",
        dtype=torch.bfloat16,
        graphs: bool = True,
        temperature: float | None = None,
    ) -> None:
        import transformers

        config = transformers.AutoConfig.from_pretrained(source)
        self.tok = transformers.AutoTokenizer.from_pretrained(source)
        cls = transformers.AutoModelForCausalLM
        if config.model_type in {"qwen3_5", "qwen3_5_text"}:
            cls, config = transformers.Qwen3_5ForCausalLM, config.get_text_config()
        self.model = cls.from_pretrained(
            source, config=config, dtype=dtype, device_map={"": device}
        ).eval()
        self.device = device
        slots = [self.tok.encode(letter, add_special_tokens=False) for letter in LETTERS]
        if any(len(ids) != 1 for ids in slots):
            raise ValueError("Every answer letter must be one token")
        self.slots = [ids[0] for ids in slots]
        self.slot_weight = self.model.lm_head.weight[self.slots].detach().contiguous()
        self.temperature = temperature if temperature is not None else _load_temperature(source)
        self.graphs: dict[int, tuple] = {}
        if graphs and os.environ.get("JEVK5_GRAPHS", "1") != "0":
            self.capture()

    def _slot_logits(self, ids: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        hidden = self.model.model(input_ids=ids, use_cache=False).last_hidden_state
        return hidden[torch.arange(ids.shape[0], device=ids.device), last] @ self.slot_weight.T

    @torch.inference_mode()
    def capture(self, lengths=GRAPH_LENGTHS) -> None:
        """Record one CUDA graph per padded length. Inputs are right-padded, which every
        (causal) layer keeps away from the last token, the only one read."""
        for n in lengths:
            ids = torch.zeros((1, n), dtype=torch.long, device=self.device)
            last = torch.full((1,), n - 1, dtype=torch.long, device=self.device)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._slot_logits(ids, last)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self._slot_logits(ids, last)
            self.graphs[n] = (graph, ids, last, out)

    def encode(self, state, criterion: str, options: list[str]) -> list[int]:
        prompt = self.tok.apply_chat_template(
            messages(state, criterion, options),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return self.tok.encode(prompt, add_special_tokens=False)

    @torch.inference_mode()
    def letter_logits(self, ids: list[int], count: int) -> np.ndarray:
        fits = [n for n in self.graphs if n >= len(ids)]
        if fits:
            graph, static_ids, last, out = self.graphs[min(fits)]
            static_ids.zero_()
            static_ids[0, : len(ids)] = torch.tensor(ids, device=self.device)
            last.fill_(len(ids) - 1)
            graph.replay()
            return out[0, :count].float().cpu().numpy()
        tensor = torch.tensor([ids], device=self.device)
        last = torch.tensor([len(ids) - 1], device=self.device)
        return self._slot_logits(tensor, last)[0, :count].float().cpu().numpy()

    def probabilities(self, state, question: dict) -> tuple[dict[str, float], int]:
        """Calibrated probability per option id, and the input token count."""
        options = decision_options(question)
        ids = self.encode(state, question["instructions"], [text for _, text in options])
        logits = self.letter_logits(ids, len(options)) / self.temperature
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return {key: float(v) for (key, _), v in zip(options, p, strict=True)}, len(ids)

    def decide(self, state, question: dict) -> dict:
        """One typed decision in TypeSafe's /v1/systemone answer shape."""
        probs, tokens = self.probabilities(state, question)
        kind = question["type"]
        answer = {"type": kind, "confidence": max(probs.values()), "input_tokens": tokens}
        if kind == "noul":
            answer["noul"] = probs["true"]
        elif kind == "choice":
            answer.update(choice=max(probs, key=probs.get), probabilities=probs)
        else:
            answer.update(score=sum(int(k) * v for k, v in probs.items()), probabilities=probs)
        return answer

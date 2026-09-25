"""JevK5-Lite (preview): a CPU encoder that scores every label of every head in one forward pass.

    from jevk5 import JevK5Lite

    lite = JevK5Lite.from_pretrained("alibiserikbay/JevK5-Lite")   # a Hub repo or a local folder
    lite.classify(
        "I was charged twice for the same order, please refund one of them.",
        {"intent": ["refund_request", "order_status", "cancel_subscription"],
         "areas": {"labels": ["billing", "shipping", "account"], "multi_label": True}},
    )
    # {"intent": {"labels": ["refund_request"], "probabilities": {...}},
    #  "areas": {"labels": ["billing"], "probabilities": {...}}}

One sequence per call: [CLS] [TASK] <task> (one|any) [LABEL] <label> ... [SEP] <text> [SEP]. Each label is
scored from the hidden state of its [LABEL] marker and the mean of the text's hidden states, by a small MLP
on [label, text, label * text]. A single-label head is a softmax over its labels (one answer); a
multi-label head is a sigmoid per label (every label at or above the threshold). Each head type has its
own calibration temperature. A model folder holds the encoder (config.json, model.safetensors), its
tokenizer (with the [TASK] and [LABEL] tokens), scorer.safetensors and lite_config.json.

CPU and float32 by default. On CPUs with bf16 support (AMX, AVX512-BF16), dtype=torch.bfloat16 roughly
halves the latency.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

TASK, LABEL = "[TASK]", "[LABEL]"


class Scorer(nn.Module):
    """Scores one label from its marker state `l` and the pooled text `t`: MLP([l, t, l * t])."""

    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, labels: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        text = text.expand_as(labels)
        return self.net(torch.cat([labels, text, labels * text], -1)).squeeze(-1)


def _heads(tasks: dict) -> list[tuple[str, list[str], bool, float | None]]:
    """(task, labels, multi_label, threshold) from {task: [labels]} or {task: {"labels", "multi_label"}}."""
    heads = []
    for task, spec in tasks.items():
        if isinstance(spec, dict):
            labels, multi, threshold = (
                spec.get("labels"),
                bool(spec.get("multi_label")),
                spec.get("threshold"),
            )
        else:
            labels, multi, threshold = spec, False, None
        if not isinstance(labels, (list, tuple)) or not all(isinstance(x, str) for x in labels):
            raise ValueError(f"task {task!r}: labels must be a list of strings")
        if len(set(labels)) != len(labels):
            raise ValueError(f"task {task!r}: labels must be unique")
        if len(labels) < (1 if multi else 2):
            raise ValueError(f"task {task!r}: a single-label task needs at least two labels")
        heads.append((str(task), list(labels), multi, threshold))
    if not heads:
        raise ValueError("no tasks given")
    return heads


class JevK5Lite:
    def __init__(self, encoder: nn.Module, scorer: Scorer, tokenizer, config: dict) -> None:
        self.encoder, self.scorer, self.tok, self.config = (
            encoder.eval(),
            scorer.eval(),
            tokenizer,
            config,
        )
        self.max_len = int(config.get("max_len", 512))
        self.temperature_single = float(config.get("temperature_single", 1.0))
        self.temperature_multi = float(config.get("temperature_multi", 1.0))
        self.task_id = tokenizer.convert_tokens_to_ids(TASK)
        self.label_id = tokenizer.convert_tokens_to_ids(LABEL)
        if tokenizer.unk_token_id in (self.task_id, self.label_id):
            raise ValueError("the tokenizer lacks the [TASK] and [LABEL] tokens")

    @classmethod
    def from_pretrained(
        cls,
        source: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        threads: int | None = None,
        revision: str | None = None,
    ) -> "JevK5Lite":
        """Load from a local folder or a Hugging Face repo; `threads` sets torch's CPU thread count."""
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer

        path = Path(source)
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(source, revision=revision))
        if threads:
            torch.set_num_threads(threads)
        config = json.loads((path / "lite_config.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(path)
        encoder = AutoModel.from_pretrained(path, dtype=dtype)
        scorer = Scorer(encoder.config.hidden_size)
        scorer.net.load_state_dict(load_file(path / "scorer.safetensors"))
        return cls(encoder.to(device), scorer.to(device=device, dtype=dtype), tokenizer, config)

    def _piece(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def encode(
        self, text: str, heads
    ) -> tuple[list[int], list[int], list[tuple[int, int]], tuple[int, int]]:
        """ids, the [LABEL] positions, each head's slice of those positions, and the text's span."""
        schema, label_pos, spans = [], [], []
        for task, labels, multi, _ in heads:
            schema += [self.task_id] + self._piece(
                f"{task.replace('_', ' ')} ({'any' if multi else 'one'})"
            )
            start = len(label_pos)
            for label in labels:
                label_pos.append(1 + len(schema))  # after [CLS]
                schema += [self.label_id] + self._piece(label)
            spans.append((start, len(label_pos)))
        room = self.max_len - len(schema) - 3
        if room < 1:
            raise ValueError(
                f"label schema is too long for max_len={self.max_len}; "
                "use fewer or shorter task and label names"
            )
        body = self.tok(text, add_special_tokens=False, truncation=True, max_length=room)[
            "input_ids"
        ]
        ids = (
            [self.tok.cls_token_id]
            + schema
            + [self.tok.sep_token_id]
            + body
            + [self.tok.sep_token_id]
        )
        start = 2 + len(schema)
        return ids, label_pos, spans, (start, start + len(body))

    @torch.inference_mode()
    def scores(self, text: str, heads) -> list[torch.Tensor]:
        """Raw label scores per head, from one encoder pass."""
        ids, label_pos, spans, (s, e) = self.encode(text, heads)
        device = next(self.encoder.parameters()).device
        x = torch.tensor([ids], device=device)
        h = self.encoder(input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state[0]
        text_state = h[s : max(e, s + 1)].mean(0, keepdim=True)
        z = self.scorer(h[label_pos], text_state).float()
        return [z[a:b] for a, b in spans]

    def classify(self, text: str, tasks: dict, threshold: float = 0.5) -> dict:
        """{task: {"labels": [...], "probabilities": {label: p}}} for every task, in one pass.

        `tasks` maps a task name to its labels, or to {"labels": [...], "multi_label": True,
        "threshold": 0.5} for a multi-label task. A single-label task always returns one label."""
        heads = _heads(tasks)
        out = {}
        for (task, labels, multi, t), z in zip(heads, self.scores(text, heads)):
            if multi:
                p = torch.sigmoid(z / self.temperature_multi)
                cut = threshold if t is None else t
                chosen = [lab for lab, q in zip(labels, p.tolist()) if q >= cut]
            else:
                p = torch.softmax(z / self.temperature_single, -1)
                chosen = [labels[int(p.argmax())]]
            out[task] = {"labels": chosen, "probabilities": dict(zip(labels, p.tolist()))}
        return out

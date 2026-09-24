"""The prompt and the option mapping, with no heavy dependencies.

Kept apart from `runtime.py` so a llama.cpp or ONNX front end can build exactly the same prompt
without importing torch. Both follow SemIf (TheoLeeCJ/SemIf, MIT).
"""

from __future__ import annotations

import json

LETTERS = "ABCDEFGHIJKLMNOP"
SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
# Qwen3.5's chat template with thinking off, rendered once and pinned here so the llama.cpp path
# is token-identical to the transformers one. Verified against
# tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False).
CHAT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


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


def prompt_text(state, criterion: str, options: list[str]) -> str:
    """The full prompt string, chat template included, for front ends without a template engine."""
    system, user = messages(state, criterion, options)
    return CHAT_TEMPLATE.format(system=system["content"], user=user["content"])


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


def answer(question: dict, probs: dict[str, float], tokens: int) -> dict:
    """TypeSafe's /v1/systemone answer shape for a distribution over the option ids."""
    kind = question["type"]
    out = {"type": kind, "confidence": max(probs.values()), "input_tokens": tokens}
    if kind == "noul":
        out["noul"] = probs["true"]
    elif kind == "choice":
        out.update(choice=max(probs, key=probs.get), probabilities=probs)
    else:
        out.update(score=sum(int(k) * v for k, v in probs.items()), probabilities=probs)
    return out

"""The prompt, the option mapping and the many-option readout, with no heavy dependencies.

Kept apart from `runtime.py` so a llama.cpp or ONNX front end can build exactly the same prompt
and combine passes exactly the same way without importing torch. Both follow SemIf
(TheoLeeCJ/SemIf, MIT).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

LETTERS = "ABCDEFGHIJKLMNOP"
METHODS = ("knockout", "tree")
# The temperature of the combined distribution over more than 16 options, per method. The letter
# temperature is fitted on questions of up to 16 options and leaves the combination
# underconfident; 0.77 was fitted for the knockout on 500 train questions of MASSIVE en-US (60
# intents, not a Decision Index benchmark) and only sharpens: no answer changes. The tree is
# unfitted.
TEMPERATURES = {"knockout": 0.77, "tree": 1.0}
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


Reader = Callable[[list[str]], Sequence[float]]


def groups(n: int, count: int) -> list[range]:
    """`count` contiguous runs covering range(n), their sizes differing by at most one."""
    base, extra = divmod(n, count)
    runs, start = [], 0
    for g in range(count):
        stop = start + base + (g < extra)
        runs.append(range(start, stop))
        start = stop
    return runs


def spread(
    read: Reader, texts: list[str], method: str = "knockout", temperature: float | None = None
) -> list[float]:
    """A probability for every option, from a reader that answers at most 16 lettered options.

    `read(texts)` is one forward pass: the calibrated distribution over those options' letters.
    Up to 16 options the result is `read(texts)` itself, untouched. Beyond that the options are
    split, in their given order, into ceil(n / 16) groups of near-equal size, and every option is
    scored by the model (nothing is pruned unread):

    - "knockout": each group is read, then a final of 16 is read: the top 16 // groups options of
      every group (at least one), the free places going to the next most likely options in any
      group. Finalists keep the final's distribution, times the chance that the answer is a
      finalist; every other option gets its group's share of the final times its in-group
      probability.
    - "tree": one pass whose letters stand for whole groups, each described by its members, then
      one pass per group: P(option) = P(its group) * P(option | its group).

    The combined distribution is then sharpened by `temperature` (default: TEMPERATURES).
    Both take ceil(n / 16) + 1 passes up to 256 options, and recurse beyond.
    """
    if len(texts) <= len(LETTERS):
        return list(read(texts))
    probs = _combine(read, texts, method)
    temperature = TEMPERATURES[method] if temperature is None else temperature
    if temperature != 1.0:
        probs = [q ** (1 / temperature) for q in probs]
        total = sum(probs)
        probs = [q / total for q in probs]
    return probs


def _combine(read: Reader, texts: list[str], method: str) -> list[float]:
    if len(texts) <= len(LETTERS):
        return list(read(texts))
    if method == "knockout":
        weights = _knockout(read, texts)
    elif method == "tree":
        weights = _tree(read, texts)
    else:
        raise ValueError(f"unknown method {method!r}; use one of {METHODS}")
    total = sum(weights)
    return [w / total for w in weights]


def _knockout(read: Reader, texts: list[str]) -> list[float]:
    runs = groups(len(texts), -(-len(texts) // len(LETTERS)))
    inner = [list(read([texts[i] for i in run])) for run in runs]
    inner = [[q / sum(p) for q in p] for p in inner]
    keep = max(1, len(LETTERS) // len(runs))
    # Ties go to the earlier option: the sorts are stable and walk the options in order.
    ranked = [sorted(range(len(p)), key=lambda j: -p[j]) for p in inner]
    chosen = {(g, j) for g, order in enumerate(ranked) for j in order[:keep]}
    rest = sorted(
        ((g, j) for g, order in enumerate(ranked) for j in order[keep:]),
        key=lambda gj: -inner[gj[0]][gj[1]],
    )
    chosen.update(rest[: max(0, len(LETTERS) - len(chosen))])
    tops = [sorted(j for h, j in chosen if h == g) for g in range(len(runs))]
    final = _combine(read, [texts[run[j]] for run, top in zip(runs, tops) for j in top], "knockout")
    shares, at = [], 0
    for top in tops:
        shares.append(dict(zip(top, final[at : at + len(top)])))
        at += len(top)
    in_final = sum(sum(f.values()) * sum(p[j] for j in f) for p, f in zip(inner, shares))
    weights = []
    for p, f in zip(inner, shares):
        mass = sum(f.values())
        weights += [f[j] * in_final if j in f else mass * q for j, q in enumerate(p)]
    return weights


def _tree(read: Reader, texts: list[str]) -> list[float]:
    runs = groups(len(texts), min(len(LETTERS), -(-len(texts) // len(LETTERS))))
    outer = read(["One of: " + "; ".join(texts[i] for i in run) for run in runs])
    weights = []
    for run, share in zip(runs, outer):
        weights += [share * q for q in _combine(read, [texts[i] for i in run], "tree")]
    return weights


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

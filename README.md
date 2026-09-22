# JevK5

**An open decision model: typed answers with calibrated probabilities, in one forward pass.**

Give it a state (a ticket, a log, a policy, a diff) and typed questions: yes/no, choice, or score.
It returns a probability for every option, with zero generated tokens, in about 13 ms on one
GPU. It answers TypeSafe's `/v1/systemone` shape, so Jev-style clients can point at it.

| | |
|---|---|
| Weights | [alibiserikbay/JevK5](https://huggingface.co/alibiserikbay/JevK5): Qwen3.5-4B + a distilled LoRA, merged (Apache-2.0) |
| Readout | SemIf's protocol: a softmax over the answer letters' next-token logits, one temperature |
| Runtime | One CUDA graph per padded input length: 13 ms vs ~70 ms eager on an H100, same answers |
| Training | Distilled from Qwen3.6-27B with thinking, on hard decisions it wrote and checked twice |

## Results on JevBench's public items

JevBench v1.2, 231 public items, through JevBench's own runner (`bench/jevk5_direct.py`):
231/231 valid, 0 failures. "Untrained" is the same base model and prompt without the LoRA or the
temperature; its answers match SemIf's official public outcomes on all 231 items.

| Split | n | Untrained Qwen3.5-4B | **JevK5 v0.1** |
|---|---:|---:|---:|
| easy | 48 | 1.000 | **1.000** |
| original (standard) | 72 | 0.986 | 0.958 |
| hard (public half) | 111 | 0.613 | **0.676** |
| hard-tier ECE | | 0.117 | **0.082** |

Latency on one H100 (in-process, batch 1): p50 13.5 ms, p95 14.9 ms on easy and standard items;
p50 30 ms, p95 161 ms on hard items with 1-4k-token documents. Per-item results are in
[results/public231](results/public231). These are public-item numbers from our own runs, not an
official JevBench score.

**Known weak spots:** two standard-tier items that the untrained model answers correctly are
wrong after training; probability fidelity on exact-distribution items is mean TVD 0.296.

## Install and use

```bash
pip install "jevk5[fast] @ git+https://github.com/allebee/jevk5@v0.1.0"
```

```python
from jevk5 import JevK5

model = JevK5("alibiserikbay/JevK5")          # ~9 GB of GPU memory in bf16
model.decide(
    "Refunds need a receipt and a purchase within 30 days. "
    "The customer bought 12 days ago and has no receipt.",
    {"type": "noul", "instructions": "Is a refund permitted under the policy?"},
)
# {'type': 'noul', 'confidence': ..., 'noul': <probability of true>, 'input_tokens': ...}
```

Question types: `noul` (yes/no, optional `criteria` {"true": ..., "false": ...}), `choice`
(`criteria` {key: description} or a list of keys), `score` (`criteria` a list of level
descriptions, lowest first).

As a server:

```bash
jevk5-serve --model alibiserikbay/JevK5 --port 8090
curl -s localhost:8090/v1/systemone -d '{"state": "Order #7120 shows delivered to No. 17; the customer lives at No. 71.",
  "questions": {"what": {"type": "choice", "instructions": "What happened to the parcel?",
                         "criteria": ["delivered", "misdelivered", "unknown"]}}}'
```

`flash-linear-attention` (the `fast` extra) speeds up long inputs; without it, Qwen3.5's
linear-attention layers fall back to transformers' PyTorch code.

## How it was made

1. **Teacher data** ([training/teacher.py](training/teacher.py)): Qwen3.6-27B with thinking writes
   realistic documents with three hard typed questions each, across 17 business domains and 11
   decision families (policy exceptions, date and number traps, multi-step lookups, judging
   answers, ambiguity, misleading notes, injected instructions, rule precedence, routing,
   extraction, rubrics). It answers every question twice, independently; a question is kept only
   when both answers match the intended one. Option keys are rebuilt from the option text.
2. **Training** ([training/lora.py](training/lora.py)): LoRA rank 16 on attention projections,
   cross-entropy on the option-letter logits, 1,635 teacher questions + 1,635 human-labelled items
   (MMLU-Pro, WANLI, MultiNLI, BoolQ, banking77, ARC, CommonsenseQA), 2 epochs, lr 3e-5.
3. **Calibration:** one temperature (1.367), fitted on teacher questions from three domains that
   training never saw, where accuracy rose from 77.6% to 84.2%.

No JevBench item and no output of Jev was used for training, tuning or model selection.

## Reproduce the JevBench run

See [bench/SUBMISSION.md](bench/SUBMISSION.md).

## Credits

Qwen3.5-4B and Qwen3.6-27B by the Qwen team (Apache-2.0). Prompt and readout adapted from
[SemIf](https://github.com/TheoLeeCJ/SemIf) by TheoLeeCJ (MIT); see [NOTICE](NOTICE).
Evaluated with [JevBench](https://github.com/fstandhartinger/jevbench) (MIT). Not affiliated with
TypeSafe AI or Jev.

## License

Apache-2.0.

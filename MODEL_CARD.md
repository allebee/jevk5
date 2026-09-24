---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
language:
- en
library_name: transformers
tags:
- decision-model
- system-one
- jev
- jev-alternative
- typed-decisions
- self-hosted
- jevbench
- calibration
- distillation
---

# JevK5 v0.2 — open-weight Jev alternative

JevK5 is an independent, Apache-2.0 open-source alternative to TypeSafe's Jev for typed
decisions. It reads a state and a yes/no (`noul`), choice, or score question and returns a
**probability for every option in one forward pass, with zero generated tokens**. The
[open weights](https://huggingface.co/alibiserikbay/JevK5) can be self-hosted using the
[JevK5 runtime](https://github.com/allebee/jevk5), which serves a TypeSafe-style
`/v1/systemone` endpoint. This is not Jev's model or architecture and is not affiliated with
TypeSafe AI.

Use the JevK5 runtime shown below to read option probabilities. Generic text-generation examples
on the Hub call `generate()` and do not perform JevK5's decision readout.

- **Base:** Qwen3.5-4B, with a LoRA (rank 16, attention projections) merged into the weights
- **Readout:** SemIf's protocol (TheoLeeCJ/SemIf, MIT): a softmax over the answer letters'
  next-token logits, divided by one calibration temperature (`jevk5_config.json`, T = 1.532)
- **Runtime:** [github.com/allebee/jevk5](https://github.com/allebee/jevk5). One CUDA graph per
  padded input length: ~13 ms per decision on an H100 (eager transformers: ~70 ms), same answers
- **License:** Apache-2.0

## Other sizes and formats

- **[JevK5-GGUF](https://huggingface.co/alibiserikbay/JevK5-GGUF)**: GGUF builds for llama.cpp, so
  JevK5 runs on NVIDIA, AMD, Intel and Apple GPUs, or on a CPU. The Q8_0 build gives the same
  answer as these weights on 228 of 231 public JevBench items.
- **[JevK5-2B](https://huggingface.co/alibiserikbay/JevK5-2B)**: the same recipe on Qwen3.5-2B, at
  half the memory (3.5 GB); 0.751 against this model's 0.804 on held-out teacher questions.

## Independent evaluation: JevBench v1.4

[JevBench](https://github.com/fstandhartinger/jevbench) ranks JevK5 v0.2 **#2 of 76 systems and
#1 among open entrants** on its v1.4 composite score: 62.04, compared with 63.29 for Jev 1.13.0.
The benchmark combines intelligence, calibration, speed, and cost. Its
[published aggregates](https://github.com/fstandhartinger/jevbench/blob/main/results/v1.4/jevbench-v1.4-results.json)
report 33.1% accuracy for JevK5 and 36.7% for Jev on 308 fresh sealed decisions, compared with
85.3% and 86.6% on the 231 public decisions. The evaluator describes the sealed set as unusually
difficult and warns that small sealed differences do not prove pairwise superiority. These are
benchmark results, not a measurement of production workflow quality.

## How it was trained

JevK5 is distilled from a model that thinks. Qwen3.6-27B (Apache-2.0), with thinking on, wrote
realistic documents with hard typed questions (policies with exceptions, date and number traps,
multi-step lookups, judging answers, ambiguity, misleading notes, injected instructions, rule
precedence, routing, extraction, rubrics) across 17 business domains. It then answered every
question twice, independently. A question was kept only when both answers matched the intended
one, and option keys were rebuilt from the option text so that no key hints at the answer.

- 3,272 of those questions (v0.1 used 1,635), plus as many human-labelled items from MMLU-Pro
  (MIT), WANLI (CC BY 4.0), MultiNLI, BoolQ (CC BY-SA 3.0), banking77 (CC BY 4.0), ARC
  (CC BY-SA 4.0) and CommonsenseQA (MIT). The 940 MMLU-Pro items came from its test split, as
  MMLU-Pro has no training split; see the 2026-09-24 correction in the repository's `CHANGELOG.md`
- Cross-entropy on the option-letter logits, 2 epochs, learning rate 3e-5, SemIf's prompt format.
  A question whose answer is a distribution trains against that exact distribution rather than a
  single letter; this checkpoint contains 9 such questions, a pilot of the family
- The temperature was fitted on teacher questions from three domains the training never saw
  (residential leases, public-sector permits, manufacturing QC), where accuracy is 80.4%. A
  temperature per question type, and averaging two option orders, were measured on that held-out
  data and on a hand-written hard set, and both were rejected
- **No JevBench item, public or held out, and no output of Jev was used for training, tuning or
  selection.** JevBench's public items were only used to report the numbers below. One correction:
  our hand-written calibration set echoed a public instruction and one public item's rule wording;
  both are rewritten and the effect is described in the repository's `CHANGELOG.md`.

The teacher itself, with thinking, answered all 111 public JevBench hard items correctly; JevK5 is
an attempt to move part of that into a single fast pass.

## Results: JevBench v1.2 public items

231 public items through JevBench's own runner (`jevk5_direct` adapter): 231/231 valid, 0 failures.
The untrained row is the same base model and prompt without the LoRA or the temperature (its
answers match SemIf's official public outcomes on 231/231 items).

| Split | n | Untrained Qwen3.5-4B | JevK5 v0.1 | **JevK5 v0.2** | v0.2 ECE |
|---|---:|---:|---:|---:|---:|
| easy | 48 | 1.000 | 1.000 | **1.000** | 0.038 |
| original (standard) | 72 | 0.986 | 0.958 | 0.958 | 0.141 |
| hard (public half) | 111 | 0.613 | 0.676 | **0.739** | **0.066** (untrained 0.117) |

On the hard tier v0.2 fixes 21 of the untrained model's items and breaks 7 (McNemar p = 0.013).
Distance to the exact gold distributions on the 10 public probability items: 0.196 (v0.1 0.296).

- **Latency** (H100, in-process, batch 1, CUDA graphs): p50 13.5 ms, p95 14.9 ms on easy and
  standard items; hard items with 1-4k-token documents p50 30 ms, p95 161 ms
- **Input tokens per decision:** 164 easy, 168 standard, 1,274 hard; 0 output tokens

## Known weak spots

- Accuracy drops sharply on JevBench's fresh sealed decisions. Real-world workflow performance
  against Jev has not yet been measured.
- Two standard-tier public items that the untrained model gets right are wrong after training
  (98.6% → 95.8%): one policy pair, unchanged since v0.1.
- The temperature is fitted on hard questions, so standard-tier answers are now underconfident
  (ECE 0.141 there, against 0.066 on the hard tier). JevBench scores calibration on the hard tier.
- Judging answers slipped on the hard tier, 0.82 → 0.76 (one item of 17).
- English only. Needs a CUDA GPU with ~9 GB for bf16. Inputs over 16,384 tokens are refused, not cut.

## Use

```python
from jevk5 import JevK5

model = JevK5("alibiserikbay/JevK5")
model.decide(
    "I was billed twice for order #4411. Please refund the duplicate charge today.",
    {"type": "choice", "instructions": "Which team should handle this?",
     "criteria": {"billing": "Payments and refunds", "tech": "Bugs", "sales": "New purchases"}},
)
```

Or as a server that answers TypeSafe-style `/v1/systemone` requests:
`jevk5-serve --model alibiserikbay/JevK5 --port 8090`.

## Credits

Qwen3.5-4B and Qwen3.6-27B by the Qwen team (Apache-2.0). The one-pass readout and prompt come
from SemIf by TheoLeeCJ (MIT). Evaluated with JevBench (github.com/fstandhartinger/jevbench, MIT). Not
affiliated with TypeSafe AI or Jev.

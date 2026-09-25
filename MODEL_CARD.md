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

# JevK5 v0.3 — open-weight Jev alternative

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
  next-token logits, divided by one calibration temperature (`jevk5_config.json`, T = 1.367)
- **Runtime:** [github.com/allebee/jevk5](https://github.com/allebee/jevk5), 0.2.2 or later. One
  CUDA graph per padded input length: ~13 ms per decision on an H100. Questions with more than 16
  options are read in several passes
- **License:** Apache-2.0
- **Previous version:** v0.2 stays available under the Hub tag `v0.2` (see Use)

## What changed from v0.2

- **Five times the teacher data, from two teachers:** 17,408 questions (v0.2: 3,272). 3,270 are
  v0.2's questions from Qwen3.6-27B; 14,138 are new, from GPT-6 Luna.
- **Wide replay, from train splits only:** 32,425 human-labelled items from the train splits of
  27 public datasets. v0.2 had 3,272 items from 7 datasets, and some were MMLU-Pro test items.
  v0.3 uses no test or validation split of any dataset, and no MMLU-Pro or BBH at all.
- **One epoch over 49,833 rows** (v0.2: two epochs over 6,544), with the same LoRA and learning
  rate. The temperature is 1.367 (v0.2: 1.532).
- **Same runtime.** jevk5 0.2.2 loads v0.3 unchanged. On the 231 public JevBench items, its
  probabilities are bit-identical to the benchmark run below.

## Other sizes and formats

- **[JevK5-9B](https://huggingface.co/alibiserikbay/JevK5-9B)**: the same v0.3 data and recipe on
  Qwen3.5-9B. It is more accurate on our held-out checks (index proxy 0.788, hand-written hard set
  0.844) but not on JevBench's public items. It needs about 19 GB in bf16 and is about 2-3x slower.
  See its card.
- **[JevK5-GGUF](https://huggingface.co/alibiserikbay/JevK5-GGUF)**: GGUF builds for llama.cpp,
  so JevK5 runs on NVIDIA, AMD, Intel and Apple GPUs, or on a CPU. The v0.3 Q8_0 build gives the
  same answer as these weights on 228 of 231 public JevBench items, and the Q5_K_M build on 221.
  A v0.3 Q4_K_M agreed on only 214 and is not published.
- **[JevK5-2B](https://huggingface.co/alibiserikbay/JevK5-2B)**: the v0.2 recipe on Qwen3.5-2B
  (not retrained for v0.3).

## Results

### Held-out checks (used to choose between models)

The dev set is held-out rows only: teacher questions from three domains that training never saw
(residential leases, public-sector permits, manufacturing QC), a hand-written hard set, and a
hashed 5% of every public train split.

| | v0.2 | **v0.3** |
|---|---:|---:|
| Index proxy (16 sources, see below) | 0.620 | **0.740** |
| Held-out teacher questions (362), accuracy | 0.804 | **0.815** |
| Hand-written hard set (64), accuracy | 0.769 | **0.797** |
| ECE on the teacher questions, calibrated | 0.034 | **0.024** |

The index proxy is our own estimate, not an index score. It is the chance-corrected skill,
averaged over the 16 dev sources that are held-out train-split rows of Jev Decision Index
benchmarks. Each source has 40 rows, so each one alone is noisy (about ±0.15). The largest gains
are GSM8K (0.29 → 0.67), RAGTruth (0.15 → 0.50), SGD (0.68 → 0.96), When2Call (0.75 → 0.95) and
iSarcasmEval (0.30 → 0.45). BANKING77 (0.91 → 0.88) and ARC (1.00 → 0.97) slipped.

### bev-decision-150K (another group's decision mix, evaluation only)

On 2,500 hashed rows of the test split of
[avbiswas/bev-decision-150K](https://huggingface.co/datasets/avbiswas/bev-decision-150K)
(4,723 questions, every question type), v0.3 is level with v0.2: accuracy 0.670 against 0.665,
ECE 0.032 against 0.036. By type: choice 0.700, yes/no 0.762, score 0.432. Leaving out the 34
questions whose document also appears in our training data changes accuracy by 0.002.

### JevBench v1.2 public items (report only)

231 public items through JevBench's own runner (`jevk5_direct` adapter): 231/231 valid, 0
failures. The untrained row is the same base model and prompt without the LoRA or the temperature.

| Split | n | Untrained Qwen3.5-4B | JevK5 v0.2 | **JevK5 v0.3** | v0.3 ECE (v0.2) |
|---|---:|---:|---:|---:|---:|
| easy | 48 | 1.000 | 1.000 | **1.000** | 0.028 (0.038) |
| original (standard) | 72 | 0.986 | 0.958 | **0.972** | 0.117 (0.141) |
| hard (public half) | 111 | 0.613 | 0.739 | **0.748** | 0.071 (0.066) |

- **The hard tier is flat.** Against v0.2, v0.3 fixes 8 hard items and breaks 7 (McNemar p = 1.0).
  By family: probability 0.70 → 0.90, multi-step lookups 0.78 → 0.89, long policies 0.58 → 0.68,
  adversarial 0.83 → 1.00. But judging answers 0.76 → 0.59, dates and numbers 0.47 → 0.33, and
  ambiguous 0.86 → 0.71.
- Distance to the exact gold distributions on the 10 public probability items: 0.172 (v0.2 0.196).
- **Latency** (H100, in-process, batch 1, CUDA graphs): p50 13.6 ms, p95 14.9 ms on easy and
  standard items; hard items with 1-4k-token documents p50 29 ms, p95 158 ms.

### More than 16 options

These use the runtime's knockout readout (groups of up to 16, then a final). The runs are 500
train-split items per dataset in the Decision Index's request shape, with every option offered.
None of these items is in v0.3's training or dev data.

| Train split | Options | Passes | v0.2 accuracy / ECE | **v0.3 accuracy / ECE** | v0.3 macro-F1 |
|---|---:|---:|---:|---:|---:|
| MASSIVE en-US | 60 | 5 | 0.754 / 0.038 | **0.768 / 0.025** | 0.757 |
| BANKING77 | 77 | 6 | 0.690 / 0.039 | 0.636 / 0.075 | 0.620 |
| CLINC150 with out-of-scope | 151 | 11 | 0.666 / 0.039 | **0.706** / 0.068 | 0.741 |

BANKING77 got worse, in both accuracy and calibration. The second temperature for more than 16
options (0.77) was fitted for v0.2 on MASSIVE and is unchanged. It still fits MASSIVE, but v0.3 is
overconfident on BANKING77 and less well calibrated on CLINC150. Out-of-scope recall on CLINC150
is 0.37 (v0.2 0.36).

## Jev Decision Index and JevBench

- **Jev Decision Index:** the index reran JevK5 0.2.2 (v0.2 weights, with the runtime that
  answers any number of options) on its previously refused rows
  ([discussion #13](https://huggingface.co/spaces/multimodalart/jev-decision-index/discussions/13)).
  It scored 36.31, 15th of 49, up from 30.02 (22nd), and its calibration was 4th best (ECE 0.031).
  v0.3 has not been run by the index yet.
- **JevBench v1.4** ranked JevK5 v0.2 #2 of 76 systems (62.04; Jev 1.13.0: 63.29). On the 308
  fresh sealed decisions, v0.2 answered 33.1% correctly and Jev 36.7%. v0.3 has not been
  submitted.

## How it was trained

**Teacher questions (17,408).** A thinking model writes realistic documents with hard typed
questions, then answers every question twice, independently. A question is kept only when both
answers match the intended one. Option keys are rebuilt from the option text, so no key hints at
the answer.

- **Qwen3.6-27B** (Apache-2.0, self-hosted, thinking on): v0.2's 3,270 questions in 11 families
  across 17 business domains, plus 9 questions whose answer is an exact distribution.
- **GPT-6 Luna** (OpenAI, through OpenAI's API): 14,138 training questions. Both of Luna's own
  answers matched the intended one for 94% of the questions it wrote. The families are dates and
  numbers (2,445), rubrics (2,130), ambiguity (1,787), paraphrases of Qwen training questions
  (1,665), multi-step lookups (1,552), causal (1,343), plausibility (1,283), stance and sarcasm
  (1,093) and tool choice (840). These outputs were generated under OpenAI's terms, which govern
  their use; review them for your use case.

**Public replay (32,425 items, train splits only).**

| Dataset (train split) | License | Rows |
|---|---|---:|
| MMLU `auxiliary_train` (`cais/mmlu`)* | MIT | 2,373 |
| GSM8K | MIT | 2,373 |
| WinoGrande (xl) | Apache-2.0 | 2,373 |
| HellaSwag | MIT | 2,373 |
| When2Call (`train_pref`) | CC BY 4.0 | 1,978 |
| HoVer (claims + the cited Wikipedia introductions) | CC BY-SA 4.0 | 1,582 |
| iSarcasmEval (task A and task C formats) | MIT | 1,581 |
| ARC (Easy + Challenge) | CC BY-SA 4.0 | 1,186 |
| CommonsenseQA | MIT | 1,186 |
| CLINC150 (`plus`) | CC BY 3.0 | 1,186 |
| Schema-Guided Dialogue (30 of 127 train files) | CC BY-SA 4.0 | 1,186 |
| Amazon ESCI (3 of 11 shards) | Apache-2.0 | 1,186 |
| RAGTruth | MIT | 1,186 |
| ToolACE | Apache-2.0 | 1,186 |
| WANLI | CC BY 4.0 | 1,186 |
| AQuA-RAT | Apache-2.0 | 791 |
| OpenBookQA | Apache-2.0 | 791 |
| Cosmos QA | CC BY 4.0 | 791 |
| SWAG | MIT | 791 |
| BoolQ | CC BY-SA 3.0 | 791 |
| MultiNLI | OANC / CC BY 3.0 / CC BY-SA 3.0 / MIT, by genre | 791 |
| BANKING77 | CC BY 4.0 | 791 |
| MASSIVE (en-US) | CC BY 4.0 | 791 |
| New Yorker caption contest (matching, fold 0) | CC BY 4.0 | 791 |
| QASC | CC BY 4.0 | 395 |
| RuleTaker | Apache-2.0 | 395 |
| Glaive function calling v2 | Apache-2.0 | 395 |

\* MMLU's `auxiliary_train` collects train items of ARC, OpenBookQA, MCTest and RACE; most of
the 2,373 rows used here are RACE reading passages. RACE's own terms limit it to non-commercial
research.

A development variant also trained on the ANLI (CC BY-NC 4.0) and NLI4CT (no stated license)
train splits. It is not released.

**Training and calibration.** Cross-entropy on the option-letter logits, SemIf's prompt format,
1 epoch, learning rate 3e-5, inputs up to 2,048 tokens. A question whose answer is a distribution
trains against that distribution. One temperature is fitted on the held-out teacher questions:
ECE 0.054 → 0.024.

**Data rules.**

- No JevBench item, public or held out, and no output of Jev was used for training, tuning or
  selection. JevBench's public items were only used to report the numbers above.
- Every public and Luna row was checked against the test and validation text of 35 Decision Index
  benchmarks, and against JevBench's public items. A row was dropped for an exact match or for any
  shared 8-word sequence. GPQA and HLE are gated and were not checked; no v0.3 source is built from
  them.
- Rows repeating an item of our many-options evaluation (above) were dropped from every file.

**Declared overlap with the Jev Decision Index.** These are train splits of index benchmarks,
deduplicated against their test and validation items:
MMLU (`auxiliary_train`), ARC, OpenBookQA, CommonsenseQA, GSM8K, WinoGrande, HellaSwag,
BANKING77, CLINC150, SGD, Amazon ESCI, When2Call, iSarcasmEval, RAGTruth, HoVer and the New
Yorker caption contest. Separately, 53 of v0.2's Qwen-written training questions share at least
one 8-word sequence with ContractNLI (41) or SGD (12) test or dev text. They were reported by the
scan and kept. v0.3 was not trained on ANLI or NLI4CT.

## Known weak spots

- **JevBench's hard tier did not improve** (0.739 → 0.748, 8 fixed / 7 broken). Judging answers
  and dates and numbers got worse there, and the hard-tier ECE is 0.071 against v0.2's 0.066.
- **BANKING77 with every intent offered is worse** (0.690 → 0.636, ECE 0.039 → 0.075), and
  CLINC150's ECE rose to 0.068. On CLINC150, "out of scope" is still recalled only 37% of the time.
- On bev-decision's test sample, v0.3 is no better than v0.2 (0.670 vs 0.665). Score questions
  are the weakest type there (0.432).
- Standard-tier answers are still underconfident (ECE 0.117), because the temperature is fitted
  on hard questions.
- Accuracy drops sharply on JevBench's fresh sealed decisions (measured for v0.2). Real-world
  workflow performance against Jev has not been measured.
- English only. Needs a CUDA GPU with ~9 GB for bf16. Inputs over 16,384 tokens are refused, not
  cut.

## Use

```python
from jevk5 import JevK5

model = JevK5("alibiserikbay/JevK5")
model.decide(
    "I was billed twice for order #4411. Please refund the duplicate charge today.",
    {"type": "choice", "instructions": "Which team should handle this?",
     "criteria": {"billing": "Payments and refunds", "tech": "Bugs", "sales": "New purchases"}},
)
# {'type': 'choice', 'choice': 'billing', 'confidence': 0.989, 'probabilities': {...}, ...}
```

Or as a server that answers TypeSafe-style `/v1/systemone` requests:
`jevk5-serve --model alibiserikbay/JevK5 --port 8090`.

For v0.2, pass a local copy of the tagged revision:

```python
from huggingface_hub import snapshot_download
model = JevK5(snapshot_download("alibiserikbay/JevK5", revision="v0.2"))
```

## Credits

Qwen3.5-4B and Qwen3.6-27B by the Qwen team (Apache-2.0). GPT-6 Luna by OpenAI. The one-pass
readout and prompt come from SemIf by TheoLeeCJ (MIT). The public datasets in the table above
belong to their authors, under the licenses listed. Evaluated with JevBench
(github.com/fstandhartinger/jevbench, MIT) and bev-decision-150K. Not affiliated with TypeSafe AI
or Jev.

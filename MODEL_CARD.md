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
  next-token logits, divided by one calibration temperature. Questions with more than 16 options
  are read in several passes and combined with a second temperature. Both are in
  `jevk5_config.json`: `temperature` 1.22 and `knockout_temperature` 0.93
- **Runtime:** [github.com/allebee/jevk5](https://github.com/allebee/jevk5), 0.3.0 or later. One
  CUDA graph per padded input length: ~13 ms per decision on an H100
- **License:** Apache-2.0
- **Previous version:** v0.2 stays available under the Hub tag `v0.2` (see Use)

## What changed from v0.2

- **Five times the teacher data, from two teachers:** 17,408 questions (v0.2: 3,272). 3,270 are
  v0.2's questions from Qwen3.6-27B; 14,138 are new, from GPT-6 Luna.
- **Wide replay, from train splits only:** 30,052 human-labelled items from the train splits of
  26 public datasets. v0.2 had 3,272 items from 7 datasets, and some were MMLU-Pro test items.
  v0.3 uses no test or validation split of any dataset, and no MMLU, MMLU-Pro or BBH at all.
- **One epoch over 47,460 rows** (v0.2: two epochs over 6,544), with the same LoRA and learning
  rate. The temperature is 1.22 (v0.2: 1.532).
- **Runtime 0.3.0** reads a second, per-model temperature for more than 16 options
  (`knockout_temperature`) from `jevk5_config.json`. Up to 16 options nothing changed: jevk5 0.2.2
  and 0.3.0 give probabilities bit-identical to the benchmark run below on all 231 public
  JevBench items. With 0.2.2, questions with more than 16 options fall back to v0.2's 0.77.

## Other sizes and formats

- **[JevK5-9B](https://huggingface.co/alibiserikbay/JevK5-9B)**: the same v0.3 data and recipe on
  Qwen3.5-9B. It is ahead of this model on our held-out checks (index proxy 0.762, teacher
  questions 0.851) and on questions with more than 16 options. It is behind on JevBench's public
  items (hard tier 0.730). It needs about 19 GB in bf16 and is about 2-3x slower. See its card.
- **[JevK5-GGUF](https://huggingface.co/alibiserikbay/JevK5-GGUF)**: GGUF builds for llama.cpp,
  so JevK5 runs on NVIDIA, AMD, Intel and Apple GPUs, or on a CPU. The v0.3 files give the same answer as these
  weights on 229 (Q8_0), 224 (Q5_K_M) and 221 (Q4_K_M) of 231 public JevBench items.
- **[JevK5-2B](https://huggingface.co/alibiserikbay/JevK5-2B)**: the v0.2 recipe on Qwen3.5-2B
  (not retrained for v0.3).

## Results

### Held-out checks (used to choose between models)

The dev set is held-out rows only: teacher questions from three domains that training never saw
(residential leases, public-sector permits, manufacturing QC), a hand-written hard set, and a
hashed 5% of every public train split.

| | v0.2 | **v0.3** |
|---|---:|---:|
| Index proxy (16 sources, see below) | 0.620 | **0.731** |
| Held-out teacher questions (362), accuracy | 0.801 | **0.834** |
| Hand-written hard set (64), accuracy | 0.766 | 0.766 |
| ECE on the teacher questions, calibrated | **0.034** | 0.035 |

Both columns are read the same way, through the runtime, on the same dev set. (v0.2's own card
reports 0.804 on the teacher questions, from its training-time evaluation.)

The index proxy is our own estimate, not an index score. It is the chance-corrected skill,
averaged over the 16 dev sources that are held-out train-split rows of Jev Decision Index
benchmarks. Each source has 40 rows, so each one alone is noisy (about ±0.15). The largest gains
are GSM8K (0.29 → 0.61), RAGTruth (0.15 → 0.50), SGD (0.68 → 0.93), When2Call (0.75 → 0.95) and
iSarcasmEval (0.30 → 0.55). BANKING77 (0.91 → 0.88) and Amazon ESCI (0.50 → 0.47) slipped, and
HoVer did not move (0.30). The hand-written hard set is level with v0.2 (49 of 64 for both).

### bev-decision-150K (another group's decision mix, evaluation only)

On 2,500 hashed rows of the test split of
[avbiswas/bev-decision-150K](https://huggingface.co/datasets/avbiswas/bev-decision-150K)
(4,723 questions, every question type), v0.3 is level with v0.2: accuracy 0.663 against 0.665,
ECE 0.036 against 0.036. By type: choice 0.697, yes/no 0.758, score 0.409. Leaving out the 34
questions whose document also appears in our training data changes accuracy by 0.002.

### JevBench v1.2 public items (report only)

231 public items through JevBench's own runner (`jevk5_direct` adapter): 231/231 valid, 0
failures. The untrained row is the same base model and prompt without the LoRA or the temperature.

| Split | n | Untrained Qwen3.5-4B | JevK5 v0.2 | **JevK5 v0.3** | v0.3 ECE (v0.2) |
|---|---:|---:|---:|---:|---:|
| easy | 48 | 1.000 | 1.000 | **1.000** | 0.018 (0.038) |
| original (standard) | 72 | 0.986 | **0.958** | 0.944 | 0.057 (0.141) |
| hard (public half) | 111 | 0.613 | 0.739 | **0.784** | 0.054 (0.066) |

- **Hard tier:** against v0.2, v0.3 fixes 10 hard items and breaks 5 (McNemar p = 0.30, so not
  significant on 111 items). By family: probability 0.70 → 1.00, adversarial 0.83 → 1.00, long
  policies 0.58 → 0.68, multi-step lookups 0.78 → 0.83. Judging answers fell 0.76 → 0.65, and
  dates and numbers stayed at 0.47.
- **Standard tier:** 3 fixed and 4 broken, one item fewer than v0.2 (68 of 72).
- **Calibration:** hard-tier ECE 0.054 (v0.2 0.066). The standard tier is much less
  underconfident than v0.2 (ECE 0.057 against 0.141). Distance to the exact gold distributions on
  the 10 public probability items: 0.164 (v0.2 0.196).
- **Latency** (H100, in-process, batch 1, CUDA graphs): p50 13.2 ms, p95 14.6 ms on easy and
  standard items; hard items with 1-4k-token documents p50 30 ms, p95 162 ms.

### More than 16 options

These use the runtime's knockout readout (groups of up to 16, then a final). The runs are 500
train-split items per dataset in the Decision Index's request shape, with every option offered.
None of these items is in v0.3's training or dev data.

| Train split | Options | Passes | v0.2 accuracy / ECE | **v0.3 accuracy / ECE** | v0.3 macro-F1 |
|---|---:|---:|---:|---:|---:|
| MASSIVE en-US (fitting set) | 60 | 5 | **0.754** / 0.038 | 0.738 / 0.045 | 0.724 |
| BANKING77 | 77 | 6 | **0.690** / 0.039 | 0.652 / 0.044 | 0.632 |
| CLINC150 with out-of-scope | 151 | 11 | 0.666 / 0.039 | **0.700** / 0.056 | 0.759 |

- CLINC150 improved (0.666 → 0.700); MASSIVE (0.754 → 0.738) and BANKING77 (0.690 → 0.652) got
  worse.
- **Second temperature:** v0.3 carries its own, 0.93, fitted by NLL on the MASSIVE items only
  (the two halves give 0.98 and 0.87). BANKING77 and CLINC150 are reported, not fitted. With
  v0.2's 0.77 this model would be overconfident (ECE 0.070, 0.101, 0.073).
- **Out of scope:** on CLINC150, "out of scope" is recalled only 27% of the time (v0.2 36%), with
  precision 0.58.

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

**Public replay (30,052 items, train splits only).**

| Dataset (train split) | License | Rows |
|---|---|---:|
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

Two development variants are not released because of their data's terms:
- one that also trained on the ANLI (CC BY-NC 4.0) and NLI4CT (no stated license) train splits;
- one that also trained on 2,373 rows of MMLU's `auxiliary_train`, most of them RACE reading
  passages (RACE is for non-commercial research only). With the same readout it scored 0.740 on
  the index proxy (this model 0.731), 0.820 on the teacher questions (0.834) and 0.781 on the
  hand-written set (0.766).

**Training and calibration.** Cross-entropy on the option-letter logits, SemIf's prompt format,
1 epoch, learning rate 3e-5, inputs up to 2,048 tokens. A question whose answer is a distribution
trains against that distribution. One temperature is fitted on the held-out teacher questions:
ECE 0.050 → 0.035.

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
ARC, OpenBookQA, CommonsenseQA, GSM8K, WinoGrande, HellaSwag, BANKING77, CLINC150, SGD, Amazon
ESCI, When2Call, iSarcasmEval, RAGTruth, HoVer and the New Yorker caption contest. Separately, 53 of v0.2's Qwen-written training questions share at least
one 8-word sequence with ContractNLI (41) or SGD (12) test or dev text. They were reported by the
scan and kept. v0.3 was not trained on any split of MMLU, MMLU-Pro, ANLI or NLI4CT.

## Known weak spots

- **The hand-written hard set did not improve** (49 of 64, the same as v0.2), and neither did
  JevBench's standard tier (0.944 against 0.958). The hard-tier gain on JevBench (10 fixed, 5
  broken) is not statistically significant.
- Judging answers got worse on JevBench's hard tier (0.76 → 0.65), and dates and numbers stay
  the weakest family (0.47).
- **Questions with more than 16 options got worse on two of three sets:** MASSIVE 0.754 → 0.738
  and BANKING77 0.690 → 0.652 (CLINC150 improved, 0.666 → 0.700). On CLINC150, "out of scope" is
  recalled only 27% of the time. Use jevk5 0.3.0 or later, so the model's own second temperature
  (0.93) is applied.
- On bev-decision's test sample, v0.3 is level with v0.2 (0.663 against 0.665). Score questions
  are the weakest type there (0.409).
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
# {'type': 'choice', 'choice': 'billing', 'confidence': 0.996, 'probabilities': {...}, ...}
```

Or as a server that answers TypeSafe-style `/v1/systemone` requests:
`jevk5-serve --model alibiserikbay/JevK5 --port 8090`. Both read `temperature` and
`knockout_temperature` from this repo's `jevk5_config.json`.

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

# Changelog

## 0.3.1

- Adds the optional JevK5Lite (`pip install "jevk5[lite]"`); changes nothing for JevK5 or JevK5-9B.
  `jevk5.JevK5Lite` is imported only when used. It runs
  [JevK5-Lite](https://huggingface.co/alibiserikbay/JevK5-Lite) (preview), a label-conditioned
  DeBERTa-v3-large that scores every label of every head in one pass on a CPU. It reproduces the evaluated
  model's probabilities: the same top label on 294 of 294 heads, max |dp| 2.9e-7.
- JevK5-Lite is a preview. Against GLiNER2.5-Decide on seven public datasets fixed in advance, its
  calibration error is lower on all six single-label sets. Its accuracy is significantly higher on two sets
  (AG News, Yahoo Answers), within noise on three, and lower on two. Decide's mean is higher (0.643 against
  0.629), and Decide is stronger on fastino/fast-decisions dev (0.637 against 0.587).

## 0.3.0

- **JevK5 v0.3 weights** at [alibiserikbay/JevK5](https://huggingface.co/alibiserikbay/JevK5)
  (Qwen3.5-4B, temperature 1.22, Apache-2.0). v0.2 stays available under the Hub tag `v0.2`.
  - Teacher data is five times larger and comes from two teachers: 17,408 questions (v0.2's 3,270
    from Qwen3.6-27B, plus 14,138 from GPT-6 Luna).
  - Replay is 30,052 items from the train splits of 26 public datasets. No test or validation
    split of any dataset is used, and MMLU, MMLU-Pro and BBH are not used at all (see the
    2026-09-24 correction below). Every row is checked against the Decision Index benchmarks'
    test and validation text and JevBench's public items.
  - One epoch over 47,460 rows.
- **New: [JevK5-9B](https://huggingface.co/alibiserikbay/JevK5-9B)**, the same data and recipe on
  Qwen3.5-9B (temperature 1.049, Apache-2.0).
- **Held-out checks, v0.2 → v0.3 4B → 9B** (same dev set and readout):
  - index proxy 0.620 → 0.731 → 0.762
  - held-out teacher questions 0.801 → 0.834 → 0.851
  - hand-written hard set 0.766 → 0.766 → 0.781
  - bev-decision-150K test sample 0.665 → 0.663 → 0.700
- **JevBench public (231), v0.2 → v0.3 4B → 9B:**
  - hard tier 0.739 → 0.784 → 0.730
  - standard 0.958 → 0.944 → 0.944
  - hard-tier ECE 0.066 → 0.054 → 0.126
  - Against v0.2, the 4B fixes 10 hard items and breaks 5 (not significant, p = 0.30). Judging
    answers got worse (0.76 → 0.65).
  - The 9B is worse than the 4B on these items: 4 hard items fixed and 10 broken, with a
    hard-tier ECE of 0.126.
- **Per-model second temperature for more than 16 options.** `JevK5` and `JevK5GGUF` read an
  optional `knockout_temperature` from `jevk5_config.json`, and a `knockout_temperature=`
  argument overrides it. Without either they keep 0.77, so v0.2 is bit-identical to 0.2.2 (231/231
  public items, and the >16-option distributions). Each value is fitted by NLL on 500 MASSIVE
  en-US train items: v0.3 4B 0.93, JevK5-9B 1.2.
- **More than 16 options, 4B v0.2 → v0.3**, each with its own second temperature:
  - CLINC150 0.666 → 0.700 (ECE 0.039 → 0.056)
  - MASSIVE 0.754 → 0.738 (ECE 0.038 → 0.045; the fitting set)
  - BANKING77 0.690 → 0.652 (ECE 0.039 → 0.044)
- **More than 16 options, JevK5-9B** (second temperature 1.2): MASSIVE 0.818, BANKING77 0.734,
  CLINC150 0.780 (ECE 0.036, 0.044, 0.061). That is 8 points above the 4B on each set.
- **GGUF**, answers the same as bf16 on the 231 public items:
  - v0.3 4B: Q8_0 229/231, Q5_K_M 224/231, Q4_K_M 221/231.
  - 9B: Q8_0 229/231, Q5_K_M 225/231.
  - Not published, below 95%: the 9B Q4_K_M (218/231).
- **Runtime:** apart from `knockout_temperature`, unchanged. 0.2.2 loads v0.3 and the 9B too,
  but uses 0.77 for more than 16 options. `JevK5GGUF()` still defaults to v0.2's values (1.532,
  0.77). With v0.3 files, pass `JevK5GGUF(temperature=1.22, knockout_temperature=0.93)` for the 4B
  or `JevK5GGUF(temperature=1.049, knockout_temperature=1.2)` for the 9B.
- **Licensing:** two development variants are not released because of their data's terms. One
  also trained on ANLI (CC BY-NC 4.0) and NLI4CT (no stated license). The other also trained on
  MMLU's `auxiliary_train`, which is mostly RACE (non-commercial research only).
  - At 4B that variant was level with the released model: index proxy 0.740 against 0.731,
    teacher questions 0.820 against 0.834.
  - At 9B it scored higher than the released JevK5-9B: index proxy 0.788 against 0.762, teacher
    questions 0.870 against 0.851, hand-written set 0.844 against 0.781.

## 0.2.2

- **Any number of options.** Questions with more than 16 options were refused: 12.7% of the Jev
  Decision Index's requests, in CLINC150, ChessBench, BANKING77, POP909, API-Bank and HLE. They are
  now read in groups of at most 16 plus a final of 16, and `decide()` returns a probability for
  every option. The weights are unchanged. See the README's "More than 16 options"; the logic
  lives in `jevk5/prompt.py`, and both runtimes use it.
- Up to 16 options, nothing changes: probabilities are bit-identical to 0.2.1 on all 231 public
  items, through CUDA and through llama.cpp (`bench/parity.py`).
- Train-split results, every option offered: BANKING77 0.690 accuracy (ECE 0.039), CLINC150 0.666
  (ECE 0.039), MASSIVE 0.754 (ECE 0.038); 116-199 ms on an H100. The combination rule and its
  temperature (0.77) were chosen and fitted on MASSIVE, which is not on the Decision Index.
- Measured and not shipped: a tree readout, whose first pass picks among groups (BANKING77 0.636,
  CLINC150 0.584), and two other ways of combining the same passes.
- `input_tokens` counts the tokens of every pass. The JevBench adapter's 16,384-token limit applies
  to the longest pass (`last_pass_tokens`).

## Correction, 2026-09-24: MMLU-Pro test items in the training data

MMLU-Pro has no training split, and our replay builder drew its items from a hashed 80% of the test
split. JevK5 v0.2 and JevK5-2B were trained on 940 MMLU-Pro test items; v0.1 on 461 of the same
items. They correspond to 980 MMLU-Pro `question_id`s (31 question texts appear under several ids),
listed in [results/mmlu_pro_training_ids.json](results/mmlu_pro_training_ids.json). The Jev Decision
Index had flagged the overlap without a count.

Measured with the design the index used for a similar case
([training/mmlu_leak.py](training/mmlu_leak.py)): JevK5 v0.2 scores 0.622 on the 940 trained items
and 0.541 on 2,000 MMLU-Pro test items it never saw; the untrained Qwen3.5-4B scores 0.431 and 0.405
on the same two sets. The difference in differences, +5.5 points (95% CI +1.8 to +9.2), is
memorization of those items. Across all 12,032 MMLU-Pro test items it is worth about 0.4 points of
accuracy.

Every other replay set came from a training split (WANLI, MultiNLI, BoolQ, banking77, ARC,
CommonsenseQA). The dev set also held 130 MMLU-Pro and 47 banking77 test items, used only for
reporting; model choices were made on held-out teacher questions and a hand-written set. The next
version uses no test split of any dataset.

## 0.2.1

- **GGUF builds and a 2B model.** [JevK5-GGUF](https://huggingface.co/alibiserikbay/JevK5-GGUF)
  holds JevK5 v0.2 at Q8_0 and Q4_K_M and JevK5-2B at Q8_0, for llama.cpp on almost any GPU or a
  CPU; [JevK5-2B](https://huggingface.co/alibiserikbay/JevK5-2B) is the v0.2 recipe on
  Qwen3.5-2B (temperature 1.42). The weights of JevK5 v0.2 itself are unchanged.
- `jevk5.JevK5GGUF`: the same readout through `llama-server`, standard library only, and
  `bench/gguf_check.py` to compare a GGUF with the bf16 model on the public items. The prompt and
  option mapping moved to `jevk5/prompt.py`, which imports no torch; `import jevk5` no longer loads
  torch until `JevK5` is used.
- A 2B Q4_K_M was built and not published: it changed 29 of 231 answers and hard-tier accuracy fell
  from 0.604 to 0.514.

## Correction, 2026-09-23

JevBench's v1.4 scan noted that our hand-written calibration set
(`training/hard_dev_items.py`, 65 items) used one generic judge instruction that also appears in
eight public hard items. Our own 8-word-sequence scan then found a second echo we had missed: one
date item whose rule wording followed a public item's closely. Both are rewritten; the scan over
every published file now returns nothing.

What it affected: that set is never trained on, but it was one of the held-out checks used to
choose between one shared temperature and one per question type. It argued for the shared
temperature we shipped - and the per-type option measures *better* on the public items (hard-tier
ECE 0.046 against 0.066), so the echo cost us rather than helped. A temperature never reorders a
distribution, so no accuracy figure in any table changes. No JevBench text has ever been in the
training data, which the teacher wrote from scratch on a machine with no benchmark files.

## 0.2.0

- Trained on 3,272 teacher questions, double v0.1's 1,635 and from the same eleven families, plus
  a 9-question pilot of a new probability family whose answers are exact distributions (too few to
  credit for anything). Public hard tier 0.676 -> 0.739, hard-tier ECE 0.082 -> 0.066, distance to
  the exact gold distributions 0.296 -> 0.196.
- Temperature 1.367 -> 1.532, fitted the same way on held-out teacher domains.
- Measured on held-out data and rejected: one temperature per question type, and averaging two
  option orders (`training/temp_choice.py`, `training/order_avg.py`).

## 0.1.1

- The JevBench adapter answers inputs up to 16,384 tokens; 0.1.0 refused anything over 4,096.
  JevBench's hard tier has policy documents of up to ~6k tokens, and the longest public prompt is
  already 4,033 tokens. Inputs past the largest CUDA graph (4,096) run the same model eagerly.
  Weights, prompt, temperature and every answer at or under 4,096 tokens are unchanged.

## 0.1.0

- First release: Qwen3.5-4B + distilled LoRA (merged), temperature 1.367, CUDA-graph runtime,
  in-process JevBench adapter and `jevk5-serve`.

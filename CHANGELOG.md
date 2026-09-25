# Changelog

## 0.3.0

- **JevK5 v0.3 weights** at [alibiserikbay/JevK5](https://huggingface.co/alibiserikbay/JevK5)
  (Qwen3.5-4B, temperature 1.367). v0.2 stays available under the Hub tag `v0.2`.
  - Teacher data is five times larger and comes from two teachers: 17,408 questions (v0.2's 3,270
    from Qwen3.6-27B, plus 14,138 from GPT-6 Luna).
  - Replay is 32,425 items from the train splits of 27 public datasets. No test or validation
    split of any dataset is used, and MMLU-Pro and BBH are not used at all (see the 2026-09-24
    correction below). Every row is checked against the Decision Index benchmarks' test and
    validation text and JevBench's public items.
  - One epoch over 49,833 rows.
- **New: [JevK5-9B](https://huggingface.co/alibiserikbay/JevK5-9B)**, the same data and recipe on
  Qwen3.5-9B (temperature 1.089).
- **Held-out checks, v0.2 → v0.3 4B → 9B:**
  - index proxy 0.620 → 0.740 → 0.788
  - held-out teacher questions 0.804 → 0.815 → 0.865
  - hand-written hard set 0.769 → 0.797 → 0.844
  - bev-decision-150K test sample 0.665 → 0.670 → 0.695
- **JevBench public (231), v0.2 → v0.3 4B → 9B:**
  - hard tier 0.739 → 0.748 → 0.757
  - standard 0.958 → 0.972 → 0.944
  - hard-tier ECE 0.066 → 0.071 → 0.095
  - The 4B's hard tier is flat against v0.2 (8 items fixed, 7 broken). Judging answers
    (0.76 → 0.59) and dates and numbers (0.47 → 0.33) got worse.
  - The 9B is not better than the 4B on these items: 6 hard items fixed, 5 broken, and 2 fewer
    standard items right.
- **More than 16 options, 4B v0.2 → v0.3:**
  - CLINC150 0.666 → 0.706
  - MASSIVE 0.754 → 0.768
  - BANKING77 0.690 → 0.636, and its ECE 0.039 → 0.075. The second temperature (0.77) is
    unchanged.
- **More than 16 options, JevK5-9B:** MASSIVE 0.808, BANKING77 0.730, CLINC150 0.772. But with the
  shared second temperature (0.77) its probabilities are overconfident (ECE 0.087, 0.118, 0.088).
  At 1.0 the same passes give 0.039, 0.056 and 0.050. The runtime does not yet take a second
  temperature per model.
- **GGUF**, answers the same as bf16 on the 231 public items:
  - v0.3 4B: Q8_0 228/231, Q5_K_M 221/231.
  - 9B: Q8_0 228/231, Q5_K_M 224/231.
  - Not published, below 95%: the v0.3 4B Q4_K_M (214/231) and the 9B Q4_K_M (218/231).
- **Runtime:** unchanged apart from the version number; 0.2.2 already loads v0.3 and the 9B.
  `JevK5GGUF()` still defaults to v0.2's temperature (1.532). With v0.3 files, pass
  `temperature=1.367` (4B) or `temperature=1.089` (9B).
- **Licensing:** a development variant that also trained on ANLI (CC BY-NC 4.0) and NLI4CT (no
  stated license) is not released. The released models' data sources and licenses are listed on
  their cards.

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

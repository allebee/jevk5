# Changelog

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

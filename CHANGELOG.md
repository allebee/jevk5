# Changelog

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

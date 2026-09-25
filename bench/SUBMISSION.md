# Running JevK5 on JevBench

Three rows, all with in-process adapters and no server. The JevBench request is
[issue #31](https://github.com/fstandhartinger/jevbench/issues/31).

| Row | Adapter | Weights (Hub) | Revision | Needs |
|---|---|---|---|---|
| **JevK5 v0.3 (4B)**, the main row | `jevk5_direct` | `alibiserikbay/JevK5` | `c4f7fdb3aeab5582336406e78d3bef11bf98833d` (tag `v0.3`) | CUDA GPU, ~10 GB |
| JevK5-9B v0.3.3 | `jevk5_direct` | `alibiserikbay/JevK5-9B` | `d6521a18a86999190e9d775c915af3d6d6772fc4` (tag `v0.3.3`) | CUDA GPU, ~20 GB |
| JevK5-Lite preview-1 (experimental) | `jevk5_lite` | `alibiserikbay/JevK5-Lite` | `315ee211f828a899161477c534cea23e84ba3568` (tag `preview-1`) | CPU |

```bash
git clone https://github.com/allebee/jevk5 && cd jevk5 && git checkout v0.3.3
pip install -e ".[fast,lite]"   # torch, transformers>=5.17, flash-linear-attention; safetensors for Lite
cp bench/jevk5_direct.py bench/jevk5_lite.py <jevbench>/jevbench/adapters/
cd <jevbench> && git apply <jevk5>/bench/jevbench-registration.patch   # registers both adapters

JEVBENCH_WARM_LOAD=1 python -m jevbench.cli run --adapter jevk5_direct \
    --endpoint alibiserikbay/JevK5 --revision c4f7fdb3aeab5582336406e78d3bef11bf98833d --model jevk5-v0.3 \
    --tasks <tasks.jsonl> --results <out.jsonl> ...
JEVBENCH_WARM_LOAD=1 python -m jevbench.cli run --adapter jevk5_direct \
    --endpoint alibiserikbay/JevK5-9B --revision d6521a18a86999190e9d775c915af3d6d6772fc4 --model jevk5-9b-v0.3.3 \
    --tasks <tasks.jsonl> --results <out.jsonl> ...
JEVK5_LITE_DTYPE=bf16 JEVK5_LITE_THREADS=16 CUDA_VISIBLE_DEVICES= JEVBENCH_WARM_LOAD=1 python -m jevbench.cli run \
    --adapter jevk5_lite --endpoint alibiserikbay/JevK5-Lite \
    --revision 315ee211f828a899161477c534cea23e84ba3568 --model jevk5-lite-preview1 \
    --tasks <tasks.jsonl> --results <out.jsonl> ...
```

Each model reads its temperatures from the `jevk5_config.json` (or, for Lite, the `lite_config.json`) at the
pinned revision:
- v0.3 4B: 1.22, with 0.93 for questions with more than 16 options.
- JevK5-9B v0.3.3: 1.316 and 1.05.
- `JEVK5_LITE_DTYPE` defaults to fp32, which is portable. The Lite numbers below are bf16 on an AMX CPU.

## System card (JevK5 v0.3 and JevK5-9B v0.3.3)

- **Weights:** Qwen3.5-4B and Qwen3.5-9B, each with a LoRA r16 on the attention projections, merged, in bf16.
  Apache-2.0.
- **Readout:** native. A softmax over the declared options' answer-letter logits at the last position,
  divided by the temperature. 0 output tokens. The option mapping is identical to `semif_direct`:
  noul → true/false, choice → criteria keys, score → level indices.
- **Limits:** up to 16 options take one pass. More options are read in groups of 16 plus a final.
  Inputs over 16,384 tokens per pass are refused and recorded as a failure, never truncated.
- **Runtime:** CUDA graphs per padded length (128 ... 4096 tokens); longer inputs run eagerly.
- **Cost basis:** your base-model price rule for Qwen3.5-4B / Qwen3.5-9B. Measured input tokens are 164 easy,
  168 standard and 1,274 hard per decision; 0 output tokens.
- **Training data:**
  - 3,270 Qwen3.6-27B questions and about 14,000 GPT-6 Luna questions, each double-solved.
  - Replay from the train splits of about 25 openly licensed datasets. No test or validation split of
    anything, and no MMLU, MMLU-Pro, RACE, ANLI or NLI4CT.
  - Every training row was scanned against JevBench's 231 public items for shared 8-word sequences, and hits
    were dropped. No Jev output was used.
  - Model choice used held-out teacher questions and held-out train-split rows; JevBench's items are only
    reported.
  - The 9B's 1.5 epochs came from a sweep whose rule was fixed before the results.

JevK5-Lite is a 437M DeBERTa-v3-large label classifier, not a JevK5 successor. Its mapping, 512-token limit
and coverage list are in the README section "JevK5-Lite on JevBench and the Decision Index".

## Reference local runs (public items)

All 231 are valid, with 0 failures, through the official runner. Per-item files are in
[results/public231](../results/public231).

| Row | Easy (48) | Standard (72) | Hard (111) | Hard ECE | TVD (10 probability items) | p50 |
|---|---:|---:|---:|---:|---:|---|
| JevK5 v0.3 (4B) | 1.000 | 0.944 | 0.784 | 0.054 | 0.164 | 13.6 ms (H100) |
| JevK5-9B v0.3.3 | 1.000 | 0.958 | 0.775 | 0.071 | 0.251 | 16.6 ms (H100) |
| JevK5-Lite preview-1 | 0.958 | 0.750 | 0.405 | 0.269 | 0.310 | 71 / 205 ms (CPU, 16 threads) |
| JevK5 v0.2 (history) | 1.000 | 0.958 | 0.739 | 0.066 | 0.196 | 13.5 ms (H100) |

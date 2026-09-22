# Running JevK5 on JevBench

In-process adapter, no server. Needs one CUDA GPU with ~10 GB free (weights ~8.5 GB in bf16).

```bash
git clone https://github.com/allebee/jevk5 && cd jevk5 && git checkout v0.1.0
pip install -e ".[fast]"        # torch, transformers>=5.17, flash-linear-attention
cp bench/jevk5_direct.py <jevbench>/jevbench/adapters/
cd <jevbench> && git apply <jevk5>/bench/jevbench-registration.patch   # against 0caa1d0
JEVBENCH_WARM_LOAD=1 python -m jevbench.cli run --adapter jevk5_direct \
    --endpoint alibiserikbay/JevK5 --revision <hub commit> --model jevk5-v0.1 \
    --tasks <tasks.jsonl> --results <out.jsonl> ...
```

## System card

- **Name:** JevK5 v0.1
- **Weights:** `alibiserikbay/JevK5` (Qwen3.5-4B + LoRA r16 on attention projections, merged,
  bf16) with `jevk5_config.json` holding the temperature 1.367
- **Readout:** native. Softmax over the declared options' answer-letter logits at the last
  position, divided by the temperature. 0 output tokens.
- **Option mapping:** identical to `semif_direct` (noul -> true/false, choice -> criteria keys,
  score -> level indices; each option shown as "<id>: <description>")
- **Limits:** at most 16 options; inputs over 4,096 tokens are refused (recorded as a failure by
  the runner), never truncated
- **Runtime:** CUDA graphs per padded length (128 ... 4096 tokens), captured at load (~3 s)
- **Cost basis:** same weights class as SemIf: deepinfra Qwen3.5-4B, $0.03/M input tokens;
  measured input tokens 164 easy / 168 standard / 1,274 hard per decision, 0 output tokens
- **Training data:** teacher-written decisions (Qwen3.6-27B, Apache-2.0) and public human-labelled
  datasets; no JevBench items and no Jev outputs were used for training, tuning or selection

## Reference local run (public items)

Official runner at `0caa1d0`, one H100, batch 1: 231/231 valid, 0 failures
([results/public231](../results/public231)).

| Split | n | Accuracy | ECE | p50 raw | p95 raw |
|---|---:|---:|---:|---:|---:|
| easy | 48 | 1.000 | 0.026 | 13.5 ms | 14.9 ms |
| original (standard) | 72 | 0.958 | 0.073 | 13.6 ms | 14.8 ms |
| hard (public half) | 111 | 0.676 | 0.082 | 30.1 ms | 160.9 ms |

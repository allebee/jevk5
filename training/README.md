# Training pipeline

The scripts that produced JevK5 v0.1. They are published for transparency; paths and settings
are the ones we used.

1. `teacher.py`: hard decisions from a teacher on any OpenAI-compatible server (we used vLLM
   serving Qwen3.6-27B-FP8 with thinking). Three questions per document, each solved twice;
   `--split test` uses three domains the train split never sees.

       python teacher.py --out data/teacher/train.jsonl --docs 3000 --parallel 96
       python teacher.py --out data/teacher/test.jsonl --split test --seed 1 --docs 150

2. `build_train.py`: kept teacher questions (option keys rebuilt from the option text) plus a
   replay of human-labelled public datasets, and a dev set (held-out-domain teacher questions,
   `hard_dev_items.py`, public dev splits). v0.1 used `--replay-share 0.5`.

3. `lora.py`: LoRA on Qwen3.5-4B, cross-entropy on the option-letter logits; the adapter is merged
   into the weights. v0.1: `--epochs 2 --lr 3e-5 --warmup 20`, rank 16. On Hopper GPUs the
   flash-linear-attention backward needs Triton >= 3.7.1; with older Triton the script falls back
   to transformers' PyTorch kernels.

The temperature is one scalar minimizing negative log-likelihood on the held-out-domain teacher
questions, written to `jevk5_config.json`.

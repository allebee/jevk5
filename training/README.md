# Training pipeline

The scripts that produced JevK5 v0.2. They are published for transparency; paths and settings
are the ones we used.

1. `teacher.py`: hard decisions from a teacher on any OpenAI-compatible server (we used vLLM
   serving Qwen3.6-27B-FP8 with thinking). Three questions per document, each solved twice;
   `--split test` uses three domains the train split never sees.

       python teacher.py --out data/teacher/train.jsonl --docs 3000 --parallel 96
       python teacher.py --out data/teacher/test.jsonl --split test --seed 1 --docs 150

2. `build_train.py`: kept teacher questions (option keys rebuilt from the option text) plus a
   replay of human-labelled public datasets, and a dev set (held-out-domain teacher questions,
   `hard_dev_items.py`, public dev splits). v0.1 and v0.2 used `--replay-share 0.5`; several
   teacher files can be passed at once.

3. `lora.py`: LoRA on Qwen3.5-4B, cross-entropy on the option-letter logits; the adapter is merged
   into the weights. v0.1 and v0.2: `--epochs 2 --lr 3e-5 --warmup 20`, rank 16. On Hopper GPUs the
   flash-linear-attention backward needs Triton >= 3.7.1; with older Triton the script falls back
   to transformers' PyTorch kernels.

The temperature is one scalar minimizing negative log-likelihood on the held-out-domain teacher
questions, written to `jevk5_config.json` (v0.2: 1.532).

`temp_choice.py` and `order_avg.py` are the two checks behind v0.2's readout, both on held-out data
only: one temperature per question type against one shared (shared kept), and averaging two option
orders against one order (one order kept, since averaging cost accuracy and doubles the work).

The `probability` family in `teacher.py` (`--families probability`) writes questions whose answer
is an exact distribution, verified by making both solutions reproduce it. v0.2 contains a 9-question
pilot of it.

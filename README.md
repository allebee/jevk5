# JevK5 — an open-weight Jev alternative

JevK5 is an independent, Apache-2.0 open-source alternative to TypeSafe's Jev for typed
decisions. Give it a state (a ticket, log, policy, or diff) and a yes/no, choice, or score question.
It returns a probability for every option in one forward pass, with zero generated tokens. The
[weights](https://huggingface.co/alibiserikbay/JevK5) run on your own GPU, and the server accepts
the TypeSafe-style `/v1/systemone` request shape. JevK5 is not affiliated with TypeSafe AI.

**v0.3** (runtime 0.3.0): five times the teacher data from two teachers, replay from public train
splits only, and a new [JevK5-9B](https://huggingface.co/alibiserikbay/JevK5-9B). See
[What changed in v0.3](#what-changed-in-v03). Most of the gains are on held-out and index-style
data; on JevBench's public items the 4B gains on the hard tier and loses one standard item.

**Independent result:** [JevBench v1.4](https://github.com/fstandhartinger/jevbench) ranks JevK5
v0.2 **second of 76 systems and first among open entrants** (62.04; Jev 1.13.0: 63.29). On its 308
fresh sealed decisions, JevK5 answered 33.1% correctly and Jev answered 36.7%; the [evaluator calls
this set unusually
difficult](https://github.com/fstandhartinger/jevbench/blob/main/docs/METHOD-v1.4.md). The ranking
measures JevBench's mix of accuracy, calibration, speed, and cost, not performance on every
production workflow. JevK5's reported H100 latency is about 13 ms for short decisions. On the [Jev
Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index), JevK5 0.2.2 (v0.2
weights) scores 36.31, 15th of 49, with the 4th-best calibration. v0.3 has not been run by either
benchmark yet.

## Watch it run

**Typed decision on an NVIDIA L40S.** A real support ticket becomes a probability over three teams;
the displayed decision took 20.60 ms after model loading. This is a paced replay of recorded GPU output.
Click the moving preview for the [full 27-second video](docs/demo/jevk5-nvidia-demo.mp4).

[![JevK5 turns a support ticket into a typed decision with probabilities](docs/demo/jevk5-preview.gif)](docs/demo/jevk5-nvidia-demo.mp4)

**Snake against seeded random.** JevK5 collected 7 apples to random's 3 in a live browser capture
using [Prompt Engineer 48's MIT-licensed arena](https://github.com/PromptEngineer48/laya-vs-jev-arena).
Click the moving preview for the [full 24-second video](docs/game-demos/snake.mp4).
This game is not a Jev comparison.

[![JevK5 plays Snake against a seeded random player](docs/game-demos/snake-preview.gif)](docs/game-demos/snake.mp4)

![JevBench v1.4: JevK5 is second of 76](docs/jevbench-v1.4.png)

| | |
|---|---|
| Weights | [alibiserikbay/JevK5](https://huggingface.co/alibiserikbay/JevK5): Qwen3.5-4B + a distilled LoRA, merged (Apache-2.0); [JevK5-9B](https://huggingface.co/alibiserikbay/JevK5-9B) on Qwen3.5-9B |
| Readout | SemIf's protocol: a softmax over the answer letters' next-token logits, one temperature |
| Runtime | One CUDA graph per padded input length: 13 ms vs ~70 ms eager on an H100, same answers |
| Training | Distilled from Qwen3.6-27B and GPT-6 Luna, on hard decisions they wrote and checked twice, plus public train splits |

## How JevK5 differs from Jev

JevK5 uses open Qwen3.5-4B (or 9B) weights and [SemIf's option-logit
readout](https://github.com/TheoLeeCJ/SemIf), not Jev's unpublished model architecture. It supports
the same three decision types—`noul` (yes/no), `choice`, and `score`—through a TypeSafe-style
endpoint. Each question is evaluated separately; the server serializes requests on one GPU. The
model is English-only, answers up to 16 options in one pass and more in several (see [More than 16
options](#more-than-16-options)), and refuses inputs over 16,384 tokens. Its quality on Jev's
published real-world workflows has not yet been measured.

## What changed in v0.3

| | v0.2 | **v0.3 (4B)** | **JevK5-9B (v0.3)** |
|---|---:|---:|---:|
| Teacher questions | 3,272 (Qwen3.6-27B) | 17,408 (Qwen3.6-27B + GPT-6 Luna) | same |
| Public replay | 3,272 items, 7 datasets | 30,052 items, 26 train splits | same |
| Index proxy (our estimate) | 0.620 | 0.731 | 0.762 |
| Held-out teacher questions | 0.801 | 0.834 | 0.851 |
| Hand-written hard set | 0.766 | 0.766 | 0.781 |
| bev-decision-150K test sample | 0.665 | 0.663 | 0.700 |
| JevBench public, hard tier | 0.739 | 0.784 | 0.730 |
| `temperature` / `knockout_temperature` | 1.532 / 0.77 | 1.22 / 0.93 | 1.049 / 1.2 |
| License | Apache-2.0 | Apache-2.0 | Apache-2.0 |
| GPU memory in bf16 | ~9 GB | ~9 GB | ~19 GB |

- The **index proxy** is our own estimate, not an index score. It is the chance-corrected skill
  averaged over held-out train-split rows of 16 Decision Index benchmarks (40 rows each). All
  held-out numbers here are read through the runtime on the same dev set.
- **bev-decision-150K** is another group's decision mix. We scored 4,723 questions from its test
  split, for evaluation only.
- **No test split of any dataset** went into v0.3: the replay comes from train splits only, and
  every row was checked against the index benchmarks' test and validation text and against
  JevBench's public items. No split of MMLU or MMLU-Pro is used: MMLU's `auxiliary_train` was
  dropped because most of it is RACE, which is for non-commercial research only.
- The full data list with licenses, the declared overlap with the Decision Index, and the weak
  spots are on the model cards: [JevK5](https://huggingface.co/alibiserikbay/JevK5) and
  [JevK5-9B](https://huggingface.co/alibiserikbay/JevK5-9B).

## Results on JevBench's public items

JevBench v1.2, 231 public items, through JevBench's own runner (`bench/jevk5_direct.py`):
231/231 valid, 0 failures. "Untrained" is the same base model and prompt without the LoRA or the
temperature; its answers match SemIf's official public outcomes on all 231 items.

| Split | n | Untrained Qwen3.5-4B | JevK5 v0.1 | JevK5 v0.2 | **JevK5 v0.3** | **JevK5-9B v0.3** |
|---|---:|---:|---:|---:|---:|---:|
| easy | 48 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| original (standard) | 72 | 0.986 | 0.958 | 0.958 | 0.944 | 0.944 |
| hard (public half) | 111 | 0.613 | 0.676 | 0.739 | **0.784** | 0.730 |
| hard-tier ECE | | 0.117 | 0.082 | 0.066 | **0.054** | 0.126 |
| distance to the exact gold distributions | | 0.298 | 0.296 | 0.196 | **0.164** | 0.284 |

Latency on one H100 (in-process, batch 1):
- 4B: p50 13.2 ms, p95 14.6 ms on easy and standard items; p50 30 ms, p95 162 ms on hard items
  with 1-4k-token documents.
- JevK5-9B: p50 32 ms, p95 36 ms on easy and standard items; p50 74 ms, p95 373 ms on hard items.
  It was measured while another job shared the GPU.

Per-item results are in [results/public231](results/public231). These are public-item numbers from
our own runs, not an official JevBench score.

**v0.3 (4B) against v0.2 on the hard tier:** 10 items fixed, 5 broken (McNemar p = 0.30, not
significant on 111 items). On the standard tier: 3 fixed, 4 broken.
- Gains: probability 0.70 -> 1.00, adversarial 0.83 -> 1.00, long policies 0.58 -> 0.68,
  multi-step lookups 0.78 -> 0.83.
- Losses: judging answers 0.76 -> 0.65. Dates and numbers stay at 0.47.

**JevK5-9B on these items:** against the untrained Qwen3.5-9B (easy 1.000, standard 0.958, hard
0.676) it fixes 15 hard items and breaks 9. **Against the v0.3 4B it is worse:** 4 hard items fixed
and 10 broken, and its hard-tier ECE (0.126) and distance to the gold distributions (0.284) are
worse than the 4B's. Its gains are on the held-out checks above, not on JevBench's public items.

**v0.2 against the untrained model and v0.1:** on the hard tier v0.2 fixes 21 of the untrained
model's items and breaks 7 (McNemar p = 0.013); against v0.1 it fixes 9 and breaks 2. Per family,
the gains are probability 0.50 -> 0.70, ambiguous 0.71 -> 0.86, trade-offs 0.83 -> 1.00, long
policies 0.47 -> 0.58, multi-step lookups 0.72 -> 0.78 and dates and numbers 0.40 -> 0.47; judging
answers slips 0.82 -> 0.76 (one item).

**Known weak spots:**
- v0.3 (4B) gets one standard-tier item fewer than v0.2 (0.944 against 0.958); its standard-tier
  calibration is much better (ECE 0.057 against 0.141).
- Judging answers is the one hard-tier family that got worse (0.76 -> 0.65).
- JevK5-9B is behind the 4B on the hard tier (0.730 against 0.784) and badly calibrated there
  (ECE 0.126).

## Install and use

```bash
pip install "jevk5[fast] @ git+https://github.com/allebee/jevk5@v0.3.0"
```

```python
from jevk5 import JevK5

model = JevK5("alibiserikbay/JevK5")          # v0.3, ~9 GB of GPU memory in bf16
# model = JevK5("alibiserikbay/JevK5-9B")     # ~19 GB
model.decide(
    "Refunds need a receipt and a purchase within 30 days. "
    "The customer bought 12 days ago and has no receipt.",
    {"type": "noul", "instructions": "Is a refund permitted under the policy?"},
)
# {'type': 'noul', 'confidence': ..., 'noul': <probability of true>, 'input_tokens': ...}
```

Question types: `noul` (yes/no, optional `criteria` {"true": ..., "false": ...}), `choice`
(`criteria` {key: description} or a list of keys), `score` (`criteria` a list of level
descriptions, lowest first).

To keep using v0.2, load the Hub tag through a local copy:
`JevK5(huggingface_hub.snapshot_download("alibiserikbay/JevK5", revision="v0.2"))`.

### More than 16 options

JevK5 answers with one of 16 letters, so up to 16 options take one pass, exactly as before: 0.2.2
returns bit-identical probabilities to 0.2.1 on all 231 public items, through both runtimes. With
more options, `decide()` still returns a probability for every option, summing to 1, with no
retraining:

1. The options are split, in order, into groups of at most 16, and each group is read.
2. A final of 16 is read: the top options of every group, the free places going to the next most
   likely options in any group.
3. Finalists keep the final's distribution, times the chance the answer is among them. Every other
   option gets its group's share of the final times its in-group probability.
4. The letter temperature is fitted on questions of up to 16 options and leaves this combination
   miscalibrated, so it is sharpened by a second temperature. Since 0.3.0 each model carries its
   own, `knockout_temperature` in its `jevk5_config.json` (v0.3 4B: 0.93, JevK5-9B: 1.2). A
   config without it (v0.2) keeps 0.77, and `knockout_temperature=` in `JevK5` or `JevK5GGUF`
   overrides both. Every value is fitted on 500 items of MASSIVE's train split, which is not a
   Decision Index benchmark. It never changes the answer.

That is ceil(n / 16) + 1 passes. `JevK5GGUF` combines its passes the same way
([jevk5/prompt.py](jevk5/prompt.py)). On train splits, with every intent offered in the Decision
Index's request shape, 500 items each and none that v0.2 trained on
([bench/many_options.py](bench/many_options.py)):

| Train split | Options | Passes | Accuracy | Macro-F1 | ECE | p50 on an H100 |
|---|---:|---:|---:|---:|---:|---:|
| MASSIVE en-US (the fitting set) | 60 | 5 | 0.754 | 0.746 | 0.038 | 89 ms |
| BANKING77 | 77 | 6 | 0.690 | 0.674 | 0.039 | 116 ms |
| CLINC150 with out-of-scope | 151 | 11 | 0.666 | 0.720 | 0.039 | 199 ms |

v0.3, on the same items (none of them in v0.3's training or dev data), each model with its own
second temperature (MASSIVE is the fitting set; BANKING77 and CLINC150 are only reported):

| Train split | v0.3 (4B) accuracy / macro-F1 / ECE | JevK5-9B accuracy / macro-F1 / ECE |
|---|---:|---:|
| MASSIVE en-US | 0.738 / 0.724 / 0.045 | 0.818 / 0.815 / 0.036 |
| BANKING77 | 0.652 / 0.632 / 0.044 | 0.734 / 0.722 / 0.044 |
| CLINC150 with out-of-scope | 0.700 / 0.759 / 0.056 | 0.780 / 0.815 / 0.061 |

The v0.3 4B is better than v0.2 on CLINC150 (0.666 -> 0.700), and worse on MASSIVE (0.754 ->
0.738) and BANKING77 (0.690 -> 0.652). With v0.2's 0.77 instead of its own 0.93 it would be
overconfident (ECE 0.070, 0.101, 0.073). JevK5-9B (second temperature 1.2) is 8 points more accurate than the 4B on each set.

The alternative we built, `method="tree"`, reads one pass whose letters stand for whole groups. It
scored 0.636 on BANKING77 and 0.584 on CLINC150, and is kept only to reproduce the comparison.
Through llama.cpp, the Q8_0 file gives the same answer as bf16 on 95 of 100 BANKING77 items and on
116 of 119 CLINC150 items (60 out-of-scope, 59 in-scope).

**Weak spot: out of scope.** On CLINC150, "none of the listed intents" reaches the final in all 91
out-of-scope items but wins it in only 33. It also wins in 58 in-scope items, which leaves recall
and precision at 0.36 each (v0.2). v0.3 (4B) recalls 0.27 with precision 0.58. JevK5-9B recalls 0.46 with precision 0.76.

As a server:

```bash
jevk5-serve --model alibiserikbay/JevK5 --port 8090
curl -s localhost:8090/v1/systemone -d '{"state": "Order #7120 shows delivered to No. 17; the customer lives at No. 71.",
  "questions": {"what": {"type": "choice", "instructions": "What happened to the parcel?",
                         "criteria": ["delivered", "misdelivered", "unknown"]}}}'
```

`flash-linear-attention` (the `fast` extra) speeds up long inputs; without it, Qwen3.5's
linear-attention layers fall back to transformers' PyTorch code.

## Run it on any GPU, a Mac, or a CPU

GGUF builds for [llama.cpp](https://github.com/ggml-org/llama.cpp), which runs on NVIDIA, AMD,
Intel and Apple GPUs and on plain CPUs, are in
[JevK5-GGUF](https://huggingface.co/alibiserikbay/JevK5-GGUF), next to a smaller
[JevK5-2B](https://huggingface.co/alibiserikbay/JevK5-2B) trained with the v0.2 recipe. Each file
was checked against its unquantized model on the 231 public items:

| File | Size | Same answer as bf16 | Hard tier (bf16) | `temperature` / `knockout_temperature` |
|---|---:|---:|---:|---:|
| `jevk5-4b-v0.3-Q8_0.gguf` | 4.48 GB | 229 / 231 | 0.784 (0.784) | 1.22 / 0.93 |
| `jevk5-4b-v0.3-Q5_K_M.gguf` | 3.07 GB | 224 / 231 | 0.784 (0.784) | 1.22 / 0.93 |
| `jevk5-4b-v0.3-Q4_K_M.gguf` | 2.71 GB | 221 / 231 | 0.766 (0.784) | 1.22 / 0.93 |
| `jevk5-9b-v0.3-Q8_0.gguf` | 9.53 GB | 229 / 231 | 0.721 (0.730) | 1.049 / 1.2 |
| `jevk5-9b-v0.3-Q5_K_M.gguf` | 6.47 GB | 225 / 231 | 0.703 (0.730) | 1.049 / 1.2 |
| `jevk5-4b-v0.2-Q8_0.gguf` | 4.48 GB | 228 / 231 | 0.721 (0.739) | 1.532 / 0.77 |
| `jevk5-4b-v0.2-Q4_K_M.gguf` | 2.71 GB | 219 / 231 | 0.730 (0.739) | 1.532 / 0.77 |
| `jevk5-2b-v0.2-Q8_0.gguf` | 2.01 GB | 226 / 231 | 0.622 (0.604) | 1.42 / 0.77 |

The 9B Q4_K_M changed 13 of 231 answers and is not published.

```bash
llama-server --hf-repo alibiserikbay/JevK5-GGUF --hf-file jevk5-4b-v0.3-Q8_0.gguf -c 8192 -ngl 99
pip install --no-deps "jevk5 @ git+https://github.com/allebee/jevk5@v0.3.0"   # standard library only
```

```python
from jevk5 import JevK5GGUF

model = JevK5GGUF(temperature=1.22, knockout_temperature=0.93)   # the file's values (table)
model.decide("Order #7120 shows delivered to No. 17; the customer lives at No. 71.",
             {"type": "choice", "instructions": "What happened to the parcel?",
              "criteria": ["delivered", "misdelivered", "unknown"]})
```

It reads the answer letters' log-probabilities from `llama-server`: the same readout as the CUDA
runtime, and on identical tokens the same probabilities. `bench/gguf_check.py` repeats the
231-item comparison on your own machine. Measured so far: about 0.25 s (2B) and 0.6 s (4B) per short
decision on a CPU alone, and 0.6 s for the 4B on an M1 Pro; consumer GPUs are not measured yet.

## JevK5-Lite: a CPU classifier (preview, experimental)

JevK5-Lite is a separate, experimental model: a 437M DeBERTa-v3-large that reads a text and any number of
label sets in one encoder pass, on a CPU. It returns a calibrated probability per label, with a softmax for
single-label heads and a sigmoid for multi-label heads.

```bash
pip install "jevk5[lite] @ git+https://github.com/allebee/jevk5@v0.3.2"
```

```python
from jevk5 import JevK5Lite

lite = JevK5Lite.from_pretrained("alibiserikbay/JevK5-Lite", threads=16)
lite.classify("I was charged twice, please refund one of them.",
              {"intent": ["refund_request", "order_status"],
               "areas": {"labels": ["billing", "shipping", "account"], "multi_label": True}})
```

Its strength is calibration, not accuracy:
- **Calibration:** on seven public datasets fixed in advance, its calibration error is lower than
  GLiNER2.5-Decide's on all six single-label sets.
- **Neutral accuracy:** it is significantly better on two sets (AG News, Yahoo Answers), within noise on three,
  and worse on two (financial sentiment, GoEmotions). Decide's mean is higher.
- **fast-decisions dev:** Decide is stronger, with head accuracy 0.637 against 0.587 and all heads right 0.488
  against 0.442. Lite's calibration is worse there too.
- **Speed:** both are DeBERTa-v3-large, with no speed difference to claim.

The model card has the tables, the training data and its licenses.

### JevK5-Lite on JevBench and the Decision Index

**JevBench.** `bench/jevk5_lite.py` is an in-process adapter that plugs into JevBench's own runner, like
`bench/jevk5_direct.py`: copy it into `jevbench/adapters/` and apply `bench/jevbench-registration.patch`, which
registers both adapters. It maps each question to one JevK5-Lite head:
- **Text:** the state.
- **Task:** the instructions.
- **noul:** the labels are "true" and "false".
- **choice:** the option descriptions, or the keys when a description is empty.
- **score:** the level texts.

The probabilities are mapped back to the option ids. Results on the 231 public items, through JevBench's runner,
CPU only (bf16, 16 threads), against JevK5 v0.2 on an H100:

| | JevK5-Lite (CPU) | JevK5 v0.2 4B (GPU) |
|---|---:|---:|
| easy (48) | 0.958 | 1.000 |
| original, standard (72) | 0.750 | 0.958 |
| hard, public half (111) | 0.405 | 0.739 |
| hard-tier ECE | 0.269 | 0.066 |
| distance to the gold distributions (10 probability items, TVD) | 0.310 | 0.196 |
| p50 latency, easy and standard / hard | 71 / 205 ms | 13.5 / 30 ms |

Per-item results are in
[results/public231/jevk5-lite-preview1.jsonl](results/public231/jevk5-lite-preview1.jsonl). To rerun them:

```bash
pip install "jevk5[lite] @ git+https://github.com/allebee/jevk5@v0.3.2"
cp bench/jevk5_lite.py <jevbench>/jevbench/adapters/
cd <jevbench> && git apply <jevk5>/bench/jevbench-registration.patch   # registers jevk5_direct and jevk5_lite
JEVK5_LITE_DTYPE=bf16 JEVK5_LITE_THREADS=16 CUDA_VISIBLE_DEVICES= python -m jevbench.cli run --adapter jevk5_lite \
    --endpoint alibiserikbay/JevK5-Lite --revision 315ee211f828a899161477c534cea23e84ba3568 \
    --model jevk5-lite-preview1 --tasks <tasks.jsonl> --results <out.jsonl> ...
```

That revision is the `preview-1` tag. The run above read the same files from a local folder: the weights, scorer,
tokenizer and configs are byte-identical. `JEVK5_LITE_DTYPE` defaults to fp32, which works on any CPU; bf16 needs
AMX or AVX512-BF16 to be fast.

**Decision Index.** Every index question is a `choice` with 2 to 255 options, and a row can hold several questions.
One `classify` call answers a whole row, one head per question:

```python
import json
from jevk5 import JevK5Lite

lite = JevK5Lite.from_pretrained("alibiserikbay/JevK5-Lite", threads=16)

def answer_row(row: dict) -> dict:
    """A Decision Index row -> {question id: {"type", "choice", "probabilities"}}, in one pass."""
    text = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], ensure_ascii=False)
    tasks, back = {}, {}
    for qid, q in row["questions"].items():
        name = q["instructions"] if isinstance(q["instructions"], str) else json.dumps(q["instructions"])
        if name in tasks:  # two questions with the same instructions
            name = f"{qid}: {name}"
        keys = list(q["criteria"])
        labels = [str(q["criteria"][k] or k) for k in keys]
        if len(set(labels)) < len(labels):
            labels = [f"{k}: {d}" for k, d in zip(keys, labels)]
        tasks[name], back[name] = labels, (qid, dict(zip(labels, keys)))
    out = {}
    for name, result in lite.classify(text, tasks).items():
        qid, key_of = back[name]
        probs = {key_of[label]: p for label, p in result["probabilities"].items()}
        out[qid] = {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs}
    return out
```

**Coverage: what JevK5-Lite handles, and how.** A refused or failed row scores 0 on both benchmarks.

| Case | Handling |
|---|---|
| Free-form numbers | Not asked by either benchmark (every question has typed options); JevK5-Lite cannot produce one. |
| Distribution targets (JevBench's 10 probability items) | Mapped: the softmax over the options is the distribution. TVD 0.310, against the 4B's 0.196. |
| More than 16 options, and long option lists | Mapped: there is no letter limit, and every option is scored in one pass. Labels come first, and the text keeps at least 16 tokens; a label list longer than the window extends the sequence past 512 tokens (see the note below). The index's BANKING77 rows (77 intents, 521 tokens of labels) are answered, not refused. |
| Several questions in one row | Mapped: one head per question, in one pass. The questions share the 512-token budget. |
| Documents past 512 tokens | Truncated: the state is cut from its end. On JevBench this hit 52 of the 111 hard items (median state 2,188 tokens); accuracy on them is 0.346, against 0.458 on the other hard items. |
| Multi-step reasoning, dates and numbers | Answered, but weakly: hard-tier multi-hop 0.11 and temporal and numeric 0.27. It is a classifier with no reasoning step. |
| Malformed questions (a choice with fewer than two options) | Refused by the runtime, recorded as a failure, and scored 0. None of the 231 public items was refused. |

**Long label sets.** When the task and label names need more than about 500 tokens, the sequence runs past 512
tokens instead of cutting labels. That is fine for this DeBERTa-v3 model, which has no absolute position
embeddings (`position_biased_input` is false) and uses relative attention with 256 log-scale buckets. It is
also how the model was trained: about 10% of training heads carried their full label set, for example all 151
CLINC150 intents. And it is how the full-label check on held-out BANKING77 items (77 labels, 0.840) was run.
Only the text is truncated, never a label.

JevK5-Lite runs on a CPU, which is what it is for. Every number above is from a CPU with 16 threads.

## How it was made

1. **Teacher data** ([training/teacher.py](training/teacher.py)): Qwen3.6-27B with thinking writes
   realistic documents with three hard typed questions each, across 17 business domains and 11
   decision families (policy exceptions, date and number traps, multi-step lookups, judging
   answers, ambiguity, misleading notes, injected instructions, rule precedence, routing,
   extraction, rubrics). It answers every question twice, independently; a question is kept only
   when both answers match the intended one. Option keys are rebuilt from the option text. v0.3
   adds 14,138 questions written and checked the same way by GPT-6 Luna through OpenAI's API
   (`--provider openai`). Its families are dates and numbers, rubrics, ambiguity, multi-step
   lookups, causal and plausibility judgements, stance and sarcasm, tool choice, and paraphrases
   of Qwen questions. Those outputs are subject to OpenAI's terms.
2. **Replay** (v0.3): 30,052 human-labelled items from the train splits of 26 public datasets,
   listed with their licenses on the model card. No test or validation split of any dataset is
   used. Every row is checked against the Decision Index benchmarks' test and validation text and
   JevBench's public items (exact match or any shared 8-word sequence).
3. **Training** ([training/lora.py](training/lora.py)): LoRA rank 16 on attention projections,
   cross-entropy on the option-letter logits, lr 3e-5. v0.3: 17,408 teacher questions + 30,052
   replay items, 1 epoch. v0.2: 3,272 + 3,272, 2 epochs. v0.2's replay included MMLU-Pro test
   items; see [the correction](CHANGELOG.md). A question that carries an exact distribution trains
   against it rather than a single letter.
4. **Calibration:** one temperature (v0.3: 1.22 for the 4B and 1.049 for the 9B; v0.2: 1.532),
   fitted on teacher questions from three domains that training never saw. A temperature per
   question type and averaging two option orders were both measured on held-out data and rejected
   ([training/temp_choice.py](training/temp_choice.py),
   [training/order_avg.py](training/order_avg.py)).

No JevBench item and no output of Jev was used for training, tuning or model selection (with one correction, see [CHANGELOG](CHANGELOG.md)).

## It also plays Tetris, badly and well

[`examples/tetris.py`](examples/tetris.py) turns Tetris into typed decisions. Reading the board as
characters, JevK5 plays at random: 0.0 lines a game, topping out after 26 pieces, usually taking the
first option offered. Asked instead which placement leaves the best board, with each option
described by what it does ("clears 1 row, buries 0 new cells, tallest column 6"), the same weights
clear **14.9 lines a game** and survive three times longer, at 20-25 ms a decision. A 20-line
heuristic still clears three times more. [What that gap means](examples/README.md).

## Reproduce the JevBench run

See [bench/SUBMISSION.md](bench/SUBMISSION.md).

## Credits

Qwen3.5-4B, Qwen3.5-9B and Qwen3.6-27B by the Qwen team (Apache-2.0). GPT-6 Luna by OpenAI wrote
most of v0.3's teacher questions. The public datasets in the replay belong to their authors (list
and licenses on the model card). Prompt and readout adapted from
[SemIf](https://github.com/TheoLeeCJ/SemIf) by TheoLeeCJ (MIT); see [NOTICE](NOTICE).
Evaluated with [JevBench](https://github.com/fstandhartinger/jevbench) (MIT). Not affiliated with
TypeSafe AI or Jev.

## License

Apache-2.0.

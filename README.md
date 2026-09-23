# JevK5

**Open 4B weights. A document in, a typed decision and probabilities out.**

Route tickets, check policies, or score options in one forward pass with zero generated tokens.
**#2 of 76 on [JevBench v1.4](https://benchmarkheaven.com/jev-models)** · Apache-2.0 ·
[Weights](https://huggingface.co/alibiserikbay/JevK5)

**Hardware:** Linux, Python 3.10+, NVIDIA GPU with bf16 support (Ampere or newer).
Start with **16 GB VRAM**; tested on an L40S, with 10.86 GiB peak PyTorch reserved memory.
Allow about 25 GB disk space. [Fresh-server setup and troubleshooting →](docs/first-run.md)

In an activated virtual environment with compatible CUDA PyTorch installed:

```bash
python -m pip install "jevk5[fast] @ git+https://github.com/allebee/jevk5@v0.2.0"
```

```python
from jevk5 import JevK5

model = JevK5()  # Downloads weights and prepares CUDA graphs on first use.
answer = model.decide(
    "I was charged twice for order #7120. Please refund the duplicate charge.",
    {"type": "choice", "instructions": "Which team should handle this support ticket?",
     "criteria": {"billing": "Payments, invoices, charges, and refunds.",
                  "technical": "Software bugs, errors, and broken integrations.",
                  "sales": "Pricing, product questions, and new purchases."}},
)
print(answer["choice"], answer["probabilities"])
```

Actual output, probabilities rounded:

```text
billing {'billing': 0.993918, 'technical': 0.004944, 'sales': 0.001138}
```

**[Watch the 27-second NVIDIA demo](docs/demo/jevk5-nvidia-demo.mp4)** ·
[Run the timed example](examples/quickstart.py) · [Raw measurements](docs/demo/l40s-run.json)

Measured on an **NVIDIA L40S**: **20.49 ms p50 / 20.89 ms p95**, 30 warm calls on this
152-token ticket, batch 1, bf16, CUDA graphs. Includes tokenization and GPU synchronization;
excludes model loading, network transport, and the demo's reading pauses. The video replays
actual timestamped GPU output. This is one example, not a benchmark-wide latency claim.

## JevBench result

**[JevBench v1.4](https://benchmarkheaven.com/jev-models) ranks JevK5 v0.2 second of 76 systems**,
behind Jev 1.13.0 (63.29 against 62.04) and ahead of every other open system. Judge tier 0.945,
identical to Jev's; the fastest Speed axis in the top five. Measured by the benchmark's author on
their own hardware, not by us.

![JevBench v1.4: JevK5 is second of 76](docs/jevbench-v1.4.png)

| | |
|---|---|
| Weights | [alibiserikbay/JevK5](https://huggingface.co/alibiserikbay/JevK5): Qwen3.5-4B + a distilled LoRA, merged (Apache-2.0) |
| Readout | SemIf's protocol: a softmax over the answer letters' next-token logits, one temperature |
| Runtime | One CUDA graph per padded input length: 13 ms vs ~70 ms eager on an H100, same answers |
| Training | Distilled from Qwen3.6-27B with thinking, on hard decisions it wrote and checked twice |

## Results on JevBench's public items

JevBench v1.2, 231 public items, through JevBench's own runner (`bench/jevk5_direct.py`):
231/231 valid, 0 failures. "Untrained" is the same base model and prompt without the LoRA or the
temperature; its answers match SemIf's official public outcomes on all 231 items.

| Split | n | Untrained Qwen3.5-4B | JevK5 v0.1 | **JevK5 v0.2** |
|---|---:|---:|---:|---:|
| easy | 48 | 1.000 | 1.000 | **1.000** |
| original (standard) | 72 | 0.986 | 0.958 | 0.958 |
| hard (public half) | 111 | 0.613 | 0.676 | **0.739** |
| hard-tier ECE | | 0.117 | 0.082 | **0.066** |
| distance to the exact gold distributions | | 0.298 | 0.296 | **0.196** |

Latency on one H100 (in-process, batch 1): p50 13.5 ms, p95 14.9 ms on easy and standard items;
p50 30 ms, p95 161 ms on hard items with 1-4k-token documents. Per-item results are in
[results/public231](results/public231). These are public-item numbers from our own runs, not an
official JevBench score.

On the hard tier v0.2 fixes 21 of the untrained model's items and breaks 7 (McNemar p = 0.013);
against v0.1 it fixes 9 and breaks 2. Per family, the gains are probability 0.50 -> 0.70, ambiguous
0.71 -> 0.86, trade-offs 0.83 -> 1.00, long policies 0.47 -> 0.58, multi-step lookups 0.72 -> 0.78
and dates and numbers 0.40 -> 0.47; judging answers slips 0.82 -> 0.76 (one item).

**Known weak spots:** two standard-tier items that the untrained model answers correctly are wrong
after training (an unchanged policy pair); the temperature is fitted on hard questions, so
standard-tier confidence is now too low (ECE 0.141 there, against 0.066 on the hard tier).

## More ways to use it

Keep the same `model` object loaded for subsequent decisions. See the
[first-run guide](docs/first-run.md) for a tested CUDA install and the HTTP interface.

Question types: `noul` (yes/no, optional `criteria` {"true": ..., "false": ...}), `choice`
(`criteria` {key: description} or a list of keys), `score` (`criteria` a list of level
descriptions, lowest first).

As a server:

```bash
jevk5-serve --model alibiserikbay/JevK5 --port 8090
curl --fail localhost:8090/v1/systemone -H 'Content-Type: application/json' -d '{"state": "Order #7120 shows delivered to No. 17; the customer lives at No. 71.",
  "questions": {"what": {"type": "choice", "instructions": "What happened to the parcel?",
                         "criteria": ["delivered", "misdelivered", "unknown"]}}}'
```

`flash-linear-attention` (the `fast` extra) speeds up long inputs; without it, Qwen3.5's
linear-attention layers fall back to transformers' PyTorch code.

## How it was made

1. **Teacher data** ([training/teacher.py](training/teacher.py)): Qwen3.6-27B with thinking writes
   realistic documents with three hard typed questions each, across 17 business domains and 11
   decision families (policy exceptions, date and number traps, multi-step lookups, judging
   answers, ambiguity, misleading notes, injected instructions, rule precedence, routing,
   extraction, rubrics). It answers every question twice, independently; a question is kept only
   when both answers match the intended one. Option keys are rebuilt from the option text.
2. **Training** ([training/lora.py](training/lora.py)): LoRA rank 16 on attention projections,
   cross-entropy on the option-letter logits, 3,272 teacher questions + 3,272 human-labelled items
   (MMLU-Pro, WANLI, MultiNLI, BoolQ, banking77, ARC, CommonsenseQA), 2 epochs, lr 3e-5. A question
   that carries an exact distribution trains against it rather than a single letter (9 so far).
3. **Calibration:** one temperature (1.532), fitted on teacher questions from three domains that
   training never saw. A temperature per question type and averaging two option orders were both
   measured on held-out data and rejected ([training/temp_choice.py](training/temp_choice.py),
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

Qwen3.5-4B and Qwen3.6-27B by the Qwen team (Apache-2.0). Prompt and readout adapted from
[SemIf](https://github.com/TheoLeeCJ/SemIf) by TheoLeeCJ (MIT); see [NOTICE](NOTICE).
Evaluated with [JevBench](https://github.com/fstandhartinger/jevbench) (MIT). Not affiliated with
TypeSafe AI or Jev.

## License

Apache-2.0.

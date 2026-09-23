# JevK5 plays three games on an NVIDIA L40S

These are real JevK5 v0.2.0 decisions from one NVIDIA L40S server. The Snake
and Kombat footage comes from [Prompt Engineer 48's MIT arena](https://github.com/PromptEngineer48/laya-vs-jev-arena/tree/b02d19a6659a04142c94efd187caac5e6950fec0),
adapted by [`arena_server.py`](../../examples/arena_server.py) to use JevK5 and a
seeded random opponent. The Tetris run uses this repo's [`tetris.py`](../../examples/tetris.py).
No Jev API calls were made in these games.

| Demo | What happened | Watch | Evidence |
|---|---|---|---|
| Snake race | JevK5 collected **7 apples**, random collected **3**, in 24 seconds | [24-second video](snake.mp4) | [Result](snake-result.json), [poster](snake-poster.png) |
| Tetris | JevK5 cleared **15 lines** in **77 pieces**, then topped out; seed 0 | [27-second replay](tetris.mp4) | [Every placement](tetris-run.json), [poster](tetris.png) |
| Kombat-style fight | JevK5 landed **0 hits**; random landed **1** and led **100% to 83%** health after 24 seconds | [24-second video](fight.mp4) | [Result](fight-result.json), [poster](fight-poster.png) |

The Snake and fight videos are browser captures of the live game. Their
[timestamped request and response trace](arena-decisions.jsonl.gz) contains 559
calls across both games, including warm-ups and a few calls that completed after
Stop. JevK5 made 102 Snake calls and 101 fight calls. The trace records each
state, typed question, answer, probability distribution, input tokens, and
server-side time. Both browser captures had zero page errors.

The Tetris video is a paced playback of the actual 77 recorded board states.
It compresses the sequence into 27 seconds; the displayed placement times come
from the run. The model made 319 typed decisions at an average **44.0 ms per
call** in this run. Each legal placement was described by rows cleared, new
holes, height, and surface roughness; the model picked among those descriptions.
That engineered representation matters to its performance.

## Read the comparisons carefully

- **JevBench** is the direct Jev comparison: JevK5 v0.2 scored 62.04 and Jev
  1.13.0 scored 63.29 in v1.4, a 1.25-point gap. The game opponent here is
  seeded random, so these videos do not establish a Jev game result.
- This Snake arena wraps at the edges and allows the snakes to pass through
  themselves. It measures collecting apples under the same game rules, not
  survival in classic Snake.
- The fight was stopped at 24 seconds. Random led on health; there was no K.O.
  JevK5 often chose to advance, retreat, block, or kick but did not land a hit.
- The browser's p50 values in the Snake and fight HUDs include HTTP through an
  SSH tunnel and the screen capture. The server trace measured **70.4 ms**
  median for two JevK5 Snake questions and **72.1 ms** for two fight questions
  in each request. These are longer prompts than the 152-token, one-question
  [quickstart demo](../demo/README.md); do not compare their timings as if the
  inputs were identical.
- The random side uses uniform sampled legal choices, and a sampled 0–1 value
  for yes/no questions. It performs no model inference. Both browser games
  make concurrent requests, so reruns can differ despite the fixed seed.

## Reproduce

First follow the [NVIDIA setup guide](../first-run.md), then clone the MIT arena
and run the adapter from this JevK5 checkout:

```bash
git clone https://github.com/PromptEngineer48/laya-vs-jev-arena.git /tmp/laya-vs-jev-arena
python examples/arena_server.py --arena /tmp/laya-vs-jev-arena --trace arena-decisions.jsonl
```

Open `http://127.0.0.1:8732/snake/` or `/fight/`, and select JevK5 on the left
and Random on the right. The adapter serves a temporary copy of the arena; the
original repository remains intact. For the Tetris run:

```bash
python examples/tetris.py --player briefed --games 1 --pieces 80 --seed 0 --record tetris-run.json
```

The browser game graphics and rules are from Prompt Engineer 48, copyright 2026,
under the [MIT license](ARENA-LICENSE.txt). The model weights and this adapter
are Apache-2.0.

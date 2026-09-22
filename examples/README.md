# JevK5 plays Tetris

A decision model is not a game player, so this is a test of where the line is. Tetris is a stream of
typed decisions, and [`tetris.py`](tetris.py) gives JevK5 two ways to make them. Nothing about the
engine is hidden from the model and nothing about the model is special-cased in the engine: only
legal options are ever offered, so a game ends when the stack reaches the top, never on a bad move.

| Player | What it sees | Lines per game | Pieces before topping out |
|---|---|---:|---:|
| random | nothing | 0.1 | 25 |
| **JevK5, reading the board** | the board as characters | **0.0** | **26** |
| **JevK5, judging outcomes** | what each placement does | **14.9** | **77** |
| a 20-line heuristic | the board, as code | 49.5 | 150+ |

10 games each for the model (200-piece cap), 20 for the baselines, on one H100. Every decision takes
20–25 ms with zero generated tokens.

**Reading the board, it plays at random.** Given the board drawn as `#` and `.` and asked which
rotation and column to use, JevK5 scores no lines and tops out after 26 pieces, where random tops
out after 25. It is not really choosing: across three games it picked the first column offered 53
times out of 73, at a mean confidence of 0.38. A single forward pass with no generated tokens does
not do spatial planning, and the model says so by defaulting to the first option.

**Judging described outcomes, it plays.** Ask instead which placement leaves the best board, with
each option described by what it does — `clears 1 row, buries 0 new cells, tallest column 6, surface
roughness 9` — and the same model clears 14.9 lines a game and survives three times longer. That is
the job it was trained for: weighing bounded options against stated criteria. The options stay in
board order and are never sorted, and when there are more than eight they are decided as a
tournament, so nothing about which option is good leaks from the ordering.

40 pieces into one such game, 13 lines cleared and 4 cells buried:

```
|..........|
|..######..|
|.##.#.####|
|.#########|
|#####..###|
+----------+
 0123456789
```

**It is still not a Tetris AI.** The 20-line heuristic in the same file, which simply counts holes,
clears three times more. The point of the comparison is the gap between the two model rows: the same
weights, the same 20 ms, and the difference is entirely in whether the question is one a decision
model can answer.

For scale, [rovle/models-playing-tetris](https://github.com/rovle/models-playing-tetris) reports
21.2 pieces for GPT-4V and 19.96 for Gemini Pro Vision from screenshots with chain-of-thought
prompting. That is a different setup — pixels, generated reasoning, another engine — so the numbers
are not comparable to these, but it is worth knowing that a random player in our engine survives 25.

```bash
cd examples
python tetris.py --player briefed --model alibiserikbay/JevK5 --games 10
python tetris.py --player model   --model alibiserikbay/JevK5 --games 10 --show
python tetris.py --player random  --games 20        # and --player greedy
```

"""JevK5 plays Tetris: every placement is two typed decisions about a board drawn as text.

    cd examples && python tetris.py --games 10 --player model    --model ../models/jevk5
    cd examples && python tetris.py --games 10 --player briefed  --model ../models/jevk5
    python tetris.py --games 20 --player random    # the baseline to beat, and --player greedy

Two ways to ask, and the gap between them is the point (results in README.md):
  * `--player model` sees the board as characters and picks a rotation, then a column. It plays at
    random level: 0.0 lines a game, topping out after 26 pieces, mostly taking the first option.
  * `--player briefed` sees what each legal placement does ("clears 1 row, buries 0 new cells,
    tallest column 6") and picks. Same weights, 14.9 lines a game, 77 pieces.
`--player random` and `--player greedy` are the baselines (0.1 and 49.5 lines).

Illegal answers are impossible - only legal options are offered - so a game ends when the stack
reaches the top, never on a malformed move.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

WIDTH, HEIGHT = 10, 20

# One entry per rotation: the filled cells as (row, column), row 0 at the top.
PIECES = {
    "I": [[(0, 0), (0, 1), (0, 2), (0, 3)], [(0, 0), (1, 0), (2, 0), (3, 0)]],
    "O": [[(0, 0), (0, 1), (1, 0), (1, 1)]],
    "T": [
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 1)],
        [(0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (1, 1), (2, 0)],
    ],
    "S": [[(0, 1), (0, 2), (1, 0), (1, 1)], [(0, 0), (1, 0), (1, 1), (2, 1)]],
    "Z": [[(0, 0), (0, 1), (1, 1), (1, 2)], [(0, 1), (1, 0), (1, 1), (2, 0)]],
    "J": [
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 2)],
        [(0, 1), (1, 1), (2, 0), (2, 1)],
    ],
    "L": [
        [(0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
    ],
}
SHAPE_ART = {
    "I": "####", "O": "##/##", "T": "###/.#.", "S": ".##/##.",
    "Z": "##./.##", "J": "#../###", "L": "..#/###",
}


class Board:
    def __init__(self) -> None:
        self.rows = [[0] * WIDTH for _ in range(HEIGHT)]
        self.lines = 0

    def draw(self) -> str:
        body = "\n".join("|" + "".join("#" if c else "." for c in row) + "|" for row in self.rows)
        return f"{body}\n+{'-' * WIDTH}+\n {''.join(str(c % 10) for c in range(WIDTH))}"

    def heights(self) -> list[int]:
        out = []
        for column in range(WIDTH):
            filled = [r for r in range(HEIGHT) if self.rows[r][column]]
            out.append(HEIGHT - filled[0] if filled else 0)
        return out

    def holes(self) -> int:
        total = 0
        for column in range(WIDTH):
            seen = False
            for row in range(HEIGHT):
                if self.rows[row][column]:
                    seen = True
                elif seen:
                    total += 1
        return total

    def landing_row(self, cells: list[tuple[int, int]], column: int) -> int | None:
        """The top row the piece comes to rest at in this column, or None if it does not fit."""
        span = max(c for _, c in cells)
        if column + span >= WIDTH:
            return None
        depth = max(r for r, _ in cells)
        for top in range(HEIGHT - depth):
            if any(self.rows[top + r][column + c] for r, c in cells):
                return top - 1 if top else None
        return HEIGHT - depth - 1

    def place(self, cells: list[tuple[int, int]], column: int, top: int) -> int:
        for r, c in cells:
            self.rows[top + r][column + c] = 1
        kept = [row for row in self.rows if not all(row)]
        cleared = HEIGHT - len(kept)
        self.rows = [[0] * WIDTH for _ in range(cleared)] + kept
        self.lines += cleared
        return cleared


def legal(board: Board, piece: str) -> dict[int, list[int]]:
    """Columns where each rotation of the piece can come to rest."""
    out = {}
    for rotation, cells in enumerate(PIECES[piece]):
        columns = [c for c in range(WIDTH) if board.landing_row(cells, c) is not None]
        if columns:
            out[rotation] = columns
    return out


def state_text(board: Board, piece: str, nxt: str) -> str:
    return (
        f"Tetris board, 10 wide and 20 tall. '#' is a filled cell, '.' is empty. "
        f"Pieces fall to the lowest free place in the columns they cover. A row that is completely "
        f"filled is cleared and scores.\n\n{board.draw()}\n\n"
        f"Column heights: {board.heights()}\nBuried empty cells: {board.holes()}\n"
        f"Piece to place: {piece} ({SHAPE_ART[piece]})\nNext piece: {nxt}"
    )


class ModelPlayer:
    """Two decisions per piece: the rotation, then the leftmost column it occupies."""

    name = "model"

    def __init__(self, model) -> None:
        self.model = model
        self.decisions = 0
        self.seconds = 0.0

    def ask(self, state: str, instructions: str, criteria: dict[str, str]) -> str:
        if len(criteria) == 1:
            return next(iter(criteria))
        start = time.perf_counter()
        answer = self.model.decide(state, {"type": "choice", "instructions": instructions, "criteria": criteria})
        self.seconds += time.perf_counter() - start
        self.decisions += 1
        return answer["choice"]

    def move(self, board: Board, piece: str, nxt: str) -> tuple[int, int]:
        options = legal(board, piece)
        state = state_text(board, piece, nxt)
        turns = {
            f"rotation_{r}": f"the piece turned {r * 90} degrees, shape {shape_art(piece, r)}"
            for r in options
        }
        rotation = int(self.ask(state, "Which rotation of the piece leaves the best board?", turns).split("_")[1])
        columns = {
            f"column_{c}": f"leftmost cell in column {c}" for c in options[rotation]
        }
        column = int(self.ask(state, f"Where should the piece go, turned {rotation * 90} degrees? "
                                     "Prefer flat stacking, no gaps underneath, and completed rows.",
                              columns).split("_")[1])
        return rotation, column


def shape_art(piece: str, rotation: int) -> str:
    cells = PIECES[piece][rotation]
    rows = max(r for r, _ in cells) + 1
    cols = max(c for _, c in cells) + 1
    grid = [["." for _ in range(cols)] for _ in range(rows)]
    for r, c in cells:
        grid[r][c] = "#"
    return "/".join("".join(row) for row in grid)


class JevClient:
    """TypeSafe's /v1/systemone, so Jev can answer the identical question (OpenRouter serves it).

    Set OPENROUTER_API_KEY (or TYPESAFE_API_KEY) in the environment; nothing is read from disk.
    """

    def __init__(self, model: str = "jev-latest") -> None:
        import os

        self.key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        self.url = "https://openrouter.ai/api/v1/systemone"
        if not self.key:
            self.key = os.environ.get("TYPESAFE_API_KEY", "").strip()
            self.url = "https://api.typesafe.ai/v1/systemone"
        if not self.key:
            raise SystemExit("set OPENROUTER_API_KEY (or TYPESAFE_API_KEY) to play as Jev")
        self.model = model
        self.input_tokens = 0

    def decide(self, state: str, question: dict) -> dict:
        import json
        import urllib.request

        body = json.dumps({"model": self.model, "state": state, "questions": {"placement": question}})
        request = urllib.request.Request(
            self.url,
            data=body.encode(),
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(9):  # a rate limit or a sleeping laptop must not end the game
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = json.loads(response.read())
                break
            except Exception as error:  # noqa: BLE001
                if attempt == 8:
                    raise
                wait = min(60, 2 ** attempt)
                print(f"  {type(error).__name__}, retrying in {wait}s", flush=True)
                time.sleep(wait)
        self.input_tokens += (payload.get("usage") or {}).get("input_tokens", 0)
        return payload["answers"]["placement"]


class BriefedPlayer:
    """Every legal placement is described by what it does, and the model judges the trade-off.

    No spatial reading is required: the engine states the consequences, the model picks. Options
    are offered in board order, never sorted, and more than `chunk` of them are decided as a
    tournament, so nothing about which option is good leaks from the ordering or the grouping.
    Both JevK5 and Jev play through this class, so only the decider differs.
    """

    name = "briefed"

    def __init__(self, model, chunk: int = 8) -> None:
        self.model = model
        self.chunk = chunk
        self.decisions = 0
        self.seconds = 0.0

    def describe(self, board: Board, cells, column: int) -> tuple[str, Board]:
        trial = Board()
        trial.rows = [row[:] for row in board.rows]
        cleared = trial.place(cells, column, trial.landing_row(cells, column))
        heights = trial.heights()
        bumps = sum(abs(a - b) for a, b in zip(heights, heights[1:]))
        new_holes = trial.holes() - board.holes()
        parts = [f"clears {cleared} row{'s' if cleared != 1 else ''}" if cleared else "clears nothing",
                 f"buries {new_holes} new cell{'s' if new_holes != 1 else ''}",
                 f"tallest column {max(heights)}", f"surface roughness {bumps}"]
        return ", ".join(parts), trial

    def ask(self, state: str, criteria: dict[str, str]) -> str:
        if len(criteria) == 1:
            return next(iter(criteria))
        start = time.perf_counter()
        answer = self.model.decide(state, {
            "type": "choice",
            "instructions": "Which placement leaves the best board? Prefer clearing rows, then "
                            "burying no cells, then keeping the stack low and flat.",
            "criteria": criteria})
        self.seconds += time.perf_counter() - start
        self.decisions += 1
        return answer["choice"]

    def move(self, board: Board, piece: str, nxt: str) -> tuple[int, int]:
        moves, criteria = [], {}
        for rotation, columns in sorted(legal(board, piece).items()):
            for column in columns:
                text, _ = self.describe(board, PIECES[piece][rotation], column)
                criteria[f"turn_{rotation}_col_{column}"] = text
                moves.append((rotation, column))
        state = state_text(board, piece, nxt)
        keys = list(criteria)
        while len(keys) > 1:  # tournament in board order, chunk by chunk
            winners = []
            for start in range(0, len(keys), self.chunk):
                group = keys[start : start + self.chunk]
                winners.append(self.ask(state, {k: criteria[k] for k in group}))
            keys = winners
        rotation, column = keys[0].split("_")[1], keys[0].split("_")[3]
        return int(rotation), int(column)


class RandomPlayer:
    name = "random"

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.decisions = 0
        self.seconds = 0.0

    def move(self, board: Board, piece: str, nxt: str) -> tuple[int, int]:
        options = legal(board, piece)
        rotation = self.rng.choice(list(options))
        return rotation, self.rng.choice(options[rotation])


class GreedyPlayer:
    """A plain heuristic, for scale: fewest new holes, then lowest landing."""

    name = "greedy"

    def __init__(self) -> None:
        self.decisions = 0
        self.seconds = 0.0

    def move(self, board: Board, piece: str, nxt: str) -> tuple[int, int]:
        best, choice = None, None
        for rotation, columns in legal(board, piece).items():
            cells = PIECES[piece][rotation]
            for column in columns:
                trial = Board()
                trial.rows = [row[:] for row in board.rows]
                top = trial.landing_row(cells, column)
                cleared = trial.place(cells, column, top)
                score = (-cleared, trial.holes(), -top)
                if best is None or score < best:
                    best, choice = score, (rotation, column)
        return choice


def play(player, seed: int, pieces: int, show: bool, frames: list | None = None) -> dict:
    rng = random.Random(seed)
    board = Board()
    bag = list(PIECES)
    current, nxt = rng.choice(bag), rng.choice(bag)
    for placed in range(pieces):
        if not legal(board, current):
            return {"pieces": placed, "lines": board.lines, "holes": board.holes(), "topped_out": True}
        started = time.perf_counter()
        rotation, column = player.move(board, current, nxt)
        elapsed = time.perf_counter() - started
        cells = PIECES[current][rotation]
        top = board.landing_row(cells, column)
        before = board.lines
        board.place(cells, column, top)
        if frames is not None:
            frames.append({
                "board": ["".join("#" if c else "." for c in row) for row in board.rows],
                "piece": current, "rotation": rotation, "column": column,
                "cleared": board.lines - before, "lines": board.lines,
                "holes": board.holes(), "ms": round(elapsed * 1000, 1),
            })
        if show:
            print(f"\n{current} -> rotation {rotation}, column {column}, lines {board.lines}")
            print(board.draw())
        current, nxt = nxt, rng.choice(bag)
    return {"pieces": pieces, "lines": board.lines, "holes": board.holes(), "topped_out": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--player", choices=["model", "briefed", "jev", "random", "greedy"], default="model")
    parser.add_argument("--model", default="alibiserikbay/JevK5")
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--pieces", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jev-model", default="jev-latest", help="model name for --player jev")
    parser.add_argument("--show", action="store_true", help="print the board after every piece")
    parser.add_argument("--record", type=Path, help="write every placement to this JSON file")
    args = parser.parse_args()
    if args.player in ("model", "briefed"):
        from jevk5 import JevK5

        loaded = JevK5(args.model)
        player = ModelPlayer(loaded) if args.player == "model" else BriefedPlayer(loaded)
    elif args.player == "jev":
        player = BriefedPlayer(JevClient(args.jev_model))
        player.name = "jev"
    games, recorded = [], []
    for game in range(args.games):
        if args.player == "random":
            player = RandomPlayer(random.Random(1000 + game))
        elif args.player == "greedy":
            player = GreedyPlayer()
        frames = [] if args.record else None
        result = play(player, args.seed + game, args.pieces, args.show, frames)
        if args.record:
            recorded.append({"seed": args.seed + game, "frames": frames, **result})
        games.append(result)
        print(f"game {game + 1}: {result['pieces']} pieces, {result['lines']} lines, "
              f"{result['holes']} buried cells" + (", topped out" if result["topped_out"] else ""))
    lines = [g["lines"] for g in games]
    holes = [g["holes"] for g in games]
    survived = sum(not g["topped_out"] for g in games)
    print(f"\n{args.player}: {sum(lines) / len(games):.1f} lines per game (best {max(lines)}), "
          f"{sum(holes) / len(games):.1f} buried cells, survived {survived}/{len(games)} games")
    if args.record:
        import json

        args.record.write_text(json.dumps({"player": player.name, "games": recorded}))
        print(f"wrote {args.record}")
    if getattr(player, "decisions", 0):
        print(f"{player.decisions} decisions, {player.seconds / player.decisions * 1000:.1f} ms each")
        tokens = getattr(getattr(player, "model", None), "input_tokens", 0)
        if tokens:
            print(f"{tokens} input tokens, {tokens / player.decisions:.0f} per decision")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

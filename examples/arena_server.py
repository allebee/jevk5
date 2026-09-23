"""Run JevK5 against a seeded random baseline in Prompt Engineer 48's MIT game arena.

Clone https://github.com/PromptEngineer48/laya-vs-jev-arena and run:
    python examples/arena_server.py --arena /path/to/laya-vs-jev-arena

Serves Snake, Tetris, and Kombat at localhost:8732. Each JevK5 answer is a real
model call. The random side samples legal options; it makes no model calls.
The original game rules and MIT license are copied to a temporary directory.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import tempfile
import threading
import time
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def prepare_arena(source: Path, target: Path) -> None:
    for name in ("snake", "fight", "tetris", "shared"):
        shutil.copytree(source / name, target / name)
    for name in ("LICENSE", "index.html"):
        shutil.copy2(source / name, target / name)
    for game in ("snake", "fight", "tetris"):
        path = target / game / "index.html"
        page = path.read_text()
        page = page.replace("Laya · local, in-process", "JevK5 · NVIDIA GPU")
        page = page.replace("Laya · local", "JevK5 · NVIDIA GPU")
        page = page.replace("TypeSafe · jev-latest", "Random · seeded baseline")
        page = re.sub(r'<option value="jev-1\.13\.0">[^<]*</option>', "", page)
        page = re.sub(r'<option value="custom">[^<]*</option>', "", page)
        page = page.replace("model: 'laya', endpoint", "model: 'jevk5', endpoint")
        page = page.replace("model:'laya', endpoint", "model:'jevk5', endpoint")
        page = page.replace("const model = sel === 'custom' ? ($('custom'+side).value.trim() || 'jev-latest') : sel;", "const model = 'random';")
        page = page.replace("{model: sel, endpoint: '/api/jev'", "{model: 'random', endpoint: '/api/jev'")
        page = page.replace("{model:v, endpoint:'/api/jev'", "{model:'random', endpoint:'/api/jev'")
        page = page.replace(">LAYA<", ">JEVK5<").replace(">JEV<", ">RANDOM<")
        page = page.replace("REMOTE / API", "SEEDED RANDOM")
        page = page.replace("model=\"jev-latest\"", "model=\"random\"")
        path.write_text(page)


def random_answer(question: dict, rng: random.Random) -> dict:
    kind = question["type"]
    if kind == "noul":
        p = rng.random()
        return {"type": kind, "noul": p, "confidence": max(p, 1 - p)}
    if kind != "choice":
        raise ValueError(f"unsupported random question type: {kind}")
    criteria = question.get("criteria") or {}
    options = list(criteria if isinstance(criteria, dict) else criteria)
    if not options:
        raise ValueError("no random options")
    choice = rng.choice(options)
    probabilities = {str(key): 1 / len(options) for key in options}
    return {"type": kind, "choice": str(choice), "confidence": 1 / len(options),
            "probabilities": probabilities}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8732)
    args = parser.parse_args()
    if not (args.arena / "LICENSE").exists():
        parser.error("--arena must point to the cloned upstream game repository")
    args.trace.parent.mkdir(parents=True, exist_ok=True)

    from jevk5 import JevK5
    import torch

    print("Loading JevK5 and CUDA graphs...", flush=True)
    model = JevK5()
    hardware = torch.cuda.get_device_name(0)
    model_lock = threading.Lock()
    trace_lock = threading.Lock()
    rng = random.Random(2026)
    run_index = 0
    workdir = Path(tempfile.mkdtemp(prefix="jevk5-arena-"))
    prepare_arena(args.arena, workdir)
    trace = args.trace.open("w")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(workdir), **kw)

        def log_message(self, *a):
            pass

        def reply(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/api/config":
                return self.reply(200, {"laya": True, "serverKey": True})
            if any(part.startswith(".") for part in self.path.split("/")):
                return self.reply(404, {"error": "not found"})
            return super().do_GET()

        def do_POST(self):
            nonlocal run_index
            endpoint = self.path.split("?")[0]
            if endpoint not in ("/api/laya", "/api/jev"):
                return self.reply(404, {"error": "not found"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1024 * 1024:
                    raise ValueError("request must be between 1 byte and 1 MiB")
                request = json.loads(self.rfile.read(size))
                questions = request["questions"]
                if not isinstance(questions, dict) or len(questions) > 32:
                    raise ValueError("provide up to 32 questions")
                state = request["state"]
                if endpoint == "/api/laya":
                    with model_lock:
                        started = time.perf_counter()
                        answers = {qid: model.decide(state, q) for qid, q in questions.items()}
                        torch.cuda.synchronize()
                        elapsed_ms = (time.perf_counter() - started) * 1000
                    tokens = sum(a.pop("input_tokens") for a in answers.values())
                    served = "jevk5"
                else:
                    with trace_lock:
                        started = time.perf_counter()
                        answers = {qid: random_answer(q, rng) for qid, q in questions.items()}
                        elapsed_ms = (time.perf_counter() - started) * 1000
                    tokens = 0
                    served = "seeded_random"
                payload = {"model": served, "answers": answers,
                           "usage": {"input_tokens": tokens, "output_tokens": 0},
                           "latency_ms": round(elapsed_ms, 2)}
                with trace_lock:
                    run_index += 1
                    record = {"index": run_index, "at_utc": datetime.now(timezone.utc).isoformat(),
                              "endpoint": endpoint, "hardware": hardware,
                              "request": request, "response": payload}
                    trace.write(json.dumps(record) + "\n")
                    trace.flush()
                return self.reply(200, payload)
            except (ValueError, KeyError, TypeError) as exc:
                return self.reply(400, {"error": str(exc)})
            except Exception as exc:
                print(f"{type(exc).__name__}: {exc}", flush=True)
                return self.reply(500, {"error": f"model call failed: {type(exc).__name__}"})

    print(f"GPU: {hardware}; Snake /snake/, Tetris /tetris/, Kombat /fight/", flush=True)
    print(f"http://127.0.0.1:{args.port}/", flush=True)
    try:
        ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    finally:
        trace.close()
        shutil.rmtree(workdir)


if __name__ == "__main__":
    main()

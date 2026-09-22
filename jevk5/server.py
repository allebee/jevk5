"""A TypeSafe-style /v1/systemone server for JevK5.

    jevk5-serve --model alibiserikbay/JevK5 --port 8090
    curl -s localhost:8090/v1/systemone -d '{"state": "I was billed twice, please refund",
        "questions": {"refund": {"type": "noul", "instructions": "Asks for money back?"}}}'

Answers carry "noul" (probability of true) for yes/no questions and "probabilities" for choice
and score questions, the shape JevBench's typesafe adapter reads. One GPU, one model: requests
are served one at a time, each question as one CUDA-graph replay.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from jevk5.runtime import JevK5


def normalized(question: dict) -> dict:
    if question.get("type") not in ("noul", "choice", "score"):
        raise ValueError(f"unknown question type {question.get('type')!r}")
    if "instructions" not in question:
        raise ValueError("question is missing instructions")
    criteria = question.get("criteria")
    if question["type"] == "choice":
        if isinstance(criteria, list):
            criteria = dict.fromkeys(criteria)
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError("choice criteria must name at least two options")
    if question["type"] == "score" and (not isinstance(criteria, list) or len(criteria) < 2):
        raise ValueError("score criteria must list at least two levels")
    return {**question, "criteria": criteria}


def make_handler(model: JevK5, name: str):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/health":
                self._send(200, {"ok": True, "model": name})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/v1/systemone":
                self._send(404, {"error": "not found"})
                return
            started = time.perf_counter()
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                with lock:
                    answers = {
                        qid: model.decide(body["state"], normalized(q))
                        for qid, q in body["questions"].items()
                    }
            except (ValueError, KeyError, TypeError) as error:
                self._send(400, {"error": str(error)})
                return
            tokens = sum(a.pop("input_tokens") for a in answers.values())
            self._send(
                200,
                {
                    "model": body.get("model") or name,
                    "answers": answers,
                    "usage": {"input_tokens": tokens, "output_tokens": 0},
                    "latency_ms": round((time.perf_counter() - started) * 1e3, 2),
                },
            )

        def log_message(self, *args) -> None:
            pass

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="alibiserikbay/JevK5")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args(argv)
    model = JevK5(args.model)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model, args.model))
    print(f"serving {args.model} on http://{args.host}:{args.port}/v1/systemone", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

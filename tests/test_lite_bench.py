"""bench/jevk5_lite.py's mapping from JevBench questions to JevK5-Lite heads (no JevBench install needed)."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def adapter():
    pkg = types.ModuleType("fakejb")
    pkg.__path__ = []
    base = types.ModuleType("fakejb.base")

    @dataclass
    class DecisionResult:  # the fields the adapter sets
        adapter: str
        ok: bool
        probs: dict | None = None
        probs_source: str = "unknown"
        model: str = ""
        error: str | None = None
        latency_s: float = 0.0
        usage: dict = field(default_factory=dict)
        raw: object = None
        request_body: object = None

    base.DecisionResult = DecisionResult
    sys.modules.update({"fakejb": pkg, "fakejb.base": base})
    path = Path(__file__).parents[1] / "bench" / "jevk5_lite.py"
    spec = importlib.util.spec_from_file_location("fakejb.jevk5_lite", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_noul(adapter):
    q = {
        "type": "noul",
        "instructions": "Is a refund permitted?",
        "criteria": {"true": "yes", "false": "no"},
    }
    assert adapter.lite_head(q) == ("Is a refund permitted?", ["true", "false"], ["true", "false"])


def test_choice_uses_descriptions_or_keys(adapter):
    q = {
        "type": "choice",
        "instructions": "What happened?",
        "criteria": {"delivered": "It arrived", "misdelivered": None, "unknown": ""},
    }
    assert adapter.lite_head(q) == (
        "What happened?",
        ["It arrived", "misdelivered", "unknown"],
        ["delivered", "misdelivered", "unknown"],
    )
    listed = {"type": "choice", "instructions": "Team?", "criteria": ["billing", "fraud"]}
    assert adapter.lite_head(listed)[1:] == (["billing", "fraud"], ["billing", "fraud"])


def test_duplicate_labels_get_their_ids(adapter):
    q = {"type": "choice", "instructions": "Pick", "criteria": {"a": "same", "b": "same"}}
    assert adapter.lite_head(q)[1] == ["a: same", "b: same"]


def test_score_levels_in_order(adapter):
    q = {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]}
    assert adapter.lite_head(q) == ("How urgent?", ["low", "medium", "high"], ["0", "1", "2"])


def test_state_serialization(adapter):
    assert adapter.state_text("plain text") == "plain text"
    assert adapter.state_text({"ticket": "Olá"}) == '{"ticket": "Olá"}'

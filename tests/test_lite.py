"""JevK5-Lite's runtime on a tiny random model built offline (no download, CPU)."""

from __future__ import annotations

import json
import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")
safetensors_torch = pytest.importorskip("safetensors.torch")

from jevk5.lite import LABEL, TASK, JevK5Lite, Scorer  # noqa: E402

WORDS = "refund order card lost balance billing shipping account urgent please my was charged twice one any".split()


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    path = tmp_path_factory.mktemp("lite")
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", TASK, LABEL]
    vocab = {w: i for i, w in enumerate(specials + WORDS + ["(", ")"])}
    core = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    core.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=core,
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        mask_token="[MASK]",
        additional_special_tokens=[TASK, LABEL],
    )
    tok.save_pretrained(path)
    torch.manual_seed(0)
    config = transformers.BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
    )
    transformers.BertModel(config).save_pretrained(path)
    safetensors_torch.save_file(Scorer(32).net.state_dict(), path / "scorer.safetensors")
    (path / "lite_config.json").write_text(
        json.dumps({"max_len": 64, "temperature_single": 1.5, "temperature_multi": 0.8})
    )
    return path


@pytest.fixture(scope="module")
def lite(folder):
    return JevK5Lite.from_pretrained(str(folder), threads=2)


TASKS = {
    "intent": ["refund", "balance", "card lost"],
    "areas": {"labels": ["billing", "shipping", "account"], "multi_label": True},
}


def test_every_task_answered(lite):
    out = lite.classify("my card was charged twice please refund one", TASKS)
    assert list(out) == ["intent", "areas"]
    single, multi = out["intent"], out["areas"]
    assert list(single["probabilities"]) == TASKS["intent"]
    assert math.isclose(sum(single["probabilities"].values()), 1.0, rel_tol=1e-5)
    assert len(single["labels"]) == 1 and single["labels"][0] == max(
        single["probabilities"], key=single["probabilities"].get
    )
    assert all(0.0 <= p <= 1.0 for p in multi["probabilities"].values())
    assert multi["labels"] == [lab for lab, p in multi["probabilities"].items() if p >= 0.5]


def test_one_encoder_pass_for_all_heads(lite):
    calls = []
    hook = lite.encoder.register_forward_hook(lambda *a: calls.append(1))
    try:
        lite.classify("order balance", {**TASKS, "urgent": ["urgent", "please"]})
    finally:
        hook.remove()
    assert len(calls) == 1


def test_sequence_layout(lite):
    heads = [("intent", ["refund", "card lost"], False, None), ("areas", ["billing"], True, None)]
    ids, label_pos, spans, (s, e) = lite.encode("my order", heads)
    assert ids[0] == lite.tok.cls_token_id and ids[-1] == lite.tok.sep_token_id
    assert [ids[i] for i in label_pos] == [lite.label_id] * 3
    assert spans == [(0, 2), (2, 3)]
    assert lite.tok.convert_ids_to_tokens(ids[s:e]) == ["my", "order"]
    assert ids[s - 1] == lite.tok.sep_token_id


def test_label_schema_cannot_exceed_model_length(lite):
    with pytest.raises(ValueError, match="label schema is too long"):
        lite.classify("refund", {"intent": [f"label{i}" for i in range(50)]})


def test_probabilities_match_a_manual_readout(lite):
    heads = [("intent", TASKS["intent"], False, None)]
    ids, label_pos, _, (s, e) = lite.encode("refund please", heads)
    with torch.no_grad():
        h = lite.encoder(input_ids=torch.tensor([ids])).last_hidden_state[0]
        t = h[s:e].mean(0)
        z = torch.stack(
            [lite.scorer.net(torch.cat([h[i], t, h[i] * t])).squeeze() for i in label_pos]
        )
    want = torch.softmax(z / lite.temperature_single, -1).tolist()
    got = list(
        lite.classify("refund please", {"intent": TASKS["intent"]})["intent"][
            "probabilities"
        ].values()
    )
    assert all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-7) for a, b in zip(got, want))


def test_threshold_per_task(lite):
    probs = lite.classify("billing", TASKS)["areas"]["probabilities"]
    low = lite.classify("billing", {"areas": {**TASKS["areas"], "threshold": 0.0}})["areas"]
    high = lite.classify("billing", {"areas": {**TASKS["areas"], "threshold": 1.01}})["areas"]
    assert low["labels"] == list(probs) and high["labels"] == []


@pytest.mark.parametrize(
    "tasks", [{}, {"t": ["only"]}, {"t": ["a", "a"]}, {"t": "abc"}, {"t": {"labels": [1, 2]}}]
)
def test_bad_tasks_are_refused(lite, tasks):
    with pytest.raises(ValueError):
        lite.classify("text", tasks)


def test_lazy_export():
    import jevk5

    assert jevk5.JevK5Lite is JevK5Lite

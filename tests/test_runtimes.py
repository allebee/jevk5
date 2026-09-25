"""The CUDA runtime and the llama.cpp client read the same passes and combine them the same way.

Neither a GPU nor a model is needed: both front ends are given one fake model, a fixed logit per
option text, and must return the same distribution from the same sequence of prompts.
"""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from jevk5.gguf import JevK5GGUF
from jevk5.prompt import CHAT_TEMPLATE, LETTERS, messages, prompt_text

np = pytest.importorskip("numpy")


def logit(description: str) -> float:
    return int(hashlib.sha1(description.encode()).hexdigest()[:6], 16) / 2**24 * 12 - 6


def descriptions(prompt: str) -> list[str]:
    user = prompt.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    return [o["description"] for o in json.loads(user)["options"]]


class FakeGGUF(JevK5GGUF):
    def __init__(self, **kwargs):
        kwargs.setdefault("temperature", 1.532)
        super().__init__(**kwargs)
        self.prompts = []

    def _logprobs(self, prompt):
        self.prompts.append(prompt)
        z = [logit(d) for d in descriptions(prompt)]
        top = max(z)
        log_total = top + math.log(sum(math.exp(v - top) for v in z))
        return {LETTERS[i]: v - log_total for i, v in enumerate(z)}, len(prompt), 0.001


def fake_cuda(method="knockout", knockout_temperature=None):
    torch = pytest.importorskip("torch")  # noqa: F841 - the runtime module imports it
    from jevk5.runtime import JevK5

    model = JevK5.__new__(JevK5)
    model.temperature, model.method, model.prompts = 1.532, method, []
    model.knockout_temperature = knockout_temperature

    def encode(state, criterion, options):
        system, user = messages(state, criterion, options)
        prompt = CHAT_TEMPLATE.format(system=system["content"], user=user["content"])
        model.prompts.append(prompt)
        return list(range(len(prompt)))

    def letter_logits(ids, count):
        return np.array([logit(d) for d in descriptions(model.prompts[-1])], dtype=np.float32)

    model.encode, model.letter_logits = encode, letter_logits
    return model


def question(n: int) -> dict:
    return {
        "type": "choice",
        "instructions": "Classify the intent of this user request:\nplay some jazz",
        "criteria": {f"option_{i}": f"intent number {i}" for i in range(n)},
    }


@pytest.mark.parametrize("method", ["knockout", "tree"])
@pytest.mark.parametrize("n", [2, 16, 17, 77, 151])
def test_cuda_and_gguf_agree(method, n):
    cuda, gguf = fake_cuda(method), FakeGGUF(method=method)
    p_cuda, _ = cuda.probabilities({}, question(n))
    p_gguf, _ = gguf.probabilities({}, question(n))
    assert cuda.prompts == gguf.prompts
    assert list(p_cuda) == list(p_gguf) == [f"option_{i}" for i in range(n)]
    for key in p_cuda:
        assert math.isclose(p_cuda[key], p_gguf[key], rel_tol=1e-5, abs_tol=1e-9)
    assert math.isclose(sum(p_cuda.values()), 1.0, abs_tol=1e-6)
    assert math.isclose(sum(p_gguf.values()), 1.0, abs_tol=1e-9)


def test_gguf_prompt_matches_prompt_text():
    gguf = FakeGGUF()
    gguf.probabilities({"note": "x"}, question(3))
    texts = [f"option_{i}: intent number {i}" for i in range(3)]
    assert gguf.prompts == [prompt_text({"note": "x"}, question(3)["instructions"], texts)]


@pytest.mark.parametrize("n,passes", [(3, 1), (77, 6)])
def test_tokens_add_up_over_passes(n, passes):
    for model in (FakeGGUF(), fake_cuda()):
        _, tokens = model.probabilities({}, question(n))
        assert len(model.prompts) == passes
        assert tokens == sum(len(p) for p in model.prompts)
        assert model.last_pass_tokens == max(len(p) for p in model.prompts)


def test_unknown_method_is_refused_up_front():
    with pytest.raises(ValueError):
        JevK5GGUF(method="vote")


# knockout_temperature: a per-model second temperature over more than 16 options (jevk5 0.3.0).


def write_config(tmp_path, **values):
    path = tmp_path / "jevk5_config.json"
    path.write_text(json.dumps(values))
    return path


def test_gguf_reads_knockout_temperature_from_config(tmp_path):
    path = write_config(tmp_path, temperature=1.089, knockout_temperature=1.0)
    model = JevK5GGUF(config=str(path))
    assert model.temperature == 1.089
    assert model.knockout_temperature == 1.0


def test_gguf_argument_overrides_config(tmp_path):
    path = write_config(tmp_path, temperature=1.089, knockout_temperature=1.0)
    model = JevK5GGUF(config=str(path), knockout_temperature=0.9)
    assert model.knockout_temperature == 0.9


def test_config_without_the_key_keeps_the_default(tmp_path):
    path = write_config(tmp_path, temperature=1.532)
    assert JevK5GGUF(config=str(path)).knockout_temperature is None  # prompt.TEMPERATURES applies
    assert JevK5GGUF().knockout_temperature is None
    assert JevK5GGUF().temperature == 1.532  # no config: v0.2's temperature, as before


def test_cuda_config_reader(tmp_path):
    pytest.importorskip("torch")
    from jevk5.runtime import _load_config, _load_temperature

    write_config(tmp_path, temperature=1.367, knockout_temperature=1.05)
    assert _load_config(str(tmp_path)) == {"temperature": 1.367, "knockout_temperature": 1.05}
    assert _load_temperature(str(tmp_path)) == 1.367


@pytest.mark.parametrize("n", [3, 16, 17, 77, 151])
def test_knockout_temperature_touches_only_wide_questions(n):
    default, explicit, custom = (
        FakeGGUF(),
        FakeGGUF(knockout_temperature=0.77),
        FakeGGUF(knockout_temperature=1.0),
    )
    p_default, _ = default.probabilities({}, question(n))
    p_explicit, _ = explicit.probabilities({}, question(n))
    p_custom, _ = custom.probabilities({}, question(n))
    assert p_explicit == p_default  # 0.77 is the default, so passing it changes nothing
    if n <= 16:
        assert p_custom == p_default  # one pass: no second temperature
    else:
        assert p_custom != p_default
        top = max(p_default, key=p_default.get)
        assert p_custom[top] < p_default[top]  # 1.0 sharpens less than 0.77


@pytest.mark.parametrize("n", [17, 77, 151])
def test_cuda_and_gguf_agree_with_a_knockout_temperature(n):
    cuda, gguf = fake_cuda(knockout_temperature=1.0), FakeGGUF(knockout_temperature=1.0)
    p_cuda, _ = cuda.probabilities({}, question(n))
    p_gguf, _ = gguf.probabilities({}, question(n))
    for key in p_cuda:
        assert math.isclose(p_cuda[key], p_gguf[key], rel_tol=1e-5, abs_tol=1e-9)


def test_tree_ignores_the_knockout_temperature():
    base, set_ = FakeGGUF(method="tree"), FakeGGUF(method="tree", knockout_temperature=0.5)
    assert base.probabilities({}, question(77))[0] == set_.probabilities({}, question(77))[0]

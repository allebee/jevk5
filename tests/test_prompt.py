"""The many-option readout in jevk5.prompt, checked against readers with known answers."""

from __future__ import annotations

import math
import random

import pytest

from jevk5.prompt import LETTERS, METHODS, TEMPERATURES, decision_options, groups, spread

GROUP = "One of: "


def luce_reader(strength: dict[str, float], calls: list[list[str]] | None = None):
    """A reader that obeys Luce's choice axiom: P(option) = strength / sum over the pass. A group
    description in a tree's first pass is as strong as its members together."""

    def weight(text: str) -> float:
        if text.startswith(GROUP):
            return sum(strength[t] for t in text[len(GROUP) :].split("; "))
        return strength[text]

    def read(texts: list[str]) -> list[float]:
        assert 2 <= len(texts) <= len(LETTERS)
        if calls is not None:
            calls.append(list(texts))
        w = [weight(t) for t in texts]
        return [x / sum(w) for x in w]

    return read


def options(n: int) -> list[str]:
    return [f"option_{i}: intent {i}" for i in range(n)]


@pytest.mark.parametrize("n,count", [(17, 2), (77, 5), (151, 10), (255, 16), (32, 2), (33, 3)])
def test_groups_cover_in_order_with_near_equal_sizes(n, count):
    runs = groups(n, count)
    assert [i for run in runs for i in run] == list(range(n))
    assert max(map(len, runs)) - min(map(len, runs)) <= 1


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n", [2, 7, 16])
def test_up_to_16_options_is_one_untouched_pass(method, n):
    calls = []
    texts = options(n)
    probs = [0.5 ** (i + 1) for i in range(n)]

    def read(given):
        calls.append(given)
        return probs

    assert spread(read, texts, method) == probs
    assert calls == [texts]


def strengths(n: int) -> tuple[list[str], dict[str, float]]:
    rng = random.Random(n)
    texts = options(n)
    return texts, {t: math.exp(rng.gauss(0, 3)) for t in texts}


@pytest.mark.parametrize("n", [17, 53, 77, 129, 151, 255, 300, 1000])
def test_tree_recovers_a_luce_model_exactly(n):
    """If every pass is a Luce choice over one set of strengths, and a group is as strong as its
    members, the tree returns the global softmax over all n options, whatever the grouping."""
    texts, strength = strengths(n)
    got = spread(luce_reader(strength), texts, "tree")
    total = sum(strength.values())
    assert TEMPERATURES["tree"] == 1.0
    for t, p in zip(texts, got):
        assert math.isclose(p, strength[t] / total, rel_tol=1e-9, abs_tol=1e-15)


@pytest.mark.parametrize("n", [17, 40, 53, 77, 129, 151, 255])
def test_knockout_keeps_the_final_and_the_groups(n):
    """Unsharpened, finalists stand in the final's proportions, every group's other options in
    their in-group proportions, and each group holds its finalists' share of the final."""
    texts, strength = strengths(n)
    calls = []
    got = dict(zip(texts, spread(luce_reader(strength, calls), texts, "knockout", temperature=1)))
    runs = groups(n, -(-n // len(LETTERS)))
    final = dict(zip(calls[-1], luce_reader(strength)(calls[-1])))
    assert len(final) == min(len(LETTERS), n) and math.isclose(sum(got.values()), 1.0)
    first = calls[-1][0]
    for t in final:
        assert math.isclose(got[t] / got[first], final[t] / final[first], rel_tol=1e-9)
    for run in runs:
        members = [texts[i] for i in run]
        others = [t for t in members if t not in final]
        for t in others:
            assert math.isclose(got[t] / got[others[0]], strength[t] / strength[others[0]])
        mass = sum(final.get(t, 0.0) for t in members)
        inner = sum(strength[t] for t in members)
        rest = sum(strength[t] for t in others) / inner
        assert math.isclose(sum(got[t] for t in others), mass * rest, rel_tol=1e-9, abs_tol=1e-15)


@pytest.mark.parametrize("n", [17, 77, 151])
def test_knockout_temperature_only_sharpens(n):
    texts, strength = strengths(n)
    plain = spread(luce_reader(strength), texts, "knockout", temperature=1)
    sharp = spread(luce_reader(strength), texts, "knockout")
    tau = TEMPERATURES["knockout"]
    total = sum(q ** (1 / tau) for q in plain)
    for a, b in zip(plain, sharp):
        assert math.isclose(b, a ** (1 / tau) / total, rel_tol=1e-9, abs_tol=1e-300)
    assert max(range(n), key=plain.__getitem__) == max(range(n), key=sharp.__getitem__)
    assert math.isclose(sum(sharp), 1.0, abs_tol=1e-12)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n", [300, 1000])
def test_recursion_beyond_256(method, n):
    texts, strength = strengths(n)
    calls = []
    got = spread(luce_reader(strength, calls), texts, method)
    assert len(got) == n and math.isclose(sum(got), 1.0, abs_tol=1e-12)
    assert all(2 <= len(c) <= len(LETTERS) for c in calls)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n,passes", [(17, 3), (77, 6), (129, 10), (151, 11), (255, 17)])
def test_passes_and_every_option_scored(method, n, passes):
    calls = []
    texts = options(n)
    spread(luce_reader({t: 1.0 + i for i, t in enumerate(texts)}, calls), texts, method)
    assert len(calls) == passes
    assert all(len(c) <= len(LETTERS) for c in calls)
    seen = {t for c in calls for t in c if not t.startswith(GROUP)}
    assert seen == set(texts)


def test_knockout_final_takes_the_top_options_of_each_group():
    texts = options(77)  # 5 groups of 16, 16, 15, 15, 15 -> 3 finalists each, 1 free place
    strength = {t: 1.0 for t in texts}
    for i in (2, 5, 9, 11, 20, 40, 76):
        strength[texts[i]] = 10.0 + i
    calls = []
    spread(luce_reader(strength, calls), texts, "knockout")
    final = calls[-1]
    assert len(final) == 16
    assert final == sorted(final, key=texts.index)  # in the given order
    # group 1 has four strong options: three by right, the fourth in the free place
    assert {texts[i] for i in (2, 5, 9, 11, 20, 40, 76)} <= set(final)


def test_free_places_fill_the_final_to_16():
    texts = options(151)  # 10 groups -> one finalist each, then 6 free places
    strength = {t: 1.0 for t in texts}
    for i in (0, 1, 2, 3, 4, 5, 6, 150):
        strength[texts[i]] = 50.0 - i
    calls = []
    spread(luce_reader(strength, calls), texts, "knockout")
    final = calls[-1]
    assert len(final) == 16
    assert {texts[i] for i in (0, 1, 2, 3, 4, 5, 6, 150)} <= set(final)


def test_ties_go_to_the_earlier_option():
    texts = options(40)  # 3 groups of 14, 13, 13 -> 5 finalists each, 1 free place
    calls = []
    spread(luce_reader({t: 1.0 for t in texts}, calls), texts, "knockout")
    # all tied within a group; the free place goes to the likeliest (1/13 beats 1/14), earliest
    assert calls[-1] == texts[0:5] + texts[14:20] + texts[27:32]


def test_unknown_method():
    with pytest.raises(ValueError):
        spread(luce_reader({t: 1.0 for t in options(20)}), options(20), "vote")


def test_index_shaped_question_maps_every_option():
    question = {
        "type": "choice",
        "instructions": "Classify the banking intent of this user request:\nwhere is my card?",
        "criteria": {f"option_{i}": f"label_{i}" for i in range(77)},
    }
    opts = decision_options(question)
    assert len(opts) == 77 and opts[76] == ("option_76", "option_76: label_76")

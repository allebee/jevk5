"""Group sequences into length-sorted, token-budgeted batches with a small set of shapes."""

from __future__ import annotations

from collections.abc import Sequence


def round_up(length: int, multiple: int, cap: int | None = None) -> int:
    padded = -(-length // multiple) * multiple
    return padded if cap is None else min(padded, max(cap, length))


def plan_batches(
    lengths: Sequence[int],
    *,
    token_budget: int,
    max_batch: int,
    bucket: int = 32,
    max_length: int | None = None,
) -> list[list[int]]:
    """Return batches of item indices.

    Items are sorted by length, so each batch pads to its last item's bucketed length. A batch
    grows while `rows * padded_length <= token_budget` and `rows <= max_batch`; an item longer
    than the budget gets a batch of its own. Rounding lengths up to `bucket` keeps the number of
    distinct batch shapes small, which lets MLX reuse buffers instead of caching new sizes.
    """
    if token_budget < 1 or max_batch < 1 or bucket < 1:
        raise ValueError("token_budget, max_batch and bucket must be positive")
    order = sorted(range(len(lengths)), key=lengths.__getitem__)
    batches: list[list[int]] = []
    current: list[int] = []
    for index in order:
        padded = round_up(lengths[index], bucket, max_length)
        if current and (len(current) == max_batch or (len(current) + 1) * padded > token_budget):
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    return batches

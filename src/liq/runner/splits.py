"""Rolling/blocked cross-validation split utilities."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    train: slice
    valid: slice


def rolling_splits(n: int, train_size: int, valid_size: int, step: int) -> Iterator[Split]:
    """Generate rolling splits over an index of length n."""
    if n <= 0 or train_size <= 0 or valid_size <= 0 or step <= 0:
        raise ValueError("n, train_size, valid_size, step must be positive")
    start = 0
    while True:
        train_start = start
        train_end = min(train_start + train_size, n)
        valid_start = train_end
        valid_end = min(valid_start + valid_size, n)
        if valid_start >= n:
            break
        yield Split(train=slice(train_start, train_end), valid=slice(valid_start, valid_end))
        if valid_end >= n:
            break
        start += step


def blocked_splits(n: int, k: int) -> Iterable[Split]:
    """Generate k blocked splits (no overlap) over n observations."""
    if n <= 0 or k <= 0:
        raise ValueError("n and k must be positive")
    block = n // k
    if block == 0:
        raise ValueError("k too large for n")
    for i in range(k):
        start = i * block
        end = start + block if i < k - 1 else n
        yield Split(train=slice(0, start), valid=slice(start, end))

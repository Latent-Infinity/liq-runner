import pytest

from liq.runner.splits import blocked_splits, rolling_splits


def test_rolling_splits_advances() -> None:
    splits = list(rolling_splits(n=10, train_size=4, valid_size=2, step=2))
    assert len(splits) >= 1
    assert splits[0].train == slice(0, 4)
    assert splits[0].valid == slice(4, 6)


def test_blocked_splits_partition() -> None:
    splits = list(blocked_splits(n=10, k=3))
    assert splits[0].valid == slice(0, 3)
    assert splits[-1].valid.stop == 10


def test_split_errors() -> None:
    with pytest.raises(ValueError):
        list(rolling_splits(n=0, train_size=1, valid_size=1, step=1))
    with pytest.raises(ValueError):
        list(rolling_splits(n=10, train_size=0, valid_size=1, step=1))
    with pytest.raises(ValueError):
        list(blocked_splits(n=0, k=1))
    with pytest.raises(ValueError):
        list(blocked_splits(n=2, k=5))

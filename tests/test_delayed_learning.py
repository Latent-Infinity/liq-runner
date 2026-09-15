"""DEV_SMOKE_ONLY clock tests; scalar inputs are not financial evidence."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone

import pytest

from liq.runner.delayed_learning import DelayedLearner, PendingOutcome


class RecordingEstimator:
    """Protocol fixture accumulating observed scalar values in arrival order."""

    def __init__(self) -> None:
        self.values: list[float] = []

    def predict(self) -> float:
        return sum(self.values)

    def learn(self, value: float) -> None:
        self.values.append(value)

    @property
    def state_hash(self) -> str:
        return repr(self.values)

    @property
    def count(self) -> int:
        return len(self.values)


@pytest.fixture
def now() -> datetime:
    return datetime(2024, 1, 2, 12, tzinfo=UTC)


def test_estimator_receives_only_matured_outcomes(now: datetime) -> None:
    estimator = RecordingEstimator()
    learner = DelayedLearner(estimator)
    initial = learner.predict(now)
    learner.schedule(PendingOutcome("event", now + timedelta(hours=2), 3.0))
    earlier = learner.predict(now + timedelta(hours=1))
    assert earlier.state_hash == initial.state_hash
    assert earlier.update_count == 0
    assert estimator.values == []
    mature = learner.predict(now + timedelta(hours=2))
    assert mature.value == 3.0
    assert mature.update_count == 1
    assert learner.pending_count == 0
    assert learner.learned_count == 1
    assert learner.advance(now + timedelta(hours=3)) == ()
    assert initial.value == 0.0


def test_future_label_mutation_preserves_prediction_prefix(now: datetime) -> None:
    prefixes = []
    suffixes = []
    for future_value in (7.0, -99.0):
        learner = DelayedLearner(RecordingEstimator())
        initial = learner.predict(now)
        learner.schedule(PendingOutcome("near", now + timedelta(hours=1), 2.0))
        learner.schedule(PendingOutcome("far", now + timedelta(hours=3), future_value))
        prefixes.append((initial, learner.predict(now + timedelta(hours=2))))
        suffixes.append(learner.predict(now + timedelta(hours=3)))
    assert prefixes[0] == prefixes[1]
    assert suffixes[0].state_hash != suffixes[1].state_hash


def test_updates_sort_by_availability_then_event_id(now: datetime) -> None:
    estimator = RecordingEstimator()
    learner = DelayedLearner(estimator)
    learner.advance(now)
    learner.schedule(PendingOutcome("z", now + timedelta(hours=1), 3.0))
    learner.schedule(PendingOutcome("b", now, 2.0))
    learner.schedule(PendingOutcome("a", now, 1.0))
    assert learner.advance(now + timedelta(hours=2)) == ("a", "b", "z")
    assert estimator.values == [1.0, 2.0, 3.0]
    assert learner.updates[0].available_at == now
    assert learner.updates[0].applied_at == now + timedelta(hours=2)
    assert learner.updates[0].state_hash == "[1.0]"
    assert learner.updates[-1].update_count == 3


def test_same_clock_schedule_waits_for_next_advance(now: datetime) -> None:
    learner = DelayedLearner(RecordingEstimator())
    initial = learner.predict(now)
    learner.schedule(PendingOutcome("event", now, 1.0))
    assert learner.learned_count == 0
    assert learner.predict(now).value == 1.0
    assert initial.value == 0.0


@pytest.mark.parametrize("apply_first", [False, True])
def test_duplicate_event_ids_are_rejected(now: datetime, apply_first: bool) -> None:
    learner = DelayedLearner(RecordingEstimator())
    learner.advance(now)
    outcome = PendingOutcome("event", now, 1.0)
    learner.schedule(outcome)
    if apply_first:
        learner.advance(now)
    with pytest.raises(ValueError):
        learner.schedule(outcome)


def test_regressing_clock_is_rejected(now: datetime) -> None:
    learner = DelayedLearner(RecordingEstimator())
    learner.advance(now)
    with pytest.raises(ValueError):
        learner.predict(now - timedelta(seconds=1))


def test_scheduling_requires_initialized_clock(now: datetime) -> None:
    learner = DelayedLearner(RecordingEstimator())
    with pytest.raises(ValueError):
        learner.schedule(PendingOutcome("event", now, 1.0))


def test_outdated_outcomes_are_rejected(now: datetime) -> None:
    learner = DelayedLearner(RecordingEstimator())
    learner.advance(now)
    with pytest.raises(ValueError):
        learner.schedule(PendingOutcome("event", now - timedelta(seconds=1), 1.0))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_outcomes_are_rejected(now: datetime, value: float) -> None:
    with pytest.raises(ValueError):
        PendingOutcome("event", now, value)


def test_empty_event_id_is_rejected(now: datetime) -> None:
    with pytest.raises(ValueError):
        PendingOutcome(" ", now, 1.0)


def test_naive_timestamps_are_rejected(now: datetime) -> None:
    with pytest.raises(ValueError):
        PendingOutcome("event", now.replace(tzinfo=None), 1.0)
    with pytest.raises(ValueError):
        DelayedLearner(RecordingEstimator()).predict(now.replace(tzinfo=None))


def test_timestamps_normalize_to_utc(now: datetime) -> None:
    local = now.astimezone(timezone(timedelta(hours=-7)))
    outcome = PendingOutcome("event", local, 1.0)
    learner = DelayedLearner(RecordingEstimator())
    prediction = learner.predict(local)
    assert prediction.timestamp.tzinfo is UTC
    assert outcome.available_at.tzinfo is UTC
    assert outcome.available_at == now


def test_prediction_and_update_evidence_is_immutable(now: datetime) -> None:
    learner = DelayedLearner(RecordingEstimator())
    prediction = learner.predict(now)
    outcome = PendingOutcome("event", now, 1.0)
    learner.schedule(outcome)
    learner.advance(now)
    for record in (prediction, outcome, learner.updates[0]):
        with pytest.raises(FrozenInstanceError):
            record.__setattr__("value", 9.0)


def test_nonfinite_estimator_prediction_is_rejected(now: datetime) -> None:
    estimator = RecordingEstimator()
    estimator.values.append(float("nan"))
    with pytest.raises(ValueError):
        DelayedLearner(estimator).predict(now)


def test_failed_update_stays_queued_without_repeating_successes(now: datetime) -> None:
    class RejectingEstimator(RecordingEstimator):
        def learn(self, value: float) -> None:
            if value == 2.0:
                raise ArithmeticError(value)
            super().learn(value)

    estimator = RejectingEstimator()
    learner = DelayedLearner(estimator)
    learner.advance(now)
    learner.schedule(PendingOutcome("a", now, 1.0))
    learner.schedule(PendingOutcome("b", now, 2.0))
    for _ in range(2):
        with pytest.raises(ArithmeticError):
            learner.advance(now)
        assert learner.pending_count == 1
        assert learner.learned_count == 1
        assert estimator.values == [1.0]

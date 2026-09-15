"""Chronological outcome delivery, separated from estimator state."""

from dataclasses import dataclass
from datetime import UTC, datetime
from heapq import heappop, heappush
from math import isfinite
from typing import Protocol


class DelayedLearningError(ValueError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _utc(value: datetime) -> datetime:
    if value.utcoffset() is None:
        raise DelayedLearningError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


class OnlineEstimator(Protocol):
    """Exclusive learner-owned estimator; learn must be atomic on exceptions."""

    def predict(self) -> float: ...

    def learn(self, value: float) -> None: ...

    @property
    def state_hash(self) -> str: ...

    @property
    def count(self) -> int: ...


@dataclass(frozen=True, slots=True)
class PendingOutcome:
    event_id: str
    available_at: datetime
    value: float

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise DelayedLearningError("event_id must be nonempty")
        if not isfinite(self.value):
            raise DelayedLearningError("outcome must be finite")
        object.__setattr__(self, "available_at", _utc(self.available_at))


@dataclass(frozen=True, slots=True)
class Prediction:
    timestamp: datetime
    value: float
    state_hash: str
    update_count: int


@dataclass(frozen=True, slots=True)
class AppliedUpdate:
    event_id: str
    available_at: datetime
    applied_at: datetime
    value: float
    state_hash: str
    update_count: int


class DelayedLearner:
    """Mutable clock and queue delivering matured outcomes before predictions.

    Initialize the clock with predict/advance, then schedule an outcome when its
    event decision is made. Future labels remain solely in the coordinator.
    Availability before the current clock is rejected; equal-clock scheduling
    is allowed and takes effect on the next advance/predict. Each queued batch
    is sorted by (available_at, event_id), including equal-time ties. Callers
    must schedule a complete batch before requesting its next prediction.

    Estimator ownership is exclusive. It must expose side-effect-free evidence
    properties and reject a failed learn without mutating its state. A failed
    advance retains that outcome for retry; earlier successful updates remain.
    """

    def __init__(self, estimator: OnlineEstimator) -> None:
        self._estimator = estimator
        self._clock: datetime | None = None
        self._pending: list[tuple[datetime, str, PendingOutcome]] = []
        self._seen: set[str] = set()
        self._updates: list[AppliedUpdate] = []

    def schedule(self, outcome: PendingOutcome) -> None:
        if self._clock is None:
            raise DelayedLearningError("initialize the decision clock before scheduling")
        if outcome.available_at < self._clock:
            raise DelayedLearningError("outcome availability precedes the current clock")
        if outcome.event_id in self._seen:
            raise DelayedLearningError(f"duplicate event_id: {outcome.event_id}")
        heappush(self._pending, (outcome.available_at, outcome.event_id, outcome))
        self._seen.add(outcome.event_id)

    def advance(self, now: datetime) -> tuple[str, ...]:
        now = _utc(now)
        if self._clock is not None and now < self._clock:
            raise DelayedLearningError("clock must not regress")
        self._clock = now
        applied: list[str] = []
        while self._pending and self._pending[0][0] <= now:
            outcome = self._pending[0][2]
            self._estimator.learn(outcome.value)
            self._updates.append(
                AppliedUpdate(
                    event_id=outcome.event_id,
                    available_at=outcome.available_at,
                    applied_at=now,
                    value=outcome.value,
                    state_hash=self._estimator.state_hash,
                    update_count=self._estimator.count,
                )
            )
            heappop(self._pending)
            applied.append(outcome.event_id)
        return tuple(applied)

    def predict(self, now: datetime) -> Prediction:
        self.advance(now)
        value = self._estimator.predict()
        if not isfinite(value):
            raise DelayedLearningError("estimator prediction must be finite")
        return Prediction(_utc(now), value, self._estimator.state_hash, self._estimator.count)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def learned_count(self) -> int:
        return len(self._updates)

    @property
    def updates(self) -> tuple[AppliedUpdate, ...]:
        return tuple(self._updates)

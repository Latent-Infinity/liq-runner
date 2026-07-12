"""Drift handling for runner to adjust sizing or trigger retrain."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class DriftAction:
    retrain: bool
    sizing_multiplier: float
    message: str | None = None


class DriftManager:
    """Considers drift results and outputs actions."""

    def __init__(
        self,
        retrain_threshold: float = 0.5,
        reduce_threshold: float = 0.2,
        min_multiplier: float = 0.5,
    ):
        self.retrain_threshold = retrain_threshold
        self.reduce_threshold = reduce_threshold
        self.min_multiplier = min_multiplier

    def evaluate(self, statistics: Iterable[float]) -> DriftAction:
        max_stat = max(statistics) if statistics else 0.0
        if max_stat >= self.retrain_threshold:
            return DriftAction(
                retrain=True,
                sizing_multiplier=self.min_multiplier,
                message="Drift exceeds retrain threshold",
            )
        if max_stat >= self.reduce_threshold:
            return DriftAction(
                retrain=False,
                sizing_multiplier=self.min_multiplier,
                message="Drift high: reduce sizing",
            )
        return DriftAction(retrain=False, sizing_multiplier=1.0, message=None)

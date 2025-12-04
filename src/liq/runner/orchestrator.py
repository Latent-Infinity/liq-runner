"""Orchestration helpers to apply pipelines and drift responses in runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Any, Callable

from liq.runner.pipeline_manager import PipelineManager, PipelineState
from liq.runner.drift_manager import DriftManager, DriftAction


@dataclass
class RunContext:
    pipeline_state: PipelineState
    model_type: str


class Orchestrator:
    """Applies persisted pipelines and evaluates drift to adjust sizing."""

    def __init__(
        self,
        pipeline_factory: Callable[[dict], Any],
        drift_manager: DriftManager | None = None,
    ) -> None:
        self.pipeline_manager = PipelineManager(pipeline_factory)
        self.drift_manager = drift_manager or DriftManager()

    def apply_pipeline(self, series: Iterable[float], ctx: RunContext) -> list[float]:
        return self.pipeline_manager.apply(series, ctx.pipeline_state)

    def evaluate_drift(self, statistics: Iterable[float]) -> DriftAction:
        return self.drift_manager.evaluate(statistics)

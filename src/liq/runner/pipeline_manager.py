"""Manage persisted feature pipelines to avoid refit and enforce train-only parameters."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass
class PipelineState:
    """Serialized pipeline state (as produced by liq-features FeaturePipeline.to_dict)."""

    data: dict[str, Any]


class PipelineManager:
    """Applies a persisted pipeline to new data without refitting."""

    def __init__(self, pipeline_factory) -> None:
        """pipeline_factory: callable that builds a pipeline from dict (e.g., FeaturePipeline.from_dict)."""
        self._factory = pipeline_factory

    def apply(self, series: Iterable[float], state: PipelineState) -> list[float]:
        pipeline = self._factory(state.data)
        return pipeline.transform(series)

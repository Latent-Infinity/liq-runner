"""Pipeline specification for orchestration runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class PipelineStageSpec:
    """Single pipeline stage specification."""

    name: str
    description: str = ""
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "enabled": self.enabled,
            "config": dict(self.config),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PipelineStageSpec":
        return cls(
            name=str(payload["name"]),
            description=str(payload.get("description", "")),
            inputs=list(payload.get("inputs", [])),
            outputs=list(payload.get("outputs", [])),
            enabled=bool(payload.get("enabled", True)),
            config=dict(payload.get("config", {})),
        )


@dataclass(frozen=True)
class PipelineSpec:
    """Pipeline specification with ordered stages."""

    name: str
    stages: list[PipelineStageSpec]
    artifact_root: str = "artifacts"

    def validate(self) -> None:
        if not self.name:
            raise ValueError("pipeline name must be set")
        if not self.stages:
            raise ValueError("pipeline must include at least one stage")
        seen: set[str] = set()
        for stage in self.stages:
            if not stage.name:
                raise ValueError("stage name must be set")
            if stage.name in seen:
                raise ValueError(f"duplicate stage name: {stage.name}")
            seen.add(stage.name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "artifact_root": self.artifact_root,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PipelineSpec":
        stages = [PipelineStageSpec.from_dict(item) for item in payload.get("stages", [])]
        return cls(
            name=str(payload["name"]),
            artifact_root=str(payload.get("artifact_root", "artifacts")),
            stages=stages,
        )


@dataclass(frozen=True)
class PipelineRunState:
    """State and hashes for a pipeline run."""

    config_hash: str
    dataset_hash: str | None = None
    model_hash: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_hash": self.config_hash,
            "dataset_hash": self.dataset_hash,
            "model_hash": self.model_hash,
            "created_at": self.created_at.isoformat(),
        }


def compute_config_hash(spec: PipelineSpec) -> str:
    """Hash the pipeline spec for reproducible runs."""
    payload = spec.to_dict()
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dry_run_steps(spec: PipelineSpec) -> list[dict[str, Any]]:
    """Return a lightweight plan of enabled stages and outputs."""
    spec.validate()
    steps = []
    for stage in spec.stages:
        if not stage.enabled:
            continue
        steps.append(
            {
                "name": stage.name,
                "inputs": list(stage.inputs),
                "outputs": list(stage.outputs),
            }
        )
    return steps


def default_pipeline() -> PipelineSpec:
    """Default stage order for SSL + finetune models (transformers, JEPA, masked TS)."""
    stages = [
        PipelineStageSpec(
            name="ingest",
            description="Ingest and normalize raw data",
            outputs=["raw_candles"],
        ),
        PipelineStageSpec(
            name="features",
            description="Build feature dataframe",
            inputs=["raw_candles"],
            outputs=["features"],
        ),
        PipelineStageSpec(
            name="datasets",
            description="Windowing + holdout split",
            inputs=["features"],
            outputs=["ssl_windows", "supervised_windows"],
        ),
        PipelineStageSpec(
            name="pretrain",
            description="Self-supervised pretrain",
            inputs=["ssl_windows"],
            outputs=["backbone_checkpoint"],
        ),
        PipelineStageSpec(
            name="finetune",
            description="Supervised finetune heads",
            inputs=["supervised_windows", "backbone_checkpoint"],
            outputs=["model_checkpoint"],
        ),
        PipelineStageSpec(
            name="eval_dev",
            description="Evaluate on dev holdout",
            inputs=["model_checkpoint"],
            outputs=["dev_metrics"],
        ),
        PipelineStageSpec(
            name="eval_lockbox",
            description="Evaluate on lockbox holdout",
            inputs=["model_checkpoint"],
            outputs=["lockbox_metrics"],
        ),
        PipelineStageSpec(
            name="publish",
            description="Publish model and metadata",
            inputs=["model_checkpoint", "dev_metrics", "lockbox_metrics"],
            outputs=["registry_entry"],
        ),
    ]
    return PipelineSpec(name="ssl_finetune", stages=stages, artifact_root="artifacts")

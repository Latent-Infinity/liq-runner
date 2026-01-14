import pytest

from liq.runner.pipeline_spec import (
    PipelineSpec,
    PipelineStageSpec,
    PipelineRunState,
    compute_config_hash,
    default_pipeline,
    dry_run_steps,
)


def test_pipeline_spec_validation() -> None:
    spec = PipelineSpec(
        name="test",
        stages=[PipelineStageSpec(name="stage")],
    )
    spec.validate()

    with pytest.raises(ValueError):
        PipelineSpec(name="", stages=[PipelineStageSpec(name="s")]).validate()
    with pytest.raises(ValueError):
        PipelineSpec(name="test", stages=[]).validate()
    with pytest.raises(ValueError):
        PipelineSpec(
            name="test",
            stages=[PipelineStageSpec(name="dup"), PipelineStageSpec(name="dup")],
        ).validate()
    with pytest.raises(ValueError):
        PipelineSpec(name="test", stages=[PipelineStageSpec(name="")]).validate()


def test_config_hash_stable() -> None:
    spec = default_pipeline()
    first = compute_config_hash(spec)
    second = compute_config_hash(spec)
    assert first == second


def test_dry_run_steps() -> None:
    spec = PipelineSpec(
        name="test",
        stages=[
            PipelineStageSpec(name="a", outputs=["x"]),
            PipelineStageSpec(name="b", enabled=False),
        ],
    )
    steps = dry_run_steps(spec)
    assert steps == [{"name": "a", "inputs": [], "outputs": ["x"]}]


def test_pipeline_run_state_to_dict() -> None:
    state = PipelineRunState(config_hash="abc", dataset_hash="data", model_hash="model")
    payload = state.to_dict()
    assert payload["config_hash"] == "abc"
    assert payload["dataset_hash"] == "data"
    assert payload["model_hash"] == "model"

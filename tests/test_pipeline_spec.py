import pytest

from liq.runner.pipeline_spec import (
    PipelineRunState,
    PipelineSpec,
    PipelineStageSpec,
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


class TestCostScenarioValidation:
    """Stages naming a cost scenario must resolve it from the cost book."""

    @staticmethod
    def _spec(cost_scenario: object) -> PipelineSpec:
        return PipelineSpec(
            name="pilot",
            stages=[
                PipelineStageSpec(
                    name="backtest",
                    config={"cost_scenario": cost_scenario},
                )
            ],
        )

    def test_known_scenario_validates(self) -> None:
        self._spec("spy_qqq_base_v1").validate()

    def test_unknown_scenario_refuses_to_start(self) -> None:
        import pytest

        from liq.runner.cost_book import UnknownCostScenarioError

        with pytest.raises(UnknownCostScenarioError):
            self._spec("not_a_scenario").validate()

    def test_unnamed_scenario_refuses_to_start(self) -> None:
        import pytest

        from liq.runner.cost_book import UnknownCostScenarioError

        with pytest.raises(UnknownCostScenarioError):
            self._spec("").validate()

    def test_stage_without_cost_scenario_still_validates(self) -> None:
        PipelineSpec(name="pipeline", stages=[PipelineStageSpec(name="stage")]).validate()

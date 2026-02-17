from liq.runner.orchestrator import Orchestrator, RunContext
from liq.runner.pipeline_manager import PipelineState


class DummyPipeline:
    def __init__(self):
        self.fitted = True

    @classmethod
    def from_dict(cls, _data):
        return cls()

    def transform(self, series):
        return [v * 2 for v in series]


def test_orchestrator_pipeline_and_drift():
    orch = Orchestrator(DummyPipeline.from_dict)
    ctx = RunContext(pipeline_state=PipelineState(data={"model_type": "nn"}), model_type="nn")
    out = orch.apply_pipeline([1, 2], ctx)
    assert out == [2, 4]
    drift = orch.evaluate_drift([0.6])
    assert drift.retrain

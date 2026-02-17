class DummyPipeline:
    def __init__(self):
        self.transformed = False
        self.fitted = True

    @classmethod
    def from_dict(cls, _data):
        return cls()

    def transform(self, series):
        self.transformed = True
        return list(series)


def test_pipeline_manager_uses_factory_and_does_not_refit():
    from liq.runner.pipeline_manager import PipelineManager, PipelineState

    mgr = PipelineManager(DummyPipeline.from_dict)
    state = PipelineState(data={"model_type": "nn"})
    out = mgr.apply([1, 2, 3], state)
    assert out == [1, 2, 3]

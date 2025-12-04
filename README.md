# liq-runner
As part of the Latent Infinity Quant (LIQ) ecosystem, `liq-runner` orchestrates high-level experiments like parameter sweeps, walk-forward validation, and results logging.

## Pipeline/Drift utilities

- `PipelineManager` + `Orchestrator`: apply persisted feature pipelines (from liq-features) without refitting.
- `DriftManager`: interpret drift statistics (from liq-features) to reduce sizing or trigger retrain.

Usage sketch:
```python
from liq.runner import Orchestrator, PipelineState
from liq.features.pipeline import FeaturePipeline

orch = Orchestrator(FeaturePipeline.from_dict)
ctx = PipelineState(data=FeaturePipeline(model_type="nn").fit_transform([1,2,3]) or {})  # placeholder state
drift_action = orch.evaluate_drift([0.3])
```

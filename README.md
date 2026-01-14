# liq-runner
As part of the Latent Infinity Quant (LIQ) ecosystem, `liq-runner` orchestrates high-level experiments like parameter sweeps, walk-forward validation, and results logging.

## Rolling strategy runner (work in progress)

The rolling runner coordinates strategy training, calibration/EV thresholding, risk sizing, simulation, and metrics:

- **Inputs**: features/labels, strategy (`liq-signals`), risk config (`liq-risk`), bars (`liq-data`/`liq-store`), sim config (`liq-sim`).
- **Processing**:
  1) Split data (rolling/blocked).
  2) Fit strategy and generate `SignalOutput`.
  3) Calibrate scores + select EV threshold (`liq-sim.calibration`).
  4) Size orders via `liq-risk` policy.
  5) Simulate via `liq-sim` (with funding/slippage/risk caps).
  6) Summarize metrics via `liq-metrics`.
- **Outputs**: fold thresholds/params, sim diagnostics (funding/slippage/rejections), metrics.

Usage sketch:
```python
import polars as pl
from liq.runner.runner import run_rolling
from liq.risk.config import RiskConfig

# strategy implements fit/predict -> SignalOutput
# bars_provider / portfolio_provider supply bars and initial PortfolioState per fold

results = run_rolling(
    features=pl.DataFrame({"f": [1,2,3]}),
    labels=pl.Series([0,1,0]),
    strategy=my_strategy,
    risk_engine=my_risk_engine,
    simulator_factory=my_sim_factory,
    bars_provider=my_bars_provider,
    portfolio_provider=my_portfolio_provider,
    risk_config=RiskConfig(),
    train_size=2,
    valid_size=1,
    step=1,
)
for fold in results:
    print(fold.threshold, fold.metrics)
```

## Pipeline CLI (spec validation + dry-run)

Use the CLI to validate pipeline specs and preview stage order without running any model code.

```bash
python -m liq.runner.cli --spec examples/pipeline.json --dry-run --print-hash
```

See `examples/rolling_runner_example.py` for a runnable end-to-end dummy wiring.

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

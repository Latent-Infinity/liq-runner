# liq-runner
As part of the Latent Infinity Quant (LIQ) ecosystem, `liq-runner` orchestrates high-level experiments like parameter sweeps, walk-forward validation, and results logging.

## Rolling strategy runner (work in progress)

The rolling runner coordinates strategy training, calibration/EV thresholding, risk sizing, simulation, and metrics:

- **Inputs**: features/labels, strategy (`liq-signals`), risk config (`liq-risk`), bars (`liq-data`/`liq-store`), sim config (`liq-sim`).
- **Processing**:
  1) Split data (rolling/blocked).
  2) Fit strategy and generate `SignalOutput`.
  3) Calibrate scores + select EV threshold (`liq-sim.calibration`).
  4) Filter actionable signals with `threshold_cfg.max_signals_per_symbol` (default: `1` to preserve strongest-signal-per-symbol behavior; `null` keeps all above-threshold signals).
  5) Size orders via `liq-risk` policy.
  6) Simulate via `liq-sim` (with funding/slippage/risk caps).
  7) Summarize metrics via `liq-metrics`.
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
uv run python -m liq.runner.cli --spec examples/pipeline.json --dry-run --print-hash
```

See `examples/rolling_runner_example.py` for a runnable end-to-end dummy wiring.

## Cost book and run provenance

Execution costs are resolved from the central cost book by scenario name —
never inlined in pilot code:

```python
from liq.runner import INTRADAY_CAMPAIGN_COST_BOOK_V1, build_run_provenance

scenario = INTRADAY_CAMPAIGN_COST_BOOK_V1.resolve("spy_qqq_base_v1")
provenance = build_run_provenance(
    run_id="run_0001",
    code_hash=code_hash,
    config_hash=config_hash,
    cost_scenario_id="spy_qqq_base_v1",
    periods_touched=[("2015-01-01", "2022-12-31")],
    seeds={"global": 42},
)
provenance.write_json(artifact_dir / "provenance.json")
```

An unknown or unnamed scenario raises `UnknownCostScenarioError` before any
data is touched — both in `build_run_provenance` and in
`PipelineSpec.validate()` when a stage config carries a `cost_scenario` key.
The resolved scenario id and cost-book version are recorded in provenance.

Before writing a signal-bearing run, reconcile every provenance period to a
citable guarded read for the same dataset:

```python
from liq.runner import reconcile_periods_touched

reconcile_periods_touched(
    provenance,
    periods_by_dataset={"oanda_fx": [("2015-01-01", "2022-12-31")]},
    guarded_windows_by_dataset={"oanda_fx": guarded_oanda_windows},
)
```

The reconciliation gate rejects invalid dates, unattributed provenance periods,
and reads from a different dataset. Supply guarded windows filtered to the same
arm and research-purpose ledger entries; `dev_smoke` cannot provide citable
coverage. This final check complements guarded access at read time—it does not
replace it.

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

## Operational Notes (Stage 6 hardening)

- `run_rolling` validates fold geometry (`train_size`, `valid_size`, `step`) and
  enforces deterministic `FoldResult.slice_id` propagation to downstream fitness
  caches.
- `fold.slice_id` is used directly by the evolution cache key; mismatched split
  generation between runner and evaluator is treated as an integration bug and should
  surface as cache misses or explicit contract violations.
- Strategy pipeline errors are normalized into runner-level exceptions:
  - missing bars/features for a fold
  - invalid risk/sizing output
  - failed strategy fitting/prediction
- For drift events, pipeline state should be regenerated before continuing with stale
  strategy snapshots.

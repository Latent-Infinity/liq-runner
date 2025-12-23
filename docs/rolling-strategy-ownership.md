# Rolling Strategy Ownership & APIs

This doc clarifies where rolling strategy pieces live across the stack.

## Boundaries
- **liq-signals** (strategy contract):
  - Define `Strategy`/`SignalProvider` interfaces that take features and emit raw scores or side/size suggestions.
  - Provide adapters to wrap model outputs (scores, metadata) into a portable `SignalOutput` that liq-risk and liq-runner can consume.
  - Calibration and EV thresholding hooks: expose optional strategy-level calibration config, but do not run orchestration loops.
- **liq-runner** (orchestration):
  - Own rolling/blocked CV orchestration: slicing data, training models, applying per-fold calibration (using `liq.sim.calibration`), running EV-based threshold selection, and producing orders via liq-risk sizing policies.
  - Drive simulation runs by constructing `OrderRequest` streams and calling `liq-sim`’s `Simulator.run(...)`.
  - Persist artifacts (calibration params, chosen thresholds, funding/slippage/risk-cap stats) and metrics, delegating storage to liq-store through liq-data/liq-features as needed.

## Minimal API surface to add
- In **liq-signals**:
  - `Strategy` interface: `predict(features) -> SignalOutput` (scores + optional aux like confidence).
  - `SignalOutput` carries `scores: pl.Series`, `labels: pl.Series | None`, `metadata`.
  - Optional `strategy.calibration_config` (pass-through to runner).
- In **liq-runner**:
  - Rolling runner that accepts: data loader (liq-data/liq-store), feature pipeline (liq-features), strategy (liq-signals), risk policy (liq-risk), sim config (liq-sim).
  - Per-fold steps: fit strategy → calibrate scores (liq-sim calibration) → EV threshold selection → size orders (liq-risk) → simulate (liq-sim) → collect metrics (liq-metrics).
  - Outputs: thresholds/params, funding/slippage stats from `SimulationResult`, metrics bundle, and serialized artifacts.

## Why this split?
- Keeps strategy definitions (and model-specific code) in liq-signals.
- Keeps orchestration concerns (data splits, calibration/threshold selection, simulation/metrics loop) in liq-runner.
- Allows liq-sim to stay focused on execution realism while being driven by runner.

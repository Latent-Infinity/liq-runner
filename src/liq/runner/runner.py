"""Rolling runner orchestration for strategies."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Protocol, cast

import polars as pl

from liq.core import Bar, OrderRequest, PortfolioState
from liq.core.fill import Fill
from liq.datasets import WalkForwardSplit
from liq.metrics import summarize_qa
from liq.risk.config import MarketState, RiskConfig
from liq.runner.calibration_adapter import calibrate_signal_output, select_threshold
from liq.runner.splits import rolling_splits
from liq.signals import Signal
from liq.signals.output import SignalOutput
from liq.sim.calibration import apply_temperature_scale, build_threshold_grid_from_scores
from liq.sim.config import SimulatorConfig
from liq.sim.simulator import SimulationResult, Simulator

logger = logging.getLogger(__name__)
SLICE_ID_PREFIX = "time_window"


class Strategy(Protocol):
    """Strategy contract for rolling runner."""

    def fit(self, features: pl.DataFrame, labels: pl.Series | None = None) -> None: ...

    def predict(self, features: pl.DataFrame) -> SignalOutput: ...


class RiskSizer(Protocol):
    """Risk sizing via liq-risk."""

    def size(
        self, scores: pl.Series, threshold: float, market: MarketState, portfolio: PortfolioState
    ) -> Iterable[OrderRequest]: ...  # pragma: no cover - protocol


class SignalRiskEngine(Protocol):
    """Risk engine contract used by the rolling runner."""

    def process_signals(
        self,
        signals: list[Signal],
        portfolio_state: PortfolioState,
        market_state: MarketState,
        risk_config: RiskConfig,
        /,
    ) -> Any: ...


class SimulatorFactory(Protocol):
    """Factory returning a configured Simulator."""

    def __call__(self) -> Simulator: ...


class BarsProvider(Protocol):
    """Provides bars for simulation."""

    def get_bars(self, split: slice) -> Iterable[Bar]: ...


class PortfolioProvider(Protocol):
    """Provides initial PortfolioState per fold/split."""

    def get_portfolio(self, split: slice) -> PortfolioState: ...


@dataclass
class FoldResult:
    threshold: float
    calibration_params: dict
    diagnostics: dict
    sim_result: SimulationResult
    metrics: dict
    signals_generated: int
    orders_submitted: int
    risk_rejected: int
    train_end: datetime | None
    oos_start: datetime | None
    oos_end: datetime | None
    constraint_violations: dict | None = None
    slice_id: str | None = None
    split_metadata: dict | None = None


def run_rolling(
    features: pl.DataFrame,
    labels: pl.Series,
    strategy: Strategy,
    risk_engine: SignalRiskEngine,
    simulator_factory: SimulatorFactory,
    bars_provider: BarsProvider,
    portfolio_provider: PortfolioProvider,
    risk_config: RiskConfig,
    train_size: int,
    valid_size: int,
    step: int,
    sim_config: SimulatorConfig | None = None,
    threshold_cfg: dict[str, Any] | None = None,
    calibration_split: float | None = None,
    splits: Sequence[WalkForwardSplit] | None = None,
) -> list[FoldResult]:
    """Simple rolling loop over features/labels."""
    threshold_cfg = threshold_cfg or {}
    fixed_threshold = threshold_cfg.get("fixed_threshold")
    top_signals = threshold_cfg.get("top_signals")
    threshold_grid = threshold_cfg.get("threshold_grid")
    threshold_grid_mode = threshold_cfg.get("threshold_grid_mode")
    threshold_grid_quantiles = threshold_cfg.get("threshold_grid_quantiles")
    threshold_grid_min = float(threshold_cfg.get("threshold_grid_min", 0.05))
    threshold_grid_max = float(threshold_cfg.get("threshold_grid_max", 0.95))
    threshold_grid_round_decimals = int(threshold_cfg.get("threshold_grid_round_decimals", 4))
    ev_cost_bps = threshold_cfg.get("ev_cost_bps")
    target_ev = -(ev_cost_bps / 10000) if ev_cost_bps else None
    calib_frac = calibration_split or threshold_cfg.get("calibration_split") or 0.0

    def _slice_bounds(window: slice) -> tuple[int, int]:
        start = 0 if window.start is None else int(window.start)
        stop = len(features) if window.stop is None else int(window.stop)
        start = max(0, min(start, len(features)))
        stop = max(start, min(stop, len(features)))
        return start, stop - start

    def _serialize_slice(window: slice) -> dict[str, int | None]:
        return {
            "start": None if window.start is None else int(window.start),
            "stop": None if window.stop is None else int(window.stop),
        }

    def _score_summary(scores: pl.Series) -> dict[str, float | int]:
        if scores.is_empty():
            return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        values = scores.cast(pl.Float64)

        def _float_or_zero(value: object) -> float:
            return 0.0 if value is None else float(cast(Any, value))

        return {
            "count": len(values),
            "min": _float_or_zero(values.min()),
            "max": _float_or_zero(values.max()),
            "mean": _float_or_zero(values.mean()),
            "std": _float_or_zero(values.std()),
        }

    def _threshold_grid_for_scores(scores: pl.Series) -> Iterable[float] | None:
        if threshold_grid is not None:
            return cast(Iterable[float], threshold_grid)
        if threshold_grid_mode != "quantile":
            return None
        grid = build_threshold_grid_from_scores(
            scores,
            quantiles=cast(Iterable[float] | None, threshold_grid_quantiles),
            minimum=threshold_grid_min,
            maximum=threshold_grid_max,
            round_decimals=threshold_grid_round_decimals,
        )
        return grid or None

    def _fallback_slice_id(
        slice_id: str | None,
        train: slice,
        validate: slice,
        test: slice,
    ) -> str:
        if slice_id and slice_id != "time_window:auto":
            return slice_id

        train_bounds = _serialize_slice(train)
        validate_bounds = _serialize_slice(validate)
        test_bounds = _serialize_slice(test)
        return (
            f"{SLICE_ID_PREFIX}:"
            f"train={train_bounds['start']}-{train_bounds['stop']}"
            f"|validate={validate_bounds['start']}-{validate_bounds['stop']}"
            f"|test={test_bounds['start']}-{test_bounds['stop']}"
        )

    if splits is not None:
        fold_specs = []
        for split in splits:
            if not isinstance(split, WalkForwardSplit):
                raise TypeError("splits must contain WalkForwardSplit instances")
            if (
                not isinstance(split.train, slice)
                or not isinstance(split.validate, slice)
                or not isinstance(split.test, slice)
            ):
                raise TypeError(
                    "walk-forward splits must be resolved to integer slices before run_rolling; convert datetime boundaries first"
                )
            if split.lockbox is not None and not isinstance(split.lockbox, slice):
                raise TypeError(
                    "walk-forward lockbox must be resolved to an integer slice before run_rolling; convert datetime boundaries first"
                )
            fold_split_id = _fallback_slice_id(
                split.slice_id, split.train, split.validate, split.test
            )
            fold_specs.append(
                (
                    split.train,
                    split.validate,
                    split.test,
                    fold_split_id,
                    {
                        "split_type": "WalkForwardSplit",
                        "slice_id": fold_split_id,
                        "train": _serialize_slice(split.train),
                        "validate": _serialize_slice(split.validate),
                        "test": _serialize_slice(split.test),
                        "lockbox": _serialize_slice(split.lockbox)
                        if split.lockbox is not None
                        else None,
                        "embargo_bars": split.embargo_bars,
                    },
                )
            )
    else:
        fold_specs = [
            (legacy_split.train, legacy_split.valid, legacy_split.valid, None, None)
            for legacy_split in rolling_splits(
                n=len(features), train_size=train_size, valid_size=valid_size, step=step
            )
        ]

    results: list[FoldResult] = []
    for train_window, validate_window, test_window, split_id, split_metadata in fold_specs:
        logger.info(
            "rolling_split",
            extra={
                "train": (train_window.start, train_window.stop),
                "validate": (validate_window.start, validate_window.stop),
                "test": (test_window.start, test_window.stop),
                "slice_id": split_id,
            },
        )
        train_offset, train_length = _slice_bounds(train_window)
        validate_offset, validate_length = _slice_bounds(validate_window)
        test_offset, test_length = _slice_bounds(test_window)
        train_df = features.slice(train_offset, train_length)
        validate_df = features.slice(validate_offset, validate_length)
        test_df = features.slice(test_offset, test_length)
        train_labels = labels.slice(train_offset, train_length)
        validate_labels = labels.slice(validate_offset, validate_length)

        def _vc(series: pl.Series) -> dict:
            vc = series.value_counts().to_dict(as_series=False)
            if not vc:
                return {}
            value_key = "label" if "label" in vc else ("value" if "value" in vc else next(iter(vc)))
            count_key = "count" if "count" in vc else next(k for k in vc if k != value_key)
            return {
                int(lbl): int(cnt) for lbl, cnt in zip(vc[value_key], vc[count_key], strict=False)
            }

        logger.info(
            "label_balance",
            extra={
                "train_counts": _vc(train_labels),
                "valid_counts": _vc(validate_labels),
            },
        )

        strategy.fit(train_df, train_labels)
        calib_size = int(len(validate_df) * calib_frac) if calib_frac > 0 else 0
        if calib_frac > 0 and calib_size == 0 and len(validate_df) > 0:
            calib_size = 1

        if calib_size > 0 and len(validate_df) > 0:
            calib_df = validate_df.slice(0, calib_size)
            calib_labels = validate_labels.slice(0, calib_size)
            sim_df = test_df
            signal_output_calib = strategy.predict(calib_df)
            calibration_raw_score_summary = _score_summary(signal_output_calib.scores)
            calib = calibrate_signal_output(signal_output_calib)
            calibration_calibrated_score_summary = _score_summary(calib.scores)
            calib_labels_used = (
                signal_output_calib.labels
                if signal_output_calib.labels is not None
                else calib_labels
            )
            diag = select_threshold(
                SignalOutput(scores=calib.scores, labels=calib_labels_used),
                min_precision=threshold_cfg.get("precision_min"),
                min_recall=threshold_cfg.get("recall_min"),
                min_trades=threshold_cfg.get("min_trades_per_window"),
                target_ev=target_ev,
                grid=_threshold_grid_for_scores(calib.scores),
            )
            signal_output = strategy.predict(sim_df)
            test_raw_score_summary = _score_summary(signal_output.scores)
            temp = calib.params.get("temperature", 1.0) if hasattr(calib, "params") else 1.0
            calibrated_scores = apply_temperature_scale(signal_output.scores, temp)
            test_calibrated_score_summary = _score_summary(calibrated_scores)
            signal_output = SignalOutput(scores=calibrated_scores, labels=signal_output.labels)
            bar_slice = test_window
        else:
            signal_source = test_df if len(test_df) > 0 else validate_df
            signal_output = strategy.predict(signal_source)
            calibration_raw_score_summary = _score_summary(signal_output.scores)
            test_raw_score_summary = _score_summary(signal_output.scores)
            calib = calibrate_signal_output(signal_output)
            calibration_calibrated_score_summary = _score_summary(calib.scores)
            test_calibrated_score_summary = _score_summary(calib.scores)
            valid_labels_used = (
                signal_output.labels if signal_output.labels is not None else validate_labels
            )
            diag = select_threshold(
                SignalOutput(scores=calib.scores, labels=valid_labels_used),
                min_precision=threshold_cfg.get("precision_min"),
                min_recall=threshold_cfg.get("recall_min"),
                min_trades=threshold_cfg.get("min_trades_per_window"),
                target_ev=target_ev,
                grid=_threshold_grid_for_scores(calib.scores),
            )
            bar_slice = test_window

        # minimal market/portfolio shims; real runner should hydrate properly
        bars = list(bars_provider.get_bars(bar_slice))
        train_bars = list(bars_provider.get_bars(train_window))
        ts = bars[0].timestamp if bars else datetime.now(UTC)
        current_bar = bars[0] if bars else None
        current_symbol = current_bar.symbol if current_bar else ""
        # Simple volatility proxy: ATR-like range over last 20 bars
        vol_val = None
        if current_bar and len(bars) > 1:
            recent = bars[max(0, len(bars) - 20) :]
            trs = []
            prev_close = recent[0].close
            for b in recent:
                high_low = float(b.high - b.low)
                high_close = float(abs(b.high - prev_close))
                low_close = float(abs(b.low - prev_close))
                tr = max(high_low, high_close, low_close)
                trs.append(tr)
                prev_close = b.close
            if trs:
                vol_val = sum(trs) / len(trs)
        market_state = MarketState(
            current_bars={current_symbol: current_bar} if current_bar else {},
            volatility={current_symbol: Decimal(str(vol_val)) if vol_val else Decimal("0.01")}
            if current_symbol
            else {},
            liquidity={},
            timestamp=ts,
        )
        portfolio = portfolio_provider.get_portfolio(test_window)
        # Wrap scores into signals for risk engine; here we use threshold as a simple long/flat filter
        # Build signals aligned to the validation bars using the calibrated scores.
        signals: list[Signal] = []
        threshold = (
            fixed_threshold
            if fixed_threshold is not None
            else diag.threshold
            if diag and diag.threshold is not None
            else 0.5
        )
        scores_iter = signal_output.scores if signal_output is not None else []
        for bar, score in zip(bars, scores_iter, strict=False):
            side = "long" if score >= threshold else "flat"
            signals.append(
                Signal(
                    symbol=bar.symbol,
                    direction=side,
                    strength=float(score),
                    timestamp=bar.timestamp,
                )
            )
        raw_actionable_signals = [s for s in signals if getattr(s, "direction", "") != "flat"]
        strongest_by_symbol: dict[str, Signal] = {}
        for signal in raw_actionable_signals:
            existing = strongest_by_symbol.get(signal.symbol)
            if existing is None or signal.strength > existing.strength:
                strongest_by_symbol[signal.symbol] = signal
        actionable_signals = list(strongest_by_symbol.values())
        if top_signals and len(actionable_signals) > top_signals:
            actionable_signals.sort(key=lambda s: getattr(s, "strength", 0.0), reverse=True)
            actionable_signals = actionable_signals[:top_signals]
        logger.debug(
            "calibration_threshold", extra={"threshold": diag.threshold, "ev": diag.expected_value}
        )
        if actionable_signals:
            risk_result = risk_engine.process_signals(
                actionable_signals,
                portfolio,
                market_state,
                risk_config,
            )
        else:
            risk_result = SimpleNamespace(
                orders=[],
                rejected_signals=[],
                constraint_violations={},
                constraint_diagnostics={},
                sizing_rejections={},
            )
        orders = list(risk_result.orders)
        long_signals = len(raw_actionable_signals)
        risk_signals = len(actionable_signals)

        sim = simulator_factory()
        if sim_config is not None:
            sim.config = sim_config
        sim_result = sim.run(orders, bars=bars)

        # Feed fills back into constraints that track frequency/history (e.g., FrequencyCap)
        try:
            get_constraints = getattr(risk_engine, "_get_constraints", None)
            constraints = get_constraints() if callable(get_constraints) else []
        except Exception:
            constraints = []
        if constraints:
            for fill in sim_result.fills:
                if not isinstance(fill, Fill):
                    continue
                for constraint in constraints:
                    record = getattr(constraint, "record_trade", None)
                    if callable(record):
                        record(fill.symbol, fill.timestamp, fill.side, fill.quantity)
        metrics = summarize_qa(cast(Any, sim_result))
        logger.info(
            "fold_result",
            extra={
                "fills": len(sim_result.fills),
                "rejected": len(sim_result.rejected_orders),
                "funding": str(getattr(sim_result, "funding_charged", 0)),
                "slippage_stats": getattr(sim_result, "slippage_stats", {}),
                "signals_generated": len(signals),
                "orders_submitted": len(orders),
                "risk_rejected": len(getattr(risk_result, "rejected_signals", [])),
                "long_signals": long_signals,
                "risk_signals": risk_signals,
                "constraint_violations": getattr(risk_result, "constraint_violations", {}),
                "constraint_diagnostics": getattr(risk_result, "constraint_diagnostics", {}),
                "sizing_rejections": getattr(risk_result, "sizing_rejections", {}),
            },
        )

        train_end_ts = train_bars[-1].timestamp if train_bars else None
        oos_start_ts = bars[0].timestamp if bars else None
        oos_end_ts = bars[-1].timestamp if bars else None

        results.append(
            FoldResult(
                threshold=threshold,
                calibration_params=calib.params,
                diagnostics={
                    "expected_value": diag.expected_value,
                    "precision": diag.precision,
                    "recall": diag.recall,
                    "trades": diag.trades,
                    "long_signals": long_signals,
                    "risk_signals": risk_signals,
                    "constraint_diagnostics": getattr(risk_result, "constraint_diagnostics", {}),
                    "sizing_rejections": getattr(risk_result, "sizing_rejections", {}),
                    "calibration_raw_scores": calibration_raw_score_summary,
                    "calibration_calibrated_scores": calibration_calibrated_score_summary,
                    "test_raw_scores": test_raw_score_summary,
                    "test_calibrated_scores": test_calibrated_score_summary,
                },
                sim_result=sim_result,
                metrics=metrics,
                signals_generated=len(signals),
                orders_submitted=len(orders),
                risk_rejected=len(getattr(risk_result, "rejected_signals", [])),
                train_end=train_end_ts,
                oos_start=oos_start_ts,
                oos_end=oos_end_ts,
                constraint_violations=getattr(risk_result, "constraint_violations", {}),
                slice_id=split_id,
                split_metadata=split_metadata,
            )
        )
    return results

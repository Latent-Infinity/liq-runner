"""Rolling runner orchestration for strategies."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Iterable, Protocol
from datetime import datetime, timezone

import polars as pl

from liq.runner.calibration_adapter import calibrate_signal_output, select_threshold
from liq.runner.splits import rolling_splits
from liq.signals.output import SignalOutput
from liq.signals import Signal
from liq.sim.config import SimulatorConfig
from liq.sim.simulator import SimulationResult, Simulator
from liq.metrics import summarize_qa
from liq.core import Bar, OrderRequest, PortfolioState
from liq.core.fill import Fill
from liq.risk.engine import RiskEngine
from liq.risk.config import RiskConfig, MarketState
import logging

logger = logging.getLogger(__name__)


class Strategy(Protocol):
    """Strategy contract for rolling runner."""

    def fit(self, features: pl.DataFrame, labels: pl.Series | None = None) -> None:
        ...

    def predict(self, features: pl.DataFrame) -> SignalOutput:
        ...


class RiskSizer(Protocol):
    """Risk sizing via liq-risk."""

    def size(self, scores: pl.Series, threshold: float, market: MarketState, portfolio: PortfolioState) -> Iterable[OrderRequest]:
        ...  # pragma: no cover - protocol


class SimulatorFactory(Protocol):
    """Factory returning a configured Simulator."""

    def __call__(self) -> Simulator:
        ...


class BarsProvider(Protocol):
    """Provides bars for simulation."""

    def get_bars(self, split: slice) -> Iterable[Bar]:
        ...


class PortfolioProvider(Protocol):
    """Provides initial PortfolioState per fold/split."""

    def get_portfolio(self, split: slice) -> PortfolioState:
        ...


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


def run_rolling(
    features: pl.DataFrame,
    labels: pl.Series,
    strategy: Strategy,
    risk_engine: RiskEngine,
    simulator_factory: SimulatorFactory,
    bars_provider: BarsProvider,
    portfolio_provider: PortfolioProvider,
    risk_config: RiskConfig,
    train_size: int,
    valid_size: int,
    step: int,
    sim_config: SimulatorConfig | None = None,
    threshold_cfg: dict | None = None,
    calibration_split: float | None = None,
) -> list[FoldResult]:
    """Simple rolling loop over features/labels."""
    threshold_cfg = threshold_cfg or {}
    fixed_threshold = threshold_cfg.get("fixed_threshold")
    top_signals = threshold_cfg.get("top_signals")
    ev_cost_bps = threshold_cfg.get("ev_cost_bps")
    target_ev = -(ev_cost_bps / 10000) if ev_cost_bps else None
    calib_frac = calibration_split or threshold_cfg.get("calibration_split") or 0.0
    results: list[FoldResult] = []
    for split in rolling_splits(n=len(features), train_size=train_size, valid_size=valid_size, step=step):
        logger.info(
            "rolling_split",
            extra={"train": (split.train.start, split.train.stop), "valid": (split.valid.start, split.valid.stop)},
        )
        train_df = features.slice(split.train.start, split.train.stop - split.train.start)
        valid_df = features.slice(split.valid.start, split.valid.stop - split.valid.start)
        train_labels = labels.slice(split.train.start, split.train.stop - split.train.start)
        valid_labels = labels.slice(split.valid.start, split.valid.stop - split.valid.start)

        def _vc(series: pl.Series) -> dict:
            vc = series.value_counts().to_dict(as_series=False)
            return {int(lbl): int(cnt) for lbl, cnt in zip(vc["label"], vc["count"])} if vc else {}

        logger.info(
            "label_balance",
            extra={
                "train_counts": _vc(train_labels),
                "valid_counts": _vc(valid_labels),
            },
        )

        strategy.fit(train_df, train_labels)
        calib_size = int(len(valid_df) * calib_frac) if calib_frac > 0 else 0
        if calib_frac > 0 and calib_size == 0 and len(valid_df) > 0:
            calib_size = 1

        if calib_size > 0:
            calib_df = valid_df.slice(0, calib_size)
            calib_labels = valid_labels.slice(0, calib_size)
            sim_df = valid_df.slice(calib_size, len(valid_df) - calib_size)
            signal_output_calib = strategy.predict(calib_df)
            calib = calibrate_signal_output(signal_output_calib)
            diag = select_threshold(
                SignalOutput(scores=calib.scores, labels=signal_output_calib.labels or calib_labels),
                min_precision=threshold_cfg.get("precision_min"),
                min_recall=threshold_cfg.get("recall_min"),
                min_trades=threshold_cfg.get("min_trades_per_window"),
                target_ev=target_ev,
            )
            signal_output = strategy.predict(sim_df)
            temp = calib.params.get("temperature", 1.0) if hasattr(calib, "params") else 1.0
            calibrated_scores = (signal_output.scores / temp).clip(0.0, 1.0)
            signal_output = SignalOutput(scores=calibrated_scores, labels=signal_output.labels)
            bar_slice = slice(split.valid.start + calib_size, split.valid.stop)
        else:
            signal_output = strategy.predict(valid_df)
            calib = calibrate_signal_output(signal_output)
            diag = select_threshold(
                SignalOutput(scores=calib.scores, labels=signal_output.labels or valid_labels),
                min_precision=threshold_cfg.get("precision_min"),
                min_recall=threshold_cfg.get("recall_min"),
                min_trades=threshold_cfg.get("min_trades_per_window"),
                target_ev=target_ev,
            )
            bar_slice = split.valid

        # minimal market/portfolio shims; real runner should hydrate properly
        bars = list(bars_provider.get_bars(bar_slice))
        train_bars = list(bars_provider.get_bars(split.train))
        ts = bars[0].timestamp if bars else datetime.now(timezone.utc)
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
            marks={},
            current_bars={current_symbol: current_bar} if current_bar else {},
            volatility={current_symbol: Decimal(str(vol_val)) if vol_val else Decimal("0.01")} if current_symbol else {},
            liquidity={},
            timestamp=ts,
        )
        portfolio = portfolio_provider.get_portfolio(split.valid)
        # Wrap scores into signals for risk engine; here we use threshold as a simple long/flat filter
        # Build signals aligned to the validation bars using the calibrated scores.
        signals: list[Signal] = []
        threshold = (
            fixed_threshold
            if fixed_threshold is not None
            else diag.threshold if diag and diag.threshold is not None else 0.5
        )
        scores_iter = signal_output.scores if signal_output is not None else []
        for bar, score in zip(bars, scores_iter):
            side = "long" if score >= threshold else "flat"
            signals.append(
                Signal(
                    symbol=bar.symbol,
                    direction=side,  # type: ignore[arg-type]
                    strength=float(score),
                    timestamp=bar.timestamp,
                )
            )
        if top_signals and len(signals) > top_signals:
            signals.sort(key=lambda s: getattr(s, "strength", 0.0), reverse=True)
            signals = signals[:top_signals]
        logger.debug("calibration_threshold", extra={"threshold": diag.threshold, "ev": diag.expected_value})
        risk_result = risk_engine.process_signals(
            signals=signals,
            portfolio_state=portfolio,
            market_state=market_state,
            risk_config=risk_config,
        )
        orders = list(risk_result.orders)
        long_signals = sum(1 for s in signals if getattr(s, "direction", "") == "long")

        sim = simulator_factory()
        if sim_config is not None:
            sim.config = sim_config
        sim_result = sim.run(orders, bars=bars)

        # Feed fills back into constraints that track frequency/history (e.g., FrequencyCap)
        try:
            constraints = risk_engine._get_constraints()  # noqa: SLF001 - intentional reuse of configured chain
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
        metrics = summarize_qa(sim_result)
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
                "constraint_violations": getattr(risk_result, "constraint_violations", {}),
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
            )
        )
    return results

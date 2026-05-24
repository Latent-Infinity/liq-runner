"""Rolling runner orchestration for strategies."""

from __future__ import annotations

import json
import logging
import os
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Literal, Protocol, cast

import polars as pl

from liq.core import Bar, OrderRequest, OrderSide, OrderType, PortfolioState
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


def _with_flat_lifecycle_exits(
    orders: list[OrderRequest],
    signals: list[Signal],
    *,
    exit_threshold: float | None = None,
    min_hold_bars: int = 0,
    max_hold_bars: int | None = None,
) -> list[OrderRequest]:
    """Append market close orders at the next qualifying exit signal after each entry."""
    if not orders:
        return orders

    signals_by_symbol: dict[str, list[Signal]] = {}
    for signal in signals:
        signals_by_symbol.setdefault(signal.symbol, []).append(signal)
    for symbol_signals in signals_by_symbol.values():
        symbol_signals.sort(key=lambda signal: signal.timestamp)

    actionable_signals = [signal for signal in signals if signal.direction != "flat"]
    matched_signal_indexes: set[int] = set()
    lifecycle_orders: list[OrderRequest] = []
    for order in orders:
        matched_signal: Signal | None = None
        for index, signal in enumerate(actionable_signals):
            if index in matched_signal_indexes:
                continue
            if signal.symbol == order.symbol and signal.strength == order.confidence:
                matched_signal = signal
                matched_signal_indexes.add(index)
                break
        if matched_signal is not None:
            order = order.model_copy(update={"timestamp": matched_signal.timestamp})
        lifecycle_orders.append(order)
        if order.side not in {OrderSide.BUY, OrderSide.SELL}:
            continue
        close_side = OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY
        symbol_signals = signals_by_symbol.get(order.symbol, [])
        entry_index = next(
            (
                index
                for index, signal in enumerate(symbol_signals)
                if signal.timestamp == order.timestamp
            ),
            None,
        )
        close_signal = next(
            (
                signal
                for index, signal in enumerate(symbol_signals)
                if signal.timestamp > order.timestamp
                and (entry_index is None or index - entry_index >= min_hold_bars)
                and signal.direction == "flat"
                and (exit_threshold is None or signal.strength <= exit_threshold)
            ),
            None,
        )
        if close_signal is None and max_hold_bars is not None and entry_index is not None:
            max_hold_index = entry_index + max_hold_bars
            if max_hold_index < len(symbol_signals):
                close_signal = symbol_signals[max_hold_index]
        if close_signal is None:
            continue
        lifecycle_orders.append(
            OrderRequest(
                symbol=order.symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                quantity=order.quantity,
                timestamp=close_signal.timestamp,
                confidence=order.confidence,
            )
        )
    return lifecycle_orders


def _apply_entry_spacing(signals: list[Signal], min_spacing_bars: int) -> list[Signal]:
    """Keep same-symbol entry signals at least ``min_spacing_bars`` apart."""
    if min_spacing_bars <= 0:
        return signals

    last_kept_index_by_symbol: dict[str, int] = {}
    kept: list[Signal] = []
    for index, signal in enumerate(signals):
        last_kept_index = last_kept_index_by_symbol.get(signal.symbol)
        if last_kept_index is not None and index - last_kept_index < min_spacing_bars:
            continue
        kept.append(signal)
        last_kept_index_by_symbol[signal.symbol] = index
    return kept


def _roc_auc(
    scores: Sequence[float] | Any,
    labels: Sequence[int] | Any,
) -> float | None:
    """Rank-based ROC AUC (Mann-Whitney U), dependency-free.

    Clean out-of-sample discrimination of raw model scores vs the true binary
    label over ALL test rows (not the post-threshold subset). Returns None when
    undefined: empty, length mismatch, non-binary labels, or a single class
    (AUC needs both a positive and a negative). Ties use average ranks.
    """
    s = [float(v) for v in scores]
    y = [int(v) for v in labels]
    if not s or len(s) != len(y):
        return None
    if set(y) - {0, 1} or len(set(y)) < 2:
        return None
    order = sorted(range(len(s)), key=lambda i: s[i])
    ranks = [0.0] * len(s)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based ranks for positions i..j
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    sum_pos = sum(rank for rank, label in zip(ranks, y, strict=True) if label == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _oos_ic(
    scores: Sequence[float] | Any,
    bars: list[Bar],
    horizon: int | None,
) -> float | None:
    """Pearson IC of raw scores vs forward h-bar return on bar.close.

    The natural monetization metric for continuous-sizing: when IC > 0, the
    score correlates with future return so a size = f(score) overlay earns
    positive PnL in expectation. Returns ``None`` when undefined (no horizon,
    too few paired points, or zero variance on either series).
    """
    if horizon is None or horizon <= 0 or not bars:
        return None
    s = [float(v) for v in scores]
    closes = [float(b.close) for b in bars]
    pairs: list[tuple[float, float]] = []
    for i in range(min(len(s), len(closes))):
        if i + horizon < len(closes) and closes[i] != 0.0:
            ret = closes[i + horizon] / closes[i] - 1.0
            pairs.append((s[i], ret))
    if len(pairs) < 2:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0.0 or sy == 0.0:
        return None
    return sxy / (sx * sy)


def _continuous_rebalance_order(
    *,
    target_size_pct: float,
    current_qty: Decimal,
    bar: Bar,
    max_notional: Decimal,
    rebalance_band_pct: float,
) -> OrderRequest | None:
    """Compute one rebalance order to move toward a continuous long target.

    target_qty = clip(target_size_pct, 0, 1) * max_notional / bar.close. Emit
    BUY if the delta is positive, SELL if negative, None if |delta·close| is
    inside the rebalance band (turnover guard). Long-only: target clipped to
    [0, 1].
    """
    t = max(0.0, min(1.0, float(target_size_pct)))
    target_notional = max_notional * Decimal(str(t))
    target_qty = target_notional / bar.close
    delta = target_qty - current_qty
    if delta == 0:
        return None
    delta_notional = abs(delta) * bar.close
    band_notional = max_notional * Decimal(str(rebalance_band_pct))
    if delta_notional <= band_notional:
        return None
    side = OrderSide.BUY if delta > 0 else OrderSide.SELL
    return OrderRequest(
        symbol=bar.symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=abs(delta),
        timestamp=bar.timestamp,
    )


def _attach_protective_brackets(
    orders: list[OrderRequest],
    market_state: MarketState,
    *,
    stop_atr_mult: float | None,
    take_atr_mult: float | None,
    bracket_prices: dict[tuple[str, datetime], tuple[Decimal, Decimal]] | None = None,
) -> list[OrderRequest]:
    """Attach stop-loss/take-profit metadata to long entry orders.

    Two paths, in precedence order:
    1. ``bracket_prices`` map keyed by (symbol, timestamp) of absolute
       (sl, tp) prices — the by-construction-coherent path: the label that
       trained the model emits these exact prices per row and execution uses
       them verbatim, eliminating label↔execution geometry mismatch.
    2. ATR-mult fallback (``stop_atr_mult``/``take_atr_mult``) — computes
       prices from the per-symbol bar close ± mult·ATR.

    Long-only: only BUY (entry) orders are bracketed; SELL (lifecycle-exit)
    orders pass through. When both the map and the mults are unset, this is
    a no-op returning the input list unchanged (byte-identical to the
    no-bracket pipeline).
    """
    if bracket_prices is None and stop_atr_mult is None and take_atr_mult is None:
        return orders
    out: list[OrderRequest] = []
    for order in orders:
        if order.side != OrderSide.BUY:
            out.append(order)
            continue
        # 1) per-row map takes precedence
        if bracket_prices is not None:
            pair = bracket_prices.get((order.symbol, order.timestamp))
            if pair is not None:
                sl, tp = pair
                md = dict(order.metadata or {})
                md["stop_loss_price"] = sl
                md["take_profit_price"] = tp
                out.append(order.model_copy(update={"metadata": md}))
                continue
        # 2) ATR-mult fallback (only if mults configured)
        if stop_atr_mult is None and take_atr_mult is None:
            out.append(order)
            continue
        bar = market_state.current_bars.get(order.symbol)
        atr = market_state.volatility.get(order.symbol)
        if bar is None or atr is None or atr <= 0:
            out.append(order)
            continue
        ref = bar.close
        md = dict(order.metadata or {})
        if stop_atr_mult is not None:
            md["stop_loss_price"] = ref - Decimal(str(stop_atr_mult)) * atr
        if take_atr_mult is not None:
            md["take_profit_price"] = ref + Decimal(str(take_atr_mult)) * atr
        out.append(order.model_copy(update={"metadata": md}))
    return out


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
    bracket_prices: dict[tuple[str, datetime], tuple[Decimal, Decimal]] | None = None,
    sizing_mode: Literal["entry", "continuous"] = "entry",
    sizing_fn: Literal["linear", "rank_normalize"] = "linear",
    max_notional_per_symbol: Decimal | None = None,
    rebalance_band_pct: float = 0.10,
    forward_horizon: int | None = None,
) -> list[FoldResult]:
    """Simple rolling loop over features/labels."""
    threshold_cfg = threshold_cfg or {}
    fixed_threshold = threshold_cfg.get("fixed_threshold")
    top_signals = threshold_cfg.get("top_signals")
    max_signals_per_symbol_raw = threshold_cfg.get("max_signals_per_symbol", 1)
    if max_signals_per_symbol_raw is None:
        max_signals_per_symbol: int | None = None
    else:
        max_signals_per_symbol = int(max_signals_per_symbol_raw)
        if max_signals_per_symbol < 1:
            raise ValueError("threshold_cfg.max_signals_per_symbol must be >= 1 or null")
    threshold_grid = threshold_cfg.get("threshold_grid")
    threshold_grid_mode = threshold_cfg.get("threshold_grid_mode")
    threshold_grid_quantiles = threshold_cfg.get("threshold_grid_quantiles")
    threshold_grid_min = float(threshold_cfg.get("threshold_grid_min", 0.05))
    threshold_grid_max = float(threshold_cfg.get("threshold_grid_max", 0.95))
    threshold_grid_round_decimals = int(threshold_cfg.get("threshold_grid_round_decimals", 4))
    exit_threshold_raw = threshold_cfg.get("exit_threshold")
    exit_threshold = None if exit_threshold_raw is None else float(exit_threshold_raw)
    bracket_stop_atr_mult_raw = threshold_cfg.get("bracket_stop_atr_mult")
    bracket_stop_atr_mult = (
        None if bracket_stop_atr_mult_raw is None else float(bracket_stop_atr_mult_raw)
    )
    bracket_take_atr_mult_raw = threshold_cfg.get("bracket_take_atr_mult")
    bracket_take_atr_mult = (
        None if bracket_take_atr_mult_raw is None else float(bracket_take_atr_mult_raw)
    )
    exit_hysteresis = float(threshold_cfg.get("exit_hysteresis", 0.0))
    if exit_hysteresis < 0:
        raise ValueError("threshold_cfg.exit_hysteresis must be >= 0")
    min_hold_bars = int(threshold_cfg.get("min_hold_bars", 0))
    if min_hold_bars < 0:
        raise ValueError("threshold_cfg.min_hold_bars must be >= 0")
    entry_threshold_margin = float(threshold_cfg.get("entry_threshold_margin", 0.0))
    if entry_threshold_margin < 0:
        raise ValueError("threshold_cfg.entry_threshold_margin must be >= 0")
    min_entry_spacing_bars = int(threshold_cfg.get("min_entry_spacing_bars", 0))
    if min_entry_spacing_bars < 0:
        raise ValueError("threshold_cfg.min_entry_spacing_bars must be >= 0")
    max_hold_bars_raw = threshold_cfg.get("max_hold_bars")
    max_hold_bars = None if max_hold_bars_raw is None else int(max_hold_bars_raw)
    if max_hold_bars is not None and max_hold_bars < 1:
        raise ValueError("threshold_cfg.max_hold_bars must be >= 1 or null")
    if max_hold_bars is not None and max_hold_bars < min_hold_bars:
        raise ValueError("threshold_cfg.max_hold_bars must be >= min_hold_bars")
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
        test_labels = labels.slice(test_offset, test_length)

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
            # Raw model scores drive ranking/selection; calibrated scores are
            # reserved for the EV-threshold GO/NO-GO decision. Capture raw
            # before the calibrated overwrite — temperature scaling can
            # saturate and destroy the raw AUC ordering.
            ranking_scores = signal_output.scores
            test_raw_score_summary = _score_summary(signal_output.scores)
            temp = calib.params.get("temperature", 1.0) if hasattr(calib, "params") else 1.0
            calibrated_scores = apply_temperature_scale(signal_output.scores, temp)
            test_calibrated_score_summary = _score_summary(calibrated_scores)
            signal_output = SignalOutput(scores=calibrated_scores, labels=signal_output.labels)
            bar_slice = test_window
        else:
            signal_source = test_df if len(test_df) > 0 else validate_df
            signal_output = strategy.predict(signal_source)
            ranking_scores = signal_output.scores
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
        # Per-symbol market snapshot: first in-window bar per symbol plus an
        # ATR-like volatility proxy over that symbol's last 20 bars. Reduces
        # exactly to single-symbol behavior when only one symbol is present
        # (panel rows are ordered (timestamp, symbol), so per-symbol bars stay
        # in ascending-timestamp order for the running TR calc).
        bars_by_symbol: dict[str, list[Bar]] = {}
        for _b in bars:
            bars_by_symbol.setdefault(_b.symbol, []).append(_b)

        def _atr_proxy(sym_bars: list[Bar]) -> float | None:
            if len(sym_bars) <= 1:
                return None
            recent = sym_bars[max(0, len(sym_bars) - 20) :]
            trs = []
            prev_close = recent[0].close
            for b in recent:
                tr = max(
                    float(b.high - b.low),
                    float(abs(b.high - prev_close)),
                    float(abs(b.low - prev_close)),
                )
                trs.append(tr)
                prev_close = b.close
            return sum(trs) / len(trs) if trs else None

        current_bars_map: dict[str, Bar] = {}
        volatility_map: dict[str, Decimal] = {}
        for _sym, _sym_bars in bars_by_symbol.items():
            current_bars_map[_sym] = _sym_bars[0]
            _v = _atr_proxy(_sym_bars)
            volatility_map[_sym] = Decimal(str(_v)) if _v else Decimal("0.01")
        market_state = MarketState(
            current_bars=current_bars_map,
            volatility=volatility_map,
            liquidity={},
            timestamp=ts,
        )
        portfolio = portfolio_provider.get_portfolio(test_window)
        # Continuous-sizing branch: build per-bar rebalance orders directly
        # from raw scores, bypassing the entry-mode threshold/risk-engine
        # path. Tracked-qty is local to the fold (assumes flat start; sim
        # executes with real cost/slippage).
        continuous_orders: list[OrderRequest] | None = None
        if sizing_mode == "continuous":
            if max_notional_per_symbol is None:
                raise ValueError(
                    "max_notional_per_symbol required when sizing_mode='continuous'"
                )
            tracked: dict[str, Decimal] = {}
            continuous_orders = []
            _ranking = ranking_scores if signal_output is not None else []
            _zero = Decimal("0")
            _one = Decimal("1")
            _two = Decimal("2")
            _half = Decimal("0.5")
            # Pre-compute window-relative ranks for rank-normalize sizing
            # (unit-invariant: monetizes the AUC/IC-defined ranking regardless
            # of the raw score's domain).
            _scores_list = [float(s) for s in _ranking]
            _sorted_scores = sorted(_scores_list)
            _n_scores = len(_sorted_scores)
            for _bar, _raw in zip(bars, _ranking, strict=False):
                # Decimal-end-to-end avoids IEEE-754 noise
                # (e.g. 2 * (0.6 - 0.5) = 0.19999... in float).
                _raw_f = float(_raw)
                if sizing_fn == "rank_normalize":
                    # Empirical CDF rank P[X <= score], unit-invariant.
                    _rank = (
                        bisect_right(_sorted_scores, _raw_f) / _n_scores
                        if _n_scores
                        else 0.0
                    )
                    _target_dec = Decimal(str(_rank))
                else:  # linear
                    _score_dec = Decimal(str(_raw_f))
                    _target_dec = max(_zero, min(_one, _two * (_score_dec - _half)))
                _cur = tracked.get(_bar.symbol, _zero)
                _od = _continuous_rebalance_order(
                    target_size_pct=float(_target_dec),
                    current_qty=_cur,
                    bar=_bar,
                    max_notional=max_notional_per_symbol,
                    rebalance_band_pct=rebalance_band_pct,
                )
                if _od is not None:
                    continuous_orders.append(_od)
                    tracked[_bar.symbol] = (
                        max_notional_per_symbol * _target_dec / _bar.close
                    )
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
        entry_threshold = float(threshold) + entry_threshold_margin
        effective_exit_threshold = (
            exit_threshold if exit_threshold is not None else float(threshold) - exit_hysteresis
        )
        # Decision score (calibrated in the calibration branch) gates the
        # long/flat GO/NO-GO via the EV-derived threshold; raw ranking score
        # carries strength so top-K / per-symbol selection and downstream
        # sizing keep the model's true (uncalibrated) ordering.
        decision_iter = signal_output.scores if signal_output is not None else []
        ranking_iter = ranking_scores if signal_output is not None else []
        for bar, decision_score, rank_score in zip(
            bars, decision_iter, ranking_iter, strict=False
        ):
            side = "long" if decision_score >= entry_threshold else "flat"
            signals.append(
                Signal(
                    symbol=bar.symbol,
                    direction=side,
                    strength=float(rank_score),
                    timestamp=bar.timestamp,
                )
            )
        raw_actionable_signals = _apply_entry_spacing(
            [s for s in signals if getattr(s, "direction", "") != "flat"],
            min_entry_spacing_bars,
        )
        if max_signals_per_symbol is None:
            actionable_signals = list(raw_actionable_signals)
        else:
            signals_by_symbol: dict[str, list[Signal]] = {}
            for signal in raw_actionable_signals:
                signals_by_symbol.setdefault(signal.symbol, []).append(signal)
            actionable_signals = []
            for symbol_signals in signals_by_symbol.values():
                symbol_signals.sort(key=lambda s: getattr(s, "strength", 0.0), reverse=True)
                actionable_signals.extend(symbol_signals[:max_signals_per_symbol])
        if top_signals and len(actionable_signals) > top_signals:
            actionable_signals.sort(key=lambda s: getattr(s, "strength", 0.0), reverse=True)
            actionable_signals = actionable_signals[:top_signals]
        logger.debug(
            "calibration_threshold", extra={"threshold": diag.threshold, "ev": diag.expected_value}
        )
        if continuous_orders is not None:
            # Continuous mode bypasses the entry-focused risk engine; sizing
            # is the risk control (target clipped to [0, 1] of max_notional).
            risk_result = SimpleNamespace(
                orders=[],
                rejected_signals=[],
                constraint_violations={},
                constraint_diagnostics={},
                sizing_rejections={},
            )
        elif actionable_signals:
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
        if continuous_orders is not None:
            # Continuous-sizing orders bypass the entry-mode lifecycle exits
            # (no entry signals to attach flat-exits to; rebalancing IS the
            # exit mechanism).
            orders = continuous_orders
        else:
            orders = _with_flat_lifecycle_exits(
                list(risk_result.orders),
                signals,
                exit_threshold=effective_exit_threshold,
                min_hold_bars=min_hold_bars,
                max_hold_bars=max_hold_bars,
            )
        if continuous_orders is None:
            orders = _attach_protective_brackets(
                orders,
                market_state,
                stop_atr_mult=bracket_stop_atr_mult,
                take_atr_mult=bracket_take_atr_mult,
                bracket_prices=bracket_prices,
            )
        # In continuous mode, sizing IS the risk control — no protective
        # brackets (they would prematurely close positions the rebalance
        # logic expects to persist, corrupting target-tracking).
        long_signals = len(raw_actionable_signals)
        risk_signals = len(actionable_signals)

        # Optional per-order autopsy sidecar (diagnostic only; off unless env
        # set). Captures the entry score: Signal.strength -> TargetPosition
        # .signal_strength -> OrderRequest.confidence. Joined to the fills
        # sidecar by client_order_id (UUID) to correlate realized win-rate
        # against model score at entry. Append mode keyed by UUID, so it is
        # window/fold-unambiguous.
        orders_dump_path = os.environ.get("LIQ_ORDERS_DUMP")
        if orders_dump_path:
            with open(orders_dump_path, "a") as _odh:
                for _o in orders:
                    _odh.write(
                        json.dumps(
                            {
                                "client_order_id": str(_o.client_order_id),
                                "symbol": _o.symbol,
                                "side": getattr(_o.side, "value", str(_o.side)),
                                "quantity": float(_o.quantity),
                                "confidence": (
                                    float(_o.confidence)
                                    if _o.confidence is not None
                                    else None
                                ),
                                "timestamp": _o.timestamp.isoformat()
                                if _o.timestamp is not None
                                else None,
                            }
                        )
                        + "\n"
                    )

        sim = simulator_factory()
        if sim_config is not None:
            sim.config = sim_config
        sim_result = sim.run(orders, bars=bars)

        # Feed fills back into constraints that track frequency/history (e.g., FrequencyCap)
        constraints: list[Any] = []
        try:
            get_constraints = getattr(risk_engine, "_get_constraints", None)
            if callable(get_constraints):
                constraints = list(cast(Iterable[Any], get_constraints()))
        except Exception:
            constraints = []
        if constraints:
            for fill in cast(Iterable[Any], sim_result.fills):
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
                    "oos_auc": _roc_auc(ranking_scores, test_labels),
                    "oos_ic": _oos_ic(ranking_scores, bars, forward_horizon),
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

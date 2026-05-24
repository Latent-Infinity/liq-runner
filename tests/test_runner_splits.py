from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID

import polars as pl
import pytest

from liq.core import Bar, OrderRequest, OrderSide, OrderType, PortfolioState
from liq.core.fill import Fill
from liq.datasets import WalkForwardSplit
from liq.risk.config import MarketState, RiskConfig
from liq.runner.runner import (
    FoldResult,
    _attach_protective_brackets,
    _continuous_rebalance_order,
    _roc_auc,
    run_rolling,
)
from liq.signals.output import SignalOutput
from liq.sim.config import ProviderConfig, SimulatorConfig
from liq.sim.simulator import Simulator


class DummyStrategy:
    def fit(self, features: pl.DataFrame, labels: pl.Series | None = None) -> None:
        self._labels = labels

    def predict(self, features: pl.DataFrame) -> SignalOutput:
        scores = pl.Series([0.9] * len(features))
        labels = self._labels if self._labels is not None else pl.Series([0] * len(features))
        return SignalOutput(scores=scores, labels=labels)


class DummyRiskEngine:
    def process_signals(self, signals, portfolio, market, config):
        # Always return one buy order
        order = OrderRequest(
            symbol="BTC_USDT",
            quantity=Decimal("1"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            timestamp=signals[0].timestamp,
        )
        return type("Res", (), {"orders": [order]})


class DummyConstraint:
    def __init__(self) -> None:
        self.records: list[tuple[str, datetime]] = []

    def record_trade(self, symbol, timestamp, side, quantity) -> None:
        self.records.append((symbol, timestamp))


class DummyBars:
    def __init__(self, bars):
        self._bars = bars

    def get_bars(self, split: slice):
        return self._bars[split]


class DummyPortfolio:
    def __init__(self, ts):
        self._ts = ts

    def get_portfolio(self, split: slice):
        return PortfolioState(cash=Decimal("10000"), positions={}, timestamp=self._ts)


def dummy_bars():
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        Bar(
            symbol="BTC_USDT",
            timestamp=ts,
            open=Decimal("10"),
            high=Decimal("10"),
            low=Decimal("10"),
            close=Decimal("10"),
            volume=Decimal("1"),
        )
        for _ in range(6)
    ]


def simulator_factory():
    cfg = SimulatorConfig()
    provider = ProviderConfig(
        name="test",
        asset_classes=["crypto"],
        fee_model="ZeroCommission",
        slippage_model="VolumeWeighted",
    )
    return Simulator(provider_config=provider, config=cfg)


class DummySimulator:
    def __init__(self) -> None:
        self.config = SimulatorConfig()

    def run(self, orders, bars):
        fill = Fill(
            fill_id=UUID("00000000-0000-0000-0000-000000000001"),
            client_order_id=UUID("00000000-0000-0000-0000-000000000002"),
            symbol=orders[0].symbol,
            side=orders[0].side,
            quantity=orders[0].quantity,
            price=Decimal("10"),
            commission=Decimal("0"),
            timestamp=bars[0].timestamp,
        )
        return SimpleNamespace(
            fills=[fill],
            rejected_orders=[],
            funding_charged=Decimal("0"),
            slippage_stats={},
            missing_ratio=0.0,
            zero_volume_ratio=0.0,
            ohlc_inconsistencies=0,
            extreme_moves=0,
            negative_volume=0,
            non_monotonic_ts=0,
        )


def dummy_simulator_factory():
    return DummySimulator()


class RiskResult:
    """Simple simulation-time risk output with explicit constraints."""

    def __init__(
        self,
        orders,
        *,
        constraints: dict[str, float] | None = None,
        rejected_signals=None,
        sizing_rejections: dict[str, list[str]] | None = None,
        constraint_diagnostics: dict[str, dict[str, object]] | None = None,
    ):
        self.orders = orders
        self.constraint_violations = constraints or {}
        self.constraint_diagnostics = constraint_diagnostics or {}
        self.sizing_rejections = sizing_rejections or {}

        self.rejected_signals = [] if rejected_signals is None else rejected_signals
        self.rejected_orders = []
        self.processed_signals = []

        self.funding_charged = Decimal("0")
        self.slippage_stats = {}
        self.missing_ratio = 0.0
        self.zero_volume_ratio = 0.0
        self.ohlc_inconsistencies = 0
        self.extreme_moves = 0
        self.negative_volume = 0
        self.non_monotonic_ts = 0


class RiskEngineWithConstraintViolation:
    """Risk engine stub that returns deterministic constraint violations."""

    def process_signals(self, signals, portfolio, market, config):  # noqa: ARG001
        order = OrderRequest(
            symbol="BTC_USDT",
            quantity=Decimal("1"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            timestamp=signals[0].timestamp,
        )
        return RiskResult(
            [order],
            constraints={"future_reference": 0.25},
        )


def test_run_rolling_returns_fold_results() -> None:
    features = pl.DataFrame({"f": [1, 2, 3, 4, 5, 6]})
    labels = pl.Series([0, 1, 0, 1, 0, 1])
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    # capture logs
    import logging

    records: list[logging.LogRecord] = []

    class ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = ListHandler()
    logger = logging.getLogger("liq.runner.runner")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    results = run_rolling(
        features=features,
        labels=labels,
        strategy=DummyStrategy(),
        risk_engine=DummyRiskEngine(),
        simulator_factory=simulator_factory,
        bars_provider=DummyBars(dummy_bars()),
        portfolio_provider=DummyPortfolio(ts),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=2,
        step=2,
        sim_config=None,
    )
    assert results
    assert results[0].threshold > 0
    assert isinstance(results[0].metrics, dict)
    assert results[0].slice_id is None
    assert results[0].split_metadata is None
    assert any("rolling_split" in rec.getMessage() for rec in records)
    assert any("fold_result" in rec.getMessage() for rec in records)
    logger.removeHandler(handler)


class NoOrderSimulator:
    def __init__(self) -> None:
        self.config = SimulatorConfig()

    def run(self, orders, bars):
        return SimpleNamespace(
            fills=[],
            rejected_orders=[],
            funding_charged=Decimal("0"),
            slippage_stats={},
            missing_ratio=0.0,
            zero_volume_ratio=0.0,
            ohlc_inconsistencies=0,
            extreme_moves=0,
            negative_volume=0,
            non_monotonic_ts=0,
        )


def no_order_simulator_factory():
    return NoOrderSimulator()


class ThresholdScoreStrategy:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def fit(self, features: pl.DataFrame, labels: pl.Series | None = None) -> None:
        self._labels = labels

    def predict(self, features: pl.DataFrame) -> SignalOutput:
        return SignalOutput(
            scores=pl.Series(self._scores[: len(features)]),
            labels=pl.Series([0] * len(features)),
        )


def test_run_rolling_sends_only_actionable_signals_to_risk() -> None:
    features = pl.DataFrame({"f": [1, 2, 3, 4]})
    labels = pl.Series([0, 1, 0, 1])
    ts = datetime(2024, 1, 1, tzinfo=UTC)

    class RejectingRiskEngine:
        def __init__(self) -> None:
            self.received_directions: list[str] = []

        def process_signals(self, signals, portfolio, market, config):
            self.received_directions = [signal.direction for signal in signals]
            return RiskResult([], constraints={}, rejected_signals=list(signals))

    risk_engine = RejectingRiskEngine()

    results = run_rolling(
        features=features,
        labels=labels,
        strategy=ThresholdScoreStrategy([0.9, 0.1]),
        risk_engine=risk_engine,
        simulator_factory=no_order_simulator_factory,
        bars_provider=DummyBars(dummy_bars()),
        portfolio_provider=DummyPortfolio(ts),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=2,
        step=2,
        sim_config=SimulatorConfig(),
        threshold_cfg={"fixed_threshold": 0.5},
    )

    assert risk_engine.received_directions == ["long"]
    assert results[0].signals_generated == 2
    assert results[0].risk_rejected == 1
    assert results[0].diagnostics["long_signals"] == 1


def test_run_rolling_sends_strongest_actionable_signal_per_symbol_to_risk() -> None:
    features = pl.DataFrame({"f": [1, 2, 3, 4]})
    labels = pl.Series([0, 1, 0, 1])
    ts = datetime(2024, 1, 1, tzinfo=UTC)

    class CapturingRiskEngine:
        def __init__(self) -> None:
            self.received_strengths: list[float] = []

        def process_signals(self, signals, portfolio, market, config):
            self.received_strengths = [signal.strength for signal in signals]
            return RiskResult(
                [],
                constraints={},
                rejected_signals=list(signals),
                sizing_rejections={"VolatilitySizer": ["BTC_USDT: No target produced by sizer"]},
                constraint_diagnostics={
                    "MinPositionValueConstraint": {
                        "rejected_notional_values": [4.0],
                        "min_position_value": 5.0,
                        "count": 1,
                    }
                },
            )

    risk_engine = CapturingRiskEngine()

    results = run_rolling(
        features=features,
        labels=labels,
        strategy=ThresholdScoreStrategy([0.6, 0.9]),
        risk_engine=risk_engine,
        simulator_factory=no_order_simulator_factory,
        bars_provider=DummyBars(dummy_bars()),
        portfolio_provider=DummyPortfolio(ts),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=2,
        step=2,
        sim_config=SimulatorConfig(),
        threshold_cfg={"fixed_threshold": 0.5},
    )

    assert risk_engine.received_strengths == [0.9]
    assert results[0].signals_generated == 2
    assert results[0].risk_rejected == 1
    assert results[0].diagnostics["long_signals"] == 2
    assert results[0].diagnostics["risk_signals"] == 1
    assert results[0].diagnostics["sizing_rejections"] == {
        "VolatilitySizer": ["BTC_USDT: No target produced by sizer"]
    }
    assert results[0].diagnostics["constraint_diagnostics"] == {
        "MinPositionValueConstraint": {
            "rejected_notional_values": [4.0],
            "min_position_value": 5.0,
            "count": 1,
        }
    }


class TestRocAuc:
    def test_perfect_separation_is_one(self) -> None:
        assert _roc_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0

    def test_inverted_separation_is_zero(self) -> None:
        assert _roc_auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0

    def test_all_tied_scores_is_half(self) -> None:
        assert _roc_auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5

    def test_known_partial_value(self) -> None:
        # pos ranks {2,4}: (6 - 2*3/2) / (2*2) = 0.75
        assert _roc_auc([1.0, 2.0, 3.0, 4.0], [0, 1, 0, 1]) == 0.75

    def test_single_class_labels_returns_none(self) -> None:
        assert _roc_auc([0.1, 0.2, 0.3], [1, 1, 1]) is None

    def test_empty_returns_none(self) -> None:
        assert _roc_auc([], []) is None

    def test_length_mismatch_returns_none(self) -> None:
        assert _roc_auc([0.1, 0.2], [1]) is None

    def test_non_binary_labels_returns_none(self) -> None:
        assert _roc_auc([0.1, 0.2, 0.3], [0, 1, 2]) is None

    def test_accepts_polars_series(self) -> None:
        assert _roc_auc(pl.Series([0.1, 0.9]), pl.Series([0, 1])) == 1.0


class TestProtectiveBrackets:
    def _ms(self, close: str = "100", atr: str = "2") -> MarketState:
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        bar = Bar(
            symbol="BTC_USDT",
            timestamp=ts,
            open=Decimal(close),
            high=Decimal(close),
            low=Decimal(close),
            close=Decimal(close),
            volume=Decimal("1"),
        )
        return MarketState(
            current_bars={"BTC_USDT": bar},
            volatility={"BTC_USDT": Decimal(atr)},
            liquidity={},
            timestamp=ts,
        )

    def _order(self, side: OrderSide) -> OrderRequest:
        return OrderRequest(
            symbol="BTC_USDT",
            side=side,
            order_type=OrderType.MARKET,
            quantity=Decimal("1"),
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        )

    def test_long_entry_gets_atr_brackets(self) -> None:
        out = _attach_protective_brackets(
            [self._order(OrderSide.BUY)], self._ms("100", "2"),
            stop_atr_mult=1.5, take_atr_mult=3.0,
        )
        md = out[0].metadata
        assert md is not None
        assert md["stop_loss_price"] == Decimal("97")
        assert md["take_profit_price"] == Decimal("106")

    def test_sell_exit_order_untouched(self) -> None:
        sell = self._order(OrderSide.SELL)
        out = _attach_protective_brackets(
            [sell], self._ms(), stop_atr_mult=1.5, take_atr_mult=3.0
        )
        assert out[0].metadata is None

    def test_disabled_when_mults_unset_is_byte_identical(self) -> None:
        orders = [self._order(OrderSide.BUY)]
        out = _attach_protective_brackets(
            orders, self._ms(), stop_atr_mult=None, take_atr_mult=None
        )
        assert out is orders  # same list object, no work done
        assert out[0].metadata is None

    def test_symbol_missing_from_market_state_is_skipped(self) -> None:
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        empty = MarketState(
            current_bars={}, volatility={}, liquidity={}, timestamp=ts
        )
        out = _attach_protective_brackets(
            [self._order(OrderSide.BUY)], empty,
            stop_atr_mult=1.5, take_atr_mult=3.0,
        )
        assert out[0].metadata is None

    def test_nonpositive_atr_is_skipped(self) -> None:
        out = _attach_protective_brackets(
            [self._order(OrderSide.BUY)], self._ms("100", "0"),
            stop_atr_mult=1.5, take_atr_mult=3.0,
        )
        assert out[0].metadata is None

    def test_per_row_bracket_prices_take_precedence_over_atr_mult(self) -> None:
        # By-construction coherence: when the label has emitted absolute SL/TP
        # prices for (symbol, timestamp), execution MUST attach those exact
        # values rather than recompute from a different ATR formula.
        order = self._order(OrderSide.BUY)
        prices: dict = {
            (order.symbol, order.timestamp): (Decimal("95.5"), Decimal("110.25")),
        }
        out = _attach_protective_brackets(
            [order], self._ms("100", "2"),
            stop_atr_mult=1.5, take_atr_mult=3.0,
            bracket_prices=prices,
        )
        md = out[0].metadata
        assert md is not None
        assert md["stop_loss_price"] == Decimal("95.5")
        assert md["take_profit_price"] == Decimal("110.25")

    def test_per_row_map_falls_back_to_atr_mult_when_key_absent(self) -> None:
        # If the per-row map is provided but does not cover this (symbol,ts),
        # the ATR-mult fallback applies (regression-safe for partial coverage).
        out = _attach_protective_brackets(
            [self._order(OrderSide.BUY)], self._ms("100", "2"),
            stop_atr_mult=1.5, take_atr_mult=3.0,
            bracket_prices={},  # empty map -> fallback
        )
        md = out[0].metadata
        assert md is not None
        assert md["stop_loss_price"] == Decimal("97")
        assert md["take_profit_price"] == Decimal("106")


class TestContinuousRebalanceOrder:
    """Pure rebalance helper for continuous-sizing mode.

    target_qty = (target_size_pct * max_notional) / bar.close
    delta_qty = target_qty - current_qty
    Emit BUY (delta>0) / SELL (delta<0) / None if |delta*close| is inside
    the rebalance band (turnover guard). Long-only: target_size_pct in [0,1].
    """

    def _bar(self) -> Bar:
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        return Bar(
            symbol="BTC_USDT",
            timestamp=ts,
            open=Decimal("100"),
            high=Decimal("100"),
            low=Decimal("100"),
            close=Decimal("100"),
            volume=Decimal("1"),
        )

    def test_increase_target_emits_buy(self) -> None:
        # max_notional=1000, target=0.5 -> target_qty=5; current 0 -> buy 5
        out = _continuous_rebalance_order(
            target_size_pct=0.5,
            current_qty=Decimal("0"),
            bar=self._bar(),
            max_notional=Decimal("1000"),
            rebalance_band_pct=0.10,
        )
        assert out is not None
        assert out.side == OrderSide.BUY
        assert out.quantity == Decimal("5")
        assert out.symbol == "BTC_USDT"

    def test_decrease_target_emits_sell(self) -> None:
        # target 0.2 -> target_qty=2; current 5 -> sell 3
        out = _continuous_rebalance_order(
            target_size_pct=0.2,
            current_qty=Decimal("5"),
            bar=self._bar(),
            max_notional=Decimal("1000"),
            rebalance_band_pct=0.0,
        )
        assert out is not None
        assert out.side == OrderSide.SELL
        assert out.quantity == Decimal("3")

    def test_target_equal_to_current_emits_nothing(self) -> None:
        out = _continuous_rebalance_order(
            target_size_pct=0.5,
            current_qty=Decimal("5"),  # already at target (5/100 = 0.5 of 1000)
            bar=self._bar(),
            max_notional=Decimal("1000"),
            rebalance_band_pct=0.0,
        )
        assert out is None

    def test_inside_band_emits_nothing(self) -> None:
        # delta_notional = 1 * 100 = $100; band = 0.10 * 1000 = $100 -> inside (strictly less than would emit; equal-to-band -> no-op).
        out = _continuous_rebalance_order(
            target_size_pct=0.5,
            current_qty=Decimal("4"),  # 4/100*100=$400 current; target $500; delta $100
            bar=self._bar(),
            max_notional=Decimal("1000"),
            rebalance_band_pct=0.10,
        )
        assert out is None

    def test_outside_band_emits(self) -> None:
        # delta_notional $150 > band $100 -> emit
        out = _continuous_rebalance_order(
            target_size_pct=0.5,
            current_qty=Decimal("3.5"),  # current $350; target $500; delta $150
            bar=self._bar(),
            max_notional=Decimal("1000"),
            rebalance_band_pct=0.10,
        )
        assert out is not None
        assert out.side == OrderSide.BUY
        assert out.quantity == Decimal("1.5")

    def test_zero_target_with_positive_position_emits_full_sell(self) -> None:
        out = _continuous_rebalance_order(
            target_size_pct=0.0,
            current_qty=Decimal("5"),
            bar=self._bar(),
            max_notional=Decimal("1000"),
            rebalance_band_pct=0.0,
        )
        assert out is not None
        assert out.side == OrderSide.SELL
        assert out.quantity == Decimal("5")

    def test_clips_target_above_one(self) -> None:
        # target>1 should clip to 1 (long-only cap).
        out = _continuous_rebalance_order(
            target_size_pct=2.0,
            current_qty=Decimal("0"),
            bar=self._bar(),
            max_notional=Decimal("1000"),
            rebalance_band_pct=0.0,
        )
        assert out is not None
        assert out.quantity == Decimal("10")  # full max_notional

    def test_clips_target_below_zero(self) -> None:
        # target<0 should clip to 0 (no shorts).
        out = _continuous_rebalance_order(
            target_size_pct=-0.5,
            current_qty=Decimal("5"),
            bar=self._bar(),
            max_notional=Decimal("1000"),
            rebalance_band_pct=0.0,
        )
        assert out is not None
        assert out.side == OrderSide.SELL
        assert out.quantity == Decimal("5")  # close to zero


def test_continuous_sizing_rank_normalize_emits_orders_when_scores_all_below_half() -> None:
    """Rank-normalize sizing is unit-invariant: it monetizes the *ranking*,
    not the raw probability scale. With all scores in [0.07, 0.22] (real TB
    case), linear sizing emits zero orders; rank-normalize sizes by
    within-window quantile so each bar gets a positive target proportional
    to its rank.
    """
    features = pl.DataFrame({"f": [1, 2, 3, 4, 5, 6]})
    labels = pl.Series([0, 1, 0, 1, 0, 1])
    ts = datetime(2024, 1, 1, tzinfo=UTC)

    class RecordingSim:
        def __init__(self) -> None:
            self.config = SimulatorConfig()
            self.recorded_orders: list[OrderRequest] = []

        def run(self, orders, bars):  # noqa: ARG002
            self.recorded_orders = list(orders)
            return SimpleNamespace(
                fills=[], rejected_orders=[], funding_charged=Decimal("0"),
                slippage_stats={}, missing_ratio=0.0, zero_volume_ratio=0.0,
                ohlc_inconsistencies=0, extreme_moves=0, negative_volume=0,
                non_monotonic_ts=0,
            )

    sim_instance = RecordingSim()

    run_rolling(
        features=features,
        labels=labels,
        # 4 test bars; all scores < 0.5 (linear would emit zero orders).
        strategy=ThresholdScoreStrategy([0.10, 0.20, 0.15, 0.25]),
        risk_engine=DummyRiskEngine(),
        simulator_factory=lambda: sim_instance,
        bars_provider=DummyBars(dummy_bars()),
        portfolio_provider=DummyPortfolio(ts),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=4,
        step=4,
        sim_config=SimulatorConfig(),
        sizing_mode="continuous",
        sizing_fn="rank_normalize",
        max_notional_per_symbol=Decimal("1000"),
        rebalance_band_pct=0.0,
    )

    # Rank-normalize: scores [0.10, 0.20, 0.15, 0.25] -> ranks 1,3,2,4 of 4.
    # rank_pct = (lo + hi)/(2*n) for each (no ties): 1/4, 3/4, 2/4, 4/4
    # = [0.25, 0.75, 0.50, 1.00].
    # Targets (max_notional=$1000, close=$10): qty = [25, 75, 50, 100].
    # Rebalances vs tracked starting at 0: BUY 25, BUY 50 (75-25), SELL 25
    # (50-75), BUY 50 (100-50).
    sides_qty = [(o.side, o.quantity) for o in sim_instance.recorded_orders]
    assert sides_qty == [
        (OrderSide.BUY, Decimal("25")),
        (OrderSide.BUY, Decimal("50")),
        (OrderSide.SELL, Decimal("25")),
        (OrderSide.BUY, Decimal("50")),
    ]


def test_run_rolling_continuous_sizing_emits_per_bar_rebalance_orders() -> None:
    """Continuous-sizing mode: per-bar rebalance against raw scores.

    Bypasses the entry-mode threshold/risk-engine path; each test bar produces
    a target = clip(2*(score-0.5), 0, 1) of max_notional, and the rebalance
    helper emits BUY/SELL deltas vs the running tracked position.
    """
    features = pl.DataFrame({"f": [1, 2, 3, 4, 5, 6]})
    labels = pl.Series([0, 1, 0, 1, 0, 1])
    ts = datetime(2024, 1, 1, tzinfo=UTC)

    class RecordingSim:
        def __init__(self) -> None:
            self.config = SimulatorConfig()
            self.recorded_orders: list[OrderRequest] = []

        def run(self, orders, bars):  # noqa: ARG002
            self.recorded_orders = list(orders)
            return SimpleNamespace(
                fills=[],
                rejected_orders=[],
                funding_charged=Decimal("0"),
                slippage_stats={},
                missing_ratio=0.0,
                zero_volume_ratio=0.0,
                ohlc_inconsistencies=0,
                extreme_moves=0,
                negative_volume=0,
                non_monotonic_ts=0,
            )

    sim_instance = RecordingSim()

    results = run_rolling(
        features=features,
        labels=labels,
        # Scores -> target_pct = [0.2, 0.8, 0, 0] on 4 test bars (price 10):
        # rebalances: BUY 20, BUY 60, SELL 80 (3rd target=0, current=80),
        # no-op (target=0, current=0).
        strategy=ThresholdScoreStrategy([0.6, 0.9, 0.3, 0.4]),
        risk_engine=DummyRiskEngine(),
        simulator_factory=lambda: sim_instance,
        bars_provider=DummyBars(dummy_bars()),
        portfolio_provider=DummyPortfolio(ts),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=4,
        step=4,
        sim_config=SimulatorConfig(),
        sizing_mode="continuous",
        max_notional_per_symbol=Decimal("1000"),
        rebalance_band_pct=0.0,
    )

    assert [(o.side, o.quantity) for o in sim_instance.recorded_orders] == [
        (OrderSide.BUY, Decimal("20")),
        (OrderSide.BUY, Decimal("60")),
        (OrderSide.SELL, Decimal("80")),
    ]
    # IC diagnostic recorded (constant-price test bars => None is fine).
    assert results
    assert "oos_ic" in results[0].diagnostics


def test_run_rolling_records_oos_auc_in_diagnostics() -> None:
    features = pl.DataFrame({"f": [1, 2, 3, 4, 5, 6]})
    labels = pl.Series([0, 1, 0, 1, 0, 1])
    ts = datetime(2024, 1, 1, tzinfo=UTC)

    results = run_rolling(
        features=features,
        labels=labels,
        strategy=ThresholdScoreStrategy([0.2, 0.8, 0.3, 0.7, 0.1, 0.9]),
        risk_engine=DummyRiskEngine(),
        simulator_factory=no_order_simulator_factory,
        bars_provider=DummyBars(dummy_bars()),
        portfolio_provider=DummyPortfolio(ts),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=2,
        step=2,
        sim_config=SimulatorConfig(),
        threshold_cfg={"fixed_threshold": 0.5},
    )

    assert results
    for r in results:
        assert "oos_auc" in r.diagnostics
        auc = r.diagnostics["oos_auc"]
        assert auc is None or (isinstance(auc, float) and 0.0 <= auc <= 1.0)


def test_run_rolling_builds_per_symbol_market_state_for_multi_symbol_panel() -> None:
    """Multi-symbol panel must yield a per-symbol MarketState.

    Panel rows are ordered (timestamp, symbol). The risk engine must receive a
    market snapshot keyed by every symbol present, each with its own ATR-proxy
    volatility (not one symbol's bar standing in for the whole book).
    """
    ts0 = datetime(2024, 1, 1, tzinfo=UTC)

    def _bar(sym: str, i: int, vary: bool) -> Bar:
        # AAA flat (vol -> 0.01 fallback); BBB varies (vol > 0.01).
        hi = Decimal("11") if vary else Decimal("10")
        lo = Decimal("9") if vary else Decimal("10")
        return Bar(
            symbol=sym,
            timestamp=ts0.replace(hour=i),
            open=Decimal("10"),
            high=hi,
            low=lo,
            close=Decimal("10"),
            volume=Decimal("1"),
        )

    # 6 timestamps x 2 symbols, canonical (timestamp, symbol) order, so each
    # fold's test window holds >=2 bars/symbol (enough for the ATR proxy).
    bars = []
    for i in range(6):
        bars.append(_bar("AAA", i, vary=False))
        bars.append(_bar("BBB", i, vary=True))

    features = pl.DataFrame({"f": list(range(12))})
    labels = pl.Series([0, 1] * 6)

    class CapturingRiskEngine:
        def __init__(self) -> None:
            self.current_bar_symbols: set[str] = set()
            self.volatility_symbols: set[str] = set()
            self.vols: dict[str, object] = {}

        def process_signals(self, signals, portfolio, market, config):  # noqa: ARG002
            self.current_bar_symbols = set(market.current_bars.keys())
            self.volatility_symbols = set(market.volatility.keys())
            self.vols = dict(market.volatility)
            return RiskResult([], constraints={}, rejected_signals=list(signals))

    risk_engine = CapturingRiskEngine()

    run_rolling(
        features=features,
        labels=labels,
        strategy=ThresholdScoreStrategy([0.9] * 12),
        risk_engine=risk_engine,
        simulator_factory=no_order_simulator_factory,
        bars_provider=DummyBars(bars),
        portfolio_provider=DummyPortfolio(ts0),
        risk_config=RiskConfig(),
        train_size=4,
        valid_size=4,
        step=4,
        sim_config=SimulatorConfig(),
        threshold_cfg={"fixed_threshold": 0.5},
    )

    assert risk_engine.current_bar_symbols == {"AAA", "BBB"}
    assert risk_engine.volatility_symbols == {"AAA", "BBB"}
    # Per-symbol volatility: flat AAA hits the 0.01 fallback, varying BBB does not.
    assert risk_engine.vols["AAA"] != risk_engine.vols["BBB"]


def test_run_rolling_adds_exit_orders_on_flat_signal_after_entry() -> None:
    features = pl.DataFrame({"f": [1, 2, 3, 4, 5, 6]})
    labels = pl.Series([0, 1, 0, 1, 0, 1])
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    bars = [
        Bar(
            symbol="BTC_USDT",
            timestamp=ts.replace(hour=i),
            open=Decimal("10"),
            high=Decimal("10"),
            low=Decimal("10"),
            close=Decimal("10"),
            volume=Decimal("1"),
        )
        for i in range(6)
    ]

    class OneOrderPerSignalRiskEngine:
        def process_signals(self, signals, portfolio, market, config):
            orders = [
                OrderRequest(
                    symbol=signal.symbol,
                    quantity=Decimal("1"),
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    timestamp=ts,
                    confidence=signal.strength,
                )
                for signal in signals
            ]
            return RiskResult(orders, constraints={})

    class CapturingSimulator:
        def __init__(self) -> None:
            self.config = SimulatorConfig()
            self.orders: list[OrderRequest] = []

        def run(self, orders, bars):
            self.orders = list(orders)
            return SimpleNamespace(
                fills=[],
                rejected_orders=[],
                funding_charged=Decimal("0"),
                slippage_stats={},
                missing_ratio=0.0,
                zero_volume_ratio=0.0,
                ohlc_inconsistencies=0,
                extreme_moves=0,
                negative_volume=0,
                non_monotonic_ts=0,
            )

    simulator = CapturingSimulator()

    results = run_rolling(
        features=features,
        labels=labels,
        strategy=ThresholdScoreStrategy([0.9, 0.1, 0.8, 0.2]),
        risk_engine=OneOrderPerSignalRiskEngine(),
        simulator_factory=lambda: simulator,
        bars_provider=DummyBars(bars),
        portfolio_provider=DummyPortfolio(ts),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=4,
        step=4,
        sim_config=SimulatorConfig(),
        threshold_cfg={"fixed_threshold": 0.5, "max_signals_per_symbol": None},
    )

    assert results[0].diagnostics["long_signals"] == 2
    assert [order.side for order in simulator.orders] == [
        OrderSide.BUY,
        OrderSide.SELL,
        OrderSide.BUY,
        OrderSide.SELL,
    ]
    assert [order.timestamp for order in simulator.orders] == [
        bars[2].timestamp,
        bars[3].timestamp,
        bars[4].timestamp,
        bars[5].timestamp,
    ]


def test_run_rolling_exit_threshold_delays_close_until_score_breaks_lower_band() -> None:
    features = pl.DataFrame({"f": [1, 2, 3, 4, 5]})
    labels = pl.Series([0, 1, 0, 1, 0])
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    bars = [
        Bar(
            symbol="BTC_USDT",
            timestamp=ts.replace(hour=i),
            open=Decimal("10"),
            high=Decimal("10"),
            low=Decimal("10"),
            close=Decimal("10"),
            volume=Decimal("1"),
        )
        for i in range(5)
    ]

    class OneOrderRiskEngine:
        def process_signals(self, signals, portfolio, market, config):
            return RiskResult(
                [
                    OrderRequest(
                        symbol=signals[0].symbol,
                        quantity=Decimal("1"),
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        timestamp=ts,
                        confidence=signals[0].strength,
                    )
                ],
                constraints={},
            )

    class CapturingSimulator:
        def __init__(self) -> None:
            self.config = SimulatorConfig()
            self.orders: list[OrderRequest] = []

        def run(self, orders, bars):
            self.orders = list(orders)
            return SimpleNamespace(
                fills=[],
                rejected_orders=[],
                funding_charged=Decimal("0"),
                slippage_stats={},
                missing_ratio=0.0,
                zero_volume_ratio=0.0,
                ohlc_inconsistencies=0,
                extreme_moves=0,
                negative_volume=0,
                non_monotonic_ts=0,
            )

    simulator = CapturingSimulator()

    run_rolling(
        features=features,
        labels=labels,
        strategy=ThresholdScoreStrategy([0.9, 0.4, 0.2]),
        risk_engine=OneOrderRiskEngine(),
        simulator_factory=lambda: simulator,
        bars_provider=DummyBars(bars),
        portfolio_provider=DummyPortfolio(ts),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=3,
        step=3,
        sim_config=SimulatorConfig(),
        threshold_cfg={"fixed_threshold": 0.5, "exit_threshold": 0.3},
    )

    assert [order.side for order in simulator.orders] == [OrderSide.BUY, OrderSide.SELL]
    assert [order.timestamp for order in simulator.orders] == [bars[2].timestamp, bars[4].timestamp]


def test_run_rolling_min_hold_bars_delays_close_after_entry() -> None:
    features = pl.DataFrame({"f": [1, 2, 3, 4, 5]})
    labels = pl.Series([0, 1, 0, 1, 0])
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    bars = [
        Bar(
            symbol="BTC_USDT",
            timestamp=ts.replace(hour=i),
            open=Decimal("10"),
            high=Decimal("10"),
            low=Decimal("10"),
            close=Decimal("10"),
            volume=Decimal("1"),
        )
        for i in range(5)
    ]

    class OneOrderRiskEngine:
        def process_signals(self, signals, portfolio, market, config):
            return RiskResult(
                [
                    OrderRequest(
                        symbol=signals[0].symbol,
                        quantity=Decimal("1"),
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        timestamp=ts,
                        confidence=signals[0].strength,
                    )
                ],
                constraints={},
            )

    class CapturingSimulator:
        def __init__(self) -> None:
            self.config = SimulatorConfig()
            self.orders: list[OrderRequest] = []

        def run(self, orders, bars):
            self.orders = list(orders)
            return SimpleNamespace(
                fills=[],
                rejected_orders=[],
                funding_charged=Decimal("0"),
                slippage_stats={},
                missing_ratio=0.0,
                zero_volume_ratio=0.0,
                ohlc_inconsistencies=0,
                extreme_moves=0,
                negative_volume=0,
                non_monotonic_ts=0,
            )

    simulator = CapturingSimulator()

    run_rolling(
        features=features,
        labels=labels,
        strategy=ThresholdScoreStrategy([0.9, 0.1, 0.1]),
        risk_engine=OneOrderRiskEngine(),
        simulator_factory=lambda: simulator,
        bars_provider=DummyBars(bars),
        portfolio_provider=DummyPortfolio(ts),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=3,
        step=3,
        sim_config=SimulatorConfig(),
        threshold_cfg={"fixed_threshold": 0.5, "min_hold_bars": 2},
    )

    assert [order.side for order in simulator.orders] == [OrderSide.BUY, OrderSide.SELL]
    assert [order.timestamp for order in simulator.orders] == [bars[2].timestamp, bars[4].timestamp]


def test_run_rolling_exit_hysteresis_delays_close_until_score_breaks_lower_band() -> None:
    features = pl.DataFrame({"f": [1, 2, 3, 4, 5]})
    labels = pl.Series([0, 1, 0, 1, 0])
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    bars = [
        Bar(
            symbol="BTC_USDT",
            timestamp=ts.replace(hour=i),
            open=Decimal("10"),
            high=Decimal("10"),
            low=Decimal("10"),
            close=Decimal("10"),
            volume=Decimal("1"),
        )
        for i in range(5)
    ]

    class OneOrderRiskEngine:
        def process_signals(self, signals, portfolio, market, config):
            return RiskResult(
                [
                    OrderRequest(
                        symbol=signals[0].symbol,
                        quantity=Decimal("1"),
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        timestamp=ts,
                        confidence=signals[0].strength,
                    )
                ],
                constraints={},
            )

    class CapturingSimulator:
        def __init__(self) -> None:
            self.config = SimulatorConfig()
            self.orders: list[OrderRequest] = []

        def run(self, orders, bars):
            self.orders = list(orders)
            return SimpleNamespace(
                fills=[],
                rejected_orders=[],
                funding_charged=Decimal("0"),
                slippage_stats={},
                missing_ratio=0.0,
                zero_volume_ratio=0.0,
                ohlc_inconsistencies=0,
                extreme_moves=0,
                negative_volume=0,
                non_monotonic_ts=0,
            )

    simulator = CapturingSimulator()

    run_rolling(
        features=features,
        labels=labels,
        strategy=ThresholdScoreStrategy([0.9, 0.46, 0.39]),
        risk_engine=OneOrderRiskEngine(),
        simulator_factory=lambda: simulator,
        bars_provider=DummyBars(bars),
        portfolio_provider=DummyPortfolio(ts),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=3,
        step=3,
        sim_config=SimulatorConfig(),
        threshold_cfg={"fixed_threshold": 0.5, "exit_hysteresis": 0.1},
    )

    assert [order.side for order in simulator.orders] == [OrderSide.BUY, OrderSide.SELL]
    assert [order.timestamp for order in simulator.orders] == [bars[2].timestamp, bars[4].timestamp]


def test_run_rolling_max_hold_bars_closes_without_flat_signal() -> None:
    features = pl.DataFrame({"f": [1, 2, 3, 4, 5]})
    labels = pl.Series([0, 1, 0, 1, 0])
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    bars = [
        Bar(
            symbol="BTC_USDT",
            timestamp=ts.replace(hour=i),
            open=Decimal("10"),
            high=Decimal("10"),
            low=Decimal("10"),
            close=Decimal("10"),
            volume=Decimal("1"),
        )
        for i in range(5)
    ]

    class OneOrderRiskEngine:
        def process_signals(self, signals, portfolio, market, config):
            return RiskResult(
                [
                    OrderRequest(
                        symbol=signals[0].symbol,
                        quantity=Decimal("1"),
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        timestamp=ts,
                        confidence=signals[0].strength,
                    )
                ],
                constraints={},
            )

    class CapturingSimulator:
        def __init__(self) -> None:
            self.config = SimulatorConfig()
            self.orders: list[OrderRequest] = []

        def run(self, orders, bars):
            self.orders = list(orders)
            return SimpleNamespace(
                fills=[],
                rejected_orders=[],
                funding_charged=Decimal("0"),
                slippage_stats={},
                missing_ratio=0.0,
                zero_volume_ratio=0.0,
                ohlc_inconsistencies=0,
                extreme_moves=0,
                negative_volume=0,
                non_monotonic_ts=0,
            )

    simulator = CapturingSimulator()

    run_rolling(
        features=features,
        labels=labels,
        strategy=ThresholdScoreStrategy([0.9, 0.8, 0.7]),
        risk_engine=OneOrderRiskEngine(),
        simulator_factory=lambda: simulator,
        bars_provider=DummyBars(bars),
        portfolio_provider=DummyPortfolio(ts),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=3,
        step=3,
        sim_config=SimulatorConfig(),
        threshold_cfg={"fixed_threshold": 0.5, "max_hold_bars": 2},
    )

    assert [order.side for order in simulator.orders] == [OrderSide.BUY, OrderSide.SELL]
    assert [order.timestamp for order in simulator.orders] == [bars[2].timestamp, bars[4].timestamp]


def test_run_rolling_rejects_invalid_exit_policy_config() -> None:
    with pytest.raises(ValueError, match="threshold_cfg.exit_hysteresis must be >= 0"):
        run_rolling(
            features=pl.DataFrame({"f": [1, 2, 3, 4]}),
            labels=pl.Series([0, 1, 0, 1]),
            strategy=ThresholdScoreStrategy([0.9, 0.1]),
            risk_engine=DummyRiskEngine(),
            simulator_factory=no_order_simulator_factory,
            bars_provider=DummyBars(dummy_bars()),
            portfolio_provider=DummyPortfolio(datetime(2024, 1, 1, tzinfo=UTC)),
            risk_config=RiskConfig(),
            train_size=2,
            valid_size=2,
            step=2,
            threshold_cfg={"fixed_threshold": 0.5, "exit_hysteresis": -0.1},
        )


def test_run_rolling_rejects_invalid_entry_policy_config() -> None:
    base_kwargs = {
        "features": pl.DataFrame({"f": [1, 2, 3, 4]}),
        "labels": pl.Series([0, 1, 0, 1]),
        "strategy": ThresholdScoreStrategy([0.9, 0.1]),
        "risk_engine": DummyRiskEngine(),
        "simulator_factory": no_order_simulator_factory,
        "bars_provider": DummyBars(dummy_bars()),
        "portfolio_provider": DummyPortfolio(datetime(2024, 1, 1, tzinfo=UTC)),
        "risk_config": RiskConfig(),
        "train_size": 2,
        "valid_size": 2,
        "step": 2,
    }

    with pytest.raises(ValueError, match="threshold_cfg.entry_threshold_margin must be >= 0"):
        run_rolling(
            **base_kwargs,
            threshold_cfg={"fixed_threshold": 0.5, "entry_threshold_margin": -0.1},
        )

    with pytest.raises(ValueError, match="threshold_cfg.min_entry_spacing_bars must be >= 0"):
        run_rolling(
            **base_kwargs,
            threshold_cfg={"fixed_threshold": 0.5, "min_entry_spacing_bars": -1},
        )


def test_run_rolling_entry_threshold_margin_filters_weak_edge_signals() -> None:
    features = pl.DataFrame({"f": [1, 2, 3, 4]})
    labels = pl.Series([0, 1, 0, 1])
    ts = datetime(2024, 1, 1, tzinfo=UTC)

    class CapturingRiskEngine:
        def __init__(self) -> None:
            self.received_strengths: list[float] = []

        def process_signals(self, signals, portfolio, market, config):
            self.received_strengths = [signal.strength for signal in signals]
            return RiskResult([], constraints={}, rejected_signals=list(signals))

    risk_engine = CapturingRiskEngine()

    results = run_rolling(
        features=features,
        labels=labels,
        strategy=ThresholdScoreStrategy([0.52, 0.56]),
        risk_engine=risk_engine,
        simulator_factory=no_order_simulator_factory,
        bars_provider=DummyBars(dummy_bars()),
        portfolio_provider=DummyPortfolio(ts),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=2,
        step=2,
        sim_config=SimulatorConfig(),
        threshold_cfg={
            "fixed_threshold": 0.5,
            "entry_threshold_margin": 0.05,
            "max_signals_per_symbol": None,
        },
    )

    assert risk_engine.received_strengths == [0.56]
    assert results[0].diagnostics["long_signals"] == 1
    assert results[0].diagnostics["risk_signals"] == 1


def test_run_rolling_min_entry_spacing_bars_reduces_same_symbol_churn() -> None:
    features = pl.DataFrame({"f": [1, 2, 3, 4, 5, 6]})
    labels = pl.Series([0, 1, 0, 1, 0, 1])
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    bars = [
        Bar(
            symbol="BTC_USDT",
            timestamp=ts.replace(hour=i),
            open=Decimal("10"),
            high=Decimal("10"),
            low=Decimal("10"),
            close=Decimal("10"),
            volume=Decimal("1"),
        )
        for i in range(6)
    ]

    class CapturingRiskEngine:
        def __init__(self) -> None:
            self.received_timestamps: list[datetime] = []

        def process_signals(self, signals, portfolio, market, config):
            self.received_timestamps = [signal.timestamp for signal in signals]
            return RiskResult([], constraints={}, rejected_signals=list(signals))

    risk_engine = CapturingRiskEngine()

    results = run_rolling(
        features=features,
        labels=labels,
        strategy=ThresholdScoreStrategy([0.9, 0.8, 0.7, 0.6]),
        risk_engine=risk_engine,
        simulator_factory=no_order_simulator_factory,
        bars_provider=DummyBars(bars),
        portfolio_provider=DummyPortfolio(ts),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=4,
        step=4,
        sim_config=SimulatorConfig(),
        threshold_cfg={
            "fixed_threshold": 0.5,
            "max_signals_per_symbol": None,
            "min_entry_spacing_bars": 2,
        },
    )

    assert risk_engine.received_timestamps == [bars[2].timestamp, bars[4].timestamp]
    assert results[0].diagnostics["long_signals"] == 2
    assert results[0].diagnostics["risk_signals"] == 2


def test_run_rolling_can_send_multiple_actionable_signals_per_symbol_to_risk() -> None:
    features = pl.DataFrame({"f": [1, 2, 3, 4]})
    labels = pl.Series([0, 1, 0, 1])
    ts = datetime(2024, 1, 1, tzinfo=UTC)

    class CapturingRiskEngine:
        def __init__(self) -> None:
            self.received_strengths: list[float] = []

        def process_signals(self, signals, portfolio, market, config):
            self.received_strengths = [signal.strength for signal in signals]
            return RiskResult([], constraints={}, rejected_signals=list(signals))

    risk_engine = CapturingRiskEngine()

    results = run_rolling(
        features=features,
        labels=labels,
        strategy=ThresholdScoreStrategy([0.6, 0.9]),
        risk_engine=risk_engine,
        simulator_factory=no_order_simulator_factory,
        bars_provider=DummyBars(dummy_bars()),
        portfolio_provider=DummyPortfolio(ts),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=2,
        step=2,
        sim_config=SimulatorConfig(),
        threshold_cfg={"fixed_threshold": 0.5, "max_signals_per_symbol": 2},
    )

    assert risk_engine.received_strengths == [0.9, 0.6]
    assert results[0].signals_generated == 2
    assert results[0].risk_rejected == 2
    assert results[0].diagnostics["long_signals"] == 2
    assert results[0].diagnostics["risk_signals"] == 2



    features = pl.DataFrame({"f": [1, 2, 3, 4]})
    labels = pl.Series([0, 1, 0, 1])
    ts = datetime(2024, 1, 1, tzinfo=UTC)

    class FailingRiskEngine:
        def process_signals(self, signals, portfolio, market, config):
            raise AssertionError("flat-only windows should not call risk")

    results = run_rolling(
        features=features,
        labels=labels,
        strategy=ThresholdScoreStrategy([0.1, 0.2]),
        risk_engine=FailingRiskEngine(),
        simulator_factory=no_order_simulator_factory,
        bars_provider=DummyBars(dummy_bars()),
        portfolio_provider=DummyPortfolio(ts),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=2,
        step=2,
        sim_config=SimulatorConfig(),
        threshold_cfg={"fixed_threshold": 0.5},
    )

    assert results[0].signals_generated == 2
    assert results[0].orders_submitted == 0
    assert results[0].risk_rejected == 0
    assert results[0].diagnostics["long_signals"] == 0


def test_run_rolling_dynamic_quantile_grid_uses_score_candidates() -> None:
    features = pl.DataFrame({"f": [1, 2, 3, 4]})
    labels = pl.Series([0, 1, 0, 1])
    ts = datetime(2024, 1, 1, tzinfo=UTC)

    class QuantileGridStrategy:
        def fit(self, features: pl.DataFrame, labels: pl.Series | None = None) -> None:
            pass

        def predict(self, features: pl.DataFrame) -> SignalOutput:
            return SignalOutput(
                scores=pl.Series([0.2, 0.8][: len(features)]),
                labels=pl.Series([1] * len(features)),
            )

    results = run_rolling(
        features=features,
        labels=labels,
        strategy=QuantileGridStrategy(),
        risk_engine=DummyRiskEngine(),
        simulator_factory=no_order_simulator_factory,
        bars_provider=DummyBars(dummy_bars()),
        portfolio_provider=DummyPortfolio(ts),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=2,
        step=2,
        sim_config=SimulatorConfig(),
        threshold_cfg={
            "threshold_grid_mode": "quantile",
            "threshold_grid_quantiles": [1.0],
            "min_trades_per_window": 1,
        },
    )

    # Strategy emits single-class labels, so calibration is unidentifiable and
    # correctly returns identity (T=1.0); the quantile-1.0 grid candidate is
    # therefore the raw max score (0.8), not an artificially spread value.
    assert results[0].threshold == pytest.approx(0.8)
    assert results[0].diagnostics["trades"] == 1


def test_run_rolling_threshold_grid_restricts_search_candidates() -> None:
    features = pl.DataFrame({"f": [1, 2, 3, 4]})
    labels = pl.Series([0, 1, 0, 1])
    ts = datetime(2024, 1, 1, tzinfo=UTC)

    class GridStrategy:
        def fit(self, features: pl.DataFrame, labels: pl.Series | None = None) -> None:
            pass

        def predict(self, features: pl.DataFrame) -> SignalOutput:
            return SignalOutput(
                scores=pl.Series([0.9, 0.8][: len(features)]),
                labels=pl.Series([1] * len(features)),
            )

    results = run_rolling(
        features=features,
        labels=labels,
        strategy=GridStrategy(),
        risk_engine=DummyRiskEngine(),
        simulator_factory=dummy_simulator_factory,
        bars_provider=DummyBars(dummy_bars()),
        portfolio_provider=DummyPortfolio(ts),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=2,
        step=2,
        sim_config=SimulatorConfig(),
        threshold_cfg={"threshold_grid": [0.5], "min_trades_per_window": 1},
    )

    assert results[0].threshold == 0.5
    assert results[0].diagnostics["precision"] == 1.0
    assert results[0].diagnostics["trades"] == 2


def test_run_rolling_with_calibration_and_constraints() -> None:
    features = pl.DataFrame({"f": [1, 2, 3, 4]})
    labels = pl.Series([0, 1, 0, 1])
    ts = datetime(2024, 1, 1, tzinfo=UTC)

    class CalibStrategy:
        def fit(self, features: pl.DataFrame, labels: pl.Series | None = None) -> None:
            self._labels = labels

        def predict(self, features: pl.DataFrame) -> SignalOutput:
            scores = pl.Series([0.9] * len(features))
            labels = pl.Series([0] * len(features))
            return SignalOutput(scores=scores, labels=labels)

    class RiskEngineWithConstraints(DummyRiskEngine):
        def __init__(self) -> None:
            super().__init__()
            self.constraint = DummyConstraint()

        def _get_constraints(self):  # noqa: SLF001 - intentionally used in runner
            return [self.constraint]

    results = run_rolling(
        features=features,
        labels=labels,
        strategy=CalibStrategy(),
        risk_engine=RiskEngineWithConstraints(),
        simulator_factory=dummy_simulator_factory,
        bars_provider=DummyBars(dummy_bars()),
        portfolio_provider=DummyPortfolio(ts),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=2,
        step=2,
        sim_config=SimulatorConfig(),
        threshold_cfg={"top_signals": 1},
        calibration_split=0.1,
    )
    assert results
    diagnostics = results[0].diagnostics
    assert diagnostics["calibration_raw_scores"]["count"] == 1
    assert diagnostics["calibration_calibrated_scores"]["count"] == 1
    assert diagnostics["test_raw_scores"]["count"] == 2
    assert diagnostics["test_calibrated_scores"]["count"] == 2
    assert diagnostics["test_calibrated_scores"]["min"] >= 0.0
    assert diagnostics["test_calibrated_scores"]["max"] <= 1.0


def test_run_rolling_uses_raw_scores_for_strength_not_saturated_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ranking/strength must use raw model scores, not calibrated probabilities.

    Temperature-scaling calibration can saturate into a near-step function
    (collapsing the model's raw AUC ordering). Calibrated probabilities are
    reserved for the EV-threshold GO/NO-GO decision; signal ``strength`` and
    the per-symbol / top-K selection that depends on it must keep the raw
    ordering so the genuinely strongest bar is the one that survives.
    """
    import liq.runner.runner as runner_mod

    def _saturating(scores: pl.Series, temp: float) -> pl.Series:  # noqa: ARG001
        # Hard step: any in-the-money raw score collapses to the same value,
        # destroying the raw ordering — the saturation failure mode.
        return pl.Series([0.99 if float(s) >= 0.5 else 0.01 for s in scores])

    monkeypatch.setattr(runner_mod, "apply_temperature_scale", _saturating)

    features = pl.DataFrame({"f": [1, 2, 3, 4]})
    labels = pl.Series([0, 1, 0, 1])
    ts = datetime(2024, 1, 1, tzinfo=UTC)

    class RawOrderingStrategy:
        def fit(self, features: pl.DataFrame, labels: pl.Series | None = None) -> None:
            self._labels = labels

        def predict(self, features: pl.DataFrame) -> SignalOutput:
            scores = pl.Series([0.55, 0.95][: len(features)])
            return SignalOutput(scores=scores, labels=pl.Series([0] * len(features)))

    class CapturingRiskEngine:
        def __init__(self) -> None:
            self.received_strengths: list[float] = []

        def process_signals(self, signals, portfolio, market, config):  # noqa: ARG002
            self.received_strengths = [signal.strength for signal in signals]
            return RiskResult([], constraints={}, rejected_signals=list(signals))

    risk_engine = CapturingRiskEngine()

    run_rolling(
        features=features,
        labels=labels,
        strategy=RawOrderingStrategy(),
        risk_engine=risk_engine,
        simulator_factory=no_order_simulator_factory,
        bars_provider=DummyBars(dummy_bars()),
        portfolio_provider=DummyPortfolio(ts),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=2,
        step=2,
        sim_config=SimulatorConfig(),
        threshold_cfg={"fixed_threshold": 0.5},
        calibration_split=0.1,
    )

    # Both bars clear the calibrated GO/NO-GO gate (0.99 >= 0.5), so both are
    # "long". Per-symbol top-1 then keeps the strongest by strength. With raw
    # ordering preserved that is the 0.95 bar; if strength used the saturated
    # calibrated score (0.99/0.99) the ordering would be lost and the weaker
    # raw bar (0.55) could win instead.
    assert risk_engine.received_strengths == [0.95]


def test_run_rolling_without_calibration_split_records_score_diagnostics() -> None:
    features = pl.DataFrame({"f": [1, 2, 3, 4]})
    labels = pl.Series([0, 1, 0, 1])
    ts = datetime(2024, 1, 1, tzinfo=UTC)

    class VaryingScoreStrategy:
        def fit(self, features: pl.DataFrame, labels: pl.Series | None = None) -> None:
            self._labels = labels

        def predict(self, features: pl.DataFrame) -> SignalOutput:
            scores = pl.Series([0.25, 0.75][: len(features)])
            labels = pl.Series([0] * len(features))
            return SignalOutput(scores=scores, labels=labels)

    results = run_rolling(
        features=features,
        labels=labels,
        strategy=VaryingScoreStrategy(),
        risk_engine=DummyRiskEngine(),
        simulator_factory=dummy_simulator_factory,
        bars_provider=DummyBars(dummy_bars()),
        portfolio_provider=DummyPortfolio(ts),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=2,
        step=2,
        sim_config=SimulatorConfig(),
        threshold_cfg={"top_signals": 1},
    )

    diagnostics = results[0].diagnostics
    assert diagnostics["calibration_raw_scores"]["count"] == 2
    assert diagnostics["calibration_calibrated_scores"]["count"] == 2
    assert diagnostics["test_raw_scores"]["count"] == 2
    assert diagnostics["test_calibrated_scores"]["count"] == 2
    assert diagnostics["test_raw_scores"]["min"] == 0.25
    assert diagnostics["test_raw_scores"]["max"] == 0.75
    assert diagnostics["test_calibrated_scores"]["min"] < 1.0
    assert diagnostics["test_calibrated_scores"]["std"] > 0.0


def test_run_rolling_with_walk_forward_splits_populates_slice_metadata() -> None:
    features = pl.DataFrame({"f": [1, 2, 3, 4, 5, 6, 7, 8]})
    labels = pl.Series([0, 1, 0, 1, 0, 1, 0, 1])
    ts = datetime(2024, 1, 1, tzinfo=UTC)

    splits = [
        WalkForwardSplit(
            train=slice(0, 2),
            validate=slice(2, 4),
            test=slice(4, 6),
            slice_id="time_window:0:start=0:end=2",
            embargo_bars=1,
        ),
        WalkForwardSplit(
            train=slice(0, 2),
            validate=slice(2, 4),
            test=slice(4, 6),
            slice_id="time_window:1:start=0:end=2",
            embargo_bars=1,
        ),
    ]

    results = run_rolling(
        features=features,
        labels=labels,
        strategy=DummyStrategy(),
        risk_engine=DummyRiskEngine(),
        simulator_factory=simulator_factory,
        bars_provider=DummyBars(dummy_bars()),
        portfolio_provider=DummyPortfolio(ts),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=2,
        step=2,
        splits=splits,
    )

    assert len(results) == 2
    assert results[0].slice_id == "time_window:0:start=0:end=2"
    assert results[1].slice_id == "time_window:1:start=0:end=2"
    assert results[0].split_metadata is not None
    assert results[0].split_metadata["split_type"] == "WalkForwardSplit"
    assert results[0].split_metadata["embargo_bars"] == 1
    assert results[0].split_metadata["train"]["start"] == 0
    assert results[0].split_metadata["train"]["stop"] == 2
    assert results[0].split_metadata["validate"]["start"] == 2
    assert results[0].split_metadata["validate"]["stop"] == 4
    assert results[0].split_metadata["test"]["start"] == 4
    assert results[0].split_metadata["test"]["stop"] == 6
    assert results[1].split_metadata is not None
    assert results[1].split_metadata["slice_id"] == "time_window:1:start=0:end=2"


def test_run_rolling_generates_deterministic_slice_id_when_missing() -> None:
    features = pl.DataFrame({"f": [1, 2, 3, 4, 5, 6, 7, 8]})
    labels = pl.Series([0, 1, 0, 1, 0, 1, 0, 1])
    ts = datetime(2024, 1, 1, tzinfo=UTC)

    split = WalkForwardSplit(
        train=slice(0, 2),
        validate=slice(2, 4),
        test=slice(4, 6),
    )

    results_a = run_rolling(
        features=features,
        labels=labels,
        strategy=DummyStrategy(),
        risk_engine=DummyRiskEngine(),
        simulator_factory=simulator_factory,
        bars_provider=DummyBars(dummy_bars()),
        portfolio_provider=DummyPortfolio(ts),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=2,
        step=2,
        splits=[split],
    )
    results_b = run_rolling(
        features=features,
        labels=labels,
        strategy=DummyStrategy(),
        risk_engine=DummyRiskEngine(),
        simulator_factory=simulator_factory,
        bars_provider=DummyBars(dummy_bars()),
        portfolio_provider=DummyPortfolio(ts),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=2,
        step=2,
        splits=[split],
    )

    assert len(results_a) == 1
    assert len(results_b) == 1
    assert results_a[0].slice_id == results_b[0].slice_id
    assert results_a[0].slice_id is not None
    assert results_a[0].slice_id == ("time_window:train=0-2|validate=2-4|test=4-6")


def test_run_rolling_splits_metadata_keeps_structural_fields_separate_from_violations() -> None:
    features = pl.DataFrame({"f": [1, 2, 3, 4, 5, 6, 7, 8]})
    labels = pl.Series([0, 1, 0, 1, 0, 1, 0, 1])
    ts = datetime(2024, 1, 1, tzinfo=UTC)

    split = WalkForwardSplit(
        train=slice(0, 2),
        validate=slice(2, 4),
        test=slice(4, 6),
        lockbox=slice(6, 7),
        embargo_bars=2,
    )

    results = run_rolling(
        features=features,
        labels=labels,
        strategy=DummyStrategy(),
        risk_engine=RiskEngineWithConstraintViolation(),
        simulator_factory=dummy_simulator_factory,
        bars_provider=DummyBars(dummy_bars()),
        portfolio_provider=DummyPortfolio(ts),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=2,
        step=2,
        splits=[split],
        sim_config=None,
    )

    assert len(results) == 1
    result = results[0]
    assert result.slice_id is not None
    assert result.split_metadata is not None
    assert result.split_metadata["embargo_bars"] == 2
    assert result.split_metadata["split_type"] == "WalkForwardSplit"
    assert result.split_metadata["lockbox"] is not None
    assert result.split_metadata["lockbox"]["start"] == 6
    assert result.split_metadata["lockbox"]["stop"] == 7
    assert result.constraint_violations == {"future_reference": 0.25}
    assert "future_reference" not in result.split_metadata


def test_downstream_cache_keying_can_use_fold_slice_id() -> None:
    features = pl.DataFrame({"f": [1, 2, 3, 4, 5, 6, 7, 8]})
    labels = pl.Series([0, 1, 0, 1, 0, 1, 0, 1])
    ts = datetime(2024, 1, 1, tzinfo=UTC)

    split = WalkForwardSplit(
        train=slice(0, 2),
        validate=slice(2, 4),
        test=slice(4, 6),
        slice_id="time_window:cache_probe",
    )

    first_run = run_rolling(
        features=features,
        labels=labels,
        strategy=DummyStrategy(),
        risk_engine=DummyRiskEngine(),
        simulator_factory=simulator_factory,
        bars_provider=DummyBars(dummy_bars()),
        portfolio_provider=DummyPortfolio(ts),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=2,
        step=2,
        splits=[split],
    )
    second_run = run_rolling(
        features=features,
        labels=labels,
        strategy=DummyStrategy(),
        risk_engine=DummyRiskEngine(),
        simulator_factory=simulator_factory,
        bars_provider=DummyBars(dummy_bars()),
        portfolio_provider=DummyPortfolio(ts),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=2,
        step=2,
        splits=[split],
    )

    strategy_hash = "strategy_hash_abc"
    cache: dict[tuple[str, str], FoldResult] = {}
    evaluation_count = 0

    for fold in first_run:
        assert fold.slice_id is not None
        key = (strategy_hash, fold.slice_id)
        if key not in cache:
            cache[key] = fold
            evaluation_count += 1

    for fold in second_run:
        assert fold.slice_id is not None
        key = (strategy_hash, fold.slice_id)
        if key not in cache:
            cache[key] = fold
            evaluation_count += 1

    assert evaluation_count == 1
    assert (strategy_hash, split.slice_id) in cache

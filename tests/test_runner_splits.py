from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID

import polars as pl

from liq.core import Bar, OrderRequest, OrderSide, OrderType, PortfolioState
from liq.core.fill import Fill
from liq.datasets import WalkForwardSplit
from liq.risk.config import RiskConfig
from liq.runner.runner import FoldResult, run_rolling
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


def test_run_rolling_skips_risk_when_no_signals_are_actionable() -> None:
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

    assert results[0].threshold > 0.8
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

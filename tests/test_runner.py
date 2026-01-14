from decimal import Decimal
from datetime import datetime, UTC
from uuid import UUID
from types import SimpleNamespace

import polars as pl
from liq.core import Bar, OrderRequest, OrderSide, OrderType, PortfolioState
from liq.core.fill import Fill

from liq.runner.runner import run_rolling
from liq.risk.config import RiskConfig, MarketState
from liq.sim.config import ProviderConfig, SimulatorConfig
from liq.sim.simulator import Simulator
from liq.signals.output import SignalOutput


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


def test_run_rolling_returns_fold_results() -> None:
    features = pl.DataFrame({"f": [1, 2, 3, 4, 5, 6]})
    labels = pl.Series([0, 1, 0, 1, 0, 1])
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    # capture logs
    records = []
    import logging
    handler = logging.StreamHandler()
    handler.emit = lambda record: records.append(record)
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
        risk_config=RiskConfig(initial_cash=Decimal("10000")),
        train_size=2,
        valid_size=2,
        step=2,
        sim_config=None,
    )
    assert results
    assert results[0].threshold > 0
    assert isinstance(results[0].metrics, dict)
    assert any("rolling_split" in rec.getMessage() for rec in records)
    assert any("fold_result" in rec.getMessage() for rec in records)
    logger.removeHandler(handler)


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
        risk_config=RiskConfig(initial_cash=Decimal("10000")),
        train_size=2,
        valid_size=2,
        step=2,
        sim_config=SimulatorConfig(),
        threshold_cfg={"top_signals": 1},
        calibration_split=0.1,
    )
    assert results

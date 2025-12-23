from decimal import Decimal
from datetime import datetime, UTC

import polars as pl
from liq.core import Bar, OrderRequest, OrderSide, OrderType, PortfolioState

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
            timestamp=signals[0]["timestamp"],
        )
        return type("Res", (), {"orders": [order]})


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

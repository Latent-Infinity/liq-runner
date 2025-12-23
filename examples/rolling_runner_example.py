"""Example showing rolling runner wiring with dummy components."""

from datetime import datetime, timezone
from decimal import Decimal

import polars as pl

from liq.core import Bar, OrderRequest, OrderSide, OrderType, PortfolioState
from liq.runner.runner import run_rolling
from liq.risk.config import RiskConfig
from liq.sim.config import ProviderConfig, SimulatorConfig
from liq.sim.simulator import Simulator
from liq.signals.output import SignalOutput


class ExampleStrategy:
    def fit(self, features: pl.DataFrame, labels: pl.Series | None = None) -> None:
        self.labels = labels

    def predict(self, features: pl.DataFrame) -> SignalOutput:
        scores = pl.Series([0.9] * len(features))
        labels = self.labels if self.labels is not None else pl.Series([0] * len(features))
        return SignalOutput(scores=scores, labels=labels)


class ExampleRiskEngine:
    def process_signals(self, signals, portfolio, market, config):
        ts = signals[0]["timestamp"]
        return type(
            "Res",
            (),
            {
                "orders": [
                    OrderRequest(
                        symbol=signals[0]["symbol"],
                        quantity=Decimal("1"),
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        timestamp=ts,
                    )
                ]
            },
        )


class ExampleBars:
    def __init__(self, bars):
        self._bars = bars

    def get_bars(self, split: slice):
        return self._bars[split]


class ExamplePortfolioProvider:
    def __init__(self, ts: datetime):
        self.ts = ts

    def get_portfolio(self, split: slice) -> PortfolioState:
        return PortfolioState(cash=Decimal("10000"), positions={}, timestamp=self.ts)


def example_sim_factory() -> Simulator:
    cfg = SimulatorConfig()
    provider = ProviderConfig(
        name="example",
        asset_classes=["crypto"],
        fee_model="ZeroCommission",
        slippage_model="VolumeWeighted",
    )
    return Simulator(provider_config=provider, config=cfg)


def build_bars() -> list[Bar]:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
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
        for _ in range(5)
    ]


def main() -> None:
    features = pl.DataFrame({"f": [1, 2, 3, 4, 5]})
    labels = pl.Series([0, 1, 0, 1, 0])
    results = run_rolling(
        features=features,
        labels=labels,
        strategy=ExampleStrategy(),
        risk_engine=ExampleRiskEngine(),
        simulator_factory=example_sim_factory,
        bars_provider=ExampleBars(build_bars()),
        portfolio_provider=ExamplePortfolioProvider(datetime(2024, 1, 1, tzinfo=timezone.utc)),
        risk_config=RiskConfig(),
        train_size=2,
        valid_size=2,
        step=1,
    )
    for fold in results:
        print("Threshold", fold.threshold, "Metrics", fold.metrics)


if __name__ == "__main__":
    main()

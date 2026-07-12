"""Central cost book: execution-cost scenarios resolved by name.

All pilot and backtest costs come from here by scenario name — cost numbers
are never inlined in pilot or strategy code. A run that names a scenario the
book does not contain refuses to start (``UnknownCostScenarioError``), and
the resolved scenario id and book version are recorded in run provenance.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType


class UnknownCostScenarioError(ValueError):
    """A run named a cost scenario the cost book does not contain."""


@dataclass(frozen=True)
class CostScenario:
    """One named execution-cost scenario for a trading surface."""

    scenario_id: str
    surface: str
    description: str
    params: Mapping[str, float | str]


@dataclass(frozen=True)
class CostBook:
    """Versioned, immutable collection of cost scenarios keyed by id."""

    version: str
    scenarios: Mapping[str, CostScenario]

    @classmethod
    def from_scenarios(cls, version: str, scenarios: Iterable[CostScenario]) -> CostBook:
        by_id: dict[str, CostScenario] = {}
        for scenario in scenarios:
            if scenario.scenario_id in by_id:
                raise ValueError(f"duplicate scenario id: {scenario.scenario_id}")
            by_id[scenario.scenario_id] = scenario
        return cls(version=version, scenarios=MappingProxyType(by_id))

    def resolve(self, scenario_id: str) -> CostScenario:
        """Return the scenario for ``scenario_id`` or refuse to proceed."""
        if not scenario_id:
            raise UnknownCostScenarioError(
                "cost scenario id is empty; every signal-bearing run must "
                "name a scenario from the cost book"
            )
        scenario = self.scenarios.get(scenario_id)
        if scenario is None:
            raise UnknownCostScenarioError(
                f"unknown cost scenario '{scenario_id}' in cost book "
                f"'{self.version}'; known scenarios: {sorted(self.scenarios)}"
            )
        return scenario


def _scenario(
    scenario_id: str, surface: str, description: str, **params: float | str
) -> CostScenario:
    return CostScenario(
        scenario_id=scenario_id,
        surface=surface,
        description=description,
        params=MappingProxyType(dict(params)),
    )


INTRADAY_CAMPAIGN_COST_BOOK_V1 = CostBook.from_scenarios(
    "intraday_campaign_v1",
    [
        _scenario(
            "spy_qqq_base_v1",
            "spy_qqq",
            "SPY/QQQ base: 0.5 bp/side (RTH, open auction, close auction)",
            per_side_bps=0.5,
        ),
        _scenario(
            "spy_qqq_stress_3x_v1",
            "spy_qqq",
            "SPY/QQQ stress: 3x multiplier on the base per-side cost",
            per_side_bps=1.5,
        ),
        _scenario(
            "spy_qqq_auction_slip_1bp_v1",
            "spy_qqq",
            "SPY/QQQ auction slippage stress: 1 bp on auction fills",
            per_side_bps=0.5,
            auction_slippage_bps=1.0,
        ),
        _scenario(
            "spy_qqq_auction_slip_2bp_v1",
            "spy_qqq",
            "SPY/QQQ auction slippage stress: 2 bps on auction fills",
            per_side_bps=0.5,
            auction_slippage_bps=2.0,
        ),
        _scenario(
            "spy_qqq_auction_slip_5bp_v1",
            "spy_qqq",
            "SPY/QQQ auction slippage stress: 5 bps on auction fills",
            per_side_bps=0.5,
            auction_slippage_bps=5.0,
        ),
        _scenario(
            "single_name_us_base_v1",
            "single_name_us",
            "US single names: 20 bps round trip plus 0.5 bp/side SPY hedge",
            round_trip_bps=20.0,
            hedge_per_side_bps=0.5,
        ),
        _scenario(
            "single_name_us_stress_30_v1",
            "single_name_us",
            "US single names stress: 30 bps round trip",
            round_trip_bps=30.0,
            hedge_per_side_bps=0.5,
        ),
        _scenario(
            "single_name_us_stress_40_v1",
            "single_name_us",
            "US single names stress: 40 bps round trip",
            round_trip_bps=40.0,
            hedge_per_side_bps=0.5,
        ),
        _scenario(
            "single_name_us_ts_optimistic_v1",
            "single_name_us",
            "US single names via TradeStation, optimistic: 2 bps round trip "
            "(low end of observed 2-6 bps RT)",
            round_trip_bps=2.0,
            hedge_per_side_bps=0.5,
        ),
        _scenario(
            "single_name_us_ts_base_v1",
            "single_name_us",
            "US single names via TradeStation, conservative realistic: 6 bps round trip "
            "(high end of observed 2-6 bps RT)",
            round_trip_bps=6.0,
            hedge_per_side_bps=0.5,
        ),
        _scenario(
            "single_name_us_ts_stress_v1",
            "single_name_us",
            "US single names via TradeStation stress: 12 bps round trip (2x the conservative base)",
            round_trip_bps=12.0,
            hedge_per_side_bps=0.5,
        ),
        _scenario(
            "oanda_fixed_spread_table_v1",
            "oanda_fx",
            "OANDA FX base: fixed per-pair spread table",
            spread_table="fixed_spread_table_v1",
            spread_multiplier=1.0,
        ),
        _scenario(
            "oanda_london_open_1p5x_v1",
            "oanda_fx",
            "OANDA FX: 1.5x spread for entries in the first 15 minutes after London open",
            spread_table="fixed_spread_table_v1",
            spread_multiplier=1.5,
            window="london_open_first_15m",
        ),
        _scenario(
            "oanda_fix_window_1p5x_v1",
            "oanda_fx",
            "OANDA FX: 1.5x spread inside the London 4pm fix window",
            spread_table="fixed_spread_table_v1",
            spread_multiplier=1.5,
            window="london_fix",
        ),
        _scenario(
            "oanda_stress_2x_v1",
            "oanda_fx",
            "OANDA FX stress: 2x spread overall",
            spread_table="fixed_spread_table_v1",
            spread_multiplier=2.0,
        ),
        _scenario(
            "binance_spot_base_v1",
            "binance_spot",
            "Binance spot: taker 10 bps / maker 2 bps",
            taker_bps=10.0,
            maker_bps=2.0,
        ),
        _scenario(
            "binance_perp_optimistic_v1",
            "binance_perp",
            "Binance perp optimistic: maker 0 / taker 5 bps",
            maker_bps=0.0,
            taker_bps=5.0,
        ),
        _scenario(
            "binance_perp_middle_v1",
            "binance_perp",
            "Binance perp middle (gate scenario): maker 2 / taker 7.5 bps",
            maker_bps=2.0,
            taker_bps=7.5,
        ),
        _scenario(
            "binance_perp_stress_v1",
            "binance_perp",
            "Binance perp stress: maker 10 / taker 10 bps (floor: PF >= 1.0)",
            maker_bps=10.0,
            taker_bps=10.0,
        ),
        _scenario(
            "coinbase_spot_maker_maker_v1",
            "coinbase_spot",
            "Coinbase Advanced spot, >$10K 30-day tier, optimistic (both legs maker): "
            "maker 25 / taker 25 bps",
            maker_bps=25.0,
            taker_bps=25.0,
        ),
        _scenario(
            "coinbase_spot_base_v1",
            "coinbase_spot",
            "Coinbase Advanced spot, >$10K 30-day tier, gate scenario (maker entry + taker exit): "
            "maker 25 / taker 40 bps (0.25% / 0.40%, 2026-07)",
            maker_bps=25.0,
            taker_bps=40.0,
        ),
        _scenario(
            "coinbase_spot_taker_taker_v1",
            "coinbase_spot",
            "Coinbase Advanced spot, >$10K 30-day tier, stress (both legs taker): "
            "maker 40 / taker 40 bps (floor: PF >= 1.0)",
            maker_bps=40.0,
            taker_bps=40.0,
        ),
    ],
)

"""Tests for the FX per-pair spread table and cost application."""

import pytest

from liq.runner.cost_book import INTRADAY_CAMPAIGN_COST_BOOK_V1
from liq.runner.fx_spread import (
    FIXED_SPREAD_TABLE_V1,
    UnknownPairError,
    pip_size,
    round_trip_cost_fraction,
    spread_cost_fraction,
)

MAJORS = (
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "AUD_USD",
    "USD_CAD",
    "USD_CHF",
    "NZD_USD",
)


class TestSpreadTable:
    def test_all_majors_present(self) -> None:
        assert set(FIXED_SPREAD_TABLE_V1.spreads_pips) == set(MAJORS)

    def test_has_version_and_provenance(self) -> None:
        assert FIXED_SPREAD_TABLE_V1.version == "fixed_spread_table_v1"
        assert FIXED_SPREAD_TABLE_V1.provenance

    def test_eur_usd_tightest(self) -> None:
        pips = FIXED_SPREAD_TABLE_V1.spreads_pips
        assert pips["EUR_USD"] <= min(pips[p] for p in MAJORS if p != "EUR_USD")


class TestPipSize:
    def test_jpy_pip_is_0p01(self) -> None:
        assert pip_size("USD_JPY") == pytest.approx(0.01)

    def test_non_jpy_pip_is_0p0001(self) -> None:
        assert pip_size("EUR_USD") == pytest.approx(0.0001)
        assert pip_size("GBP_USD") == pytest.approx(0.0001)

    def test_unknown_pair_raises(self) -> None:
        with pytest.raises(UnknownPairError, match="XXX_YYY"):
            pip_size("XXX_YYY")


class TestSpreadCostFraction:
    def test_eur_usd_base(self) -> None:
        # 1.0 pip / price. At price 1.10, cost fraction = 0.0001/1.10.
        frac = spread_cost_fraction("EUR_USD", price=1.10, multiplier=1.0)
        assert frac == pytest.approx(0.0001 / 1.10)

    def test_jpy_uses_0p01_pip(self) -> None:
        # USD_JPY 1.0 pip at price 150 => 0.01/150.
        frac = spread_cost_fraction("USD_JPY", price=150.0, multiplier=1.0)
        assert frac == pytest.approx(0.01 / 150.0)

    def test_multiplier_scales(self) -> None:
        base = spread_cost_fraction("EUR_USD", price=1.10, multiplier=1.0)
        scaled = spread_cost_fraction("EUR_USD", price=1.10, multiplier=1.5)
        assert scaled == pytest.approx(1.5 * base)

    def test_unknown_pair_raises(self) -> None:
        with pytest.raises(UnknownPairError):
            spread_cost_fraction("XXX_YYY", price=1.0, multiplier=1.0)


class TestRoundTripCostFraction:
    def test_resolves_multiplier_from_scenario(self) -> None:
        base = INTRADAY_CAMPAIGN_COST_BOOK_V1.resolve("oanda_fixed_spread_table_v1")
        stress = INTRADAY_CAMPAIGN_COST_BOOK_V1.resolve("oanda_stress_2x_v1")
        base_frac = round_trip_cost_fraction("EUR_USD", price=1.10, scenario=base)
        stress_frac = round_trip_cost_fraction("EUR_USD", price=1.10, scenario=stress)
        assert stress_frac == pytest.approx(2.0 * base_frac)

    def test_london_open_scenario_is_1p5x(self) -> None:
        base = INTRADAY_CAMPAIGN_COST_BOOK_V1.resolve("oanda_fixed_spread_table_v1")
        london = INTRADAY_CAMPAIGN_COST_BOOK_V1.resolve("oanda_london_open_1p5x_v1")
        base_frac = round_trip_cost_fraction("GBP_USD", price=1.25, scenario=base)
        london_frac = round_trip_cost_fraction("GBP_USD", price=1.25, scenario=london)
        assert london_frac == pytest.approx(1.5 * base_frac)

    def test_non_fx_scenario_rejected(self) -> None:
        spy = INTRADAY_CAMPAIGN_COST_BOOK_V1.resolve("spy_qqq_base_v1")
        with pytest.raises(ValueError, match="spread_table"):
            round_trip_cost_fraction("EUR_USD", price=1.10, scenario=spy)

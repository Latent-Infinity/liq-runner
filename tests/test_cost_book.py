"""Tests for the central cost book."""

import pytest

from liq.runner.cost_book import (
    INTRADAY_CAMPAIGN_COST_BOOK_V1,
    CostBook,
    CostScenario,
    UnknownCostScenarioError,
)


class TestCostBookResolution:
    def test_resolve_known_scenario(self) -> None:
        scenario = INTRADAY_CAMPAIGN_COST_BOOK_V1.resolve("spy_qqq_base_v1")
        assert isinstance(scenario, CostScenario)
        assert scenario.scenario_id == "spy_qqq_base_v1"
        assert scenario.surface == "spy_qqq"
        assert scenario.params["per_side_bps"] == 0.5

    def test_unknown_scenario_raises(self) -> None:
        with pytest.raises(UnknownCostScenarioError, match="no_such_scenario"):
            INTRADAY_CAMPAIGN_COST_BOOK_V1.resolve("no_such_scenario")

    def test_unnamed_scenario_raises(self) -> None:
        with pytest.raises(UnknownCostScenarioError, match="empty"):
            INTRADAY_CAMPAIGN_COST_BOOK_V1.resolve("")

    def test_error_lists_known_scenarios(self) -> None:
        with pytest.raises(UnknownCostScenarioError, match="spy_qqq_base_v1"):
            INTRADAY_CAMPAIGN_COST_BOOK_V1.resolve("typo")

    def test_book_has_version(self) -> None:
        assert INTRADAY_CAMPAIGN_COST_BOOK_V1.version == "intraday_campaign_v1"


class TestCampaignScenarios:
    """The default book encodes the campaign cost table."""

    @pytest.mark.parametrize(
        "scenario_id",
        [
            "spy_qqq_base_v1",
            "spy_qqq_stress_3x_v1",
            "spy_qqq_auction_slip_1bp_v1",
            "spy_qqq_auction_slip_2bp_v1",
            "spy_qqq_auction_slip_5bp_v1",
            "single_name_us_base_v1",
            "single_name_us_stress_30_v1",
            "single_name_us_stress_40_v1",
            "single_name_us_ts_optimistic_v1",
            "single_name_us_ts_base_v1",
            "single_name_us_ts_stress_v1",
            "oanda_fixed_spread_table_v1",
            "oanda_london_open_1p5x_v1",
            "oanda_fix_window_1p5x_v1",
            "oanda_stress_2x_v1",
            "binance_spot_base_v1",
            "binance_perp_optimistic_v1",
            "binance_perp_middle_v1",
            "binance_perp_stress_v1",
            "coinbase_spot_maker_maker_v1",
            "coinbase_spot_base_v1",
            "coinbase_spot_taker_taker_v1",
        ],
    )
    def test_scenario_present(self, scenario_id: str) -> None:
        scenario = INTRADAY_CAMPAIGN_COST_BOOK_V1.resolve(scenario_id)
        assert scenario.scenario_id == scenario_id
        assert scenario.description

    def test_spy_stress_is_three_times_base(self) -> None:
        base = INTRADAY_CAMPAIGN_COST_BOOK_V1.resolve("spy_qqq_base_v1")
        stress = INTRADAY_CAMPAIGN_COST_BOOK_V1.resolve("spy_qqq_stress_3x_v1")
        assert stress.params["per_side_bps"] == 3 * base.params["per_side_bps"]

    def test_single_name_includes_hedge_leg(self) -> None:
        base = INTRADAY_CAMPAIGN_COST_BOOK_V1.resolve("single_name_us_base_v1")
        assert base.params["round_trip_bps"] == 20.0
        assert base.params["hedge_per_side_bps"] == 0.5

    def test_single_name_ts_realistic_costs(self) -> None:
        """TradeStation single-name equity: observed ~2-6 bps RT, base = 6 (conservative)."""
        opt = INTRADAY_CAMPAIGN_COST_BOOK_V1.resolve("single_name_us_ts_optimistic_v1")
        base = INTRADAY_CAMPAIGN_COST_BOOK_V1.resolve("single_name_us_ts_base_v1")
        stress = INTRADAY_CAMPAIGN_COST_BOOK_V1.resolve("single_name_us_ts_stress_v1")
        assert opt.params["round_trip_bps"] == 2.0
        assert base.params["round_trip_bps"] == 6.0
        assert stress.params["round_trip_bps"] == 12.0
        # all on the single_name_us surface, stress = 2x the conservative base
        assert base.surface == "single_name_us"
        assert stress.params["round_trip_bps"] == 2 * base.params["round_trip_bps"]

    def test_perp_gate_scenario_is_the_middle_one(self) -> None:
        middle = INTRADAY_CAMPAIGN_COST_BOOK_V1.resolve("binance_perp_middle_v1")
        assert middle.params["maker_bps"] == 2.0
        assert middle.params["taker_bps"] == 7.5
        assert "gate" in middle.description.lower()

    def test_coinbase_spot_tier_and_gate(self) -> None:
        """Coinbase Advanced spot, >$10K 30-day tier: maker 25 / taker 40 bps."""
        opt = INTRADAY_CAMPAIGN_COST_BOOK_V1.resolve("coinbase_spot_maker_maker_v1")
        base = INTRADAY_CAMPAIGN_COST_BOOK_V1.resolve("coinbase_spot_base_v1")
        stress = INTRADAY_CAMPAIGN_COST_BOOK_V1.resolve("coinbase_spot_taker_taker_v1")
        assert base.surface == "coinbase_spot"
        # base is the gate scenario: maker entry + taker exit
        assert base.params["maker_bps"] == 25.0
        assert base.params["taker_bps"] == 40.0
        assert "gate" in base.description.lower()
        # optimistic = both legs maker; stress = both legs taker (PF >= 1.0 floor)
        assert opt.params["taker_bps"] == 25.0
        assert stress.params["maker_bps"] == 40.0
        assert stress.params["taker_bps"] == 40.0


class TestCustomBook:
    def test_from_scenarios(self) -> None:
        scenario = CostScenario(
            scenario_id="test_v1",
            surface="test",
            description="test scenario",
            params={"round_trip_bps": 5.0},
        )
        book = CostBook.from_scenarios("custom_v1", [scenario])
        assert book.resolve("test_v1") is scenario

    def test_duplicate_scenario_ids_raise(self) -> None:
        scenario = CostScenario(scenario_id="dup_v1", surface="s", description="d", params={})
        with pytest.raises(ValueError, match="duplicate"):
            CostBook.from_scenarios("custom_v1", [scenario, scenario])

"""Tests for run provenance emission."""

import json
from pathlib import Path

import pytest

from liq.runner.cost_book import (
    INTRADAY_CAMPAIGN_COST_BOOK_V1,
    UnknownCostScenarioError,
)
from liq.runner.provenance import RunProvenance, build_run_provenance


def _build(**overrides: object) -> RunProvenance:
    kwargs: dict = {
        "run_id": "run_0001",
        "code_hash": "abc123",
        "config_hash": "def456",
        "cost_scenario_id": "spy_qqq_base_v1",
        "data_hash": "0011aa",
        "periods_touched": (("2015-01-01", "2022-12-31"),),
        "seeds": {"global": 42},
    }
    kwargs.update(overrides)
    return build_run_provenance(**kwargs)


class TestBuildRunProvenance:
    def test_includes_cost_scenario_and_book_version(self) -> None:
        prov = _build()
        assert prov.cost_scenario_id == "spy_qqq_base_v1"
        assert prov.cost_book_version == INTRADAY_CAMPAIGN_COST_BOOK_V1.version
        assert prov.run_id == "run_0001"

    def test_unknown_scenario_refuses_to_start(self) -> None:
        with pytest.raises(UnknownCostScenarioError):
            _build(cost_scenario_id="not_in_book")

    def test_unnamed_scenario_refuses_to_start(self) -> None:
        with pytest.raises(UnknownCostScenarioError):
            _build(cost_scenario_id="")

    def test_required_fields_enforced(self) -> None:
        with pytest.raises(ValueError, match="run_id"):
            _build(run_id="")
        with pytest.raises(ValueError, match="code_hash"):
            _build(code_hash="")
        with pytest.raises(ValueError, match="config_hash"):
            _build(config_hash="")

    def test_created_at_is_utc(self) -> None:
        prov = _build()
        assert prov.created_at.tzinfo is not None
        assert prov.created_at.utcoffset().total_seconds() == 0


class TestSerialization:
    def test_to_dict_round_trip_fields(self) -> None:
        payload = _build().to_dict()
        assert payload["cost_scenario_id"] == "spy_qqq_base_v1"
        assert payload["periods_touched"] == [["2015-01-01", "2022-12-31"]]
        assert payload["seeds"] == {"global": 42}
        assert "created_at" in payload

    def test_write_json(self, tmp_path: Path) -> None:
        path = tmp_path / "provenance.json"
        prov = _build()
        prov.write_json(path)
        loaded = json.loads(path.read_text())
        assert loaded["run_id"] == "run_0001"
        assert loaded["cost_book_version"] == INTRADAY_CAMPAIGN_COST_BOOK_V1.version

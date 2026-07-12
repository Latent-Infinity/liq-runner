"""TDD pins for the six-curve harness (orchestration + IO over liq-metrics).

The harness assembles aligned inputs, delegates curve math to
``liq.metrics.six_curves``, and emits one JSON-able artifact carrying every
curve plus the policy/spec hash slots that later lock null-simulation runs.
A converter keeps legacy per-cell verdict JSONs (the ``cells`` /
``ablation_results`` / top-level-list shapes) re-loadable as diagnostic
records so historical trial accounting stays valid.
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from liq.metrics.six_curves import SixCurveInputs
from liq.metrics.tax_curves import TaxPolicy, TaxRates
from liq.runner.six_curve_harness import (
    convert_legacy_verdict,
    run_six_curve_harness,
)

D = Decimal


def _inputs() -> SixCurveInputs:
    return SixCurveInputs(
        dates=(date(2026, 1, 31), date(2026, 2, 28)),
        starting_capital=D("50000"),
        baseline_returns=(D("0.01"), D("0.02")),
        overlay_returns=(D("0.03"), D("-0.01")),
        sleeve_weight=D("0.2"),
        measured_costs=(D("0.0005"), D("0.0005")),
        tax_policy=TaxPolicy(rates=TaxRates.zero(), no_tax_smoke=True),
    )


class TestHarnessArtifact:
    def test_artifact_carries_all_curves_and_hash_slots(self, tmp_path: Path) -> None:
        out = tmp_path / "six-curves.json"
        artifact = run_six_curve_harness(
            _inputs(),
            output_path=out,
            metadata={
                "experiment_id": "demo",
                "account_policy_hash": "unpinned-smoke",
                "optimizer_spec_hash": "unpinned-smoke",
            },
        )
        assert out.exists()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded == artifact
        for key in ("a", "b", "c", "d", "e", "a3"):
            assert len(loaded["curves"][key]) == 2
        assert set(loaded["f_views"]) >= {"f1_nav", "f2_nav", "f3_nav"}
        assert loaded["metadata"]["account_policy_hash"] == "unpinned-smoke"
        assert loaded["dates"] == ["2026-01-31", "2026-02-28"]
        assert loaded["no_tax_smoke"] is True
        assert loaded["performance"]["candidate_curve"] == "e"
        assert loaded["performance"]["baseline_curve"] == "a3"

    def test_metadata_hash_slots_are_required(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="account_policy_hash"):
            run_six_curve_harness(
                _inputs(),
                output_path=tmp_path / "x.json",
                metadata={"experiment_id": "demo"},
            )

    def test_curve_values_serialize_as_strings(self, tmp_path: Path) -> None:
        artifact = run_six_curve_harness(
            _inputs(),
            output_path=tmp_path / "x.json",
            metadata={
                "experiment_id": "demo",
                "account_policy_hash": "h1",
                "optimizer_spec_hash": "h2",
            },
        )
        assert all(isinstance(v, str) for v in artifact["curves"]["a"])  # Decimal fidelity

    def test_performance_report_reuses_metrics_analyzer(self, tmp_path: Path) -> None:
        artifact = run_six_curve_harness(
            _inputs(),
            output_path=tmp_path / "x.json",
            metadata={
                "experiment_id": "demo",
                "account_policy_hash": "h1",
                "optimizer_spec_hash": "h2",
            },
        )
        performance = artifact["performance"]
        assert performance["candidate"]["aggregate"]["num_bars"] == 3
        assert performance["baseline"]["aggregate"]["num_bars"] == 3
        assert "outperforms_aggregate" in performance
        assert performance["outperforms_per_regime"] == {"aggregate": False}


class TestLegacyVerdictConverter:
    # verbatim shape excerpts from committed CRV/INT verdict artifacts
    CELLS_SHAPE = {
        "cells": [
            {"cell_id": "alpha_cell0542", "profit_factor": 2.08, "nav_return_pct": 167.7},
            {"cell_id": "alpha_cell0001", "profit_factor": 0.91, "nav_return_pct": -3.2},
        ],
        "top_k_by_pf": [{"cell_id": "alpha_cell0542"}],
        "cost_bps_rt": 20,
        "fold_label": "F3F4F5-pooled",
    }
    ABLATION_SHAPE = {"ablation_results": [{"K": 6, "net_sharpe": 0.4}], "dsr_probability": 0.1}
    LIST_SHAPE = [{"cell_id": "z1", "alpha_floor": 1.145}]

    def test_cells_shape_loads_as_diagnostic_records(self) -> None:
        legacy = convert_legacy_verdict(self.CELLS_SHAPE)
        assert legacy.n_cells == 2  # top_k subset not double-counted
        assert legacy.records[0]["cell_id"] == "alpha_cell0542"
        assert legacy.diagnostic_only is True

    def test_ablation_and_list_shapes_load(self) -> None:
        assert convert_legacy_verdict(self.ABLATION_SHAPE).n_cells == 1
        assert convert_legacy_verdict(self.LIST_SHAPE).n_cells == 1

    def test_non_cell_payload_yields_empty(self) -> None:
        legacy = convert_legacy_verdict({"summary": {"n_covered": 491}})
        assert legacy.n_cells == 0
        assert legacy.records == []

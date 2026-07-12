"""Pure cash-conversion floor and metric-shape contract tests."""

from __future__ import annotations

import pytest

from liq.runner.fce_cash_conversion import (
    CashConversionStatus,
    ExposureUtilizationMetrics,
    derive_cash_conversion_status,
    gamma_flow_floor,
    no_trade_filter_floor,
    sizer_floor,
)


def test_exposure_utilization_requires_cash_drag_and_full_metric_shape() -> None:
    payload = _exposure_payload()
    metrics = ExposureUtilizationMetrics.from_mapping(payload)

    assert metrics.cash_drag == 0.18
    assert set(metrics.as_mapping()) == {
        "average_gross",
        "average_net",
        "utilization",
        "cash_drag",
        "realized_vol",
        "turnover",
        "trade_count",
        "holding_period",
        "active_time",
        "pnl_per_unit_gross",
        "pnl_per_unit_realized_risk",
    }

    payload.pop("cash_drag")
    with pytest.raises(ValueError, match="cash_drag"):
        ExposureUtilizationMetrics.from_mapping(payload)


def test_no_trade_filter_floor_requires_net_avoided_loss_at_equal_count() -> None:
    assert no_trade_filter_floor(
        avoided_loss=12.0,
        missed_profit=8.0,
        equal_trade_count=True,
        sharpe_delta=0.05,
        sharpe_delta_floor=0.0,
    )
    assert not no_trade_filter_floor(
        avoided_loss=12.0,
        missed_profit=8.0,
        equal_trade_count=False,
        sharpe_delta=0.05,
        sharpe_delta_floor=0.0,
    )
    assert not no_trade_filter_floor(
        avoided_loss=7.0,
        missed_profit=8.0,
        equal_trade_count=True,
        sharpe_delta=0.05,
        sharpe_delta_floor=0.0,
    )
    assert not no_trade_filter_floor(
        avoided_loss=12.0,
        missed_profit=8.0,
        equal_trade_count=True,
        sharpe_delta=0.0,
        sharpe_delta_floor=0.0,
    )


def test_sizer_floor_requires_utility_target_error_and_non_worse_drawdown() -> None:
    assert sizer_floor(
        utility_gain=0.01,
        vol_target_error_improved=True,
        drawdown_non_worse=True,
    )
    assert not sizer_floor(
        utility_gain=0.0,
        vol_target_error_improved=True,
        drawdown_non_worse=True,
    )
    assert not sizer_floor(
        utility_gain=0.01,
        vol_target_error_improved=False,
        drawdown_non_worse=True,
    )
    assert not sizer_floor(
        utility_gain=0.01,
        vol_target_error_improved=True,
        drawdown_non_worse=False,
    )


def test_gamma_flow_floor_requires_a3_over_a1_and_both_stresses() -> None:
    assert gamma_flow_floor(
        a3_utility=1.1,
        a1_utility=1.0,
        tail_loss_reduced=True,
        doubled_cost_survives=True,
        oi_lag_stress_survives=True,
    )
    assert not gamma_flow_floor(
        a3_utility=1.0,
        a1_utility=1.0,
        tail_loss_reduced=True,
        doubled_cost_survives=True,
        oi_lag_stress_survives=True,
    )
    assert not gamma_flow_floor(
        a3_utility=1.1,
        a1_utility=1.0,
        tail_loss_reduced=True,
        doubled_cost_survives=False,
        oi_lag_stress_survives=True,
    )
    assert not gamma_flow_floor(
        a3_utility=1.1,
        a1_utility=1.0,
        tail_loss_reduced=False,
        doubled_cost_survives=True,
        oi_lag_stress_survives=True,
    )
    assert not gamma_flow_floor(
        a3_utility=1.1,
        a1_utility=1.0,
        tail_loss_reduced=True,
        doubled_cost_survives=True,
        oi_lag_stress_survives=False,
    )


def test_status_is_derived_only_for_frozen_signal_bearing_results() -> None:
    with pytest.raises(ValueError, match="frozen signal-bearing"):
        derive_cash_conversion_status(
            object_type="no_trade_filter",
            floor_satisfied=True,
            frozen_signal_bearing_run=False,
        )

    assert (
        derive_cash_conversion_status(
            object_type="pure_forecast",
            floor_satisfied=True,
            frozen_signal_bearing_run=True,
        )
        is CashConversionStatus.FORECAST_ONLY
    )
    assert (
        derive_cash_conversion_status(
            object_type="no_trade_filter",
            floor_satisfied=True,
            frozen_signal_bearing_run=True,
        )
        is CashConversionStatus.ECONOMIC_CHAMPION
    )
    assert (
        derive_cash_conversion_status(
            object_type="sizer",
            floor_satisfied=False,
            frozen_signal_bearing_run=True,
        )
        is CashConversionStatus.NULL
    )

    with pytest.raises(ValueError, match="object_type"):
        derive_cash_conversion_status(
            object_type="unknown",
            floor_satisfied=True,
            frozen_signal_bearing_run=True,
        )


def _exposure_payload() -> dict[str, float | int]:
    return {
        "average_gross": 0.80,
        "average_net": 0.62,
        "utilization": 0.82,
        "cash_drag": 0.18,
        "realized_vol": 0.12,
        "turnover": 1.4,
        "trade_count": 320,
        "holding_period": 4.0,
        "active_time": 0.72,
        "pnl_per_unit_gross": 0.03,
        "pnl_per_unit_realized_risk": 0.20,
    }

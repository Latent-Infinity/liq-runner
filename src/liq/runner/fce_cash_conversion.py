"""Pure cash-conversion floors and exposure-utilization metric contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite

_EXPOSURE_FIELDS = (
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
)
EXPOSURE_UTILIZATION_FIELDS = _EXPOSURE_FIELDS
_ECONOMIC_OBJECT_TYPES = frozenset({"no_trade_filter", "sizer", "gamma_flow_gate"})


class CashConversionStatus(StrEnum):
    """State-machine statuses available after a frozen signal-bearing run."""

    FORECAST_ONLY = "forecast_only"
    ECONOMIC_CHAMPION = "economic_champion"
    NULL = "null"


@dataclass(frozen=True, slots=True)
class ExposureUtilizationMetrics:
    """R&D §15.4 exposure, activity, cash-drag, and normalized-PnL panel."""

    average_gross: float
    average_net: float
    utilization: float
    cash_drag: float
    realized_vol: float
    turnover: float
    trade_count: int
    holding_period: float
    active_time: float
    pnl_per_unit_gross: float
    pnl_per_unit_realized_risk: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float | int]) -> ExposureUtilizationMetrics:
        """Validate an exact metric panel, refusing omitted cash-drag accounting."""

        missing = set(_EXPOSURE_FIELDS).difference(values)
        extra = set(values).difference(_EXPOSURE_FIELDS)
        if missing or extra:
            raise ValueError(
                f"exposure-utilization metric shape mismatch; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        numeric = {key: _finite_number(values[key], key) for key in _EXPOSURE_FIELDS}
        trade_count = values["trade_count"]
        if isinstance(trade_count, bool) or not isinstance(trade_count, int) or trade_count < 0:
            raise ValueError("trade_count must be a non-negative integer")
        return cls(
            average_gross=numeric["average_gross"],
            average_net=numeric["average_net"],
            utilization=numeric["utilization"],
            cash_drag=numeric["cash_drag"],
            realized_vol=numeric["realized_vol"],
            turnover=numeric["turnover"],
            trade_count=trade_count,
            holding_period=numeric["holding_period"],
            active_time=numeric["active_time"],
            pnl_per_unit_gross=numeric["pnl_per_unit_gross"],
            pnl_per_unit_realized_risk=numeric["pnl_per_unit_realized_risk"],
        )

    def as_mapping(self) -> dict[str, float | int]:
        """Return the complete serializable metric panel."""

        return asdict(self)


def no_trade_filter_floor(
    *,
    avoided_loss: float,
    missed_profit: float,
    equal_trade_count: bool,
    sharpe_delta: float,
    sharpe_delta_floor: float,
) -> bool:
    """Whether a no-trade filter clears its equal-count economic floor."""

    return (
        avoided_loss - missed_profit > 0.0
        and equal_trade_count
        and sharpe_delta > sharpe_delta_floor
    )


def sizer_floor(
    *, utility_gain: float, vol_target_error_improved: bool, drawdown_non_worse: bool
) -> bool:
    """Whether a sizer improves utility without worsening target error or drawdown."""

    return utility_gain > 0.0 and vol_target_error_improved and drawdown_non_worse


def gamma_flow_floor(
    *,
    a3_utility: float,
    a1_utility: float,
    tail_loss_reduced: bool,
    doubled_cost_survives: bool,
    oi_lag_stress_survives: bool,
) -> bool:
    """Whether the fractal gamma overlay improves on the plain-vol control."""

    return (
        a3_utility > a1_utility
        and tail_loss_reduced
        and doubled_cost_survives
        and oi_lag_stress_survives
    )


def derive_cash_conversion_status(
    *, object_type: str, floor_satisfied: bool, frozen_signal_bearing_run: bool
) -> CashConversionStatus:
    """Derive status only after the human-frozen signal-bearing evaluation."""

    if not frozen_signal_bearing_run:
        raise ValueError("status derivation requires a frozen signal-bearing run")
    if object_type == "pure_forecast":
        return CashConversionStatus.FORECAST_ONLY
    if object_type not in _ECONOMIC_OBJECT_TYPES:
        raise ValueError(f"unsupported object_type: {object_type!r}")
    return CashConversionStatus.ECONOMIC_CHAMPION if floor_satisfied else CashConversionStatus.NULL


def _finite_number(value: float | int, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
        raise ValueError(f"{field_name} must be a finite number")
    return float(value)


__all__ = [
    "CashConversionStatus",
    "EXPOSURE_UTILIZATION_FIELDS",
    "ExposureUtilizationMetrics",
    "derive_cash_conversion_status",
    "gamma_flow_floor",
    "no_trade_filter_floor",
    "sizer_floor",
]

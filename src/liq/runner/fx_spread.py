"""FX per-pair spread table and cost application.

The cost book's ``oanda_*`` scenarios name a spread table (``spread_table``)
and a multiplier (``spread_multiplier``); this module holds the concrete
per-pair pip spreads and converts them to a per-round-trip return fraction.

Spread values are OANDA published typical spreads for the majors (pips),
transcribed conservatively; see ``FIXED_SPREAD_TABLE_V1.provenance``. The
quoted spread is charged once per round trip (enter at the far side of the
spread, exit at the near side), converted pips -> price -> return at the
trade price.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from liq.runner.cost_book import CostScenario

_JPY_QUOTE = "JPY"
_PIP_JPY = 0.01
_PIP_DEFAULT = 0.0001


class UnknownPairError(ValueError):
    """A pair is not present in the FX spread table."""


@dataclass(frozen=True)
class FxSpreadTable:
    """Versioned per-pair typical spreads in pips."""

    version: str
    provenance: str
    spreads_pips: Mapping[str, float]


FIXED_SPREAD_TABLE_V1 = FxSpreadTable(
    version="fixed_spread_table_v1",
    provenance=(
        "OANDA published typical spreads for USD majors (pips), conservative "
        "end of the typical range, retrieved 2026-07-05. Charged once per "
        "round trip."
    ),
    spreads_pips=MappingProxyType(
        {
            "EUR_USD": 1.0,
            "USD_JPY": 1.0,
            "AUD_USD": 1.2,
            "GBP_USD": 1.4,
            "USD_CAD": 1.6,
            "USD_CHF": 1.6,
            "NZD_USD": 1.8,
        }
    ),
)


def pip_size(pair: str) -> float:
    """Price increment of one pip for ``pair`` (0.01 for JPY quotes)."""
    if pair not in FIXED_SPREAD_TABLE_V1.spreads_pips:
        raise UnknownPairError(f"unknown FX pair '{pair}'")
    return _PIP_JPY if pair.endswith(_JPY_QUOTE) else _PIP_DEFAULT


def spread_cost_fraction(pair: str, *, price: float, multiplier: float) -> float:
    """Round-trip spread cost as a return fraction at ``price``."""
    spread_pips = FIXED_SPREAD_TABLE_V1.spreads_pips.get(pair)
    if spread_pips is None:
        raise UnknownPairError(f"unknown FX pair '{pair}'")
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}")
    spread_price = spread_pips * pip_size(pair) * multiplier
    return spread_price / price


def round_trip_cost_fraction(pair: str, *, price: float, scenario: CostScenario) -> float:
    """Round-trip FX cost for ``pair`` under a cost-book ``scenario``."""
    if scenario.params.get("spread_table") != FIXED_SPREAD_TABLE_V1.version:
        raise ValueError(
            f"scenario '{scenario.scenario_id}' does not reference "
            f"spread_table '{FIXED_SPREAD_TABLE_V1.version}'; wrong surface"
        )
    multiplier = float(scenario.params.get("spread_multiplier", 1.0))
    return spread_cost_fraction(pair, price=price, multiplier=multiplier)

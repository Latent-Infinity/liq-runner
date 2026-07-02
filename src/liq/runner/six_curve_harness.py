"""Six-curve harness: orchestration + IO over ``liq.metrics.six_curves``.

Assembles aligned experiment inputs, delegates all curve math to liq-metrics,
and writes one JSON artifact per experiment carrying every curve (A/B/C/D/E,
the A3 leverage-matched comparator, and the F1/F2/F3 after-tax views) plus the
metadata hash slots (``account_policy_hash``, ``optimizer_spec_hash``) that
after-tax and null-simulation consumers use to verify which frozen policy
produced the numbers. Decimal values serialize as strings for fidelity.

``convert_legacy_verdict`` keeps historical per-cell verdict payloads (the
``cells`` / ``ablation_results`` / top-level-list shapes) re-loadable as
diagnostic-only records so retroactive trial accounting remains valid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from liq.metrics.six_curves import SixCurveInputs, SixCurveResult, compute_six_curves

REQUIRED_METADATA = ("experiment_id", "account_policy_hash", "optimizer_spec_hash")
_LEGACY_CELL_KEYS = ("cells", "ablation_results", "results", "tracks")


@dataclass(frozen=True)
class LegacyVerdictSummary:
    """Historical verdict payload converted to diagnostic-only records."""

    records: list[dict[str, Any]]
    diagnostic_only: bool = True

    @property
    def n_cells(self) -> int:
        return len(self.records)


def _curve_json(values: tuple) -> list[str]:
    return [str(v) for v in values]


def _artifact(inputs: SixCurveInputs, result: SixCurveResult, metadata: dict[str, Any]) -> dict:
    return {
        "dates": [d.isoformat() for d in result.dates],
        "starting_capital": str(inputs.starting_capital),
        "sleeve_weight": str(inputs.sleeve_weight),
        "leverage": str(inputs.leverage),
        "no_tax_smoke": inputs.tax_policy.no_tax_smoke,
        "curves": {
            "a": _curve_json(result.a),
            "b": _curve_json(result.b),
            "c": _curve_json(result.c),
            "d": _curve_json(result.d),
            "e": _curve_json(result.e),
            "a3": _curve_json(result.a3),
        },
        "f_views": {
            "pre_tax_nav": str(result.f.pre_tax_nav),
            "f1_realized_tax": str(result.f.f1_realized_tax),
            "f2_liquidation_tax": str(result.f.f2_liquidation_tax),
            "f3_terminal_tax": str(result.f.f3_terminal_tax),
            "f1_nav": str(result.f.f1_nav),
            "f2_nav": str(result.f.f2_nav),
            "f3_nav": str(result.f.f3_nav),
        },
        "metadata": dict(metadata),
    }


def run_six_curve_harness(
    inputs: SixCurveInputs,
    *,
    output_path: str | Path,
    metadata: dict[str, Any],
) -> dict:
    """Compute the curve set and persist one JSON artifact; returns the artifact.

    ``metadata`` must carry ``experiment_id``, ``account_policy_hash``, and
    ``optimizer_spec_hash`` — after-tax consumers refuse unlabeled outputs, so
    the harness refuses to produce them.
    """
    missing = [key for key in REQUIRED_METADATA if key not in metadata]
    if missing:
        raise ValueError(f"harness metadata missing required hash slots: {missing}")

    result = compute_six_curves(inputs)
    artifact = _artifact(inputs, result, metadata)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def convert_legacy_verdict(payload: Any) -> LegacyVerdictSummary:
    """Convert a historical verdict payload into diagnostic-only records.

    Mirrors the retroactive-seeding cell detection: a top-level list of dicts,
    or the first present of the known cell-list keys. ``top_k_by_pf`` subsets
    are never double-counted. Anything else converts to zero records.
    """
    if isinstance(payload, list):
        return LegacyVerdictSummary(records=[c for c in payload if isinstance(c, dict)])
    if isinstance(payload, dict):
        for key in _LEGACY_CELL_KEYS:
            value = payload.get(key)
            if isinstance(value, list) and value and all(isinstance(c, dict) for c in value):
                return LegacyVerdictSummary(records=list(value))
    return LegacyVerdictSummary(records=[])

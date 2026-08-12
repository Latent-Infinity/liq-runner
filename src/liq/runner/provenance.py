"""Run provenance: the identity record every research run must emit.

A result without provenance does not exist. ``build_run_provenance`` resolves
the cost scenario from the cost book, so a run with an unknown or unnamed
scenario refuses to start before touching any data.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from liq.runner.cost_book import INTRADAY_CAMPAIGN_COST_BOOK_V1, CostBook


@dataclass(frozen=True)
class RunProvenance:
    """Identity of one research run: what code, config, data, and costs."""

    run_id: str
    code_hash: str
    config_hash: str
    cost_scenario_id: str
    cost_book_version: str
    data_hash: str | None = None
    periods_touched: tuple[tuple[str, str], ...] = ()
    seeds: Mapping[str, int] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "code_hash": self.code_hash,
            "config_hash": self.config_hash,
            "cost_scenario_id": self.cost_scenario_id,
            "cost_book_version": self.cost_book_version,
            "data_hash": self.data_hash,
            "periods_touched": [list(period) for period in self.periods_touched],
            "seeds": dict(self.seeds) if self.seeds is not None else None,
            "created_at": self.created_at.isoformat(),
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))


def build_run_provenance(
    *,
    run_id: str,
    code_hash: str,
    config_hash: str,
    cost_scenario_id: str,
    data_hash: str | None = None,
    periods_touched: Sequence[Sequence[str]] = (),
    seeds: Mapping[str, int] | None = None,
    cost_book: CostBook = INTRADAY_CAMPAIGN_COST_BOOK_V1,
) -> RunProvenance:
    """Validate identity fields, resolve the cost scenario, and build provenance."""
    if not run_id:
        raise ValueError("run_id must be set")
    if not code_hash:
        raise ValueError("code_hash must be set")
    if not config_hash:
        raise ValueError("config_hash must be set")
    scenario = cost_book.resolve(cost_scenario_id)
    return RunProvenance(
        run_id=run_id,
        code_hash=code_hash,
        config_hash=config_hash,
        cost_scenario_id=scenario.scenario_id,
        cost_book_version=cost_book.version,
        data_hash=data_hash,
        periods_touched=tuple((str(start), str(end)) for start, end in periods_touched),
        seeds=seeds,
    )


class RunReconciliationError(Exception):
    """Claimed dataset periods do not reconcile to eligible guarded reads."""


def _parse_period(period: Sequence[str], *, label: str) -> tuple[date, date]:
    try:
        start_text, end_text = period
        start = date.fromisoformat(str(start_text))
        end = date.fromisoformat(str(end_text))
    except (TypeError, ValueError) as exc:
        raise RunReconciliationError(f"invalid {label} period {period!r}") from exc
    if start > end:
        raise RunReconciliationError(f"invalid {label} period {period!r}: start is after end")
    return start, end


def reconcile_periods_touched(
    provenance: RunProvenance,
    *,
    periods_by_dataset: Mapping[str, Sequence[tuple[str, str]]],
    guarded_windows_by_dataset: Mapping[str, Sequence[tuple[str, str]]],
) -> None:
    """Reconcile every claimed dataset period to that dataset's guarded reads.

    ``periods_by_dataset`` binds each provenance period to the dataset it came
    from. ``guarded_windows_by_dataset`` must come from the lockbox usage log
    for the same arm, filtered to citable research purposes. Every dataset
    period must appear in ``provenance.periods_touched`` and be contained by a
    guarded window for that exact dataset. Conversely, every provenance period
    must be attributed to at least one dataset.

    This is a final consistency gate; it complements rather than replaces
    guarded data access at read time.
    """
    provenance_periods = {
        _parse_period(period, label="provenance") for period in provenance.periods_touched
    }
    attributed_periods: set[tuple[date, date]] = set()

    for dataset, periods in periods_by_dataset.items():
        guarded = [
            _parse_period(period, label=f"guarded {dataset}")
            for period in guarded_windows_by_dataset.get(dataset, ())
        ]
        for period in periods:
            claimed_start, claimed_end = _parse_period(period, label=f"claimed {dataset}")
            claimed = (claimed_start, claimed_end)
            if claimed not in provenance_periods:
                raise RunReconciliationError(
                    f"dataset {dataset!r} period {claimed_start}..{claimed_end} "
                    "is absent from provenance.periods_touched"
                )
            attributed_periods.add(claimed)
            if not any(
                guarded_start <= claimed_start and claimed_end <= guarded_end
                for guarded_start, guarded_end in guarded
            ):
                raise RunReconciliationError(
                    f"dataset {dataset!r} period {claimed_start}..{claimed_end} has no "
                    "covering guarded research read and is non-citable"
                )

    unattributed = provenance_periods - attributed_periods
    if unattributed:
        rendered = ", ".join(f"{start}..{end}" for start, end in sorted(unattributed))
        raise RunReconciliationError(
            f"provenance periods lack dataset attribution: {rendered}; run is non-citable"
        )

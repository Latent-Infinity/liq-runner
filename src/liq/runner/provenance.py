"""Run provenance: the identity record every research run must emit.

A result without provenance does not exist. ``build_run_provenance`` resolves
the cost scenario from the cost book, so a run with an unknown or unnamed
scenario refuses to start before touching any data.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
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

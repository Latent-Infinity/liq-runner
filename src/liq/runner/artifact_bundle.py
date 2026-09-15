"""Publish research artifacts with a completion manifest and immutable run paths."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from liq.runner.provenance import RunProvenance, reconcile_periods_touched


class ArtifactBundleError(Exception):
    """An artifact bundle cannot be safely published."""


def _check_name(name: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None:
        raise ArtifactBundleError(f"Unsafe artifact path component: {name!r}")


def write_run_bundle(
    root: Path,
    provenance: RunProvenance,
    artifacts: Mapping[str, bytes],
    *,
    periods_by_dataset: Mapping[str, Sequence[tuple[str, str]]],
    guarded_windows_by_dataset: Mapping[str, Sequence[tuple[str, str]]],
) -> Path:
    """Write an exclusive run directory, completing its manifest last.

    Guard windows must come from the eligible usage ledger; reconciliation does
    not authorize access. Failed writes retain incomplete output for diagnosis.
    """
    _check_name(provenance.run_id)
    if not provenance.data_hash or not provenance.data_hash.strip():
        raise ArtifactBundleError("A source data identity is required")
    payloads = dict(artifacts)
    if not payloads:
        raise ArtifactBundleError("At least one artifact is required")
    names = {"provenance.json", "artifact-manifest.json"}
    for name in payloads:
        _check_name(name)
        folded = name.casefold()
        if folded in names:
            raise ArtifactBundleError(f"Reserved or colliding artifact name: {name!r}")
        names.add(folded)
    reconcile_periods_touched(
        provenance,
        periods_by_dataset=periods_by_dataset,
        guarded_windows_by_dataset=guarded_windows_by_dataset,
    )
    payloads["provenance.json"] = json.dumps(provenance.to_dict(), indent=2, sort_keys=True).encode(
        "utf-8"
    )
    manifest = json.dumps(
        {
            "run_id": provenance.run_id,
            "sha256": {
                name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()
            },
        },
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    destination = root / provenance.run_id
    destination.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        with (destination / name).open("xb") as stream:
            stream.write(payload)
    temporary_manifest = destination / ".completion-pending"
    with temporary_manifest.open("xb") as stream:
        stream.write(manifest)
    temporary_manifest.rename(destination / "artifact-manifest.json")
    return destination

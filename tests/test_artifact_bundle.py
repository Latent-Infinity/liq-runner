import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from liq.runner.artifact_bundle import ArtifactBundleError, write_run_bundle
from liq.runner.provenance import RunProvenance, RunReconciliationError, build_run_provenance

PERIOD = ("2024-01-02", "2024-01-03")
WINDOWS = {"tradestation_cohort_1m": [PERIOD]}


def provenance() -> RunProvenance:
    return build_run_provenance(
        run_id="calendar-check",
        code_hash="code-identity",
        config_hash="config-identity",
        data_hash="source-identity",
        cost_scenario_id="spy_qqq_base_v1",
        periods_touched=[PERIOD],
        seeds={"control": 20260910},
    )


def test_writes_bound_artifacts_and_completion_manifest(tmp_path: Path) -> None:
    payloads = {"memo.md": b"Software artifact test.\n", "details.json": b"{}\n"}
    directory = write_run_bundle(
        tmp_path,
        provenance(),
        payloads,
        periods_by_dataset=WINDOWS,
        guarded_windows_by_dataset=WINDOWS,
    )
    manifest = json.loads((directory / "artifact-manifest.json").read_text())
    assert manifest["run_id"] == "calendar-check"
    assert set(manifest["sha256"]) == {"memo.md", "details.json", "provenance.json"}
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    assert json.loads((directory / "provenance.json").read_text())["data_hash"] == "source-identity"


def test_existing_run_is_never_overwritten(tmp_path: Path) -> None:
    directory = tmp_path / "calendar-check"
    directory.mkdir()
    (directory / "memo.md").write_bytes(b"Original")
    with pytest.raises(FileExistsError):
        write_run_bundle(
            tmp_path,
            provenance(),
            {"memo.md": b"Replacement"},
            periods_by_dataset=WINDOWS,
            guarded_windows_by_dataset=WINDOWS,
        )
    assert (directory / "memo.md").read_bytes() == b"Original"


@pytest.mark.parametrize(
    "name",
    ["../escape", "/absolute", "nested/file", ".", "", "provenance.json", "ARTIFACT-MANIFEST.JSON"],
)
def test_invalid_artifact_names_rejected_before_publication(tmp_path: Path, name: str) -> None:
    with pytest.raises(ArtifactBundleError):
        write_run_bundle(
            tmp_path,
            provenance(),
            {name: b"test"},
            periods_by_dataset=WINDOWS,
            guarded_windows_by_dataset=WINDOWS,
        )
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("run_id", ["../escape", "/absolute", ".", ""])
def test_invalid_run_identity_rejected(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ArtifactBundleError):
        write_run_bundle(
            tmp_path,
            replace(provenance(), run_id=run_id),
            {"memo.md": b"test"},
            periods_by_dataset=WINDOWS,
            guarded_windows_by_dataset=WINDOWS,
        )
    assert not tuple(tmp_path.iterdir())


def test_periods_must_reconcile_before_writing(tmp_path: Path) -> None:
    with pytest.raises(RunReconciliationError):
        write_run_bundle(
            tmp_path,
            provenance(),
            {"memo.md": b"test"},
            periods_by_dataset=WINDOWS,
            guarded_windows_by_dataset={},
        )
    assert not tuple(tmp_path.iterdir())


def test_missing_data_identity_rejected(tmp_path: Path) -> None:
    with pytest.raises(ArtifactBundleError):
        write_run_bundle(
            tmp_path,
            replace(provenance(), data_hash=None),
            {"memo.md": b"test"},
            periods_by_dataset=WINDOWS,
            guarded_windows_by_dataset=WINDOWS,
        )


def test_case_colliding_names_rejected_before_writing(tmp_path: Path) -> None:
    with pytest.raises(ArtifactBundleError):
        write_run_bundle(
            tmp_path,
            provenance(),
            {"Memo.md": b"one", "memo.md": b"two"},
            periods_by_dataset=WINDOWS,
            guarded_windows_by_dataset=WINDOWS,
        )
    assert not tuple(tmp_path.iterdir())


def test_empty_bundle_rejected(tmp_path: Path) -> None:
    with pytest.raises(ArtifactBundleError):
        write_run_bundle(
            tmp_path,
            provenance(),
            {},
            periods_by_dataset=WINDOWS,
            guarded_windows_by_dataset=WINDOWS,
        )


def test_interrupted_publication_has_no_completion_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupt(source: Path, target: Path) -> Path:
        raise OSError("Simulated publication interruption")

    monkeypatch.setattr(Path, "rename", interrupt)
    with pytest.raises(OSError, match="publication interruption"):
        write_run_bundle(
            tmp_path,
            provenance(),
            {"memo.md": b"Software check"},
            periods_by_dataset=WINDOWS,
            guarded_windows_by_dataset=WINDOWS,
        )
    directory = tmp_path / "calendar-check"
    assert (directory / "memo.md").read_bytes() == b"Software check"
    assert not (directory / "artifact-manifest.json").exists()

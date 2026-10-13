"""Supervisor claims remain attached to their original directory identities."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from liq.runner import bounded_attempt
from liq.runner.bounded_attempt import AttemptSpec, run_once


def test_parent_claim_records_directory_identity(tmp_path: Path) -> None:
    destination = tmp_path / "attempt"
    spec = AttemptSpec((sys.executable, "-c", "pass"), b"{}", 5.0)

    run_once(destination, spec)

    identity = destination.stat()
    assert json.loads((destination / "claim.json").read_bytes()) == {
        "device": identity.st_dev,
        "inode": identity.st_ino,
    }


@pytest.mark.parametrize("replacement", ["symlink", "directory"])
def test_replaced_directory_retains_claim_without_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement: str
) -> None:
    destination = tmp_path / "attempt"
    retained = tmp_path / "retained"

    def replace_claim(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert (destination / "attempt.json").read_bytes() == b"{}"
        assert (destination / "claim.json").is_file()
        destination.rename(retained)
        if replacement == "symlink":
            destination.symlink_to(retained, target_is_directory=True)
        else:
            destination.mkdir()
        return subprocess.CompletedProcess([sys.executable], 0)

    monkeypatch.setattr(bounded_attempt.subprocess, "run", replace_claim)
    spec = AttemptSpec((sys.executable, "-c", "pass"), b"{}", 5.0)
    with pytest.raises(PermissionError, match="Claim directory path changed"):
        run_once(destination, spec)

    assert (retained / "attempt.json").read_bytes() == b"{}"
    assert (retained / "claim.json").is_file()
    assert not (retained / "supervisor.json").exists()
    assert not (destination / "supervisor.json").exists()

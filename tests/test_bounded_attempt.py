"""One-use process supervision without broker access or market observations."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from liq.runner.bounded_attempt import AttemptSpec, publish_once, run_once


def test_nested_claim_entries_are_durable_before_worker_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from liq.runner import bounded_attempt

    destination = tmp_path / "new-parent" / "attempt"
    synced: set[tuple[int, int]] = set()
    original_sync = os.fsync

    def track(descriptor: int) -> None:
        identity = os.fstat(descriptor)
        original_sync(descriptor)
        synced.add((identity.st_dev, identity.st_ino))

    def launch(
        command: tuple[str, ...],
        *,
        stdin: int,
        stdout: int,
        stderr: int,
        check: bool,
        timeout: float,
        cwd: Path,
    ) -> subprocess.CompletedProcess[bytes]:
        for parent in (tmp_path, destination.parent):
            identity = parent.stat()
            assert (identity.st_dev, identity.st_ino) in synced
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(os, "fsync", track)
    monkeypatch.setattr(bounded_attempt.subprocess, "run", launch)
    run_once(destination, AttemptSpec((sys.executable, "-c", "pass"), b"{}", 5.0))


def test_run_records_claim_before_child_starts(tmp_path: Path) -> None:
    destination = tmp_path / "attempt"
    code = "from pathlib import Path; assert Path('attempt.json').read_bytes() == b'{}'"
    spec = AttemptSpec((sys.executable, "-c", code), b"{}", 5.0)

    result = run_once(destination, spec)

    assert result.category == "COMPLETED"
    assert result.returncode == 0
    assert json.loads((destination / "supervisor.json").read_bytes())["returncode"] == 0


def test_claim_prevents_second_process_even_after_failure(tmp_path: Path) -> None:
    destination = tmp_path / "attempt"
    spec = AttemptSpec((sys.executable, "-c", "raise SystemExit(3)"), b"{}", 5.0)
    run_once(destination, spec)
    original = (destination / "supervisor.json").read_bytes()

    with pytest.raises(FileExistsError):
        run_once(destination, spec)

    assert (destination / "supervisor.json").read_bytes() == original


def test_timeout_kills_worker_and_retains_claim(tmp_path: Path) -> None:
    spec = AttemptSpec((sys.executable, "-c", "import signal; signal.pause()"), b"{}", 0.2)

    result = run_once(tmp_path / "attempt", spec)

    assert result.category == "TIMED_OUT"
    assert result.returncode is None
    assert (tmp_path / "attempt" / "attempt.json").read_bytes() == b"{}"


def test_spawn_failure_is_redacted_and_retained(tmp_path: Path) -> None:
    spec = AttemptSpec(("/missing/private-account-program",), b"{}", 5.0)

    result = run_once(tmp_path / "attempt", spec)

    assert result.category == "SPAWN_ERROR"
    assert b"private-account" not in (tmp_path / "attempt" / "supervisor.json").read_bytes()


def test_publish_never_replaces_existing_record(tmp_path: Path) -> None:
    target = tmp_path / "record.json"
    publish_once(target, b"original")

    with pytest.raises(FileExistsError):
        publish_once(target, b"replacement")

    assert target.read_bytes() == b"original"


@pytest.mark.parametrize("seconds", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_deadline_refuses_before_claim(tmp_path: Path, seconds: float) -> None:
    with pytest.raises(ValueError):
        AttemptSpec((sys.executable,), b"{}", seconds)

    assert list(tmp_path.iterdir()) == []

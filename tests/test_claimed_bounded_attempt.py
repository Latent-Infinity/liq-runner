"""Actual subprocess claim identity, never broker access or financial data."""

import json
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import pytest

from liq.runner import bounded_attempt as attempts
from liq.runner.pinned_directory import PinnedDirectory


@pytest.mark.parametrize("exitcode", [0, 7])
def test_claimed_result_preserves_original_identity_and_supervisor_schema(
    tmp_path: Path,
    exitcode: int,
) -> None:
    # Given one real child and a fresh private claim path.
    destination = tmp_path / "attempt"
    spec = attempts.AttemptSpec((sys.executable, "-c", f"raise SystemExit({exitcode})"), b"{}", 5)
    # When the claimed-result supervisor executes once.
    result = attempts.run_once_claimed(destination, spec, private=True)
    # Then the original claim and legacy three-field outcome are returned separately.
    claim = json.loads((destination / "claim.json").read_bytes())
    assert (result.device, result.inode) == (claim["device"], claim["inode"])
    assert result.outcome.category == "COMPLETED" and result.outcome.returncode == exitcode
    assert json.loads((destination / "supervisor.json").read_bytes()) == asdict(result.outcome)
    with pytest.raises(FileExistsError):
        attempts.run_once_claimed(destination, spec, private=True)


def test_timeout_retains_the_original_claim_without_relaunch(tmp_path: Path) -> None:
    # Given a genuinely blocked child and a finite supervisor deadline.
    destination = tmp_path / "attempt"
    spec = attempts.AttemptSpec((sys.executable, "-c", "import signal; signal.pause()"), b"{}", 0.2)
    # When the supervisor deadline expires.
    result = attempts.run_once_claimed(destination, spec, private=True)
    # Then the timed-out outcome and original inode remain available without reuse.
    claim = json.loads((destination / "claim.json").read_bytes())
    assert result.outcome.category == "TIMED_OUT"
    assert (result.device, result.inode) == (claim["device"], claim["inode"])
    with pytest.raises(FileExistsError):
        attempts.run_once_claimed(destination, spec, private=True)


def test_spawn_error_retains_only_redacted_legacy_supervisor_facts(tmp_path: Path) -> None:
    # Given an unavailable program, never an account or secret-bearing command.
    destination = tmp_path / "attempt"
    spec = attempts.AttemptSpec(("/missing/unit-claim-program",), b"{}", 5)
    # When child spawning fails.
    result = attempts.run_once_claimed(destination, spec, private=True)
    # Then no native command details enter the unchanged supervisor record.
    assert result.outcome.category == "SPAWN_ERROR"
    assert b"unit-claim-program" not in (destination / "supervisor.json").read_bytes()
    claim = json.loads((destination / "claim.json").read_bytes())
    assert (result.device, result.inode) == (claim["device"], claim["inode"])


@pytest.mark.parametrize(
    "code",
    [
        "import os; os.chmod('.', 0o755)",
        "from pathlib import Path; p=Path.cwd(); p.rename(p.with_name('retained')); p.mkdir(mode=0o700)",
    ],
)
def test_private_drift_cannot_return_a_successful_claim(tmp_path: Path, code: str) -> None:
    # Given a child that changes the actual private claim rather than a mock result.
    destination = tmp_path / "attempt"
    spec = attempts.AttemptSpec((sys.executable, "-c", code), b"{}", 5)
    # When the supervisor reconciles the held original directory.
    with pytest.raises(PermissionError):
        attempts.run_once_claimed(destination, spec, private=True)
    # Then no successful supervisor record is published into the changed path.
    assert not (destination / "supervisor.json").exists()


def test_returned_identity_is_captured_before_release_not_from_reopened_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given real supervision with a path replaced only after the original capability closes.
    destination = tmp_path / "attempt"
    original_pin = attempts.pin_directory
    identities: list[tuple[int, int]] = []

    @contextmanager
    def replaced_after_release(
        path: Path, *, create: bool = False, private: bool = False
    ) -> Iterator[PinnedDirectory]:
        with original_pin(path, create=create, private=private) as directory:
            identities.append(directory.identity)
            yield directory
        retained = path.with_name("retained")
        path.rename(retained)
        shutil.copytree(retained, path)

    monkeypatch.setattr(attempts, "pin_directory", replaced_after_release)
    spec = attempts.AttemptSpec((sys.executable, "-c", "pass"), b"{}", 5)
    # When the caller receives the result after that path replacement.
    result = attempts.run_once_claimed(destination, spec, private=True)
    # Then the returned identity remains the pre-launch original, not the new directory.
    current = destination.stat()
    assert (result.device, result.inode) == identities[0]
    assert (result.device, result.inode) != (current.st_dev, current.st_ino)


def test_failed_claim_publication_returns_no_original_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given a publication failure before the child may launch.
    destination = tmp_path / "attempt"

    def refused_publish(self: PinnedDirectory, name: str, payload: bytes) -> None:
        raise OSError("unit-publication-control")

    monkeypatch.setattr(PinnedDirectory, "publish", refused_publish)
    spec = attempts.AttemptSpec((sys.executable, "-c", "pass"), b"{}", 5)
    # When the claimed-result API cannot durably publish its claim.
    with pytest.raises(OSError):
        attempts.run_once_claimed(destination, spec, private=True)
    # Then no attempt or supervisor record exists to imply successful launch.
    assert not (destination / "attempt.json").exists()
    assert not (destination / "supervisor.json").exists()

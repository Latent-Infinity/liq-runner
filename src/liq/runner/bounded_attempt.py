"""One-use, failure-retaining supervision for bounded verification workers."""

import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from liq.runner.pinned_directory import pin_directory


@dataclass(frozen=True, slots=True)
class ClaimedAttemptOutcome:
    """Original pre-launch claim identity, separate from legacy supervisor facts."""

    outcome: "AttemptOutcome"
    device: int
    inode: int


def claim_directory(destination: Path) -> None:
    """Exclusively create a one-use directory, durably syncing every new entry."""
    with pin_directory(destination, create=True):
        pass


@dataclass(frozen=True, slots=True)
class AttemptSpec:
    """A child command, pre-request receipt, and finite wall-time budget."""

    command: tuple[str, ...]
    receipt: bytes
    timeout_seconds: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("A finite positive timeout is required")
        if not self.command:
            raise ValueError("A worker command is required")


@dataclass(frozen=True, slots=True)
class AttemptOutcome:
    """Supervisor facts only; worker output must be separately reconciled."""

    category: Literal["COMPLETED", "TIMED_OUT", "SPAWN_ERROR"]
    returncode: int | None
    elapsed_seconds: float


def publish_once(path: Path, payload: bytes) -> None:
    """Atomically publish a durable complete file without replacing any target."""
    with pin_directory(path.parent) as directory:
        directory.publish(path.name, payload)


def run_once(destination: Path, spec: AttemptSpec, *, private: bool = False) -> AttemptOutcome:
    """Claim before launch, kill overdue child, and retain every claimed attempt.

    The worker must not spawn descendants. Its stdout/stderr are discarded to
    prevent accidental credential disclosure; it publishes minimized artifacts
    separately. A supervisor receipt is not a claim that the worker succeeded.
    Timeout bounds worker execution; durable supervisor bookkeeping follows it.
    Private mode uses owner-only claims and refuses permission drift; it does
    not grant data-access authority or defend against a malicious same-UID child.
    """
    return _run_once(destination, spec, private=private).outcome


def run_once_claimed(
    destination: Path, spec: AttemptSpec, *, private: bool = False
) -> ClaimedAttemptOutcome:
    """Retain the original held claim identity without changing supervisor records."""
    return _run_once(destination, spec, private=private)


def _run_once(destination: Path, spec: AttemptSpec, *, private: bool) -> ClaimedAttemptOutcome:
    started = time.monotonic()
    with pin_directory(destination, create=True, private=private) as directory:
        device, inode = directory.identity
        directory.publish("claim.json", json.dumps({"device": device, "inode": inode}).encode())
        directory.publish("attempt.json", spec.receipt)
        directory.check()
        outcome = _supervise(destination, spec, started)
        directory.publish("supervisor.json", json.dumps(asdict(outcome)).encode())
        return ClaimedAttemptOutcome(outcome, device, inode)


def _supervise(destination: Path, spec: AttemptSpec, started: float) -> AttemptOutcome:
    remaining = spec.timeout_seconds - (time.monotonic() - started)
    category: Literal["COMPLETED", "TIMED_OUT", "SPAWN_ERROR"] = "TIMED_OUT"
    returncode: int | None = None
    if remaining > 0:
        try:
            completed = subprocess.run(
                spec.command,
                cwd=destination,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=remaining,
                check=False,
            )
            category = "COMPLETED"
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            category = "TIMED_OUT"
        except OSError:
            category = "SPAWN_ERROR"
    return AttemptOutcome(category, returncode, time.monotonic() - started)

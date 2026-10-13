"""Private publication validates the new file before payload and final linking."""

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, NoReturn

import pytest

from liq.runner.pinned_directory import pin_directory


class _WriteInterrupted(OSError):
    """A test-only stream failure after a real prefix has been flushed."""


@dataclass(frozen=True, slots=True)
class _InterruptedWriter:
    stream: BinaryIO

    def fileno(self) -> int:
        return self.stream.fileno()

    def write(self, payload: bytes) -> NoReturn:
        self.stream.write(payload[:3])
        self.stream.flush()
        assert os.fstat(self.stream.fileno()).st_size == 3
        raise _WriteInterrupted("WRITE_INTERRUPTED")


def test_partial_payload_write_failure_removes_pending_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_fdopen = os.fdopen

    @contextmanager
    def interrupted(descriptor: int, mode: str) -> Iterator[_InterruptedWriter]:
        assert mode == "wb"
        with original_fdopen(descriptor, "wb") as stream:
            yield _InterruptedWriter(stream)

    with pin_directory(tmp_path, private=True) as directory:
        directory.publish("claim.bin", b"possible exposure")
        monkeypatch.setattr(os, "fdopen", interrupted)
        with pytest.raises(_WriteInterrupted, match="^WRITE_INTERRUPTED$"):
            directory.publish("observation.bin", b"partial nonfinancial test payload")
    assert not (tmp_path / "observation.bin").exists()
    assert not list(tmp_path.glob(".pending-*"))
    assert (tmp_path / "claim.bin").read_bytes() == b"possible exposure"
    assert stat.S_IMODE((tmp_path / "claim.bin").stat().st_mode) == 0o600


def test_unsafe_pending_file_is_refused_before_payload_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_fchmod = os.fchmod
    retained: list[int] = []

    def wrong_mode(descriptor: int, mode: int) -> None:
        original_fchmod(descriptor, 0o644)
        retained.append(os.dup(descriptor))

    monkeypatch.setattr(os, "fchmod", wrong_mode)
    try:
        with pin_directory(tmp_path, private=True) as directory, pytest.raises(PermissionError):
            directory.publish("receipt.bin", b"must not write this payload")
        assert len(retained) == 1
        assert os.fstat(retained[0]).st_size == 0
        assert not list(tmp_path.iterdir())
    finally:
        for descriptor in retained:
            os.close(descriptor)


def test_foreign_pending_owner_is_refused_before_payload_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_fstat = os.fstat
    observed_sizes: list[int] = []

    def wrong_owner(descriptor: int) -> os.stat_result:
        info = original_fstat(descriptor)
        if stat.S_ISREG(info.st_mode):
            observed_sizes.append(info.st_size)
            fields = list(info)
            fields[4] = os.geteuid() + 1
            return os.stat_result(fields)
        return info

    monkeypatch.setattr(os, "fstat", wrong_owner)
    with pin_directory(tmp_path, private=True) as directory, pytest.raises(PermissionError):
        directory.publish("receipt.bin", b"must not write this payload")
    assert observed_sizes == [0]
    assert not list(tmp_path.iterdir())


def test_private_file_mode_drift_before_link_refuses_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_fsync = os.fsync

    def change_file_mode(descriptor: int) -> None:
        original_fsync(descriptor)
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.fchmod(descriptor, 0o644)

    monkeypatch.setattr(os, "fsync", change_file_mode)
    with pin_directory(tmp_path, private=True) as directory, pytest.raises(PermissionError):
        directory.publish("receipt.bin", b"nonfinancial test payload")
    assert not list(tmp_path.iterdir())


def test_fchmod_error_refuses_without_pending_or_final_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(descriptor: int, mode: int) -> None:
        raise PermissionError("Permission change refused")

    monkeypatch.setattr(os, "fchmod", refuse)
    with pin_directory(tmp_path, private=True) as directory, pytest.raises(PermissionError):
        directory.publish("receipt.bin", b"must not write this payload")
    assert not list(tmp_path.iterdir())


def test_private_read_rechecks_file_permissions_before_returning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_fstat = os.fstat
    calls = 0
    with pin_directory(tmp_path, private=True) as directory:
        directory.publish("receipt.bin", b"nonfinancial test payload")

        def drift(descriptor: int) -> os.stat_result:
            nonlocal calls
            info = original_fstat(descriptor)
            if stat.S_ISREG(info.st_mode):
                calls += 1
                if calls == 2:
                    os.fchmod(descriptor, 0o644)
                    return original_fstat(descriptor)
            return info

        monkeypatch.setattr(os, "fstat", drift)
        with pytest.raises(PermissionError):
            directory.read("receipt.bin")
    assert calls == 2
    assert stat.S_IMODE((tmp_path / "receipt.bin").stat().st_mode) == 0o644

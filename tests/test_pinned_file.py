"""Bounded authority-file witnesses use real filesystem fault injection."""

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Literal, assert_never

import pytest

from liq.runner import pinned_file
from liq.runner.pinned_file import PinnedFileRefused, read_pinned_file


def test_regular_file_returns_exact_hidden_bytes_and_identity(tmp_path: Path) -> None:
    target = tmp_path / "authority.json"
    payload = b'{"nonfinancial-control":true}\n'
    target.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    witness = read_pinned_file(target, expected, len(payload))

    assert witness.payload == payload and witness.sha256 == expected
    assert witness.state.inode == target.stat().st_ino
    assert witness.ancestors[-1] == (tmp_path.stat().st_dev, tmp_path.stat().st_ino)
    assert payload.decode().strip() not in repr(witness)


@pytest.mark.parametrize("fault", ["symlink", "hardlink", "fifo", "writable"])
def test_unsafe_record_kind_or_permissions_is_refused(
    tmp_path: Path, fault: Literal["symlink", "hardlink", "fifo", "writable"]
) -> None:
    target = tmp_path / "record.bin"
    original = tmp_path / "original.bin"
    payload = b"nonfinancial control"
    original.write_bytes(payload)
    match fault:
        case "symlink":
            target.symlink_to(original)
        case "hardlink":
            os.link(original, target)
        case "fifo":
            os.mkfifo(target)
        case "writable":
            target.write_bytes(payload)
            target.chmod(0o666)
        case unreachable:
            assert_never(unreachable)

    with pytest.raises(PinnedFileRefused):
        read_pinned_file(target, hashlib.sha256(payload).hexdigest(), 1024)


@pytest.mark.parametrize("fault", ["file", "ancestor", "mutation"])
def test_boundary_replacement_is_refused_even_when_replacement_bytes_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: Literal["file", "ancestor", "mutation"],
) -> None:
    parent = tmp_path / "records"
    parent.mkdir()
    target = parent / "record.bin"
    payload = b"original nonfinancial control"
    target.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    original_fstat = os.fstat
    file_checks = 0

    def interfere(descriptor: int) -> os.stat_result:
        nonlocal file_checks
        info = original_fstat(descriptor)
        if stat.S_ISREG(info.st_mode):
            file_checks += 1
            if file_checks == 2:
                match fault:
                    case "file":
                        target.rename(parent / "retained.bin")
                        target.write_bytes(payload)
                    case "ancestor":
                        parent.rename(tmp_path / "retained-directory")
                        parent.mkdir()
                        target.write_bytes(payload)
                    case "mutation":
                        target.write_bytes(b"changed nonfinancial control")
                    case unreachable:
                        assert_never(unreachable)
                return original_fstat(descriptor) if fault == "mutation" else info
        return info

    monkeypatch.setattr(pinned_file.os, "fstat", interfere)

    with pytest.raises(PinnedFileRefused):
        read_pinned_file(target, expected, 1024)


def test_oversize_record_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "record.bin"
    target.write_bytes(b"nonfinancial control")

    with pytest.raises(PinnedFileRefused):
        read_pinned_file(target, hashlib.sha256(target.read_bytes()).hexdigest(), 1)


def test_digest_mismatch_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "record.bin"
    target.write_bytes(b"nonfinancial control")

    with pytest.raises(PinnedFileRefused):
        read_pinned_file(target, "a" * 64, 1024)


@pytest.mark.parametrize("fault", ["short", "growth"])
def test_bounded_read_refuses_truncation_and_growth_at_the_read_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: Literal["short", "growth"]
) -> None:
    target = tmp_path / "record.bin"
    payload = b"nonfinancial control"
    target.write_bytes(payload)
    original_fdopen = os.fdopen

    @dataclass(frozen=True, slots=True)
    class ControlledReader:
        source: BinaryIO

        def fileno(self) -> int:
            return self.source.fileno()

        def read(self, maximum: int) -> bytes:
            match fault:
                case "short":
                    return self.source.read(1)
                case "growth":
                    with target.open("ab") as growing:
                        growing.write(b"additional nonfinancial control")
                    return self.source.read(maximum)
                case unreachable:
                    assert_never(unreachable)

        def __enter__(self) -> "ControlledReader":
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            self.source.close()

    def controlled_fdopen(descriptor: int, mode: Literal["rb"]) -> ControlledReader:
        return ControlledReader(original_fdopen(descriptor, mode))

    monkeypatch.setattr(pinned_file.os, "fdopen", controlled_fdopen)

    with pytest.raises(PinnedFileRefused):
        read_pinned_file(target, hashlib.sha256(payload).hexdigest(), len(payload))


def test_wrong_effective_owner_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "record.bin"
    payload = b"nonfinancial control"
    target.write_bytes(payload)
    current_owner = os.geteuid()
    monkeypatch.setattr(pinned_file.os, "geteuid", lambda: current_owner + 1)

    with pytest.raises(PinnedFileRefused):
        read_pinned_file(target, hashlib.sha256(payload).hexdigest(), 1024)


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_byte_cap_is_refused_before_opening(tmp_path: Path, limit: int) -> None:
    with pytest.raises(PinnedFileRefused) as stopped:
        read_pinned_file(tmp_path / "absent.bin", "a" * 64, limit)

    assert stopped.value.reason == "LIMIT_REFUSED"


@pytest.mark.parametrize("path", ["nonfinancial.bin", "//nonfinancial.bin", "/nonfinancial\0.bin"])
def test_noncanonical_path_is_refused_before_opening(path: str) -> None:
    with pytest.raises(PinnedFileRefused) as stopped:
        read_pinned_file(Path(path), "a" * 64, 1024)

    assert stopped.value.reason == "PATH_REFUSED"


def test_missing_record_refusal_excludes_native_path_and_exception_context(tmp_path: Path) -> None:
    with pytest.raises(PinnedFileRefused) as stopped:
        read_pinned_file(tmp_path / "nonfinancial-sensitive-name.bin", "a" * 64, 1024)

    assert stopped.value.reason == "FILE_ACCESS_REFUSED"
    assert "nonfinancial-sensitive-name" not in str(stopped.value)
    assert stopped.value.__suppress_context__ is True


def test_missing_posix_facility_is_refused_before_opening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as controlled:
        controlled.delattr(pinned_file.os, "O_NOFOLLOW")

        with pytest.raises(PinnedFileRefused) as stopped:
            read_pinned_file(tmp_path / "absent.bin", "a" * 64, 1024)

    assert stopped.value.reason == "PLATFORM_REFUSED"


@pytest.mark.parametrize("digest", ["short", "A" * 64])
def test_malformed_reviewed_digest_is_refused_before_opening(tmp_path: Path, digest: str) -> None:
    with pytest.raises(PinnedFileRefused) as stopped:
        read_pinned_file(tmp_path / "absent.bin", digest, 1024)

    assert stopped.value.reason == "DIGEST_REFUSED"

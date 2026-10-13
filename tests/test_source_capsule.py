"""Reviewed code snapshots, with no market data or transport."""

import hashlib
import tarfile
from pathlib import Path
from typing import Literal, assert_never

import pytest

from liq.runner.source_capsule import SourceCapsuleError, SourcePin, archive_sources


def test_capsule_retains_exact_reviewed_bytes_after_workspace_changes(tmp_path: Path) -> None:
    original = b"VALUE = 1\n"
    source = tmp_path / "example.py"
    source.write_bytes(original)
    pin = SourcePin("example.py", hashlib.sha256(original).hexdigest())
    target = tmp_path / "snapshot.tar.gz"

    digest = archive_sources(tmp_path, target, (pin,))
    source.write_bytes(b"VALUE = 2\n")

    assert digest == hashlib.sha256(target.read_bytes()).hexdigest()
    with tarfile.open(target) as archive:
        assert archive.getnames() == ["example.py"]
        member = archive.extractfile("example.py")
        assert member is not None
        with member:
            assert member.read() == original


def test_nested_inventory_is_exact_and_independent_of_input_order(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_bytes(b"a = 1\n")
    (tmp_path / "b.py").write_bytes(b"b = 2\n")
    (tmp_path / "unreviewed.txt").write_bytes(b"not in inventory")
    pins = (
        SourcePin("src/a.py", hashlib.sha256(b"a = 1\n").hexdigest()),
        SourcePin("b.py", hashlib.sha256(b"b = 2\n").hexdigest()),
    )

    digest = archive_sources(tmp_path, tmp_path / "first.tar.gz", pins)
    reordered = archive_sources(tmp_path, tmp_path / "second.tar.gz", tuple(reversed(pins)))

    assert digest == reordered
    with tarfile.open(tmp_path / "first.tar.gz") as archive:
        assert archive.getnames() == ["b.py", "src/a.py"]


def test_capsule_is_deterministic_and_write_once(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_bytes(b"x = 1\n")
    pin = SourcePin("code.py", hashlib.sha256(b"x = 1\n").hexdigest())
    target = tmp_path / "first.tar.gz"
    first = archive_sources(tmp_path, target, (pin,))

    second = archive_sources(tmp_path, tmp_path / "second.tar.gz", (pin,))

    assert first == second
    with pytest.raises(FileExistsError):
        archive_sources(tmp_path, target, (pin,))
    assert hashlib.sha256(target.read_bytes()).hexdigest() == first


@pytest.mark.parametrize(
    "name", ["../code.py", "/code.py", "a/../code.py", "./code.py", "a//b", "a\\b", ""]
)
def test_noncanonical_source_names_are_refused_before_publication(
    tmp_path: Path, name: str
) -> None:
    target = tmp_path / "snapshot.tar.gz"

    with pytest.raises(SourceCapsuleError):
        archive_sources(tmp_path, target, (SourcePin(name, "0" * 64),))

    assert not target.exists()


@pytest.mark.parametrize(
    "mode", ["mismatch", "duplicate", "empty", "symlink", "ancestor_symlink", "directory"]
)
def test_unreviewed_or_ambiguous_sources_never_publish(
    tmp_path: Path,
    mode: Literal["mismatch", "duplicate", "empty", "symlink", "ancestor_symlink", "directory"],
) -> None:
    source = tmp_path / "code.py"
    source.write_bytes(b"x = 1\n")
    pin = SourcePin("code.py", hashlib.sha256(source.read_bytes()).hexdigest())
    pins = (pin,)
    match mode:
        case "mismatch":
            source.write_bytes(b"x = 2\n")
        case "duplicate":
            pins = (pin, pin)
        case "empty":
            pins = ()
        case "symlink":
            source.rename(tmp_path / "original.py")
            source.symlink_to(tmp_path / "original.py")
        case "ancestor_symlink":
            (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
            pins = (SourcePin("link/code.py", pin.sha256),)
        case "directory":
            pins = (SourcePin("directory", pin.sha256),)
            (tmp_path / "directory").mkdir()
        case unreachable:
            assert_never(unreachable)
    target = tmp_path / "snapshot.tar.gz"

    with pytest.raises(SourceCapsuleError):
        archive_sources(tmp_path, target, pins)

    assert not target.exists()

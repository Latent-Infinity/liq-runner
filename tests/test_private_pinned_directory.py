"""Private quarantine storage refuses unsafe permissions without repairing them."""

import os
import stat
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

from liq.runner.pinned_directory import pin_directory


def test_private_creation_is_atomic_under_permissive_umask(tmp_path: Path) -> None:
    result = subprocess.run(
        (
            sys.executable,
            "-c",
            dedent("""
                import os
                import stat
                import sys
                from pathlib import Path
                from liq.runner.pinned_directory import pin_directory

                previous_mask = os.umask(0)
                target = Path(sys.argv[1]) / "private" / "attempt"
                try:
                    with pin_directory(target, create=True, private=True) as directory:
                        assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
                        assert stat.S_IMODE(target.stat().st_mode) == 0o700
                        directory.publish("receipt.bin", b"nonfinancial test payload")
                        assert stat.S_IMODE((target / "receipt.bin").stat().st_mode) == 0o600
                finally:
                    os.umask(previous_mask)
            """),
            str(tmp_path),
        ),
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "CoverageWarning" not in result.stderr


def test_private_publication_sets_exact_file_mode_under_restrictive_umask(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        (
            sys.executable,
            "-c",
            dedent("""
                import os
                import stat
                import sys
                from pathlib import Path
                from liq.runner.pinned_directory import pin_directory

                target = Path(sys.argv[1]) / "attempt"
                with pin_directory(target, create=True, private=True) as directory:
                    previous_mask = os.umask(0o777)
                    try:
                        directory.publish("receipt.bin", b"nonfinancial test payload")
                        assert stat.S_IMODE((target / "receipt.bin").stat().st_mode) == 0o600
                        assert directory.read("receipt.bin") == b"nonfinancial test payload"
                    finally:
                        os.umask(previous_mask)
            """),
            str(tmp_path),
        ),
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "CoverageWarning" not in result.stderr


@pytest.mark.parametrize("mode", [0o750, 0o755, 0o770, 0o777])
def test_private_open_refuses_existing_unsafe_directory_without_chmod(
    tmp_path: Path, mode: int
) -> None:
    target = tmp_path / "existing"
    target.mkdir()
    target.chmod(mode)
    with pytest.raises(PermissionError), pin_directory(target, private=True):
        pytest.fail("Unsafe directory must be refused before yielding")
    assert stat.S_IMODE(target.stat().st_mode) == mode
    assert not list(target.iterdir())


def test_private_open_refuses_a_different_effective_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    effective_uid = os.geteuid()
    monkeypatch.setattr(os, "geteuid", lambda: effective_uid + 1)
    with pytest.raises(PermissionError), pin_directory(tmp_path, private=True):
        pytest.fail("Foreign-owned directory must be refused before yielding")


def test_private_directory_permission_drift_refuses_publication(tmp_path: Path) -> None:
    target = tmp_path / "attempt"
    with (
        pytest.raises(PermissionError),
        pin_directory(target, create=True, private=True) as directory,
    ):
        directory.publish("receipt.bin", b"possible exposure")
        target.chmod(0o755)
        directory.publish("observation.bin", b"must not publish")
    assert (target / "receipt.bin").read_bytes() == b"possible exposure"
    assert not (target / "observation.bin").exists()


@pytest.mark.parametrize("mode", [0o400, 0o644, 0o666])
def test_private_read_refuses_unsafe_record_mode_without_repair(tmp_path: Path, mode: int) -> None:
    with pin_directory(tmp_path, private=True) as directory:
        directory.publish("receipt.bin", b"nonfinancial test payload")
        target = tmp_path / "receipt.bin"
        target.chmod(mode)
        with pytest.raises(PermissionError):
            directory.read("receipt.bin")
        assert stat.S_IMODE(target.stat().st_mode) == mode


def test_private_publication_remains_write_once(tmp_path: Path) -> None:
    with pin_directory(tmp_path, private=True) as directory:
        directory.publish("receipt.bin", b"original")
        with pytest.raises(FileExistsError):
            directory.publish("receipt.bin", b"replacement")
        assert directory.read("receipt.bin") == b"original"
        assert not list(tmp_path.glob(".pending-*"))


def test_public_directory_default_remains_compatible(tmp_path: Path) -> None:
    target = tmp_path / "public"
    target.mkdir()
    target.chmod(0o755)
    with pin_directory(target) as directory:
        directory.publish("receipt.bin", b"nonfinancial test payload")
        assert directory.read("receipt.bin") == b"nonfinancial test payload"
    assert stat.S_IMODE(target.stat().st_mode) == 0o755


def test_private_creation_refuses_restrictive_directory_umask(tmp_path: Path) -> None:
    result = subprocess.run(
        (
            sys.executable,
            "-c",
            dedent("""
                import os
                import stat
                import sys
                from pathlib import Path
                from liq.runner.pinned_directory import pin_directory

                target = Path(sys.argv[1]) / "attempt"
                previous_mask = os.umask(0o777)
                try:
                    with pin_directory(target, create=True, private=True) as directory:
                        directory.publish("receipt.bin", b"must not publish")
                except PermissionError:
                    assert stat.S_IMODE(target.stat().st_mode) == 0
                else:
                    raise AssertionError("Restrictive umask must not publish")
                finally:
                    os.umask(previous_mask)
            """),
            str(tmp_path),
        ),
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    target = tmp_path / "attempt"
    try:
        assert result.returncode == 0, result.stderr
        assert "CoverageWarning" not in result.stderr
    finally:
        # Restore only this test's newly created empty path for fixture cleanup.
        if target.exists():
            target.chmod(0o700)
    assert not list(target.iterdir())


@pytest.mark.parametrize("substitution", ["directory", "symlink", "ancestor"])
def test_private_path_substitution_preserves_only_the_original_receipt(
    tmp_path: Path, substitution: str
) -> None:
    parent = tmp_path / "private"
    target = parent / "attempt"
    with (
        pytest.raises(PermissionError),
        pin_directory(target, create=True, private=True) as directory,
    ):
        directory.publish("receipt.bin", b"possible exposure")
        changed = parent if substitution == "ancestor" else target
        retained = changed.with_name("retained")
        changed.rename(retained)
        if substitution == "directory":
            changed.mkdir(mode=0o700)
        else:
            changed.symlink_to(retained, target_is_directory=True)
        directory.publish("observation.bin", b"must not publish")
    original = retained / "attempt" if substitution == "ancestor" else retained
    assert (original / "receipt.bin").read_bytes() == b"possible exposure"
    assert not (original / "observation.bin").exists()
    assert not (target / "observation.bin").exists()


def test_private_read_refuses_a_symlink_record(tmp_path: Path) -> None:
    with pin_directory(tmp_path, private=True) as directory:
        directory.publish("receipt.bin", b"nonfinancial test payload")
        (tmp_path / "alias.bin").symlink_to(tmp_path / "receipt.bin")
        with pytest.raises(OSError):
            directory.read("alias.bin")


def test_private_read_refuses_a_foreign_owned_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_fstat = os.fstat
    with pin_directory(tmp_path, private=True) as directory:
        directory.publish("receipt.bin", b"nonfinancial test payload")

        def foreign_file(descriptor: int) -> os.stat_result:
            info = original_fstat(descriptor)
            if stat.S_ISREG(info.st_mode):
                fields = list(info)
                fields[4] = os.geteuid() + 1
                return os.stat_result(fields)
            return info

        monkeypatch.setattr(os, "fstat", foreign_file)
        with pytest.raises(PermissionError):
            directory.read("receipt.bin")

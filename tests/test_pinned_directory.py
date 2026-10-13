"""One-use publication identities survive path substitution without following it."""

from pathlib import Path

import pytest

from liq.runner.pinned_directory import pin_directory


def test_pinned_publication_is_durable_exclusive_and_readable(tmp_path: Path) -> None:
    with pin_directory(tmp_path / "new" / "attempt", create=True) as directory:
        directory.publish("receipt.json", b"{}")
        assert directory.read("receipt.json") == b"{}"
        with pytest.raises(FileExistsError):
            directory.publish("receipt.json", b"changed")


@pytest.mark.parametrize("substitution", ["directory", "symlink", "ancestor"])
def test_path_substitution_refuses_publication_and_retains_original_claim(
    tmp_path: Path, substitution: str
) -> None:
    parent = tmp_path / "parent"
    target = parent / "attempt"
    with pytest.raises(PermissionError), pin_directory(target, create=True) as directory:
        directory.publish("receipt.json", b"possible exposure")
        changed = parent if substitution == "ancestor" else target
        saved = changed.with_name(changed.name + "-saved")
        changed.rename(saved)
        if substitution == "directory":
            changed.mkdir()
        else:
            changed.symlink_to(saved, target_is_directory=True)
        directory.publish("outcome.json", b"must not publish")

    retained = saved / "attempt" if substitution == "ancestor" else saved
    assert (retained / "receipt.json").read_bytes() == b"possible exposure"
    assert not (retained / "outcome.json").exists()
    assert not (target / "outcome.json").exists()


@pytest.mark.parametrize("name", ["../escape", "/escape", "", ".", "..", "a/b", "a\\b"])
def test_publication_requires_a_single_safe_name(tmp_path: Path, name: str) -> None:
    with pin_directory(tmp_path) as directory, pytest.raises(ValueError):
        directory.publish(name, b"data")


def test_read_refuses_a_symlink_record(tmp_path: Path) -> None:
    (tmp_path / "original").write_bytes(b"data")
    (tmp_path / "alias").symlink_to(tmp_path / "original")

    with pin_directory(tmp_path) as directory, pytest.raises(OSError):
        directory.read("alias")

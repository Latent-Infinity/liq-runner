"""A check/link race cannot redirect bytes into a substituted directory."""

import os
from pathlib import Path

import pytest

from liq.runner.pinned_directory import pin_directory


def test_link_race_refuses_success_and_never_writes_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "attempt"
    retained = tmp_path / "retained"
    original_link = os.link

    def substitute(
        source: str, target: str, *, src_dir_fd: int, dst_dir_fd: int, follow_symlinks: bool
    ) -> None:
        destination.rename(retained)
        destination.mkdir()
        original_link(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    with (
        pytest.raises(PermissionError, match="Claim directory path changed"),
        pin_directory(destination, create=True) as directory,
    ):
        directory.publish("attempt.json", b'{"potential_exposure":true}')
        monkeypatch.setattr(os, "link", substitute)
        directory.publish("observation.json", b"{}")

    assert list(destination.iterdir()) == []
    assert (retained / "attempt.json").read_bytes() == b'{"potential_exposure":true}'
    assert not list(retained.glob(".pending-*"))

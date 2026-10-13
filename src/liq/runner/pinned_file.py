"""Bounded reviewed-file witnesses with held ancestor and pathname reconciliation."""

import hashlib
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from liq.runner.pinned_directory import pin_directory

FileRefusal = Literal[
    "PLATFORM_REFUSED",
    "PATH_REFUSED",
    "LIMIT_REFUSED",
    "DIGEST_REFUSED",
    "FILE_ACCESS_REFUSED",
    "FILE_POLICY_REFUSED",
    "FILE_CHANGED",
    "FILE_OVERSIZE",
]


class PinnedFileRefused(PermissionError):
    """Category-only diagnostic, never a path, payload or native exception message."""

    def __init__(self, reason: FileRefusal) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class FileState:
    device: int
    inode: int
    mode: int
    owner: int
    links: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def from_stat(cls, info: os.stat_result) -> "FileState":
        return cls(
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_uid,
            info.st_nlink,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )


@dataclass(frozen=True, slots=True)
class PinnedFile:
    path: Path
    payload: bytes = field(repr=False)
    sha256: str
    state: FileState
    ancestors: tuple[tuple[int, int], ...]


def read_pinned_file(path: Path, expected: str, max_bytes: int) -> PinnedFile:
    """Read one owned, non-writable-by-others, single-link reviewed regular file.

    The file descriptor and complete ancestor chain stay pinned through the
    bounded read and pathname reconciliation. This establishes an identity
    witness, not permission to consume the file for research or transport.
    """
    if os.name != "posix" or any(
        not hasattr(os, name) for name in ("O_NOFOLLOW", "O_NONBLOCK", "O_DIRECTORY", "geteuid")
    ):
        raise PinnedFileRefused("PLATFORM_REFUSED")
    if (
        path.anchor != "/"
        or ".." in path.parts
        or "\0" in str(path)
        or path.name in {"", ".", ".."}
    ):
        raise PinnedFileRefused("PATH_REFUSED")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise PinnedFileRefused("LIMIT_REFUSED")
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise PinnedFileRefused("DIGEST_REFUSED")
    try:
        with pin_directory(path.parent) as parent:
            descriptor = os.open(
                path.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=parent._descriptor,
            )
            with os.fdopen(descriptor, "rb") as stream:
                before = FileState.from_stat(os.fstat(stream.fileno()))
                if (
                    not stat.S_ISREG(before.mode)
                    or before.owner != os.geteuid()
                    or before.links != 1
                    or before.mode & 0o022
                ):
                    raise PinnedFileRefused("FILE_POLICY_REFUSED")
                if before.size > max_bytes:
                    raise PinnedFileRefused("FILE_OVERSIZE")
                pathname = FileState.from_stat(
                    os.stat(path.name, dir_fd=parent._descriptor, follow_symlinks=False)
                )
                if pathname != before:
                    raise PinnedFileRefused("FILE_CHANGED")
                parent.check()
                payload = stream.read(max_bytes + 1)
                after = FileState.from_stat(os.fstat(stream.fileno()))
                pathname = FileState.from_stat(
                    os.stat(path.name, dir_fd=parent._descriptor, follow_symlinks=False)
                )
                parent.check()
                if len(payload) > max_bytes:
                    raise PinnedFileRefused("FILE_OVERSIZE")
                if after != before or pathname != before or len(payload) != before.size:
                    raise PinnedFileRefused("FILE_CHANGED")
                digest = hashlib.sha256(payload).hexdigest()
                if digest != expected:
                    raise PinnedFileRefused("DIGEST_REFUSED")
                return PinnedFile(
                    path, payload, digest, before, tuple(edge.identity for edge in parent._edges)
                )
    except PinnedFileRefused:
        raise
    except OSError:
        raise PinnedFileRefused("FILE_ACCESS_REFUSED") from None

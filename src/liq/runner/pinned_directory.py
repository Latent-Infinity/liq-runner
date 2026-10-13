"""Descriptor-relative, one-use publication with directory-chain reconciliation."""

import os
import stat
import uuid
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class _Edge:
    parent: int
    name: str
    identity: tuple[int, int]


def _name(name: str) -> None:
    if name in {"", ".", ".."} or any(char in name for char in ("/", "\\", "\0")):
        raise ValueError("A single record filename is required")


def _require_private(info: os.stat_result, mode: int) -> None:
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != mode:
        raise PermissionError("Private claim permissions refused")


@dataclass(frozen=True, slots=True)
class PinnedDirectory:
    """A live directory capability; usable only inside its owning context."""

    path: Path
    identity: tuple[int, int]
    _descriptor: int
    _edges: tuple[_Edge, ...]
    _private: bool = False

    def check(self) -> None:
        for edge in self._edges:
            try:
                current = os.stat(edge.name, dir_fd=edge.parent, follow_symlinks=False)
            except OSError:
                raise PermissionError("Claim directory path changed") from None
            if (
                not stat.S_ISDIR(current.st_mode)
                or (current.st_dev, current.st_ino) != edge.identity
            ):
                raise PermissionError("Claim directory path changed")
            if self._private and edge is self._edges[-1]:
                _require_private(current, 0o700)
        if self._private:
            _require_private(os.fstat(self._descriptor), 0o700)

    def publish(self, name: str, payload: bytes) -> None:
        """Link a complete synced record into the pinned directory, never another path."""
        _name(name)
        self.check()
        pending = f".pending-{uuid.uuid4().hex}"
        descriptor = os.open(
            pending,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=self._descriptor,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                if self._private:
                    os.fchmod(stream.fileno(), 0o600)
                    _require_private(os.fstat(stream.fileno()), 0o600)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
                if self._private:
                    _require_private(os.fstat(stream.fileno()), 0o600)
            self.check()
            os.link(
                pending,
                name,
                src_dir_fd=self._descriptor,
                dst_dir_fd=self._descriptor,
                follow_symlinks=False,
            )
            os.fsync(self._descriptor)
            self.check()
        finally:
            os.unlink(pending, dir_fd=self._descriptor)

    def read(self, name: str) -> bytes:
        _name(name)
        self.check()
        descriptor = os.open(
            name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self._descriptor
        )
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise PermissionError("Claim record is not a regular file")
            if self._private:
                _require_private(info, 0o600)
            payload = stream.read()
            if self._private:
                _require_private(os.fstat(stream.fileno()), 0o600)
        self.check()
        return payload


@contextmanager
def pin_directory(
    destination: Path, *, create: bool = False, private: bool = False
) -> Iterator[PinnedDirectory]:
    """Pin ancestors; private leaves require owner-only access without repairing modes.

    New private directories use mode 0700, subject to umask. A restrictive mask
    may leave an inaccessible empty path and refuse before publication. Existing
    directories are never chmoded. This checks POSIX mode/UID, not filesystem ACLs.
    """
    destination = destination.absolute()
    with ExitStack() as handles:
        parent = os.open(destination.anchor, os.O_RDONLY | os.O_DIRECTORY)
        handles.callback(os.close, parent)
        edges: list[_Edge] = []
        parts = destination.parts[1:]
        for index, part in enumerate(parts):
            if create:
                try:
                    os.mkdir(part, mode=0o700 if private else 0o777, dir_fd=parent)
                except FileExistsError:
                    if index == len(parts) - 1:
                        raise
                else:
                    os.fsync(parent)
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            handles.callback(os.close, child)
            info = os.fstat(child)
            edges.append(_Edge(parent, part, (info.st_dev, info.st_ino)))
            parent = child
        info = os.fstat(parent)
        directory = PinnedDirectory(
            destination, (info.st_dev, info.st_ino), parent, tuple(edges), private
        )
        directory.check()
        yield directory
        directory.check()

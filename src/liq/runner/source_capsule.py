"""Deterministic, write-once archives of reviewed source bytes before execution."""

import gzip
import hashlib
import io
import os
import stat
import tarfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from liq.runner.bounded_attempt import publish_once


class SourceCapsuleError(ValueError):
    """A source set cannot be archived as its reviewed identity."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class SourcePin:
    """One reviewed workspace-relative source identity."""

    path: str
    sha256: str


def archive_sources(workspace: Path, target: Path, pins: tuple[SourcePin, ...]) -> str:
    """Publish exactly the reviewed file bytes, returning the capsule SHA-256.

    The caller owns the approved inventory and must finish this operation before
    transport. This archives source identity; it does not authorize execution or
    prove that a later running process still uses the archived files.
    """
    capsule = build_source_capsule(workspace, pins)
    publish_once(target, capsule)
    return hashlib.sha256(capsule).hexdigest()


def build_source_capsule(workspace: Path, pins: tuple[SourcePin, ...]) -> bytes:
    """Build deterministic reviewed bytes without selecting a publication path."""
    if not pins or len({pin.path for pin in pins}) != len(pins):
        raise SourceCapsuleError("Empty or duplicate source inventory")
    for pin in pins:
        if (
            not pin.path
            or PurePosixPath(pin.path).is_absolute()
            or "\\" in pin.path
            or any(part in {"", ".", ".."} for part in pin.path.split("/"))
        ):
            raise SourceCapsuleError("Noncanonical source path")
    output = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        for pin in sorted(pins, key=lambda item: item.path):
            with ExitStack() as handles:
                parent = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY)
                handles.callback(os.close, parent)
                parts = pin.path.split("/")
                try:
                    for part in parts[:-1]:
                        parent = os.open(
                            part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
                        )
                        handles.callback(os.close, parent)
                    descriptor = os.open(
                        parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
                    )
                except OSError:
                    raise SourceCapsuleError("Source path unavailable or redirected") from None
                handles.callback(os.close, descriptor)
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise SourceCapsuleError("Source is not a regular file")
                with os.fdopen(descriptor, "rb", closefd=False) as source:
                    payload = source.read()
            if hashlib.sha256(payload).hexdigest() != pin.sha256:
                raise SourceCapsuleError("Source content differs from reviewed identity")
            member = tarfile.TarInfo(pin.path)
            member.size = len(payload)
            member.mode = 0o444
            archive.addfile(member, io.BytesIO(payload))
    return output.getvalue()

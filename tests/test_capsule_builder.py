"""Capsule construction is independent of its publication destination."""

import hashlib
from pathlib import Path

from liq.runner.source_capsule import SourcePin, archive_sources, build_source_capsule


def test_builder_matches_compatible_archive(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    payload = b'"""Reviewed source fixture."""\n'
    source.write_bytes(payload)
    pins = (SourcePin(source.name, hashlib.sha256(payload).hexdigest()),)

    capsule = build_source_capsule(tmp_path, pins)
    assert sorted(path.name for path in tmp_path.iterdir()) == [source.name]
    target = tmp_path / "source.tar.gz"
    digest = archive_sources(tmp_path, target, pins)

    assert capsule == target.read_bytes()
    assert digest == hashlib.sha256(capsule).hexdigest()

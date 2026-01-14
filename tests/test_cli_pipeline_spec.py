import json
from pathlib import Path

import pytest

from liq.runner.cli import main


def _write_spec(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(payload))
    return path


def test_cli_dry_run_and_hash(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    payload = {
        "name": "test",
        "artifact_root": "artifacts",
        "stages": [
            {"name": "ingest", "outputs": ["raw"], "enabled": True},
            {"name": "disabled", "enabled": False},
        ],
    }
    spec_path = _write_spec(tmp_path, payload)

    code = main(["--spec", str(spec_path), "--dry-run", "--print-hash"])
    assert code == 0

    output = json.loads(capsys.readouterr().out)
    assert output["name"] == "test"
    assert "config_hash" in output
    assert output["dry_run"] == [{"name": "ingest", "inputs": [], "outputs": ["raw"]}]


def test_cli_missing_spec(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        main(["--spec", str(missing)])


def test_cli_invalid_spec(tmp_path: Path) -> None:
    payload = {"name": "", "stages": []}
    spec_path = _write_spec(tmp_path, payload)
    with pytest.raises(ValueError):
        main(["--spec", str(spec_path), "--dry-run"])

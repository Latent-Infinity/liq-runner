"""One-use private supervision retains redacted claims on every child outcome."""

import json
import os
import stat
import sys
from pathlib import Path
from textwrap import dedent

import pytest

from liq.runner.bounded_attempt import AttemptSpec, run_once


def test_child_observes_durable_private_claims_before_work(tmp_path: Path) -> None:
    destination = tmp_path / "attempt"
    code = dedent("""
        import json
        import os
        import stat
        from pathlib import Path

        assert stat.S_IMODE(Path.cwd().stat().st_mode) == 0o700
        assert Path.cwd().stat().st_uid == os.geteuid()
        for name in ("claim.json", "attempt.json"):
            assert stat.S_IMODE(Path(name).stat().st_mode) == 0o600
        claim = json.loads(Path("claim.json").read_bytes())
        assert claim == {"device": Path.cwd().stat().st_dev, "inode": Path.cwd().stat().st_ino}
        assert Path("attempt.json").read_bytes() == b'{"potential_exposure":true}'
    """)
    spec = AttemptSpec((sys.executable, "-c", code), b'{"potential_exposure":true}', 5.0)

    outcome = run_once(destination, spec, private=True)

    assert outcome.category == "COMPLETED" and outcome.returncode == 0
    supervisor = destination / "supervisor.json"
    assert stat.S_IMODE(supervisor.stat().st_mode) == 0o600
    assert json.loads(supervisor.read_bytes())["returncode"] == 0


@pytest.mark.parametrize("exitcode", [0, 7])
def test_private_attempt_cannot_be_relaunched_after_child_exit(
    tmp_path: Path, exitcode: int
) -> None:
    destination = tmp_path / "attempt"
    spec = AttemptSpec((sys.executable, "-c", f"raise SystemExit({exitcode})"), b"{}", 5.0)
    first = run_once(destination, spec, private=True)
    receipt = (destination / "attempt.json").read_bytes()
    supervisor = (destination / "supervisor.json").read_bytes()

    with pytest.raises(FileExistsError):
        run_once(destination, spec, private=True)

    assert first.category == "COMPLETED" and first.returncode == exitcode
    assert (destination / "attempt.json").read_bytes() == receipt
    assert (destination / "supervisor.json").read_bytes() == supervisor
    assert stat.S_IMODE(destination.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in destination.iterdir())


def test_private_timeout_retains_claim_and_refuses_second_launch(tmp_path: Path) -> None:
    destination = tmp_path / "attempt"
    spec = AttemptSpec((sys.executable, "-c", "import signal; signal.pause()"), b"{}", 0.2)

    outcome = run_once(destination, spec, private=True)

    assert outcome.category == "TIMED_OUT" and outcome.returncode is None
    assert (destination / "attempt.json").read_bytes() == b"{}"
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in destination.iterdir())
    with pytest.raises(FileExistsError):
        run_once(destination, spec, private=True)


def test_private_spawn_failure_retains_only_redacted_private_records(tmp_path: Path) -> None:
    destination = tmp_path / "attempt"
    spec = AttemptSpec(("/missing/sensitive-program",), b"{}", 5.0)

    outcome = run_once(destination, spec, private=True)

    assert outcome.category == "SPAWN_ERROR" and outcome.returncode is None
    assert b"sensitive" not in (destination / "supervisor.json").read_bytes()
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in destination.iterdir())


def test_private_root_permission_drift_refuses_post_child_completion(tmp_path: Path) -> None:
    destination = tmp_path / "attempt"
    spec = AttemptSpec((sys.executable, "-c", "import os; os.chmod('.', 0o755)"), b"{}", 5.0)

    with pytest.raises(PermissionError):
        run_once(destination, spec, private=True)

    assert (destination / "attempt.json").read_bytes() == b"{}"
    assert not (destination / "supervisor.json").exists()


def test_private_root_substitution_refuses_post_child_completion(tmp_path: Path) -> None:
    destination = tmp_path / "attempt"
    code = dedent("""
        from pathlib import Path
        target = Path.cwd()
        target.rename(target.with_name("retained"))
        target.mkdir(mode=0o700)
    """)
    spec = AttemptSpec((sys.executable, "-c", code), b"{}", 5.0)

    with pytest.raises(PermissionError):
        run_once(destination, spec, private=True)

    assert (tmp_path / "retained" / "attempt.json").read_bytes() == b"{}"
    assert not list(destination.iterdir())
    assert not (tmp_path / "retained" / "supervisor.json").exists()


def test_private_owner_refusal_precedes_supervision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from liq.runner import bounded_attempt

    def forbidden_launch(destination: Path, spec: AttemptSpec, started: float) -> None:
        pytest.fail("Permission refusal must precede child launch")

    owner = os.geteuid()
    monkeypatch.setattr(os, "geteuid", lambda: owner + 1)
    monkeypatch.setattr(bounded_attempt, "_supervise", forbidden_launch)
    destination = tmp_path / "attempt"
    with pytest.raises(PermissionError):
        run_once(destination, AttemptSpec((sys.executable, "-c", "pass"), b"{}", 5), private=True)
    assert not list(destination.iterdir())


def test_private_mode_refusal_precedes_supervision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from liq.runner import bounded_attempt

    original_mkdir = os.mkdir

    def public_creation(path: str, mode: int, *, dir_fd: int | None = None) -> None:
        original_mkdir(path, mode=0o755, dir_fd=dir_fd)

    def forbidden_launch(destination: Path, spec: AttemptSpec, started: float) -> None:
        pytest.fail("Permission refusal must precede child launch")

    monkeypatch.setattr(os, "mkdir", public_creation)
    monkeypatch.setattr(bounded_attempt, "_supervise", forbidden_launch)
    destination = tmp_path / "attempt"
    with pytest.raises(PermissionError):
        run_once(destination, AttemptSpec((sys.executable, "-c", "pass"), b"{}", 5), private=True)
    assert not list(destination.iterdir())


def test_public_default_keeps_legacy_record_and_mode_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_private_chmod(descriptor: int, mode: int) -> None:
        pytest.fail("Public default must not enable private-mode fchmod")

    monkeypatch.setattr(os, "fchmod", forbidden_private_chmod)
    destination = tmp_path / "attempt"
    outcome = run_once(destination, AttemptSpec((sys.executable, "-c", "pass"), b"{}", 5))
    assert outcome.returncode == 0
    assert (destination / "attempt.json").read_bytes() == b"{}"
    assert {path.name for path in destination.iterdir()} == {
        "claim.json",
        "attempt.json",
        "supervisor.json",
    }

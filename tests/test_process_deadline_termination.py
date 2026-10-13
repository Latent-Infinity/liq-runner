"""Real OS kills retain one-use claims; no market data or network is involved."""

import json
import signal
import sys
from pathlib import Path
from textwrap import dedent

import pytest

from liq.runner.bounded_attempt import AttemptSpec, run_once


@pytest.mark.parametrize(
    "operation",
    [
        "read_fd, write_fd = os.pipe(); os.read(read_fd, 1)",
        "while True: os.write(write_fd, b'x'); os.read(read_fd, 1)",
        "with closing_stream(): pass",
    ],
    ids=["kernel-blocked-headers", "progressing-body", "blocked-response-close"],
)
def test_request_wall_kills_worker_and_retains_potential_exposure(
    tmp_path: Path, operation: str
) -> None:
    # Given a claimed worker with a bounded request and an unending operation.
    destination = tmp_path / "attempt"
    code = (
        dedent("""
        import os, signal
        from contextlib import contextmanager
        from pathlib import Path
        from liq.runner.process_deadline import hard_process_deadline
        from liq.runner.bounded_attempt import publish_once
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        assert Path('attempt.json').read_bytes() == b'{"potential_exposure": true}'
        read_fd, write_fd = os.pipe()
        @contextmanager
        def closing_stream():
            try:
                yield
            finally:
                os.read(read_fd, 1)
        with hard_process_deadline(0.15):
    """)
        + "    "
        + operation
        + "\npublish_once(Path('success.json'), b'{}')\n"
    )
    spec = AttemptSpec((sys.executable, "-c", code), b'{"potential_exposure": true}', 4.0)
    # When the worker enters its own OS request wall, inside parent supervision.
    result = run_once(destination, spec)
    # Then the worker is dead, the claim survives, and success is impossible.
    assert result.category == "COMPLETED" and result.returncode == -signal.SIGALRM
    assert not (destination / "success.json").exists()
    assert json.loads((destination / "attempt.json").read_bytes())["potential_exposure"] is True
    assert (
        json.loads((destination / "supervisor.json").read_bytes())["returncode"] == -signal.SIGALRM
    )
    with pytest.raises(FileExistsError):
        run_once(destination, spec)


def test_nested_long_request_cannot_extend_short_outer_wall(tmp_path: Path) -> None:
    # Given a short outer budget surrounding a much longer request budget.
    code = """
import os, signal
from liq.runner.process_deadline import hard_process_deadline
signal.signal(signal.SIGALRM, signal.SIG_DFL)
with hard_process_deadline(0.15):
    with hard_process_deadline(3):
        read_fd, write_fd = os.pipe()
        os.read(read_fd, 1)
"""
    spec = AttemptSpec((sys.executable, "-c", code), b"{}", 1.5)
    # When the nested request blocks indefinitely.
    result = run_once(tmp_path / "attempt", spec)
    # Then the inner timer cannot postpone the outer kill until parent timeout.
    assert result.category == "COMPLETED" and result.returncode == -signal.SIGALRM
    assert result.elapsed_seconds < 1.2


def test_restore_does_not_reset_elapsed_outer_wall(tmp_path: Path) -> None:
    # Given a parent scope with work on both sides of a successful inner scope.
    code = """
import os, signal, time
from pathlib import Path
from liq.runner.process_deadline import hard_process_deadline
from liq.runner.bounded_attempt import publish_once
signal.signal(signal.SIGALRM, signal.SIG_DFL)
with hard_process_deadline(0.3):
    with hard_process_deadline(0.2):
        until = time.monotonic() + 0.12
        while time.monotonic() < until:
            pass
    read_fd, write_fd = os.pipe()
    os.read(read_fd, 1)
publish_once(Path('success.json'), b'{}')
"""
    # When the request completes but later case work blocks.
    result = run_once(tmp_path / "attempt", AttemptSpec((sys.executable, "-c", code), b"{}", 2))
    # Then the outer deadline remains absolute and the success file is absent.
    assert result.category == "COMPLETED" and result.returncode == -signal.SIGALRM
    assert not (tmp_path / "attempt" / "success.json").exists()


@pytest.mark.parametrize("ticks", ["[0.0, 2.0]", "[0.0, 0.0, 0.0, 2.0]"])
def test_exhausted_request_budget_kills_instead_of_disarming(tmp_path: Path, ticks: str) -> None:
    # Given a deterministic clock advancing beyond the request wall at entry/exit.
    code = f"""
import signal
from pathlib import Path
from liq.runner import process_deadline
from liq.runner.bounded_attempt import publish_once
signal.signal(signal.SIGALRM, signal.SIG_DFL)
ticks = iter({ticks})
process_deadline.time.monotonic = lambda: next(ticks)
with process_deadline.hard_process_deadline(1):
    pass
publish_once(Path('success.json'), b'{{}}')
"""
    # When an exhausted duration would otherwise become a zero/disarming timer.
    result = run_once(tmp_path / "attempt", AttemptSpec((sys.executable, "-c", code), b"{}", 2))
    # Then process death occurs instead of a catchable timeout or success.
    assert result.category == "COMPLETED" and result.returncode == -signal.SIGALRM
    assert not (tmp_path / "attempt" / "success.json").exists()


@pytest.mark.parametrize("parent", [False, True])
@pytest.mark.parametrize("body", ["pass", "raise LookupError('controlled')"])
def test_expiry_during_cleanup_cannot_disarm_or_replace_request(
    tmp_path: Path, parent: bool, body: str
) -> None:
    # Given a clock that crosses the request deadline during disarm/restore.
    code = f"""
import signal
from pathlib import Path
from liq.runner import process_deadline
from liq.runner.bounded_attempt import publish_once
signal.signal(signal.SIGALRM, signal.SIG_DFL)
if {parent}:
    signal.setitimer(signal.ITIMER_REAL, 3)
clock = [0.0]
process_deadline.time.monotonic = lambda: clock[0]
original = signal.setitimer
def change_timer(which, delay, interval=0.0):
    if delay == 0 or delay > 2:
        clock[0] = 2.0
    return original(which, delay, interval)
process_deadline.signal.setitimer = change_timer
try:
    with process_deadline.hard_process_deadline(1):
        {body}
except LookupError:
    pass
publish_once(Path('success.json'), b'{{}}')
"""
    # When cleanup cancels/replaces the alarm after its absolute expiry.
    result = run_once(tmp_path / "attempt", AttemptSpec((sys.executable, "-c", code), b"{}", 2))
    # Then even exceptional cleanup cannot return after the expired request.
    assert result.category == "COMPLETED" and result.returncode == -signal.SIGALRM
    assert not (tmp_path / "attempt" / "success.json").exists()


@pytest.mark.parametrize("restore", [False, True])
def test_subresolution_positive_timer_cannot_allow_success(tmp_path: Path, restore: bool) -> None:
    # Given a strictly positive remainder below the native timer resolution.
    code = f"""
import signal
from pathlib import Path
from liq.runner import process_deadline
from liq.runner.bounded_attempt import publish_once
signal.signal(signal.SIGALRM, signal.SIG_DFL)
clock = [0.0]
process_deadline.time.monotonic = lambda: clock[0]
parent = []
original = signal.getitimer
original_arm = signal.setitimer
def arm_timer(which, delay, interval=0.0):
    # Reproduce the observed native truncation without a signal-delivery race.
    return original_arm(which, 0.0 if 0 < delay < 1e-8 else delay, interval)
process_deadline.signal.setitimer = arm_timer
def read_timer(which):
    result = original(which)
    if not parent:
        parent.append(result[0])
    return result
process_deadline.signal.getitimer = read_timer
if {restore}:
    signal.setitimer(signal.ITIMER_REAL, 3)
with process_deadline.hard_process_deadline(10 if {restore} else 1e-9):
    if {restore}:
        clock[0] = parent[0] - 1e-9
publish_once(Path('success.json'), b'{{}}')
"""
    # When entry or parent restoration attempts to arm that native duration.
    result = run_once(tmp_path / "attempt", AttemptSpec((sys.executable, "-c", code), b"{}", 2))
    # Then an unrepresentable positive interval is fatal, never rounded upward.
    assert result.category == "COMPLETED" and result.returncode == -signal.SIGALRM
    assert not (tmp_path / "attempt" / "success.json").exists()

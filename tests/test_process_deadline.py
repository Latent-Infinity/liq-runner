"""Process-wall policy and nesting, without market data or broker access."""

import json
import subprocess
import sys
from textwrap import dedent

import pytest

from liq.runner.process_deadline import ProcessDeadlineRefused


def worker(code: str) -> subprocess.CompletedProcess[str]:
    """Run an isolated worker so pytest never changes its own signal policy."""
    return subprocess.run(
        (sys.executable, "-c", dedent(code)),
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )


@pytest.mark.parametrize("seconds", ["0", "-1", "float('nan')", "float('inf')"])
def test_invalid_budget_refuses_before_entering(seconds: str) -> None:
    # Given a default-handler worker with an invalid request budget.
    code = f"""
import signal
from liq.runner.process_deadline import hard_process_deadline, ProcessDeadlineRefused
signal.signal(signal.SIGALRM, signal.SIG_DFL)
try:
    with hard_process_deadline({seconds}):
        raise AssertionError('Transport entered')
except ProcessDeadlineRefused as error:
    print(error.reason)
assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
"""
    # When the worker attempts to enter the deadline.
    result = worker(code)
    # Then it refuses without transport or an installed timer.
    assert result.returncode == 0 and result.stdout.strip() == "INVALID_BUDGET"


@pytest.mark.parametrize("handler", ["signal.SIG_IGN", "lambda signum, frame: None"])
def test_nondefault_handler_refuses_without_changing_policy(handler: str) -> None:
    # Given a worker whose alarm could be ignored or cooperatively delayed.
    code = f"""
import signal
from liq.runner.process_deadline import hard_process_deadline, ProcessDeadlineRefused
handler = {handler}
signal.signal(signal.SIGALRM, handler)
try:
    with hard_process_deadline(1):
        raise AssertionError('Transport entered')
except ProcessDeadlineRefused as error:
    print(error.reason)
assert signal.getsignal(signal.SIGALRM) == handler
assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
"""
    # When the worker attempts to enter the deadline.
    result = worker(code)
    # Then the incompatible policy remains untouched.
    assert result.returncode == 0 and result.stdout.strip() == "HANDLER_REFUSED"


def test_background_thread_refuses_before_timer_work() -> None:
    # Given a worker using the default process alarm policy.
    code = """
import signal
from concurrent.futures import ThreadPoolExecutor
from liq.runner.process_deadline import hard_process_deadline, ProcessDeadlineRefused
signal.signal(signal.SIGALRM, signal.SIG_DFL)
def request():
    try:
        with hard_process_deadline(1):
            raise AssertionError('Transport entered')
    except ProcessDeadlineRefused as error:
        return error.reason
with ThreadPoolExecutor(max_workers=1) as pool:
    print(pool.submit(request).result())
assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
"""
    # When a background thread requests a process-wide deadline.
    result = worker(code)
    # Then transport cannot begin.
    assert result.returncode == 0 and result.stdout.strip() == "RUNTIME_REFUSED"


def test_unsupported_platform_refuses_before_transport() -> None:
    # Given a worker whose platform is outside the supported POSIX contract.
    code = """
import os
from liq.runner.process_deadline import hard_process_deadline, ProcessDeadlineRefused
os.name = 'nt'
try:
    with hard_process_deadline(1):
        raise AssertionError('Transport entered')
except ProcessDeadlineRefused as error:
    print(error.reason)
"""
    # When the worker tries to install a process alarm.
    result = worker(code)
    # Then the unsupported configuration refuses before transport.
    assert result.returncode == 0 and result.stdout.strip() == "RUNTIME_REFUSED"


def test_periodic_timer_refuses_without_replacing_it() -> None:
    # Given an incompatible periodic alarm in an isolated worker.
    code = """
import signal
from liq.runner.process_deadline import hard_process_deadline, ProcessDeadlineRefused
signal.signal(signal.SIGALRM, signal.SIG_DFL)
signal.setitimer(signal.ITIMER_REAL, 3, 3)
try:
    with hard_process_deadline(1):
        raise AssertionError('Transport entered')
except ProcessDeadlineRefused as error:
    print(error.reason)
assert signal.getitimer(signal.ITIMER_REAL)[1] == 3
signal.setitimer(signal.ITIMER_REAL, 0)
"""
    # When the request deadline encounters the periodic timer.
    result = worker(code)
    # Then it refuses rather than changing that timer's semantics.
    assert result.returncode == 0 and result.stdout.strip() == "PERIODIC_TIMER_REFUSED"


@pytest.mark.parametrize("pending", [False, True])
def test_blocked_or_pending_alarm_refuses_before_transport(pending: bool) -> None:
    # Given a worker where the default alarm cannot be delivered immediately.
    code = f"""
import os, signal
from liq.runner.process_deadline import hard_process_deadline, ProcessDeadlineRefused
signal.signal(signal.SIGALRM, signal.SIG_DFL)
signal.pthread_sigmask(signal.SIG_BLOCK, {{signal.SIGALRM}})
if {pending}:
    os.kill(os.getpid(), signal.SIGALRM)
try:
    with hard_process_deadline(1):
        raise AssertionError('Transport entered')
except ProcessDeadlineRefused as error:
    print(error.reason)
assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
"""
    # When the worker attempts to enter the deadline.
    result = worker(code)
    # Then it refuses without unmasking a pending signal or entering transport.
    assert result.returncode == 0 and result.stdout.strip() == "SIGNAL_STATE_REFUSED"


def test_missing_timer_facility_refuses_before_transport() -> None:
    # Given a platform lacking one required POSIX timer facility.
    code = """
import signal
from liq.runner.process_deadline import hard_process_deadline, ProcessDeadlineRefused
del signal.setitimer
try:
    with hard_process_deadline(1):
        raise AssertionError('Transport entered')
except ProcessDeadlineRefused as error:
    print(error.reason)
"""
    # When the worker tries to enter the deadline.
    result = worker(code)
    # Then there is no fallback to a cooperative timer.
    assert result.returncode == 0 and result.stdout.strip() == "RUNTIME_REFUSED"


def test_pending_alarm_snapshot_refuses_even_with_empty_mask() -> None:
    # Given a pending-state snapshot at the otherwise unblocked OS boundary.
    code = """
import signal
from liq.runner.process_deadline import hard_process_deadline, ProcessDeadlineRefused
signal.signal(signal.SIGALRM, signal.SIG_DFL)
signal.sigpending = lambda: {signal.SIGALRM}
try:
    with hard_process_deadline(1):
        raise AssertionError('Transport entered')
except ProcessDeadlineRefused as error:
    print(error.reason)
assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
"""
    # When that snapshot is returned without delivering a fatal test signal.
    result = worker(code)
    # Then the pending-state check independently prevents transport.
    assert result.returncode == 0 and result.stdout.strip() == "SIGNAL_STATE_REFUSED"


@pytest.mark.parametrize("body", ["pass", "raise LookupError('controlled')"])
def test_exit_restores_remaining_parent_budget(body: str) -> None:
    # Given a shorter parent timer, whose elapsed budget must not be replenished.
    code = f"""
import json, signal, time
from liq.runner.process_deadline import hard_process_deadline
signal.signal(signal.SIGALRM, signal.SIG_DFL)
signal.setitimer(signal.ITIMER_REAL, 3)
started = time.monotonic()
try:
    with hard_process_deadline(1):
        until = time.monotonic() + 0.04
        while time.monotonic() < until:
            pass
        {body}
except LookupError:
    pass
remaining = signal.getitimer(signal.ITIMER_REAL)[0]
elapsed = time.monotonic() - started
signal.setitimer(signal.ITIMER_REAL, 0)
print(json.dumps([remaining, elapsed]))
"""
    # When the inner scope finishes normally or with a body exception.
    result = worker(code)
    # Then the absolute parent budget remains elapsed, not reset to three seconds.
    assert result.returncode == 0, result.stderr
    remaining, elapsed = json.loads(result.stdout)
    assert 2.8 < remaining < 2.98 and abs(remaining + elapsed - 3) < 0.015


def test_success_without_parent_disarms_request_timer() -> None:
    # Given an isolated default-handler worker with no parent timer.
    code = """
import signal
from liq.runner.process_deadline import hard_process_deadline
signal.signal(signal.SIGALRM, signal.SIG_DFL)
with hard_process_deadline(1):
    pass
assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
"""
    # When its deadline scope completes.
    result = worker(code)
    # Then no timer remains armed after successful work.
    assert result.returncode == 0, result.stderr


def test_refusal_message_contains_only_category() -> None:
    # Given a structured refusal without arbitrary inputs.
    error = ProcessDeadlineRefused("INVALID_BUDGET")
    # When a boundary serializes the failure.
    message = str(error)
    # Then the receipt contains only its category.
    assert message == "INVALID_BUDGET"

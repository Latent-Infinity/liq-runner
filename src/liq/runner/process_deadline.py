"""OS-default process walls for single-owner, descendant-free POSIX workers."""

import math
import os
import signal
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final, Literal, NoReturn

DeadlineRefusalReason = Literal[
    "INVALID_BUDGET",
    "RUNTIME_REFUSED",
    "HANDLER_REFUSED",
    "SIGNAL_STATE_REFUSED",
    "PERIODIC_TIMER_REFUSED",
]
_FACILITIES: Final = (
    "SIGALRM",
    "ITIMER_REAL",
    "SIG_BLOCK",
    "getsignal",
    "getitimer",
    "setitimer",
    "pthread_sigmask",
    "sigpending",
)


class ProcessDeadlineRefused(ValueError):
    """Safe refusal before protected work; never a substitute for process death."""

    def __init__(self, reason: DeadlineRefusalReason) -> None:
        self.reason = reason
        super().__init__(reason)


def _require_runtime() -> None:
    if (
        os.name != "posix"
        or threading.current_thread() is not threading.main_thread()
        or not all(hasattr(signal, name) for name in _FACILITIES)
    ):
        raise ProcessDeadlineRefused("RUNTIME_REFUSED")
    if signal.getsignal(signal.SIGALRM) != signal.SIG_DFL:
        raise ProcessDeadlineRefused("HANDLER_REFUSED")
    if (
        signal.SIGALRM in signal.pthread_sigmask(signal.SIG_BLOCK, [])
        or signal.SIGALRM in signal.sigpending()
    ):
        raise ProcessDeadlineRefused("SIGNAL_STATE_REFUSED")


def _terminate() -> NoReturn:
    os.kill(os.getpid(), signal.SIGALRM)
    # Never return even if another owner violates the signal-state contract.
    os._exit(128 + signal.SIGALRM)


def _arm_deadline(deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        _terminate()
    signal.setitimer(signal.ITIMER_REAL, remaining)
    # Positive sub-resolution durations can silently disarm the native timer.
    if signal.getitimer(signal.ITIMER_REAL)[0] <= 0 or time.monotonic() >= deadline:
        _terminate()


@contextmanager
def hard_process_deadline(seconds: float) -> Iterator[None]:
    """Kill the worker on expiry; preserve any earlier absolute parent wall.

    Call only from the main thread of a descendant-free worker that explicitly
    installed SIG_DFL. The caller owns signal/timer/mask exclusivity throughout
    the region and must retain its exposure claim before entry. Wrap all work
    including resource close. No Python alarm handler or catchable timeout is
    installed; a supervisor must interpret nonzero child exits as incomplete.
    """
    if not math.isfinite(seconds) or seconds <= 0:
        raise ProcessDeadlineRefused("INVALID_BUDGET")
    _require_runtime()
    started = time.monotonic()
    parent_remaining, interval = signal.getitimer(signal.ITIMER_REAL)
    if interval != 0:
        raise ProcessDeadlineRefused("PERIODIC_TIMER_REFUSED")
    parent_deadline = started + parent_remaining if parent_remaining > 0 else None
    request_deadline = started + seconds
    effective_deadline = (
        min(request_deadline, parent_deadline) if parent_deadline is not None else request_deadline
    )
    _arm_deadline(effective_deadline)
    try:
        yield
    finally:
        if time.monotonic() >= effective_deadline:
            _terminate()
        if parent_deadline is None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        else:
            _arm_deadline(parent_deadline)
        # Expiry can occur between the earlier check and timer replacement.
        if time.monotonic() >= effective_deadline:
            _terminate()

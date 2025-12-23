"""Adapters to bridge SignalOutput to liq-sim calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from liq.signals.output import SignalOutput
from liq.sim.calibration import CalibrationResult, ThresholdDiagnostics, ev_threshold_search, temperature_scale


@dataclass(frozen=True)
class CalibratedScores:
    scores: object
    params: dict
    threshold: float | None = None
    diagnostics: ThresholdDiagnostics | None = None


def calibrate_signal_output(signal_output: SignalOutput) -> CalibrationResult:
    """Calibrate SignalOutput scores using temperature scaling."""
    labels = signal_output.labels if signal_output.labels is not None else signal_output.scores * 0
    return temperature_scale(signal_output.scores, labels)


def select_threshold(
    signal_output: SignalOutput,
    *,
    min_precision: float | None = None,
    min_recall: float | None = None,
    min_trades: int | None = None,
    target_ev: float | None = None,
    grid: Iterable[float] | None = None,
) -> ThresholdDiagnostics:
    """Run EV-based threshold search on SignalOutput."""
    labels = signal_output.labels if signal_output.labels is not None else signal_output.scores * 0
    return ev_threshold_search(
        signal_output.scores,
        labels,
        min_precision=min_precision,
        min_recall=min_recall,
        min_trades=min_trades,
        target_ev=target_ev,
        grid=grid,
    )

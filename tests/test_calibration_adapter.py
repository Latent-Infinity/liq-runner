import polars as pl

from liq.runner.calibration_adapter import calibrate_signal_output, select_threshold
from liq.signals.output import SignalOutput


def test_calibrate_signal_output_returns_params() -> None:
    sig = SignalOutput(scores=pl.Series([0.1, 0.9]), labels=pl.Series([0, 1]))
    res = calibrate_signal_output(sig)
    assert "temperature" in res.params


def test_select_threshold_runs() -> None:
    sig = SignalOutput(scores=pl.Series([0.1, 0.9]), labels=pl.Series([0, 1]))
    diag = select_threshold(sig, min_precision=0.1)
    assert 0 < diag.threshold < 1

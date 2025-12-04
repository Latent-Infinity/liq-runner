from liq.runner.drift_manager import DriftManager


def test_drift_manager_triggers_retrain_on_high_stat():
    mgr = DriftManager(retrain_threshold=0.5, reduce_threshold=0.2)
    action = mgr.evaluate([0.6])
    assert action.retrain
    assert action.sizing_multiplier < 1.0


def test_drift_manager_reduces_sizing_on_medium_stat():
    mgr = DriftManager(retrain_threshold=0.5, reduce_threshold=0.2, min_multiplier=0.7)
    action = mgr.evaluate([0.3])
    assert not action.retrain
    assert action.sizing_multiplier == 0.7


def test_drift_manager_no_action_on_low_stat():
    mgr = DriftManager(retrain_threshold=0.5, reduce_threshold=0.2)
    action = mgr.evaluate([0.05])
    assert not action.retrain
    assert action.sizing_multiplier == 1.0

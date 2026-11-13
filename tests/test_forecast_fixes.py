"""Focused regressions for targeted XFiRE v3.3 forecasting fixes."""
import numpy as np
from xfire.forecast import _fit_horizon_weights, _gbm


def test_horizon_groups_without_enough_validation_rows_use_climatology():
    pred = np.array([120.0, 130.0])
    clim = np.array([100.0, 110.0])
    obs = np.array([115.0, 125.0])
    hh = np.array([1, 2])
    weights = _fit_horizon_weights(pred, clim, obs, hh, np.ones(2, dtype=bool))
    assert weights == {"1-5": 0.0, "6-13": 0.0, "14-26": 0.0}


def test_horizon_weights_are_bounded_when_validation_data_are_sufficient():
    rng = np.random.default_rng(7)
    hh = np.tile(np.arange(1, 27), 5)
    clim = np.full(len(hh), 100.0)
    obs = clim + rng.normal(0, 2, len(hh))
    pred = obs + rng.normal(0, 2, len(hh))
    weights = _fit_horizon_weights(pred, clim, obs, hh, np.ones(len(hh), dtype=bool))
    assert set(weights) == {"1-5", "6-13", "14-26"}
    assert all(0.0 <= value <= 1.0 for value in weights.values())


def test_gradient_boosting_parameter_adapter_runs():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 3))
    model = _gbm(n_estimators=20, num_leaves=5, min_child_samples=10, reg_lambda=1.0)
    model.fit(X, X[:, 0])
    assert np.isfinite(model.predict(X)).all()

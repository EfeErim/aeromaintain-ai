import numpy as np
import pytest

from aeromaintain.evaluation.bootstrap import engine_bootstrap
from aeromaintain.models import v5


def _two_metrics(per_engine):
    return {
        "rmse": (per_engine, np.sqrt),
        "rate": (per_engine / per_engine.max(), lambda v: v),
    }


def test_interval_resamples_every_engine():
    # Engines 0 and 1 are identical, so an interval drawn from them alone has
    # zero width. Any real engine resample must see the spread in the rest.
    per_engine = np.array([4.0, 4.0, 1.0, 9.0, 16.0, 25.0])
    low, high = engine_bootstrap(_two_metrics(per_engine), 2000, seed=0)["rmse"]
    assert low < 2.0 < 3.0 < high


def test_interval_depends_on_seed():
    per_engine = np.arange(1.0, 36.0)
    first = engine_bootstrap(_two_metrics(per_engine), 500, seed=0)
    second = engine_bootstrap(_two_metrics(per_engine), 500, seed=1)
    assert first["rmse"] != second["rmse"]


def test_interval_contains_full_sample_value():
    per_engine = np.random.default_rng(3).gamma(2.0, 50.0, size=35)
    low, high = engine_bootstrap(_two_metrics(per_engine), 2000, seed=44)["rmse"]
    assert low < np.sqrt(per_engine.mean()) < high


def test_metrics_must_share_engine_count():
    with pytest.raises(ValueError, match="same number of engines"):
        engine_bootstrap(
            {"a": (np.ones(3), np.sqrt), "b": (np.ones(4), np.sqrt)}, 10, seed=0
        )


def test_frozen_v5_bootstrap_only_sees_two_engines():
    # Pins the defect described in docs/results_v5.md: the frozen function sizes
    # its resample from the (values, transform) tuple, so only engines 0 and 1
    # are ever drawn. If this starts failing, the erratum is out of date.
    per_engine = np.array([4.0, 4.0, 1.0, 9.0, 16.0, 25.0])
    low, high = v5.engine_bootstrap(_two_metrics(per_engine), 2000, seed=0)["rmse"]
    assert low == high == 2.0

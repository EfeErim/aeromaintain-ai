"""XGBoost thread count is part of a C-MAPSS run's specification.

With row subsampling, XGBoost's ``hist`` builder draws rows per thread, so the same
seed gives different trees at different ``xgboost_n_jobs``. The published configs fix
the value and their hashes are in the locks; these tests record the behaviour, so a
library upgrade that changes it is noticed.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aeromaintain.models import rul


def _model_bytes(subsample: float, threads: int) -> bytes:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 10))
    y = 3.0 * x[:, 0] + rng.normal(size=500)
    config = replace(rul.ModelingConfig(), xgboost_n_jobs=threads)
    parameters = {"max_depth": 3, "subsample": subsample}
    model = rul._new_xgboost(
        parameters, config, seed=42, n_estimators=20, early_stopping=False
    )
    model.fit(x, y)
    return bytes(model.get_booster().save_raw("ubj"))


def test_subsampled_trees_depend_on_thread_count() -> None:
    assert _model_bytes(0.8, 2) == _model_bytes(0.8, 2)
    assert _model_bytes(0.8, 1) != _model_bytes(0.8, 2)


@pytest.mark.parametrize("threads", [2, 4])
def test_trees_without_subsampling_ignore_thread_count(threads: int) -> None:
    assert _model_bytes(1.0, 1) == _model_bytes(1.0, threads)

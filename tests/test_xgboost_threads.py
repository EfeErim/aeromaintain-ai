"""XGBoost thread count is part of a C-MAPSS run's specification.

With row subsampling, XGBoost's ``hist`` builder may give different trees for the same
seed at different ``xgboost_n_jobs``: on Windows the trees differ, on GitHub's Ubuntu
runner they are identical. XGBoost promises neither, so these tests pin only what holds
everywhere: a fixed thread count repeats, and without subsampling the thread count
does not matter. The published configs fix the value and their hashes are in the locks.
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


def test_subsampled_trees_repeat_at_a_fixed_thread_count() -> None:
    assert _model_bytes(0.8, 2) == _model_bytes(0.8, 2)


@pytest.mark.parametrize("threads", [2, 4])
def test_trees_without_subsampling_ignore_thread_count(threads: int) -> None:
    assert _model_bytes(1.0, 1) == _model_bytes(1.0, threads)

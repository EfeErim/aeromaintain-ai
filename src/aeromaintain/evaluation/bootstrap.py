"""Engine-level percentile bootstrap.

This replaces ``models.v5.engine_bootstrap``, which stays frozen under the v5 lock and
resamples only two engines (see ``docs/results_v5.md``, erratum).
"""

from collections.abc import Callable, Mapping

import numpy as np

Metric = tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]


def engine_bootstrap(
    per_engine: Mapping[str, Metric], resamples: int, seed: int
) -> dict[str, list[float]]:
    """95% percentile intervals from resampling engines with replacement.

    Each metric is a per-engine value (for RMSE, the engine's mean squared error) and
    a transform applied to the resampled mean. All metrics share one set of resampled
    engine indices, so their intervals come from the same draws.
    """
    counts = {len(values) for values, _ in per_engine.values()}
    if len(counts) != 1:
        raise ValueError("All metrics must have the same number of engines")
    (count,) = counts
    columns = np.random.default_rng(seed).integers(0, count, size=(resamples, count))
    return {
        name: np.percentile(
            transform(np.asarray(values)[columns].mean(axis=1)), [2.5, 97.5]
        )
        .astype(float)
        .tolist()
        for name, (values, transform) in per_engine.items()
    }

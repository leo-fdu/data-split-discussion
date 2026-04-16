from __future__ import annotations

from decimal import Decimal

import numpy as np

from .clustering import run_butina_clustering, summarize_clusters
from .distance import combine_distances
from .split import clusters_to_splits
from .types import SweepResult


def generate_param_grid(min_value: float, max_value: float, gap: float) -> list[float]:
    """
    Generate a numeric sweep grid that includes both endpoints.

    Decimal arithmetic is used to avoid dropping the upper endpoint because of
    floating-point accumulation error. If the step does not land exactly on the
    upper endpoint, `max_value` is appended explicitly.
    """
    if gap <= 0:
        raise ValueError("gap must be positive.")
    if min_value > max_value:
        raise ValueError("min_value must be less than or equal to max_value.")

    start = Decimal(str(min_value))
    stop = Decimal(str(max_value))
    step = Decimal(str(gap))
    values: list[float] = []
    current = start

    while current <= stop:
        values.append(float(current))
        current += step

    stop_float = float(stop)
    if not values or abs(values[-1] - stop_float) > 1e-12:
        values.append(stop_float)

    deduplicated: list[float] = []
    for value in values:
        if not deduplicated or abs(value - deduplicated[-1]) > 1e-12:
            deduplicated.append(value)
    return deduplicated


def sweep_tunable_splits(
    d_scaffold: np.ndarray,
    d_fg: np.ndarray,
    lambda_min: float,
    lambda_max: float,
    lambda_gap: float,
    cutoff_min: float,
    cutoff_max: float,
    cutoff_gap: float,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> list[SweepResult]:
    """
    Sweep `(lambda_, cutoff)` pairs and return structured split configurations.

    The sweep only produces clustering and split outputs. It does not train or
    evaluate any predictive model.
    """
    lambda_grid = generate_param_grid(lambda_min, lambda_max, lambda_gap)
    cutoff_grid = generate_param_grid(cutoff_min, cutoff_max, cutoff_gap)
    results: list[SweepResult] = []

    for lambda_ in lambda_grid:
        total_distance = combine_distances(d_scaffold, d_fg, lambda_=lambda_)
        for cutoff in cutoff_grid:
            clusters = run_butina_clustering(total_distance, cutoff=cutoff)
            split_result = clusters_to_splits(
                clusters=clusters,
                train_frac=train_frac,
                val_frac=val_frac,
                test_frac=test_frac,
            )
            cluster_summary = summarize_clusters(clusters)
            summary = {
                "lambda": float(lambda_),
                "cutoff": float(cutoff),
                **cluster_summary,
                "split_summary": split_result.summary,
            }
            results.append(
                SweepResult(
                    lambda_=float(lambda_),
                    cutoff=float(cutoff),
                    clusters=tuple(tuple(cluster) for cluster in clusters),
                    split_result=split_result,
                    summary=summary,
                )
            )
    return results

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from rdkit import Chem

from .config import (
    DEFAULT_MAX_MCS_ROUNDS,
    DEFAULT_NUMPY_DTYPE,
    DEFAULT_SCAFFOLD_FAILURE_DISTANCE,
)
from .io_utils import load_distance_cache, save_distance_cache
from .scaffold import compute_scaffold_similarity
from .types import (
    PairwiseComputationFailure,
    PairwiseDistanceMatrixResult,
    ScaffoldExtractionResult,
)


LOGGER = logging.getLogger(__name__)


def compute_fg_distance(counts_a: np.ndarray, counts_b: np.ndarray) -> float:
    """
    Compute the leaf-level functional-group distance between two count vectors.

    Active dimensions are those where `counts_a[i] + counts_b[i] > 0`. If there
    are no active dimensions, the distance is defined as `0.0`.
    """
    vector_a = np.asarray(counts_a, dtype=np.float64)
    vector_b = np.asarray(counts_b, dtype=np.float64)
    if vector_a.shape != vector_b.shape:
        raise ValueError("FG count vectors must have the same shape.")

    active_mask = (vector_a + vector_b) > 0
    if not np.any(active_mask):
        return 0.0

    active_a = vector_a[active_mask]
    active_b = vector_b[active_mask]
    numerator = np.abs(active_a - active_b)
    denominator = np.maximum(active_a, active_b)
    distance = float(np.mean(numerator / denominator))
    return float(min(1.0, max(0.0, distance)))


def compute_pairwise_fg_distance_matrix(
    fg_count_matrix: np.ndarray,
    dtype: np.dtype = DEFAULT_NUMPY_DTYPE,
) -> np.ndarray:
    """Compute the full pairwise FG distance matrix from a count matrix."""
    counts = np.asarray(fg_count_matrix)
    if counts.ndim != 2:
        raise ValueError("fg_count_matrix must be a 2D array.")

    num_items = counts.shape[0]
    matrix = np.zeros((num_items, num_items), dtype=dtype)
    for i in range(num_items):
        for j in range(i + 1, num_items):
            distance = compute_fg_distance(counts[i], counts[j])
            matrix[i, j] = distance
            matrix[j, i] = distance
    np.fill_diagonal(matrix, 0.0)
    return matrix


def combine_distance_pair(d_scaffold: float, d_fg: float, lambda_: float) -> float:
    """Linearly combine one scaffold distance and one FG distance."""
    _validate_lambda(lambda_)
    combined = lambda_ * float(d_scaffold) + (1.0 - lambda_) * float(d_fg)
    return float(min(1.0, max(0.0, combined)))


def combine_distances(
    d_scaffold: np.ndarray,
    d_fg: np.ndarray,
    lambda_: float,
) -> np.ndarray:
    """
    Linearly combine full scaffold and FG distance matrices.

    The returned matrix is symmetrized, clipped into `[0, 1]`, and has a zero diagonal.
    """
    _validate_lambda(lambda_)
    scaffold_matrix = np.asarray(d_scaffold, dtype=np.float64)
    fg_matrix = np.asarray(d_fg, dtype=np.float64)
    if scaffold_matrix.shape != fg_matrix.shape:
        raise ValueError("d_scaffold and d_fg must have the same shape.")
    if scaffold_matrix.ndim != 2 or scaffold_matrix.shape[0] != scaffold_matrix.shape[1]:
        raise ValueError("Distance inputs must be square matrices.")

    combined = lambda_ * scaffold_matrix + (1.0 - lambda_) * fg_matrix
    combined = np.clip(combined, 0.0, 1.0)
    combined = 0.5 * (combined + combined.T)
    np.fill_diagonal(combined, 0.0)
    return combined.astype(DEFAULT_NUMPY_DTYPE, copy=False)


def compute_pairwise_scaffold_distance_matrix(
    scaffolds: Sequence[Chem.Mol | ScaffoldExtractionResult],
    max_rounds: int = DEFAULT_MAX_MCS_ROUNDS,
    dtype: np.dtype = DEFAULT_NUMPY_DTYPE,
    logger: logging.Logger | None = None,
) -> PairwiseDistanceMatrixResult:
    """
    Compute the full pairwise iterative scaffold distance matrix.

    If one pairwise scaffold MCS computation fails, the pair distance is set to
    `1.0`, a warning is logged, and the failure metadata is returned.
    """
    active_logger = logger or LOGGER
    scaffold_mols = [_coerce_scaffold_mol(item) for item in scaffolds]
    num_items = len(scaffold_mols)
    matrix = np.zeros((num_items, num_items), dtype=dtype)
    failures: list[PairwiseComputationFailure] = []

    for i in range(num_items):
        for j in range(i + 1, num_items):
            try:
                match_result = compute_scaffold_similarity(
                    scaffold_mols[i],
                    scaffold_mols[j],
                    max_rounds=max_rounds,
                )
                if match_result.status.startswith("failed"):
                    raise RuntimeError(
                        f"Scaffold similarity returned failure status: {match_result.status}"
                    )
                distance = match_result.distance
            except Exception as exc:
                active_logger.warning(
                    "Scaffold distance failed for pair (%d, %d): %s",
                    i,
                    j,
                    exc,
                )
                distance = DEFAULT_SCAFFOLD_FAILURE_DISTANCE
                failures.append(
                    PairwiseComputationFailure(
                        i=i,
                        j=j,
                        status="failed",
                        message=str(exc),
                    )
                )
            matrix[i, j] = distance
            matrix[j, i] = distance

    np.fill_diagonal(matrix, 0.0)
    return PairwiseDistanceMatrixResult(
        matrix=matrix,
        metric_name="scaffold",
        failures=tuple(failures),
        metadata={
            "num_items": num_items,
            "max_rounds": max_rounds,
            "dtype": str(np.dtype(dtype)),
            "num_failures": len(failures),
        },
    )


def load_or_compute_scaffold_distance_matrix(
    scaffolds: Sequence[Chem.Mol | ScaffoldExtractionResult],
    cache_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    max_rounds: int = DEFAULT_MAX_MCS_ROUNDS,
    dtype: np.dtype = DEFAULT_NUMPY_DTYPE,
    logger: logging.Logger | None = None,
) -> PairwiseDistanceMatrixResult:
    """Load a cached scaffold distance matrix or compute and cache it."""
    if cache_path is not None and Path(cache_path).exists():
        matrix, metadata = load_distance_cache(cache_path, metadata_path, dtype=dtype)
        failures = _failures_from_metadata(metadata)
        return PairwiseDistanceMatrixResult(
            matrix=matrix,
            metric_name="scaffold",
            failures=failures,
            metadata=metadata or {},
        )

    result = compute_pairwise_scaffold_distance_matrix(
        scaffolds=scaffolds,
        max_rounds=max_rounds,
        dtype=dtype,
        logger=logger,
    )
    if cache_path is not None:
        metadata = _result_to_metadata(result)
        save_distance_cache(result.matrix, cache_path, metadata=metadata, metadata_path=metadata_path)
        result = PairwiseDistanceMatrixResult(
            matrix=result.matrix,
            metric_name=result.metric_name,
            failures=result.failures,
            metadata=metadata,
        )
    return result


def load_or_compute_fg_distance_matrix(
    fg_count_matrix: np.ndarray,
    cache_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    dtype: np.dtype = DEFAULT_NUMPY_DTYPE,
) -> PairwiseDistanceMatrixResult:
    """Load a cached FG distance matrix or compute and cache it."""
    if cache_path is not None and Path(cache_path).exists():
        matrix, metadata = load_distance_cache(cache_path, metadata_path, dtype=dtype)
        return PairwiseDistanceMatrixResult(
            matrix=matrix,
            metric_name="fg",
            failures=(),
            metadata=metadata or {},
        )

    matrix = compute_pairwise_fg_distance_matrix(fg_count_matrix, dtype=dtype)
    metadata = {
        "num_items": int(matrix.shape[0]),
        "dtype": str(np.dtype(dtype)),
        "num_features": int(np.asarray(fg_count_matrix).shape[1]) if np.asarray(fg_count_matrix).ndim == 2 else 0,
    }
    if cache_path is not None:
        save_distance_cache(matrix, cache_path, metadata=metadata, metadata_path=metadata_path)
    return PairwiseDistanceMatrixResult(
        matrix=matrix,
        metric_name="fg",
        failures=(),
        metadata=metadata,
    )


def load_or_compute_total_distance_matrix(
    d_scaffold: np.ndarray,
    d_fg: np.ndarray,
    lambda_: float,
    cache_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> PairwiseDistanceMatrixResult:
    """Load a cached total distance matrix or build it from scaffold and FG distances."""
    if cache_path is not None and Path(cache_path).exists():
        matrix, metadata = load_distance_cache(cache_path, metadata_path, dtype=DEFAULT_NUMPY_DTYPE)
        return PairwiseDistanceMatrixResult(
            matrix=matrix,
            metric_name="total",
            failures=(),
            metadata=metadata or {},
        )

    matrix = combine_distances(d_scaffold, d_fg, lambda_=lambda_)
    metadata = {
        "num_items": int(matrix.shape[0]),
        "dtype": str(matrix.dtype),
        "lambda": float(lambda_),
    }
    if cache_path is not None:
        save_distance_cache(matrix, cache_path, metadata=metadata, metadata_path=metadata_path)
    return PairwiseDistanceMatrixResult(
        matrix=matrix,
        metric_name="total",
        failures=(),
        metadata=metadata,
    )


def _validate_lambda(lambda_: float) -> None:
    if not 0.0 <= float(lambda_) <= 1.0:
        raise ValueError("lambda_ must be within the closed interval [0, 1].")


def _coerce_scaffold_mol(item: Chem.Mol | ScaffoldExtractionResult) -> Chem.Mol:
    if isinstance(item, ScaffoldExtractionResult):
        return item.scaffold_mol
    return item


def _result_to_metadata(result: PairwiseDistanceMatrixResult) -> dict[str, object]:
    metadata = dict(result.metadata)
    metadata["metric_name"] = result.metric_name
    metadata["failures"] = [
        {
            "i": failure.i,
            "j": failure.j,
            "status": failure.status,
            "message": failure.message,
        }
        for failure in result.failures
    ]
    return metadata


def _failures_from_metadata(
    metadata: dict[str, object] | None,
) -> tuple[PairwiseComputationFailure, ...]:
    if not metadata:
        return ()
    failures = metadata.get("failures", [])
    return tuple(
        PairwiseComputationFailure(
            i=int(item["i"]),
            j=int(item["j"]),
            status=str(item["status"]),
            message=str(item["message"]),
        )
        for item in failures
    )

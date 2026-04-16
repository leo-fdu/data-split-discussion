from __future__ import annotations

import numpy as np
from rdkit.ML.Cluster import Butina


def distance_matrix_to_condensed(distance_matrix: np.ndarray) -> list[float]:
    """Convert a full symmetric distance matrix into RDKit's condensed lower triangle."""
    matrix = np.asarray(distance_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix.")
    if matrix.shape[0] == 0:
        return []
    if not np.allclose(matrix, matrix.T, atol=1e-8):
        raise ValueError("distance_matrix must be symmetric.")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-8):
        raise ValueError("distance_matrix must have a zero diagonal.")

    condensed: list[float] = []
    for i in range(1, matrix.shape[0]):
        for j in range(i):
            condensed.append(float(matrix[i, j]))
    return condensed


def run_butina_clustering(
    distance_matrix: np.ndarray,
    cutoff: float,
) -> list[list[int]]:
    """
    Cluster a distance matrix with RDKit Butina clustering.

    `cutoff` is a distance cutoff, not a similarity cutoff. Smaller cutoff values
    create stricter and more fragmented clusters; larger cutoff values create
    looser and larger clusters.
    """
    if cutoff < 0:
        raise ValueError("cutoff must be non-negative.")

    matrix = np.asarray(distance_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix.")
    if matrix.shape[0] == 0:
        return []
    if matrix.shape[0] == 1:
        return [[0]]

    condensed = distance_matrix_to_condensed(matrix)
    raw_clusters = Butina.ClusterData(
        condensed,
        nPts=matrix.shape[0],
        distThresh=float(cutoff),
        isDistData=True,
        reordering=False,
    )
    normalized_clusters = [sorted(cluster) for cluster in raw_clusters]
    normalized_clusters.sort(key=lambda cluster: (-len(cluster), cluster[0], tuple(cluster)))
    return normalized_clusters


def summarize_clusters(clusters: list[list[int]]) -> dict[str, object]:
    """Return a lightweight deterministic summary for a cluster list."""
    cluster_sizes = [len(cluster) for cluster in clusters]
    return {
        "num_clusters": len(clusters),
        "cluster_sizes": cluster_sizes,
        "largest_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "smallest_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
    }

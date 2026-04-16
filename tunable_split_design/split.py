from __future__ import annotations

import math

from .config import FLOAT_TOLERANCE, SPLIT_NAMES
from .types import AssignedCluster, SplitResult


def clusters_to_splits(
    clusters: list[list[int]],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> SplitResult:
    """
    Convert clusters into deterministic train/val/test splits with a greedy assignment.

    Clusters are ordered by descending size, then assigned one by one to the
    split whose post-assignment size is closest to its target while preferring
    splits that are still below target size.
    """
    _validate_split_fractions(train_frac, val_frac, test_frac)
    ordered_clusters = [
        tuple(sorted(cluster))
        for cluster in sorted(clusters, key=lambda item: (-len(item), min(item), tuple(item)))
    ]

    seen_indices: set[int] = set()
    total_count = 0
    for cluster in ordered_clusters:
        cluster_set = set(cluster)
        if len(cluster_set) != len(cluster):
            raise ValueError(f"Cluster contains duplicate sample indices: {cluster}")
        if seen_indices & cluster_set:
            raise ValueError("Input clusters overlap; every sample must belong to exactly one cluster.")
        seen_indices.update(cluster_set)
        total_count += len(cluster)

    target_counts = compute_target_counts(total_count, train_frac, val_frac, test_frac)
    split_to_indices = {split_name: [] for split_name in SPLIT_NAMES}
    current_counts = {split_name: 0 for split_name in SPLIT_NAMES}
    assignments: list[AssignedCluster] = []

    for cluster_id, cluster in enumerate(ordered_clusters):
        split_name = _choose_split_for_cluster(current_counts, target_counts, len(cluster))
        split_to_indices[split_name].extend(cluster)
        current_counts[split_name] += len(cluster)
        assignments.append(
            AssignedCluster(
                cluster_id=cluster_id,
                split_name=split_name,
                indices=cluster,
            )
        )

    train_indices = tuple(sorted(split_to_indices["train"]))
    val_indices = tuple(sorted(split_to_indices["val"]))
    test_indices = tuple(sorted(split_to_indices["test"]))
    summary = {
        "total_count": total_count,
        "target_counts": target_counts,
        "actual_counts": {
            "train": len(train_indices),
            "val": len(val_indices),
            "test": len(test_indices),
        },
        "actual_fractions": {
            "train": len(train_indices) / total_count if total_count else 0.0,
            "val": len(val_indices) / total_count if total_count else 0.0,
            "test": len(test_indices) / total_count if total_count else 0.0,
        },
        "num_clusters": len(ordered_clusters),
        "cluster_sizes": [len(cluster) for cluster in ordered_clusters],
    }
    return SplitResult(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        cluster_assignments=tuple(assignments),
        summary=summary,
    )


def compute_target_counts(
    total_count: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> dict[str, int]:
    """Round train/val/test target counts with a largest-remainder rule."""
    _validate_split_fractions(train_frac, val_frac, test_frac)
    fractions = {
        "train": train_frac,
        "val": val_frac,
        "test": test_frac,
    }
    raw_counts = {
        split_name: total_count * fraction
        for split_name, fraction in fractions.items()
    }
    rounded_counts = {
        split_name: math.floor(raw_counts[split_name])
        for split_name in SPLIT_NAMES
    }
    remainder = total_count - sum(rounded_counts.values())
    remainder_order = sorted(
        SPLIT_NAMES,
        key=lambda split_name: (
            raw_counts[split_name] - rounded_counts[split_name],
            -SPLIT_NAMES.index(split_name),
        ),
        reverse=True,
    )
    for offset in range(remainder):
        rounded_counts[remainder_order[offset]] += 1
    return rounded_counts


def _validate_split_fractions(train_frac: float, val_frac: float, test_frac: float) -> None:
    fractions = [train_frac, val_frac, test_frac]
    if any(fraction < 0.0 or fraction > 1.0 for fraction in fractions):
        raise ValueError("Split fractions must each lie in the closed interval [0, 1].")
    if abs(sum(fractions) - 1.0) > FLOAT_TOLERANCE:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0.")


def _choose_split_for_cluster(
    current_counts: dict[str, int],
    target_counts: dict[str, int],
    cluster_size: int,
) -> str:
    ranked_candidates = []
    for split_name in SPLIT_NAMES:
        current_count = current_counts[split_name]
        target_count = target_counts[split_name]
        after_assignment = current_count + cluster_size
        ranked_candidates.append(
            (
                0 if current_count < target_count else 1,
                abs(target_count - after_assignment),
                -max(target_count - current_count, 0),
                SPLIT_NAMES.index(split_name),
                split_name,
            )
        )
    ranked_candidates.sort()
    return ranked_candidates[0][-1]

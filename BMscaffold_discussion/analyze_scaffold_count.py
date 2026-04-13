from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "scaffold_counts"
INPUT_PATTERN = "*_scaffold_counts.csv"
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_PATH = RESULTS_DIR / "scaffold_summary.csv"
NO_SCAFFOLD_TOKEN = "[NO_SCAFFOLD]"
OUTPUT_COLUMNS = [
    "dataset_name",
    "total_molecule_count",
    "total_scaffold_count",
    "no_scaffold_ratio",
    "top1_ratio",
    "top5_ratio",
    "singleton_ratio",
    "doubleton_ratio",
    "tripleton_ratio",
    "gini_coefficient",
]


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a ratio while guarding against division by zero."""
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def dataset_name_from_path(path: Path) -> str:
    """Extract the dataset name from a *_scaffold_counts.csv filename."""
    suffix = "_scaffold_counts"
    stem = path.stem
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def gini_coefficient(counts: np.ndarray) -> float:
    """Compute the Gini coefficient from scaffold counts."""
    if counts.size == 0:
        return 0.0

    sorted_counts = np.sort(counts.astype(float))
    total = sorted_counts.sum()
    n = sorted_counts.size

    if total == 0 or n == 0:
        return 0.0

    indices = np.arange(1, n + 1, dtype=float)
    gini = (2.0 * np.sum(indices * sorted_counts)) / (n * total) - (n + 1.0) / n
    return float(gini)


def analyze_dataset(csv_path: Path) -> dict[str, float | int | str]:
    """Read one scaffold-count CSV and compute the requested summary metrics."""
    df = pd.read_csv(csv_path)

    required_columns = {"scaffold_smiles", "count"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_display = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"{csv_path.name} is missing required column(s): {missing_display}"
        )

    if df.empty:
        counts = np.array([], dtype=int)
        no_scaffold_count = 0
    else:
        counts = pd.to_numeric(df["count"], errors="coerce").fillna(0).to_numpy(dtype=int)
        no_scaffold_mask = df["scaffold_smiles"].astype(str) == NO_SCAFFOLD_TOKEN
        no_scaffold_count = int(counts[no_scaffold_mask.to_numpy()].sum())

    total_molecule_count = int(counts.sum())
    total_scaffold_count = int(len(df))
    sorted_desc = np.sort(counts)[::-1]

    summary = {
        "dataset_name": dataset_name_from_path(csv_path),
        "total_molecule_count": total_molecule_count,
        "total_scaffold_count": total_scaffold_count,
        "no_scaffold_ratio": safe_ratio(no_scaffold_count, total_molecule_count),
        "top1_ratio": safe_ratio(int(sorted_desc[:1].sum()), total_molecule_count),
        "top5_ratio": safe_ratio(int(sorted_desc[:5].sum()), total_molecule_count),
        "singleton_ratio": safe_ratio(int(np.sum(counts == 1)), total_scaffold_count),
        "doubleton_ratio": safe_ratio(int(np.sum(counts == 2)), total_scaffold_count),
        "tripleton_ratio": safe_ratio(int(np.sum(counts == 3)), total_scaffold_count),
        "gini_coefficient": gini_coefficient(counts),
    }

    print(
        f"{summary['dataset_name']}: molecules={summary['total_molecule_count']}, "
        f"scaffolds={summary['total_scaffold_count']}, "
        f"top1={summary['top1_ratio']:.4f}, "
        f"top5={summary['top5_ratio']:.4f}, "
        f"gini={summary['gini_coefficient']:.4f}"
    )

    return summary


def main() -> int:
    """Analyze all scaffold-count CSV files in the scaffold_counts directory."""
    csv_files = sorted(
        path
        for path in INPUT_DIR.glob(INPUT_PATTERN)
        if path.name != OUTPUT_PATH.name
    )

    if not csv_files:
        print(f"No files matching '{INPUT_PATTERN}' were found in {INPUT_DIR}")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        empty_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
        empty_df.to_csv(OUTPUT_PATH, index=False)
        print(f"Saved empty summary to {OUTPUT_PATH}")
        return 0

    summaries = [analyze_dataset(csv_path) for csv_path in csv_files]
    summary_df = pd.DataFrame(summaries, columns=OUTPUT_COLUMNS)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved dataset summary to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

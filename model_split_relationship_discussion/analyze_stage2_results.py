from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - environment-dependent fallback
    sns = None


SPLIT_ORDER = [
    "random",
    "bm_scaffold",
    "butina_morgan",
    "butina_maccs",
    "butina_atompair",
]
BUTINA_ORDER = ["butina_morgan", "butina_maccs", "butina_atompair"]
FINGERPRINT_ORDER = ["morgan", "maccs", "atompair"]
CLASSIFICATION_METRICS = ["roc_auc", "pr_auc"]
REGRESSION_METRICS = ["mae", "rmse"]
PARTITION_LABELS = {"val": "Validation", "test": "Test"}
BASE_COLUMNS = ["dataset", "task_type", "fingerprint_type", "split_method"]
SEABORN_FALLBACK_WARNED = False


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "summary_results.csv",
        help="Path to results/summary_results.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "analysis",
        help="Directory for analysis outputs.",
    )
    return parser


def load_summary_results(summary_csv: Path) -> pd.DataFrame:
    """Load the aggregated results CSV."""
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary results file: {summary_csv}")

    df = pd.read_csv(summary_csv)
    if df.empty:
        raise ValueError(f"Summary results file is empty: {summary_csv}")

    return df


def validate_columns(df: pd.DataFrame) -> None:
    """Validate the base schema required for Stage 2 analysis."""
    missing = [column for column in BASE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "summary_results.csv is missing required columns: "
            f"{missing}"
        )


def get_metrics_for_task(task_type: str, columns: list[str]) -> list[str]:
    """Return the valid metrics for the task type that exist in the real schema."""
    if task_type == "classification":
        candidate_metrics = CLASSIFICATION_METRICS
    elif task_type == "regression":
        candidate_metrics = REGRESSION_METRICS
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    available_metrics = []
    for metric in candidate_metrics:
        required = [f"val_{metric}_mean", f"test_{metric}_mean"]
        if all(column in columns for column in required):
            available_metrics.append(metric)
        else:
            warnings.warn(
                f"Skipping metric '{metric}' for task_type '{task_type}' because "
                f"required columns are missing: {required}",
                stacklevel=2,
            )

    if not available_metrics:
        raise ValueError(f"No valid metrics available for task_type '{task_type}'.")

    return available_metrics


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    """Create analysis output folders."""
    heatmap_dir = output_dir / "heatmaps"
    csv_dir = output_dir / "csv"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    return {"root": output_dir, "heatmaps": heatmap_dir, "csv": csv_dir}


def get_dataset_task_type(df: pd.DataFrame, dataset: str) -> str:
    """Get the unique task type for one dataset."""
    task_types = sorted(df.loc[df["dataset"] == dataset, "task_type"].dropna().unique())
    if len(task_types) != 1:
        raise ValueError(
            f"Expected exactly one task_type for dataset '{dataset}', found {task_types}."
        )
    return task_types[0]


def build_metric_column(partition: str, metric: str) -> str:
    """Build the metric mean column name from partition and metric."""
    return f"{partition}_{metric}_mean"


def make_global_table(dataset_df: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    """Create the split x fingerprint table for absolute metric values."""
    table = dataset_df.pivot_table(
        index="split_method",
        columns="fingerprint_type",
        values=metric_column,
        aggfunc="first",
    )
    return table.reindex(index=SPLIT_ORDER, columns=FINGERPRINT_ORDER)


def compute_delta_vs_random(dataset_df: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    """Compute Butina split deltas relative to the random split under the same fingerprint."""
    table = make_global_table(dataset_df, metric_column)
    random_baseline = table.loc["random"]
    if random_baseline.isna().any():
        warnings.warn(
            f"Random baseline is incomplete for '{metric_column}'. Delta heatmap may contain NaN values.",
            stacklevel=2,
        )

    butina_table = table.reindex(index=BUTINA_ORDER, columns=FINGERPRINT_ORDER)
    return butina_table.subtract(random_baseline, axis="columns")


def save_heatmap(
    table: pd.DataFrame,
    title: str,
    output_path: Path,
    cmap: str,
    center: float | None = None,
) -> None:
    """Save one annotated heatmap."""
    global SEABORN_FALLBACK_WARNED
    plt.figure(figsize=(8, 4.8))

    if sns is not None:
        ax = sns.heatmap(
            table,
            annot=True,
            fmt=".3f",
            cmap=cmap,
            center=center,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"shrink": 0.9},
        )
    else:
        if not SEABORN_FALLBACK_WARNED:
            warnings.warn(
                "seaborn is not installed; using a matplotlib heatmap fallback.",
                stacklevel=2,
            )
            SEABORN_FALLBACK_WARNED = True
        ax = plt.gca()
        values = table.to_numpy(dtype=float)
        masked = np.ma.masked_invalid(values)

        if center is None or np.isnan(values).all():
            im = ax.imshow(masked, cmap=cmap, aspect="auto")
        else:
            max_delta = np.nanmax(np.abs(values - center))
            im = ax.imshow(
                masked,
                cmap=cmap,
                aspect="auto",
                vmin=center - max_delta,
                vmax=center + max_delta,
            )

        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels(table.columns)
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels(table.index)

        for row_idx in range(table.shape[0]):
            for col_idx in range(table.shape[1]):
                value = values[row_idx, col_idx]
                label = "" if np.isnan(value) else f"{value:.3f}"
                ax.text(col_idx, row_idx, label, ha="center", va="center", color="black")

        plt.colorbar(im, ax=ax, shrink=0.9)

    ax.set_xlabel("Fingerprint Representation")
    ax.set_ylabel("Split Method")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def make_global_heatmap(
    dataset: str,
    partition: str,
    metric: str,
    table: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Create and save the global split x fingerprint heatmap."""
    output_path = output_dir / f"{dataset}_{partition}_{metric}_global_heatmap.png"
    title = (
        f"{dataset} {PARTITION_LABELS[partition]} {metric} Mean\n"
        "Global Split x Fingerprint Landscape"
    )
    save_heatmap(table, title, output_path, cmap="viridis")
    return output_path


def make_butina_delta_heatmap(
    dataset: str,
    partition: str,
    metric: str,
    delta_table: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Create and save the Butina-only delta-to-random heatmap."""
    output_path = output_dir / f"{dataset}_{partition}_{metric}_butina_delta_heatmap.png"
    title = (
        f"{dataset} {PARTITION_LABELS[partition]} {metric} Delta vs Random\n"
        "Butina Split Effect by Fingerprint"
    )
    save_heatmap(delta_table, title, output_path, cmap="coolwarm", center=0.0)
    return output_path


def split_fingerprint_from_method(split_method: str) -> str | None:
    """Map Butina split method name to its fingerprint family."""
    if split_method.startswith("butina_"):
        return split_method.removeprefix("butina_")
    return None


def compute_relative_effect(
    aligned_minus_unaligned: float,
    unaligned_mean_delta: float,
) -> float:
    """Compute the normalized aligned-vs-unaligned effect safely."""
    if pd.isna(unaligned_mean_delta) or unaligned_mean_delta == 0.0:
        return np.nan
    return float(aligned_minus_unaligned / abs(unaligned_mean_delta))


def summarize_aligned_vs_unaligned(
    dataset: str,
    task_type: str,
    partition: str,
    metric: str,
    delta_table: pd.DataFrame,
) -> dict[str, object]:
    """Summarize aligned vs unaligned Butina delta cells."""
    aligned_values = []
    unaligned_values = []

    for split_method in BUTINA_ORDER:
        split_fp = split_fingerprint_from_method(split_method)
        for fingerprint_type in FINGERPRINT_ORDER:
            value = delta_table.loc[split_method, fingerprint_type]
            if pd.isna(value):
                continue
            if split_fp == fingerprint_type:
                aligned_values.append(float(value))
            else:
                unaligned_values.append(float(value))

    aligned_mean = sum(aligned_values) / len(aligned_values) if aligned_values else float("nan")
    unaligned_mean = sum(unaligned_values) / len(unaligned_values) if unaligned_values else float("nan")
    aligned_minus_unaligned = aligned_mean - unaligned_mean
    relative_effect = compute_relative_effect(aligned_minus_unaligned, unaligned_mean)

    return {
        "dataset": dataset,
        "task_type": task_type,
        "partition": partition,
        "metric": metric,
        "aligned_mean_delta": aligned_mean,
        "unaligned_mean_delta": unaligned_mean,
        "aligned_minus_unaligned": aligned_minus_unaligned,
        "relative_effect": relative_effect,
    }


def main() -> int:
    args = build_parser().parse_args()
    summary_csv = args.summary_csv.resolve()
    output_dirs = ensure_output_dirs(args.output_dir.resolve())

    summary_df = load_summary_results(summary_csv)
    validate_columns(summary_df)

    if sns is not None:
        sns.set_theme(style="white")

    saved_heatmaps = []
    saved_csvs = []
    aligned_rows = []

    for dataset in sorted(summary_df["dataset"].dropna().unique()):
        dataset_df = summary_df.loc[summary_df["dataset"] == dataset].copy()
        task_type = get_dataset_task_type(summary_df, dataset)
        metrics = get_metrics_for_task(task_type, summary_df.columns.tolist())

        for partition in ["val", "test"]:
            for metric in metrics:
                metric_column = build_metric_column(partition, metric)
                if metric_column not in dataset_df.columns:
                    warnings.warn(
                        f"Skipping {dataset} / {partition} / {metric} because "
                        f"'{metric_column}' is missing.",
                        stacklevel=2,
                    )
                    continue

                global_table = make_global_table(dataset_df, metric_column)
                if global_table.isna().all().all():
                    warnings.warn(
                        f"Skipping {dataset} / {partition} / {metric} because the global table is empty.",
                        stacklevel=2,
                    )
                    continue

                global_path = make_global_heatmap(
                    dataset=dataset,
                    partition=partition,
                    metric=metric,
                    table=global_table,
                    output_dir=output_dirs["heatmaps"],
                )
                saved_heatmaps.append(global_path)

                delta_table = compute_delta_vs_random(dataset_df, metric_column)
                delta_csv_path = (
                    output_dirs["csv"] / f"{dataset}_{partition}_{metric}_butina_delta_table.csv"
                )
                delta_table.to_csv(delta_csv_path)
                saved_csvs.append(delta_csv_path)

                if delta_table.isna().all().all():
                    warnings.warn(
                        f"Skipping delta heatmap for {dataset} / {partition} / {metric} "
                        "because all delta values are NaN.",
                        stacklevel=2,
                    )
                    continue

                delta_path = make_butina_delta_heatmap(
                    dataset=dataset,
                    partition=partition,
                    metric=metric,
                    delta_table=delta_table,
                    output_dir=output_dirs["heatmaps"],
                )
                saved_heatmaps.append(delta_path)

                aligned_rows.append(
                    summarize_aligned_vs_unaligned(
                        dataset=dataset,
                        task_type=task_type,
                        partition=partition,
                        metric=metric,
                        delta_table=delta_table,
                    )
                )
                summary_row = aligned_rows[-1]
                relative_effect = summary_row["relative_effect"]
                relative_effect_text = (
                    f"{relative_effect:.4f}" if pd.notna(relative_effect) else "nan"
                )
                print(
                    f"{dataset} / {partition} / {metric} -> "
                    f"relative_effect={relative_effect_text}"
                )

    aligned_df = pd.DataFrame(
        aligned_rows,
        columns=[
            "dataset",
            "task_type",
            "partition",
            "metric",
            "aligned_mean_delta",
            "unaligned_mean_delta",
            "aligned_minus_unaligned",
            "relative_effect",
        ],
    )
    aligned_df = aligned_df.sort_values(["dataset", "partition", "metric"])
    aligned_df["relative_effect_pct"] = (
        aligned_df["relative_effect"]
        .mul(100.0)
        .round(2)
        .map(lambda value: "NaN" if pd.isna(value) else f"{value:.2f}%")
    )
    aligned_summary_path = output_dirs["csv"] / "aligned_unaligned_summary.csv"
    aligned_df.to_csv(aligned_summary_path, index=False)
    saved_csvs.append(aligned_summary_path)

    print(f"Loaded summary results: {summary_csv}")
    print(f"Saved heatmaps: {len(saved_heatmaps)}")
    print(f"Saved CSV files: {len(saved_csvs)}")
    print(f"Heatmap output dir: {output_dirs['heatmaps']}")
    print(f"CSV output dir: {output_dirs['csv']}")
    print(f"Aligned summary CSV: {aligned_summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

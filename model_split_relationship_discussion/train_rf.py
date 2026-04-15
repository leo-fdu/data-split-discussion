from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from data_utils import (
    SPLIT_NAMES,
    TARGET_DATASETS,
    TASK_TYPES,
    discover_runs,
    load_dataset_frame,
    load_split_indices,
    load_targets,
)
from fingerprints import FINGERPRINT_TYPES, build_fingerprint_matrix
from metrics import METRIC_NAMES, classification_metrics, regression_metrics


COMMON_RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 0.5,
    "n_jobs": -1,
    "random_state": 42,
}
CLASSIFIER_RF_PARAMS = {**COMMON_RF_PARAMS, "class_weight": "balanced"}
REGRESSOR_RF_PARAMS = dict(COMMON_RF_PARAMS)
RESULT_COLUMNS = [
    "dataset",
    "task_type",
    "fingerprint_type",
    "split_method",
    "run_label",
    "model_name",
    "hyperparameters",
    "train_size",
    "val_size",
    "test_size",
] + [f"{split_name}_{metric_name}" for split_name in SPLIT_NAMES for metric_name in METRIC_NAMES]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Path to the results directory.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=f"Optional subset of datasets from: {', '.join(TARGET_DATASETS)}",
    )
    parser.add_argument(
        "--fingerprints",
        nargs="+",
        default=None,
        help=f"Optional subset of fingerprints from: {', '.join(FINGERPRINT_TYPES)}",
    )
    parser.add_argument(
        "--split_methods",
        nargs="+",
        default=None,
        help="Optional subset of split methods.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing result files.",
    )
    return parser


def validate_fingerprints(fingerprints: list[str] | None) -> list[str]:
    selected = fingerprints or FINGERPRINT_TYPES
    invalid = sorted(set(selected) - set(FINGERPRINT_TYPES))
    if invalid:
        raise ValueError(f"Unsupported fingerprints requested: {invalid}")
    return selected


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_files = [
        output_dir / "per_run_results.csv",
        output_dir / "summary_results.csv",
        output_dir / "metadata.json",
    ]
    existing = [path for path in result_files if path.exists()]
    if existing and not overwrite:
        existing_text = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output files already exist: {existing_text}. Use --overwrite to replace them."
        )


def make_model(task_type: str):
    if task_type == "classification":
        return RandomForestClassifier(**CLASSIFIER_RF_PARAMS), "RandomForestClassifier"
    if task_type == "regression":
        return RandomForestRegressor(**REGRESSOR_RF_PARAMS), "RandomForestRegressor"
    raise ValueError(f"Unsupported task type: {task_type}")


def get_hyperparameters_text(task_type: str) -> str:
    if task_type == "classification":
        return json.dumps(CLASSIFIER_RF_PARAMS, sort_keys=True)
    return json.dumps(REGRESSOR_RF_PARAMS, sort_keys=True)


def evaluate_model(model, task_type: str, X_split, y_split, context: str) -> dict[str, float]:
    if task_type == "classification":
        y_pred = model.predict(X_split)
        classes = list(model.classes_)
        positive_label = max(set(classes) | set(pd.unique(y_split).tolist()))
        if positive_label in classes:
            positive_index = classes.index(positive_label)
            y_score = model.predict_proba(X_split)[:, positive_index]
        else:
            y_score = np.zeros(len(y_split), dtype=float)
        return classification_metrics(y_split, y_pred, y_score, context, positive_label)

    y_pred = model.predict(X_split)
    return regression_metrics(y_split, y_pred, context)


def build_result_row(
    dataset: str,
    task_type: str,
    fingerprint_type: str,
    split_method: str,
    run_label: str,
    model_name: str,
    hyperparameters: str,
    split_indices: dict[str, object],
    split_metrics: dict[str, dict[str, float]],
) -> dict[str, object]:
    row = {
        "dataset": dataset,
        "task_type": task_type,
        "fingerprint_type": fingerprint_type,
        "split_method": split_method,
        "run_label": run_label,
        "model_name": model_name,
        "hyperparameters": hyperparameters,
        "train_size": len(split_indices["train"]),
        "val_size": len(split_indices["val"]),
        "test_size": len(split_indices["test"]),
    }

    for split_name in SPLIT_NAMES:
        for metric_name in METRIC_NAMES:
            row[f"{split_name}_{metric_name}"] = split_metrics[split_name][metric_name]

    return row


def summarise_results(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()

    group_columns = [
        "dataset",
        "task_type",
        "fingerprint_type",
        "split_method",
        "model_name",
        "hyperparameters",
    ]
    numeric_columns = results_df.select_dtypes(include=["number"]).columns.tolist()

    summary_df = (
        results_df.groupby(group_columns, dropna=False)
        .agg(
            run_count=("run_label", "count"),
            **{f"{column}_mean": (column, "mean") for column in numeric_columns},
            **{f"{column}_std": (column, "std") for column in numeric_columns},
        )
        .reset_index()
    )
    return summary_df


def main() -> int:
    args = build_parser().parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    selected_fingerprints = validate_fingerprints(args.fingerprints)
    prepare_output_dir(output_dir, args.overwrite)

    runs = discover_runs(
        data_dir=data_dir,
        datasets=args.datasets,
        split_methods=args.split_methods,
    )

    selected_datasets = sorted({run["dataset"] for run in runs})
    selected_split_methods = sorted({run["split_method"] for run in runs})

    runs_by_dataset: dict[str, list[dict[str, str | Path]]] = {
        dataset: [] for dataset in selected_datasets
    }
    for run in runs:
        runs_by_dataset[str(run["dataset"])].append(run)

    results = []
    failed_runs = []
    total_runs = 0

    for dataset in selected_datasets:
        task_type = TASK_TYPES[dataset]
        full_df = load_dataset_frame(data_dir, dataset)
        y = load_targets(full_df, task_type)
        smiles = full_df["smiles"].tolist()

        fingerprint_cache = {}
        for fingerprint_type in selected_fingerprints:
            fingerprint_cache[fingerprint_type] = build_fingerprint_matrix(
                smiles,
                fingerprint_type,
            )

        for fingerprint_type in selected_fingerprints:
            X = fingerprint_cache[fingerprint_type]

            for run in runs_by_dataset[dataset]:
                split_method = str(run["split_method"])
                run_label = str(run["run_label"])
                split_path = Path(run["split_path"])
                total_runs += 1

                try:
                    split_indices = load_split_indices(split_path, dataset_size=len(full_df))
                    X_train = X[split_indices["train"]]
                    X_val = X[split_indices["val"]]
                    X_test = X[split_indices["test"]]
                    y_train = y[split_indices["train"]]
                    y_val = y[split_indices["val"]]
                    y_test = y[split_indices["test"]]

                    model, model_name = make_model(task_type)
                    model.fit(X_train, y_train)

                    context_base = f"{dataset} / {fingerprint_type} / {split_method} / {run_label}"
                    split_metrics = {
                        "train": evaluate_model(
                            model,
                            task_type,
                            X_train,
                            y_train,
                            f"{context_base} / train",
                        ),
                        "val": evaluate_model(
                            model,
                            task_type,
                            X_val,
                            y_val,
                            f"{context_base} / val",
                        ),
                        "test": evaluate_model(
                            model,
                            task_type,
                            X_test,
                            y_test,
                            f"{context_base} / test",
                        ),
                    }

                    results.append(
                        build_result_row(
                            dataset=dataset,
                            task_type=task_type,
                            fingerprint_type=fingerprint_type,
                            split_method=split_method,
                            run_label=run_label,
                            model_name=model_name,
                            hyperparameters=get_hyperparameters_text(task_type),
                            split_indices=split_indices,
                            split_metrics=split_metrics,
                        )
                    )
                    print(f"{dataset} / {fingerprint_type} / {split_method} / {run_label} -> done")
                except Exception as exc:
                    failed_runs.append(
                        {
                            "dataset": dataset,
                            "fingerprint_type": fingerprint_type,
                            "split_method": split_method,
                            "run_label": run_label,
                            "error": str(exc),
                        }
                    )
                    print(
                        f"{dataset} / {fingerprint_type} / {split_method} / {run_label} "
                        f"-> FAILED: {exc}"
                    )

    per_run_df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    if not per_run_df.empty:
        per_run_df = per_run_df.sort_values(
            ["dataset", "fingerprint_type", "split_method", "run_label"]
        )
    summary_df = summarise_results(per_run_df)

    per_run_path = output_dir / "per_run_results.csv"
    summary_path = output_dir / "summary_results.csv"
    metadata_path = output_dir / "metadata.json"

    per_run_df.to_csv(per_run_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    metadata = {
        "datasets": selected_datasets,
        "fingerprints": selected_fingerprints,
        "split_methods": selected_split_methods,
        "config": {
            "model_family": "RandomForest",
            "classification_model": "RandomForestClassifier",
            "regression_model": "RandomForestRegressor",
            "classification_hyperparameters": CLASSIFIER_RF_PARAMS,
            "regression_hyperparameters": REGRESSOR_RF_PARAMS,
        },
        "total_runs": total_runs,
        "failed_runs": failed_runs,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"\nTotal runs: {total_runs}")
    print(f"Failed runs: {len(failed_runs)}")
    print(f"Saved per-run results: {per_run_path}")
    print(f"Saved summary results: {summary_path}")
    print(f"Saved metadata: {metadata_path}")

    return 1 if failed_runs else 0


if __name__ == "__main__":
    raise SystemExit(main())

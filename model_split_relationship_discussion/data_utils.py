from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_DATASETS = ["bbbp", "freesolv", "hiv", "lipophilicity", "solv"]
TASK_TYPES = {
    "bbbp": "classification",
    "hiv": "classification",
    "freesolv": "regression",
    "lipophilicity": "regression",
    "solv": "regression",
}
SPLIT_NAMES = ["train", "val", "test"]
REQUIRED_COLUMNS = ["smiles", "y"]


def load_dataset_frame(data_dir: Path, dataset: str) -> pd.DataFrame:
    csv_path = data_dir / "labelled_cleaned" / f"{dataset}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dataset CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    return df.loc[:, REQUIRED_COLUMNS].copy().reset_index(drop=True)


def load_targets(df: pd.DataFrame, task_type: str) -> np.ndarray:
    y = pd.to_numeric(df["y"], errors="raise")
    if y.isna().any():
        raise ValueError("Target column 'y' contains missing values.")

    if task_type == "classification":
        return y.astype(int).to_numpy()

    return y.astype(float).to_numpy()


def load_split_indices(split_path: Path, dataset_size: int) -> dict[str, np.ndarray]:
    with split_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    indices = {}
    for split_name in SPLIT_NAMES:
        key = f"{split_name}_idx"
        if key not in payload:
            raise ValueError(f"{split_path} is missing key '{key}'.")

        split_idx = np.asarray(payload[key], dtype=int)
        if np.any(split_idx < 0) or np.any(split_idx >= dataset_size):
            raise ValueError(f"{split_path} contains out-of-range indices in '{key}'.")
        if len(split_idx) != len(set(split_idx.tolist())):
            raise ValueError(f"{split_path} contains duplicate indices in '{key}'.")
        indices[split_name] = split_idx

    if set(indices["train"]) & set(indices["val"]):
        raise ValueError(f"{split_path} has overlapping train/val indices.")
    if set(indices["train"]) & set(indices["test"]):
        raise ValueError(f"{split_path} has overlapping train/test indices.")
    if set(indices["val"]) & set(indices["test"]):
        raise ValueError(f"{split_path} has overlapping val/test indices.")

    return indices


def discover_runs(
    data_dir: Path,
    datasets: list[str] | None = None,
    split_methods: list[str] | None = None,
) -> list[dict[str, str | Path]]:
    split_root = data_dir / "split_data_index"
    if not split_root.exists():
        raise FileNotFoundError(f"Missing split index directory: {split_root}")

    available_datasets = [
        dataset
        for dataset in TARGET_DATASETS
        if (split_root / dataset).is_dir()
    ]
    selected_datasets = datasets or available_datasets

    invalid_datasets = sorted(set(selected_datasets) - set(TARGET_DATASETS))
    if invalid_datasets:
        raise ValueError(f"Unsupported datasets requested: {invalid_datasets}")

    missing_datasets = sorted(set(selected_datasets) - set(available_datasets))
    if missing_datasets:
        raise ValueError(f"Missing split index folders for datasets: {missing_datasets}")

    runs: list[dict[str, str | Path]] = []
    for dataset in selected_datasets:
        dataset_dir = split_root / dataset
        method_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())

        if split_methods is not None:
            requested_methods = set(split_methods)
            method_dirs = [path for path in method_dirs if path.name in requested_methods]
            missing_methods = sorted(
                requested_methods - {path.name for path in method_dirs}
            )
            if missing_methods:
                raise ValueError(
                    f"{dataset} is missing requested split methods: {missing_methods}"
                )

        for method_dir in method_dirs:
            for run_path in sorted(method_dir.glob("*.json")):
                runs.append(
                    {
                        "dataset": dataset,
                        "split_method": method_dir.name,
                        "run_label": run_path.stem,
                        "split_path": run_path,
                    }
                )

    if not runs:
        raise ValueError("No split runs found for the requested filters.")

    return runs

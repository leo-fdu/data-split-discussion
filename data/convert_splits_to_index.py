from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["smiles", "y"]
SPLIT_NAMES = ["train", "val", "test"]


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df.loc[:, REQUIRED_COLUMNS].copy()


def choose_match_columns(full_df: pd.DataFrame, dataset_name: str) -> list[str]:
    if full_df["smiles"].is_unique:
        return ["smiles"]

    if not full_df.duplicated(subset=["smiles", "y"]).any():
        return ["smiles", "y"]

    examples = (
        full_df.loc[
            full_df.duplicated(subset=["smiles", "y"], keep=False),
            ["smiles", "y"],
        ]
        .drop_duplicates()
        .head()
    )
    raise ValueError(
        f"{dataset_name}: full dataset is not unique by ['smiles'] or ['smiles', 'y'].\n"
        f"Example duplicate keys:\n{examples.to_string(index=False)}"
    )


def map_split(split_path: Path, full_df: pd.DataFrame, match_columns: list[str]) -> list[int]:
    split_df = load_csv(split_path).reset_index(drop=True)
    split_df["_row_order"] = split_df.index

    merged = split_df.merge(
        full_df[match_columns + ["idx"]],
        on=match_columns,
        how="left",
        sort=False,
        validate="many_to_one",
    ).sort_values("_row_order")

    if merged["idx"].isna().any():
        missing_rows = merged.loc[merged["idx"].isna(), REQUIRED_COLUMNS].head()
        raise ValueError(
            f"Could not match rows from {split_path} using {match_columns}.\n"
            f"Example unmatched rows:\n{missing_rows.to_string(index=False)}"
        )

    indices = merged["idx"].astype(int).tolist()
    if len(indices) != len(set(indices)):
        duplicate_rows = merged.loc[
            merged["idx"].duplicated(keep=False),
            ["idx", "smiles", "y"],
        ].head()
        raise ValueError(
            f"Duplicate mapping detected in {split_path}.\n"
            f"Example duplicate rows:\n{duplicate_rows.to_string(index=False)}"
        )

    return indices


def validate_disjoint(split_indices: dict[str, list[int]], run_label: str) -> None:
    train_idx = set(split_indices["train"])
    val_idx = set(split_indices["val"])
    test_idx = set(split_indices["test"])

    overlaps = []
    if train_idx & val_idx:
        overlaps.append(("train", "val", sorted(train_idx & val_idx)[:5]))
    if train_idx & test_idx:
        overlaps.append(("train", "test", sorted(train_idx & test_idx)[:5]))
    if val_idx & test_idx:
        overlaps.append(("val", "test", sorted(val_idx & test_idx)[:5]))

    if overlaps:
        parts = [
            f"{left}/{right} overlap example indices: {example}"
            for left, right, example in overlaps
        ]
        raise ValueError(f"{run_label}: train/val/test indices are not disjoint.\n" + "\n".join(parts))


def save_split_json(path: Path, split_indices: dict[str, list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_idx": split_indices["train"],
        "val_idx": split_indices["val"],
        "test_idx": split_indices["test"],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def process_dataset(dataset_dir: Path, data_dir: Path, output_dir: Path) -> list[str]:
    dataset_name = dataset_dir.name
    full_path = data_dir / "labelled_cleaned" / f"{dataset_name}.csv"
    if not full_path.exists():
        raise FileNotFoundError(f"Missing full dataset CSV for {dataset_name}: {full_path}")

    full_df = load_csv(full_path).reset_index(drop=True)
    full_df["idx"] = full_df.index
    match_columns = choose_match_columns(full_df, dataset_name)

    dataset_output_dir = output_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    full_df[["idx", "smiles", "y"]].to_csv(dataset_output_dir / "index_mapping.csv", index=False)

    summary = []
    print(f"{dataset_name}: matching on {match_columns}")

    method_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    if not method_dirs:
        raise ValueError(f"No split methods found for {dataset_name}: {dataset_dir}")

    for method_dir in method_dirs:
        run_dirs = sorted(path for path in method_dir.iterdir() if path.is_dir())
        if not run_dirs:
            raise ValueError(f"No split runs found in {method_dir}")

        for run_dir in run_dirs:
            split_indices = {}
            for split_name in SPLIT_NAMES:
                split_path = run_dir / f"{split_name}.csv"
                if not split_path.exists():
                    raise FileNotFoundError(f"Missing split file: {split_path}")
                split_indices[split_name] = map_split(split_path, full_df, match_columns)

            run_label = f"{dataset_name} / {method_dir.name} / {run_dir.name}"
            validate_disjoint(split_indices, run_label)

            output_path = dataset_output_dir / method_dir.name / f"{run_dir.name}.json"
            save_split_json(output_path, split_indices)

            total = sum(len(split_indices[name]) for name in SPLIT_NAMES)
            summary.append(
                f"{dataset_name} / {method_dir.name} / {run_dir.name} -> "
                f"{total} samples "
                f"(train={len(split_indices['train'])}, "
                f"val={len(split_indices['val'])}, "
                f"test={len(split_indices['test'])})"
            )

    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to the data/ directory.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    split_root = data_dir / "split_data"
    output_dir = data_dir / "split_data_index"

    if not split_root.exists():
        raise FileNotFoundError(f"Missing split_data directory: {split_root}")

    dataset_dirs = sorted(path for path in split_root.iterdir() if path.is_dir())
    if not dataset_dirs:
        raise ValueError(f"No dataset directories found in {split_root}")

    summary = []
    for dataset_dir in dataset_dirs:
        summary.extend(process_dataset(dataset_dir, data_dir, output_dir))

    print("\nSummary")
    for line in summary:
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

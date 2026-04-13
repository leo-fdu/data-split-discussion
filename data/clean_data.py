from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import Chem, RDLogger

try:
    from rdkit.Chem.MolStandardize import rdMolStandardize
except ImportError:
    try:
        from rdkit.Chem import MolStandardize

        rdMolStandardize = MolStandardize.rdMolStandardize
    except ImportError:
        rdMolStandardize = None


PROJECT_DATA_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DATA_DIR / "labelled"
OUTPUT_DIR = PROJECT_DATA_DIR / "labelled_cleaned"
SMILES_COLUMN = "smiles"
MISSING_LABEL_SENTINEL = object()

DATASET_TYPE_MAP = {
    "bbbp.csv": "classification",
    "tox21.csv": "classification",
    "hiv.csv": "classification",
    "freesolv.csv": "regression",
    "lipophilicity.csv": "regression",
    "solv.csv": "regression",
    "qm9_smiles.csv": "smiles_only",
}


@dataclass
class DatasetSummary:
    name: str
    original_rows: int = 0
    missing_blank_smiles_removed: int = 0
    invalid_smiles_removed: int = 0
    duplicate_groups_found: int = 0
    classification_conflict_rows_removed: int = 0
    final_rows: int = 0


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Parse, sanitize, fragment-standardize, and canonicalize a SMILES string.

    Returns None if parsing or sanitization fails.
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None

        Chem.SanitizeMol(mol)
        parent_mol = get_parent_fragment_mol(mol)
        if parent_mol is None:
            return None

        Chem.SanitizeMol(parent_mol)
        return Chem.MolToSmiles(parent_mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def get_parent_fragment_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Collapse multi-fragment inputs to a parent/main fragment before
    canonicalization, without adding tautomer standardization.
    """
    try:
        if rdMolStandardize is not None:
            if hasattr(rdMolStandardize, "FragmentParent"):
                try:
                    # Prefer fragment-parent handling without extra cleanup
                    # so we focus on salt/fragment removal only.
                    parent_mol = rdMolStandardize.FragmentParent(
                        mol,
                        skipStandardize=True,
                    )
                except TypeError:
                    parent_mol = rdMolStandardize.FragmentParent(mol)

                if parent_mol is not None:
                    return parent_mol

            if hasattr(rdMolStandardize, "LargestFragmentChooser"):
                chooser = rdMolStandardize.LargestFragmentChooser()
                parent_mol = chooser.choose(mol)
                if parent_mol is not None:
                    return parent_mol

        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if not fragments:
            return None

        return max(
            fragments,
            key=lambda frag: (frag.GetNumHeavyAtoms(), frag.GetNumAtoms()),
        )
    except Exception:
        return None


def determine_dataset_type(filename: str) -> str:
    """Determine dataset type from a filename-based mapping."""
    dataset_type = DATASET_TYPE_MAP.get(filename)
    if dataset_type is None:
        raise ValueError(
            f"Unknown dataset type for '{filename}'. "
            "Please add it to DATASET_TYPE_MAP."
        )
    return dataset_type


def normalize_smiles_value(value: object) -> Optional[str]:
    """Return a stripped SMILES string or None for missing/blank entries."""
    if pd.isna(value):
        return None

    smiles = str(value).strip()
    if not smiles:
        return None

    if smiles.lower() in {"nan", "none", "null"}:
        return None

    return smiles


def normalize_label_value(value: object) -> object:
    """Normalize label values so missing values compare consistently."""
    if pd.isna(value):
        return MISSING_LABEL_SENTINEL
    return value


def resolve_smiles_only_duplicates(
    group: pd.DataFrame,
    label_columns: list[str],
) -> tuple[Optional[pd.Series], int]:
    """Deduplicate smiles-only data by keeping the first row."""
    del label_columns
    return group.iloc[0].copy(), 0


def resolve_regression_duplicates(
    group: pd.DataFrame,
    label_columns: list[str],
) -> tuple[Optional[pd.Series], int]:
    """Deduplicate regression data by averaging target values."""
    merged_row = group.iloc[0].copy()

    for column in label_columns:
        numeric_values = pd.to_numeric(group[column], errors="coerce")
        merged_row[column] = numeric_values.mean()

    return merged_row, 0


def resolve_classification_duplicates(
    group: pd.DataFrame,
    label_columns: list[str],
) -> tuple[Optional[pd.Series], int]:
    """
    Deduplicate single-task or standard classification data.

    Keep one row if duplicate labels are identical; otherwise remove all rows
    in the conflicting duplicate group.
    """
    unique_rows = {
        tuple(normalize_label_value(row[column]) for column in label_columns)
        for _, row in group.iterrows()
    }

    if len(unique_rows) > 1:
        return None, len(group)

    return group.iloc[0].copy(), 0


def resolve_tox21_duplicates(
    group: pd.DataFrame,
    label_columns: list[str],
) -> tuple[Optional[pd.Series], int]:
    """
    Deduplicate Tox21 with overlap-aware conflict checking.

    Duplicate rows are consistent if all overlapping non-missing labels match.
    If consistent, merge by taking the first non-missing value per label column.
    """
    merged_row = group.iloc[0].copy()

    for column in label_columns:
        non_missing_values = group[column].dropna()
        unique_values = pd.unique(non_missing_values)

        if len(unique_values) > 1:
            return None, len(group)

        if len(non_missing_values) > 0:
            merged_row[column] = non_missing_values.iloc[0]
        else:
            merged_row[column] = pd.NA

    return merged_row, 0


def resolve_duplicate_groups(
    df: pd.DataFrame,
    dataset_type: str,
    filename: str,
) -> tuple[pd.DataFrame, int, int]:
    """
    Resolve duplicate canonical SMILES groups for a dataset.

    Returns:
        cleaned_df,
        duplicate_group_count,
        classification_conflict_rows_removed
    """
    label_columns = [column for column in df.columns if column != SMILES_COLUMN]
    grouped = df.groupby(SMILES_COLUMN, sort=False, dropna=False)
    duplicate_group_count = sum(len(group) > 1 for _, group in grouped)

    kept_rows: list[pd.Series] = []
    conflict_rows_removed = 0

    for _, group in df.groupby(SMILES_COLUMN, sort=False, dropna=False):
        if len(group) == 1:
            kept_rows.append(group.iloc[0].copy())
            continue

        if dataset_type == "smiles_only":
            resolved_row, removed_count = resolve_smiles_only_duplicates(
                group, label_columns
            )
        elif dataset_type == "regression":
            resolved_row, removed_count = resolve_regression_duplicates(
                group, label_columns
            )
        elif filename == "tox21.csv":
            resolved_row, removed_count = resolve_tox21_duplicates(
                group, label_columns
            )
        else:
            resolved_row, removed_count = resolve_classification_duplicates(
                group, label_columns
            )

        conflict_rows_removed += removed_count

        if resolved_row is not None:
            kept_rows.append(resolved_row)

    if not kept_rows:
        cleaned_df = df.iloc[0:0].copy()
    else:
        cleaned_df = pd.DataFrame(kept_rows, columns=df.columns)

    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df, duplicate_group_count, conflict_rows_removed


def clean_dataset(input_path: Path, output_dir: Path) -> DatasetSummary:
    """Clean one dataset CSV and save the result."""
    filename = input_path.name
    dataset_type = determine_dataset_type(filename)
    summary = DatasetSummary(name=filename)

    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV: {exc}") from exc

    if SMILES_COLUMN not in df.columns:
        raise ValueError(f"Missing required '{SMILES_COLUMN}' column.")

    summary.original_rows = len(df)

    valid_indices: list[int] = []
    canonical_smiles: list[str] = []

    for index, raw_smiles in df[SMILES_COLUMN].items():
        normalized_smiles = normalize_smiles_value(raw_smiles)
        if normalized_smiles is None:
            summary.missing_blank_smiles_removed += 1
            continue

        canonical_smiles_value = canonicalize_smiles(normalized_smiles)
        if canonical_smiles_value is None:
            summary.invalid_smiles_removed += 1
            continue

        valid_indices.append(index)
        canonical_smiles.append(canonical_smiles_value)

    cleaned_df = df.loc[valid_indices].copy().reset_index(drop=True)
    cleaned_df[SMILES_COLUMN] = canonical_smiles

    (
        cleaned_df,
        summary.duplicate_groups_found,
        summary.classification_conflict_rows_removed,
    ) = resolve_duplicate_groups(
        cleaned_df,
        dataset_type=dataset_type,
        filename=filename,
    )

    summary.final_rows = len(cleaned_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    try:
        cleaned_df.to_csv(output_path, index=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to write cleaned CSV: {exc}") from exc

    return summary


def print_dataset_summary(summary: DatasetSummary) -> None:
    """Print a clear per-dataset cleaning summary."""
    print(f"[DATASET] {summary.name}")
    print(f"  original row count: {summary.original_rows}")
    print(
        "  rows removed due to missing/blank smiles: "
        f"{summary.missing_blank_smiles_removed}"
    )
    print(
        "  rows removed due to invalid RDKit parsing/sanitization: "
        f"{summary.invalid_smiles_removed}"
    )
    print(f"  duplicate groups found: {summary.duplicate_groups_found}")
    print(
        "  rows removed because of classification label conflicts: "
        f"{summary.classification_conflict_rows_removed}"
    )
    print(f"  final row count: {summary.final_rows}")
    print()


def print_global_summary(summaries: list[DatasetSummary]) -> None:
    """Print an aggregate summary across all processed datasets."""
    total = DatasetSummary(name="GLOBAL")

    for summary in summaries:
        total.original_rows += summary.original_rows
        total.missing_blank_smiles_removed += summary.missing_blank_smiles_removed
        total.invalid_smiles_removed += summary.invalid_smiles_removed
        total.duplicate_groups_found += summary.duplicate_groups_found
        total.classification_conflict_rows_removed += (
            summary.classification_conflict_rows_removed
        )
        total.final_rows += summary.final_rows

    print("[GLOBAL SUMMARY]")
    print(f"  datasets processed: {len(summaries)}")
    print(f"  original row count: {total.original_rows}")
    print(
        "  rows removed due to missing/blank smiles: "
        f"{total.missing_blank_smiles_removed}"
    )
    print(
        "  rows removed due to invalid RDKit parsing/sanitization: "
        f"{total.invalid_smiles_removed}"
    )
    print(f"  duplicate groups found: {total.duplicate_groups_found}")
    print(
        "  rows removed because of classification label conflicts: "
        f"{total.classification_conflict_rows_removed}"
    )
    print(f"  final row count: {total.final_rows}")


def main() -> int:
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")

    if not INPUT_DIR.exists():
        print(f"[ERROR] Input directory does not exist: {INPUT_DIR}")
        return 1

    csv_files = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        print(f"[WARNING] No CSV files found in: {INPUT_DIR}")
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Input dir : {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Found {len(csv_files)} CSV file(s).\n")

    summaries: list[DatasetSummary] = []
    had_errors = False

    for csv_file in csv_files:
        try:
            summary = clean_dataset(csv_file, OUTPUT_DIR)
            print_dataset_summary(summary)
            summaries.append(summary)
        except Exception as exc:
            had_errors = True
            print(f"[ERROR] {csv_file.name}: {exc}\n")

    if summaries:
        print_global_summary(summaries)

    return 1 if had_errors else 0


if __name__ == "__main__":
    sys.exit(main())

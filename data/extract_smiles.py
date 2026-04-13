# data/extract_smiles.py

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_DATA_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DATA_DIR / "labelled"
OUTPUT_DIR = PROJECT_DATA_DIR / "smilesonly"


def extract_smiles_from_csv(
    input_path: Path,
    output_dir: Path,
    smiles_col: str = "smiles",
    drop_duplicates: bool = True,
) -> None:
    """
    Read a labelled CSV, extract the smiles column, and save it as a
    single-column CSV named <stem>_smiles.csv.

    Parameters
    ----------
    input_path : Path
        Path to the input CSV file.
    output_dir : Path
        Directory where the output CSV will be saved.
    smiles_col : str
        Name of the SMILES column.
    drop_duplicates : bool
        Whether to drop duplicate SMILES strings.
    """
    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        print(f"[ERROR] Failed to read {input_path.name}: {exc}")
        return

    if smiles_col not in df.columns:
        print(
            f"[WARNING] Skipping {input_path.name}: "
            f"column '{smiles_col}' not found."
        )
        return

    smiles_df = df[[smiles_col]].copy()

    # Normalize missing/blank values
    smiles_df[smiles_col] = smiles_df[smiles_col].astype(str).str.strip()
    smiles_df = smiles_df[smiles_df[smiles_col].notna()]
    smiles_df = smiles_df[smiles_df[smiles_col] != ""]
    smiles_df = smiles_df[smiles_df[smiles_col].str.lower() != "nan"]

    n_before = len(smiles_df)

    if drop_duplicates:
        smiles_df = smiles_df.drop_duplicates(subset=[smiles_col])

    n_after = len(smiles_df)

    # Ensure output column is exactly named "smiles"
    smiles_df = smiles_df.rename(columns={smiles_col: "smiles"})

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_smiles.csv"

    try:
        smiles_df.to_csv(output_path, index=False)
    except Exception as exc:
        print(f"[ERROR] Failed to save {output_path.name}: {exc}")
        return

    removed = n_before - n_after
    print(
        f"[OK] {input_path.name} -> {output_path.name} | "
        f"rows kept: {n_after}"
        + (f" | duplicates removed: {removed}" if drop_duplicates else "")
    )


def main() -> int:
    if not INPUT_DIR.exists():
        print(f"[ERROR] Input directory does not exist: {INPUT_DIR}")
        return 1

    csv_files = sorted(INPUT_DIR.glob("*.csv"))

    if not csv_files:
        print(f"[WARNING] No CSV files found in: {INPUT_DIR}")
        return 0

    print(f"Input dir : {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Found {len(csv_files)} CSV file(s).\n")

    for csv_file in csv_files:
        extract_smiles_from_csv(
            input_path=csv_file,
            output_dir=OUTPUT_DIR,
            smiles_col="smiles",
            drop_duplicates=True,
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
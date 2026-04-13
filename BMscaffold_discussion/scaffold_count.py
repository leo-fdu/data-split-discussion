from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_DIR = PROJECT_ROOT / "data" / "smilesonly"
OUTPUT_DIR = SCRIPT_DIR / "scaffold_counts"
SMILES_COLUMN = "smiles"


@dataclass
class DatasetSummary:
    dataset_name: str
    total_rows: int
    valid_molecules: int
    invalid_smiles: int
    unique_scaffolds: int
    output_path: Path
    top_scaffolds: list[tuple[str, int]]


def normalize_smiles_value(value: object) -> Optional[str]:
    """Return a stripped SMILES string or None for missing or blank entries."""
    if pd.isna(value):
        return None

    smiles = str(value).strip()
    if not smiles:
        return None

    if smiles.lower() in {"nan", "none", "null"}:
        return None

    return smiles


def smiles_to_scaffold_smiles(smiles: str) -> Optional[str]:
    """Convert a SMILES string to its canonical Bemis-Murcko scaffold SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold_mol, canonical=True)
        if scaffold_smiles == "":
            scaffold_smiles = "[NO_SCAFFOLD]"
        return scaffold_smiles
    except Exception:
        return None


def compute_scaffold_counts(smiles_series: pd.Series) -> tuple[Counter[str], int]:
    """Count canonical Bemis-Murcko scaffold SMILES from a SMILES series."""
    scaffold_counts: Counter[str] = Counter()
    invalid_smiles = 0

    for value in smiles_series:
        smiles = normalize_smiles_value(value)
        if smiles is None:
            invalid_smiles += 1
            continue

        scaffold_smiles = smiles_to_scaffold_smiles(smiles)
        if scaffold_smiles is None:
            invalid_smiles += 1
            continue

        scaffold_counts[scaffold_smiles] += 1

    return scaffold_counts, invalid_smiles


def dataset_name_from_path(input_path: Path) -> str:
    """Derive the dataset name from an input filename."""
    stem = input_path.stem
    if stem.endswith("_smiles"):
        return stem[: -len("_smiles")]
    return stem


def scaffold_counts_to_dataframe(
    scaffold_counts: Counter[str],
) -> pd.DataFrame:
    """Build a sorted scaffold-count dataframe."""
    rows = sorted(scaffold_counts.items(), key=lambda item: (-item[1], item[0]))
    return pd.DataFrame(rows, columns=["scaffold_smiles", "count"])


def format_scaffold_for_log(scaffold_smiles: str) -> str:
    """Make scaffold console output easier to read."""
    return scaffold_smiles if scaffold_smiles else "<empty scaffold>"


def process_file(input_path: Path, output_dir: Path) -> Optional[DatasetSummary]:
    """Process one CSV file and write scaffold counts to disk."""
    dataset_name = dataset_name_from_path(input_path)
    print(f"\n[START] Processing {input_path.name}")

    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        print(f"[ERROR] Failed to read {input_path}: {exc}")
        return None

    if SMILES_COLUMN not in df.columns:
        print(
            f"[ERROR] Skipping {input_path.name}: "
            f"missing required column '{SMILES_COLUMN}'."
        )
        return None

    total_rows = len(df)
    scaffold_counts, invalid_smiles = compute_scaffold_counts(df[SMILES_COLUMN])
    valid_molecules = total_rows - invalid_smiles

    output_df = scaffold_counts_to_dataframe(scaffold_counts)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_name}_scaffold_counts.csv"

    try:
        output_df.to_csv(output_path, index=False)
    except Exception as exc:
        print(f"[ERROR] Failed to save {output_path}: {exc}")
        return None

    top_scaffolds = output_df.head(5).itertuples(index=False, name=None)
    summary = DatasetSummary(
        dataset_name=dataset_name,
        total_rows=total_rows,
        valid_molecules=valid_molecules,
        invalid_smiles=invalid_smiles,
        unique_scaffolds=len(scaffold_counts),
        output_path=output_path,
        top_scaffolds=list(top_scaffolds),
    )

    print(f"[DONE] Saved {output_path.name}")
    print(f"  total molecules   : {summary.total_rows}")
    print(f"  valid molecules   : {summary.valid_molecules}")
    print(f"  invalid skipped   : {summary.invalid_smiles}")
    print(f"  unique scaffolds  : {summary.unique_scaffolds}")

    if summary.top_scaffolds:
        print("  top 5 scaffolds   :")
        for scaffold_smiles, count in summary.top_scaffolds:
            print(f"    {format_scaffold_for_log(scaffold_smiles)} -> {count}")
    else:
        print("  top 5 scaffolds   : none")

    return summary


def main() -> int:
    """Compute scaffold counts for all smiles-only datasets."""
    RDLogger.DisableLog("rdApp.*")

    if not INPUT_DIR.exists():
        print(f"[ERROR] Input directory does not exist: {INPUT_DIR}")
        return 1

    csv_files = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        print(f"[WARNING] No CSV files found in: {INPUT_DIR}")
        return 0

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input dir   : {INPUT_DIR}")
    print(f"Output dir  : {OUTPUT_DIR}")
    print(f"Found {len(csv_files)} dataset(s).")

    summaries: list[DatasetSummary] = []
    failed_files = 0

    for csv_file in csv_files:
        summary = process_file(csv_file, OUTPUT_DIR)
        if summary is None:
            failed_files += 1
            continue
        summaries.append(summary)

    print("\n[SUMMARY]")
    print(f"  processed datasets: {len(summaries)}")
    print(f"  failed datasets   : {failed_files}")
    print(
        "  total molecules   : "
        f"{sum(summary.total_rows for summary in summaries)}"
    )
    print(
        "  total scaffolds   : "
        f"{sum(summary.unique_scaffolds for summary in summaries)}"
    )

    return 0 if failed_files == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

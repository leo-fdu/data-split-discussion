from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from .types import MoleculeTable


def smiles_to_mol(smiles: str) -> Chem.Mol:
    """Parse one SMILES string into an RDKit molecule or raise a clear error."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles!r}")
    return mol


def canonicalize_smiles(smiles: str) -> str:
    """Return the canonical RDKit SMILES string for one input SMILES."""
    mol = smiles_to_mol(smiles)
    return Chem.MolToSmiles(mol, canonical=True)


def load_molecule_table(
    source: str | Path | pd.DataFrame,
    smiles_column: str = "smiles",
    invalid_smiles: str = "raise",
    sort_by_canonical_smiles: bool = True,
) -> MoleculeTable:
    """
    Load molecules from a CSV path or DataFrame.

    The returned DataFrame is copied, gets a `canonical_smiles` column, and is
    optionally sorted by `(canonical_smiles, source_row_index)` using a stable sort.
    Invalid SMILES rows can either raise immediately or be dropped.
    """
    if invalid_smiles not in {"raise", "drop"}:
        raise ValueError("invalid_smiles must be either 'raise' or 'drop'.")

    if isinstance(source, pd.DataFrame):
        df = source.copy()
        source_label = "<dataframe>"
    else:
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Input CSV does not exist: {source_path}")
        df = pd.read_csv(source_path)
        source_label = str(source_path)

    if smiles_column not in df.columns:
        raise ValueError(f"Input data is missing required column {smiles_column!r}.")

    records: list[dict[str, Any]] = []
    mols: list[Chem.Mol] = []
    canonical_smiles_list: list[str] = []
    dropped_invalid_indices: list[int] = []

    for row_position, raw_value in enumerate(df[smiles_column].tolist()):
        smiles = str(raw_value).strip()
        if not smiles or smiles.lower() in {"nan", "none", "null"}:
            if invalid_smiles == "raise":
                raise ValueError(f"Row {row_position} has an empty or missing SMILES value.")
            dropped_invalid_indices.append(row_position)
            continue

        try:
            mol = smiles_to_mol(smiles)
        except ValueError as exc:
            if invalid_smiles == "raise":
                raise ValueError(f"Row {row_position} has an invalid SMILES: {smiles!r}") from exc
            dropped_invalid_indices.append(row_position)
            continue

        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        record = df.iloc[row_position].to_dict()
        record["source_row_index"] = row_position
        record["canonical_smiles"] = canonical_smiles
        records.append(record)
        mols.append(mol)
        canonical_smiles_list.append(canonical_smiles)

    if not records:
        raise ValueError("No valid molecules were retained after SMILES parsing.")

    cleaned_df = pd.DataFrame(records)
    if sort_by_canonical_smiles:
        cleaned_df = cleaned_df.sort_values(
            by=["canonical_smiles", "source_row_index"],
            kind="mergesort",
        ).reset_index(drop=True)
        order = cleaned_df["source_row_index"].tolist()
        row_to_position = {
            original_idx: position
            for position, original_idx in enumerate(
                record["source_row_index"] for record in records
            )
        }
        mols = [mols[row_to_position[int(original_idx)]] for original_idx in order]
        canonical_smiles_list = cleaned_df["canonical_smiles"].tolist()

    return MoleculeTable(
        dataframe=cleaned_df,
        smiles_column=smiles_column,
        canonical_smiles=tuple(canonical_smiles_list),
        mols=tuple(mols),
        source=source_label,
        dropped_invalid_indices=tuple(sorted(dropped_invalid_indices)),
    )


def save_distance_matrix(matrix: np.ndarray, path: str | Path) -> Path:
    """Save one distance matrix as a `.npy` file and return the resolved path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.asarray(matrix))
    return output_path


def load_distance_matrix(path: str | Path, dtype: np.dtype | None = None) -> np.ndarray:
    """Load one `.npy` distance matrix, optionally casting it to a target dtype."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Distance matrix cache does not exist: {input_path}")
    matrix = np.load(input_path)
    if dtype is not None:
        matrix = matrix.astype(dtype, copy=False)
    return matrix


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    """Save one JSON payload with stable formatting and return the output path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    return output_path


def load_json(path: str | Path) -> dict[str, Any]:
    """Load one JSON file into a Python dictionary."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {input_path}")
    with input_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_distance_cache(
    matrix: np.ndarray,
    matrix_path: str | Path,
    metadata: dict[str, Any] | None = None,
    metadata_path: str | Path | None = None,
) -> None:
    """Save a distance matrix and optional metadata sidecar."""
    save_distance_matrix(matrix, matrix_path)
    if metadata is not None and metadata_path is not None:
        save_json(metadata, metadata_path)


def load_distance_cache(
    matrix_path: str | Path,
    metadata_path: str | Path | None = None,
    dtype: np.dtype | None = None,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Load a cached distance matrix and an optional metadata sidecar."""
    matrix = load_distance_matrix(matrix_path, dtype=dtype)
    metadata = None
    if metadata_path is not None and Path(metadata_path).exists():
        metadata = load_json(metadata_path)
    return matrix, metadata

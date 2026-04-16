from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from .clustering import run_butina_clustering
from .distance import (
    combine_distances,
    compute_pairwise_fg_distance_matrix,
    compute_pairwise_scaffold_distance_matrix,
)
from .fg_features import build_fg_count_matrix, build_leaf_functional_group_vocabulary
from .io_utils import load_molecule_table
from .scaffold import extract_expanded_scaffold
from .split import clusters_to_splits
from .sweep import sweep_tunable_splits


DEMO_ROWS = [
    {"smiles": "CCO"},
    {"smiles": "CC(=O)O"},
    {"smiles": "c1ccccc1O"},
    {"smiles": "c1ccncc1"},
    {"smiles": "CCN(CC)CC"},
    {"smiles": "O=C(Nc1ccccc1)C"},
]


def main(argv: list[str] | None = None) -> None:
    """Run a minimal end-to-end demo of the tunable split framework."""
    parser = argparse.ArgumentParser(description="Run a minimal tunable split demo.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to a CSV file containing at least a 'smiles' column.",
    )
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        csv_path = args.csv or _write_demo_csv(tmp_dir / "demo_molecules.csv")

        table = load_molecule_table(csv_path, smiles_column="smiles", invalid_smiles="raise")
        scaffolds = [extract_expanded_scaffold(mol) for mol in table.mols]
        vocabulary = build_leaf_functional_group_vocabulary()
        fg_count_matrix, vocabulary = build_fg_count_matrix(table.mols, vocabulary=vocabulary)
        scaffold_distance_result = compute_pairwise_scaffold_distance_matrix(scaffolds)
        fg_distance_matrix = compute_pairwise_fg_distance_matrix(fg_count_matrix)
        total_distance_matrix = combine_distances(
            scaffold_distance_result.matrix,
            fg_distance_matrix,
            lambda_=0.5,
        )
        clusters = run_butina_clustering(total_distance_matrix, cutoff=0.45)
        split_result = clusters_to_splits(clusters)
        sweep_results = sweep_tunable_splits(
            d_scaffold=scaffold_distance_result.matrix,
            d_fg=fg_distance_matrix,
            lambda_min=0.25,
            lambda_max=0.75,
            lambda_gap=0.25,
            cutoff_min=0.35,
            cutoff_max=0.55,
            cutoff_gap=0.10,
        )

        print(f"Loaded {table.size} molecules from: {csv_path}")
        print("Expanded scaffold SMILES:")
        for scaffold in scaffolds:
            print(f"  - {scaffold.scaffold_smiles or '<empty>'}")
        print(f"Leaf FG vocabulary size: {vocabulary.size}")
        print(f"Number of clusters at cutoff=0.45: {len(clusters)}")
        print(f"Split summary: {split_result.summary}")
        print(f"Generated {len(sweep_results)} sweep configurations.")
        if sweep_results:
            print(f"First sweep summary: {sweep_results[0].summary}")


def _write_demo_csv(path: Path) -> Path:
    demo_df = pd.DataFrame(DEMO_ROWS)
    demo_df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    main()

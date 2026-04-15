from __future__ import annotations

import heapq
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "labelled_cleaned"
OUTPUT_DIR = SCRIPT_DIR / "split_data"
SMILES_COLUMN = "smiles"
SPLIT_NAMES = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEEDED_SPLITS = (0, 1, 2, 3, 4)
EXCLUDED_DATASETS = {"tox21.csv", "qm9_smiles.csv"}
NO_SCAFFOLD_LABEL = "[NO_SCAFFOLD]"
BUTINA_DISTANCE_CUTOFF = 0.4

MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
    includeChirality=True,
)
ATOMPAIR_GENERATOR = rdFingerprintGenerator.GetAtomPairGenerator(
    fpSize=2048,
    includeChirality=True,
)


@dataclass(frozen=True)
class LoadedDataset:
    dataset_name: str
    source_path: Path
    df: pd.DataFrame
    mols: list[Chem.Mol]
    dropped_missing_smiles: int
    dropped_invalid_smiles: int

    @property
    def total_rows(self) -> int:
        return len(self.df)


@dataclass(frozen=True)
class FingerprintSpec:
    split_method: str
    fingerprint_type: str
    builder: Callable[[Chem.Mol], DataStructs.cDataStructs.ExplicitBitVect]
    distance_cutoff: float = BUTINA_DISTANCE_CUTOFF


FINGERPRINT_SPECS = {
    "butina_morgan": FingerprintSpec(
        split_method="butina_morgan",
        fingerprint_type="morgan",
        builder=lambda mol: MORGAN_GENERATOR.GetFingerprint(mol),
    ),
    "butina_maccs": FingerprintSpec(
        split_method="butina_maccs",
        fingerprint_type="maccs",
        builder=MACCSkeys.GenMACCSKeys,
    ),
    "butina_atompair": FingerprintSpec(
        split_method="butina_atompair",
        fingerprint_type="atompair",
        builder=lambda mol: ATOMPAIR_GENERATOR.GetFingerprint(mol),
    ),
}


def normalize_smiles_value(value: object) -> Optional[str]:
    """Return a stripped SMILES string or None for missing/blank values."""
    if pd.isna(value):
        return None

    smiles = str(value).strip()
    if not smiles:
        return None

    if smiles.lower() in {"nan", "none", "null"}:
        return None

    return smiles


def compute_target_counts(
    total_count: int,
    split_ratios: dict[str, float] = SPLIT_RATIOS,
) -> dict[str, int]:
    """Round target split counts with a largest-remainder rule."""
    raw_counts = {
        split_name: total_count * split_ratios[split_name]
        for split_name in SPLIT_NAMES
    }
    rounded_counts = {
        split_name: math.floor(raw_counts[split_name])
        for split_name in SPLIT_NAMES
    }
    remainder = total_count - sum(rounded_counts.values())

    remainder_order = sorted(
        SPLIT_NAMES,
        key=lambda split_name: (
            raw_counts[split_name] - rounded_counts[split_name],
            -SPLIT_NAMES.index(split_name),
        ),
        reverse=True,
    )

    for offset in range(remainder):
        rounded_counts[remainder_order[offset]] += 1

    return rounded_counts


def counts_to_fractions(split_counts: dict[str, int], total_count: int) -> dict[str, float]:
    """Convert train/val/test counts to fractions."""
    if total_count == 0:
        return {split_name: 0.0 for split_name in SPLIT_NAMES}

    return {
        split_name: split_counts[split_name] / total_count
        for split_name in SPLIT_NAMES
    }


def load_dataset(csv_path: Path) -> LoadedDataset:
    """
    Read one cleaned dataset and keep only rows with a valid RDKit-parsable SMILES.

    The original non-SMILES columns are preserved unchanged.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read {csv_path}: {exc}") from exc

    if SMILES_COLUMN not in df.columns:
        raise ValueError(
            f"Dataset '{csv_path.name}' is missing required column '{SMILES_COLUMN}'."
        )

    valid_row_indices: list[int] = []
    normalized_smiles: list[str] = []
    valid_mols: list[Chem.Mol] = []
    dropped_missing_smiles = 0
    dropped_invalid_smiles = 0

    for row_idx, row in df.iterrows():
        smiles = normalize_smiles_value(row[SMILES_COLUMN])
        if smiles is None:
            dropped_missing_smiles += 1
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            dropped_invalid_smiles += 1
            continue

        valid_row_indices.append(row_idx)
        normalized_smiles.append(smiles)
        valid_mols.append(mol)

    if not valid_row_indices:
        raise ValueError(
            f"Dataset '{csv_path.name}' has no usable rows after SMILES validation."
        )

    cleaned_df = df.loc[valid_row_indices].copy().reset_index(drop=True)
    cleaned_df[SMILES_COLUMN] = normalized_smiles
    return LoadedDataset(
        dataset_name=csv_path.stem,
        source_path=csv_path,
        df=cleaned_df,
        mols=valid_mols,
        dropped_missing_smiles=dropped_missing_smiles,
        dropped_invalid_smiles=dropped_invalid_smiles,
    )


def random_split(dataset: LoadedDataset, seed: int) -> dict[str, list[int]]:
    """Create a reproducible row-level random split."""
    indices = list(range(dataset.total_rows))
    rng = random.Random(seed)
    rng.shuffle(indices)

    target_counts = compute_target_counts(dataset.total_rows)
    train_end = target_counts["train"]
    val_end = train_end + target_counts["val"]

    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }


def scaffold_from_mol(mol: Chem.Mol) -> str:
    """Return the canonical Bemis-Murcko scaffold string for a molecule."""
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold_mol, canonical=True)
    return scaffold_smiles if scaffold_smiles else NO_SCAFFOLD_LABEL


def order_group_entries(
    group_entries: Sequence[tuple[str, list[int]]],
    seed: Optional[int] = None,
) -> list[tuple[str, list[int]]]:
    """
    Order groups by descending size.

    Deterministic grouped methods use a lexical tiebreaker; seeded grouped
    methods shuffle ties reproducibly before the stable size sort.
    """
    ordered_entries = list(group_entries)
    if seed is None:
        ordered_entries.sort(key=lambda entry: (-len(entry[1]), entry[0]))
        return ordered_entries

    rng = random.Random(seed)
    rng.shuffle(ordered_entries)
    ordered_entries.sort(key=lambda entry: len(entry[1]), reverse=True)
    return ordered_entries


def choose_split_for_group(
    current_counts: dict[str, int],
    target_counts: dict[str, int],
) -> str:
    """Select the split with the largest remaining relative quota."""
    return max(
        SPLIT_NAMES,
        key=lambda split_name: (
            target_counts[split_name] - current_counts[split_name] > 0,
            (
                (target_counts[split_name] - current_counts[split_name])
                / max(target_counts[split_name], 1)
            ),
            target_counts[split_name] - current_counts[split_name],
            -SPLIT_NAMES.index(split_name),
        ),
    )


def assign_grouped_entries(
    group_entries: Sequence[tuple[str, list[int]]],
    total_count: int,
    seed: Optional[int] = None,
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Greedily assign whole groups/clusters to train/val/test."""
    split_indices = {split_name: [] for split_name in SPLIT_NAMES}
    split_group_counts = {split_name: 0 for split_name in SPLIT_NAMES}
    current_counts = {split_name: 0 for split_name in SPLIT_NAMES}
    target_counts = compute_target_counts(total_count)

    for group_id, member_indices in order_group_entries(group_entries, seed=seed):
        del group_id
        target_split = choose_split_for_group(current_counts, target_counts)
        split_indices[target_split].extend(member_indices)
        split_group_counts[target_split] += 1
        current_counts[target_split] += len(member_indices)

    return split_indices, split_group_counts


def bm_scaffold_split(
    dataset: LoadedDataset,
) -> tuple[dict[str, list[int]], dict[str, object]]:
    """Create one deterministic Bemis-Murcko scaffold split."""
    scaffold_groups: defaultdict[str, list[int]] = defaultdict(list)
    no_scaffold_count = 0

    for idx, mol in enumerate(dataset.mols):
        scaffold = scaffold_from_mol(mol)
        if scaffold == NO_SCAFFOLD_LABEL:
            no_scaffold_count += 1
        scaffold_groups[scaffold].append(idx)

    split_indices, split_group_counts = assign_grouped_entries(
        group_entries=sorted(scaffold_groups.items()),
        total_count=dataset.total_rows,
        seed=None,
    )

    metadata = {
        "group_type": "scaffold",
        "num_groups": len(scaffold_groups),
        "num_scaffold_groups": len(scaffold_groups),
        "group_counts": split_group_counts,
        "num_no_scaffold_molecules": no_scaffold_count,
    }
    return split_indices, metadata


def fingerprint_factory(split_method: str) -> FingerprintSpec:
    """Return fingerprint settings for a Butina split method."""
    try:
        return FINGERPRINT_SPECS[split_method]
    except KeyError as exc:
        raise ValueError(f"Unsupported fingerprint split method: {split_method}") from exc


def build_fingerprints(
    mols: Sequence[Chem.Mol],
    spec: FingerprintSpec,
) -> list[DataStructs.cDataStructs.ExplicitBitVect]:
    """Compute fingerprints for one dataset and fingerprint type."""
    return [spec.builder(mol) for mol in mols]


def build_neighbor_lists(
    fingerprints: Sequence[DataStructs.cDataStructs.ExplicitBitVect],
    distance_cutoff: float,
) -> list[list[int]]:
    """
    Build thresholded Butina neighbor lists from pairwise Tanimoto distances.

    Distances are evaluated row-by-row to avoid materializing a dense NxN matrix,
    which becomes impractical for larger datasets such as HIV.
    """
    similarity_cutoff = 1.0 - distance_cutoff
    total = len(fingerprints)
    neighbor_lists = [[idx] for idx in range(total)]
    previous_fingerprints: list[DataStructs.cDataStructs.ExplicitBitVect] = []
    progress_step = max(1, total // 10)

    for idx, fingerprint in enumerate(fingerprints):
        if previous_fingerprints:
            similarities = DataStructs.BulkTanimotoSimilarity(
                fingerprint,
                previous_fingerprints,
            )
            for other_idx, similarity in enumerate(similarities):
                if similarity >= similarity_cutoff:
                    neighbor_lists[idx].append(other_idx)
                    neighbor_lists[other_idx].append(idx)

        previous_fingerprints.append(fingerprint)

        if total >= 1000 and (idx + 1) % progress_step == 0:
            print(f"      pairwise rows processed: {idx + 1}/{total}")

    return neighbor_lists


def cluster_neighbor_lists(neighbor_lists: Sequence[Sequence[int]]) -> list[tuple[int, ...]]:
    """Cluster thresholded neighbor lists with Butina-style reordering."""
    total = len(neighbor_lists)
    if total == 0:
        return []

    unassigned = [True] * total
    heap: list[tuple[int, int]] = [
        (-len(neighbors), idx) for idx, neighbors in enumerate(neighbor_lists)
    ]
    heapq.heapify(heap)
    clusters: list[tuple[int, ...]] = []

    while heap:
        neg_count, idx = heapq.heappop(heap)
        if not unassigned[idx]:
            continue

        current_members = [member for member in neighbor_lists[idx] if unassigned[member]]
        current_count = len(current_members)

        if current_count != -neg_count:
            heapq.heappush(heap, (-current_count, idx))
            continue

        if current_count <= 1:
            break

        for member in current_members:
            unassigned[member] = False
        clusters.append(tuple(current_members))

    for idx, is_unassigned in enumerate(unassigned):
        if is_unassigned:
            clusters.append((idx,))

    return clusters


def compute_butina_clusters(
    fingerprints: Sequence[DataStructs.cDataStructs.ExplicitBitVect],
    distance_cutoff: float,
) -> list[tuple[int, ...]]:
    """Compute Butina clusters from fingerprints."""
    neighbor_lists = build_neighbor_lists(fingerprints, distance_cutoff=distance_cutoff)
    return cluster_neighbor_lists(neighbor_lists)


def butina_split(
    dataset: LoadedDataset,
    split_method: str,
    seed: int,
    clusters: Optional[Sequence[Sequence[int]]] = None,
) -> tuple[dict[str, list[int]], dict[str, object]]:
    """Create a seeded cluster-level Butina split."""
    spec = fingerprint_factory(split_method)
    cluster_members = clusters
    if cluster_members is None:
        fingerprints = build_fingerprints(dataset.mols, spec)
        cluster_members = compute_butina_clusters(
            fingerprints,
            distance_cutoff=spec.distance_cutoff,
        )

    group_entries = [
        (f"cluster_{cluster_idx}", list(member_indices))
        for cluster_idx, member_indices in enumerate(cluster_members)
    ]
    split_indices, split_group_counts = assign_grouped_entries(
        group_entries=group_entries,
        total_count=dataset.total_rows,
        seed=seed,
    )

    metadata = {
        "group_type": "cluster",
        "num_groups": len(group_entries),
        "num_clusters": len(group_entries),
        "group_counts": split_group_counts,
        "fingerprint_type": spec.fingerprint_type,
        "butina_distance_cutoff": spec.distance_cutoff,
    }
    return split_indices, metadata


def save_split_csvs(
    dataset: LoadedDataset,
    split_method: str,
    run_label: str,
    split_indices: dict[str, list[int]],
    metadata: dict[str, object],
    seed_value: int | str,
) -> Path:
    """Write train/val/test CSVs and the matching split summary JSON."""
    run_dir = OUTPUT_DIR / dataset.dataset_name / split_method / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    split_counts = {split_name: len(split_indices[split_name]) for split_name in SPLIT_NAMES}
    split_fractions = counts_to_fractions(split_counts, dataset.total_rows)

    for split_name in SPLIT_NAMES:
        subset = dataset.df.iloc[sorted(split_indices[split_name])].reset_index(drop=True)
        subset.to_csv(run_dir / f"{split_name}.csv", index=False)

    summary = {
        "dataset_name": dataset.dataset_name,
        "source_csv": str(dataset.source_path),
        "split_method": split_method,
        "seed": seed_value,
        "total_sample_count": dataset.total_rows,
        "dropped_missing_smiles": dataset.dropped_missing_smiles,
        "dropped_invalid_smiles": dataset.dropped_invalid_smiles,
        "counts": split_counts,
        "fractions": split_fractions,
    }
    summary.update(metadata)

    with (run_dir / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return run_dir


def print_saved_run(
    split_method: str,
    run_label: str,
    run_dir: Path,
    split_indices: dict[str, list[int]],
) -> None:
    """Print a concise progress line for a saved split run."""
    print(
        f"    {split_method}/{run_label} -> {run_dir} | "
        f"train={len(split_indices['train'])}, "
        f"val={len(split_indices['val'])}, "
        f"test={len(split_indices['test'])}"
    )


def iter_included_datasets(input_dir: Path) -> list[Path]:
    """List input CSVs while honoring the requested exclusions."""
    return sorted(
        csv_path
        for csv_path in input_dir.glob("*.csv")
        if csv_path.name not in EXCLUDED_DATASETS
    )


def main() -> int:
    """Generate all requested dataset split CSVs."""
    RDLogger.DisableLog("rdApp.*")

    if not INPUT_DIR.exists():
        print(f"[ERROR] Input directory does not exist: {INPUT_DIR}")
        return 1

    dataset_paths = iter_included_datasets(INPUT_DIR)
    if not dataset_paths:
        print(f"[WARNING] No input CSVs found in {INPUT_DIR}")
        return 0

    print(f"Input dir : {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Datasets  : {', '.join(path.name for path in dataset_paths)}")

    failed_datasets = 0

    for dataset_path in dataset_paths:
        print(f"\n[DATASET] {dataset_path.name}")
        try:
            dataset = load_dataset(dataset_path)
        except Exception as exc:
            failed_datasets += 1
            print(f"  [ERROR] {exc}")
            continue

        print(f"  rows used for splitting : {dataset.total_rows}")
        if dataset.dropped_missing_smiles:
            print(f"  dropped blank smiles    : {dataset.dropped_missing_smiles}")
        if dataset.dropped_invalid_smiles:
            print(f"  dropped invalid smiles  : {dataset.dropped_invalid_smiles}")
        try:
            for seed in SEEDED_SPLITS:
                split_indices = random_split(dataset, seed=seed)
                run_dir = save_split_csvs(
                    dataset=dataset,
                    split_method="random",
                    run_label=f"seed_{seed}",
                    split_indices=split_indices,
                    metadata={},
                    seed_value=seed,
                )
                print_saved_run("random", f"seed_{seed}", run_dir, split_indices)

            scaffold_indices, scaffold_metadata = bm_scaffold_split(dataset)
            scaffold_run_dir = save_split_csvs(
                dataset=dataset,
                split_method="bm_scaffold",
                run_label="deterministic",
                split_indices=scaffold_indices,
                metadata=scaffold_metadata,
                seed_value="deterministic",
            )
            print_saved_run(
                "bm_scaffold",
                "deterministic",
                scaffold_run_dir,
                scaffold_indices,
            )

            for split_method in FINGERPRINT_SPECS:
                spec = fingerprint_factory(split_method)
                print(
                    "  "
                    f"{split_method}: computing {spec.fingerprint_type} fingerprints "
                    f"and Butina clusters once"
                )
                fingerprints = build_fingerprints(dataset.mols, spec)
                clusters = compute_butina_clusters(
                    fingerprints,
                    distance_cutoff=spec.distance_cutoff,
                )
                print(f"    clusters found: {len(clusters)}")

                for seed in SEEDED_SPLITS:
                    split_indices, split_metadata = butina_split(
                        dataset=dataset,
                        split_method=split_method,
                        seed=seed,
                        clusters=clusters,
                    )
                    run_dir = save_split_csvs(
                        dataset=dataset,
                        split_method=split_method,
                        run_label=f"seed_{seed}",
                        split_indices=split_indices,
                        metadata=split_metadata,
                        seed_value=seed,
                    )
                    print_saved_run(split_method, f"seed_{seed}", run_dir, split_indices)
        except Exception as exc:
            failed_datasets += 1
            print(f"  [ERROR] Split generation failed for {dataset.dataset_name}: {exc}")
            continue

    print("\n[SUMMARY]")
    print(f"  processed datasets: {len(dataset_paths) - failed_datasets}")
    print(f"  failed datasets   : {failed_datasets}")
    print(f"  output root       : {OUTPUT_DIR}")

    return 0 if failed_datasets == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

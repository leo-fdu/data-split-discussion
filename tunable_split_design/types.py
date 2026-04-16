from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem


@dataclass(frozen=True)
class ScaffoldExtractionResult:
    """Structured result for one expanded scaffold extraction."""

    scaffold_mol: Chem.Mol
    scaffold_smiles: str
    scaffold_atom_indices: tuple[int, ...]
    scaffold_bond_indices: tuple[int, ...]
    num_atoms: int
    num_bonds: int


@dataclass(frozen=True)
class ScaffoldMatchRound:
    """One deterministic MCS round used inside scaffold similarity."""

    round_index: int
    bond_count: int
    atom_indices_a: tuple[int, ...]
    bond_indices_a: tuple[int, ...]
    atom_indices_b: tuple[int, ...]
    bond_indices_b: tuple[int, ...]


@dataclass(frozen=True)
class ScaffoldMatchResult:
    """Iterative scaffold overlap summary between two scaffold graphs."""

    round_bond_counts: tuple[int, ...]
    matched_bond_total: int
    similarity: float
    distance: float
    status: str
    rounds: tuple[ScaffoldMatchRound, ...] = ()


@dataclass(frozen=True)
class FunctionalGroupVocabularyEntry:
    """One leaf-level RDKit functional-group vocabulary entry."""

    index: int
    name: str
    label: str
    smarts: str
    pattern: Chem.Mol = field(repr=False)


@dataclass(frozen=True)
class FunctionalGroupVocabulary:
    """Ordered leaf-level functional-group vocabulary."""

    entries: tuple[FunctionalGroupVocabularyEntry, ...]

    @property
    def labels(self) -> tuple[str, ...]:
        """Return the fixed dimension order used for FG vectors."""
        return tuple(entry.label for entry in self.entries)

    @property
    def size(self) -> int:
        """Return the number of leaf-level functional-group dimensions."""
        return len(self.entries)


@dataclass(frozen=True)
class FunctionalGroupCountResult:
    """Functional-group counts and deduplicated match details for one molecule."""

    counts: np.ndarray
    count_dict: dict[str, int]
    matched_atom_sets: dict[str, tuple[tuple[int, ...], ...]]


@dataclass(frozen=True)
class PairwiseComputationFailure:
    """Failure metadata for one pairwise distance computation."""

    i: int
    j: int
    status: str
    message: str


@dataclass(frozen=True)
class PairwiseDistanceMatrixResult:
    """Distance matrix plus lightweight computation metadata."""

    matrix: np.ndarray
    metric_name: str
    failures: tuple[PairwiseComputationFailure, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssignedCluster:
    """One cluster assignment produced during split generation."""

    cluster_id: int
    split_name: str
    indices: tuple[int, ...]


@dataclass(frozen=True)
class SplitResult:
    """Train/val/test split indices derived from clusters."""

    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    cluster_assignments: tuple[AssignedCluster, ...]
    summary: dict[str, Any]


@dataclass(frozen=True)
class SweepResult:
    """One tunable split configuration produced during a parameter sweep."""

    lambda_: float
    cutoff: float
    clusters: tuple[tuple[int, ...], ...]
    split_result: SplitResult
    summary: dict[str, Any]


@dataclass(frozen=True)
class MoleculeTable:
    """Canonicalized molecule table loaded from a CSV or DataFrame."""

    dataframe: pd.DataFrame
    smiles_column: str
    canonical_smiles: tuple[str, ...]
    mols: tuple[Chem.Mol, ...]
    source: str | None = None
    dropped_invalid_indices: tuple[int, ...] = ()

    @property
    def size(self) -> int:
        """Return the number of retained molecules."""
        return len(self.mols)

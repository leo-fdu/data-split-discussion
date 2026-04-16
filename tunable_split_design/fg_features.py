from __future__ import annotations

from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import FunctionalGroups

from .config import MAX_SUBSTRUCT_MATCHES
from .types import (
    FunctionalGroupCountResult,
    FunctionalGroupVocabulary,
    FunctionalGroupVocabularyEntry,
)


@lru_cache(maxsize=1)
def build_leaf_functional_group_vocabulary() -> FunctionalGroupVocabulary:
    """
    Build a fixed-order leaf-level RDKit functional-group vocabulary.

    Only leaf nodes from the RDKit FunctionalGroups hierarchy are retained.
    Entries are ordered deterministically by `(label, name, smarts)`.
    """
    hierarchy = FunctionalGroups.BuildFuncGroupHierarchy()
    leaves = []
    for root in hierarchy:
        leaves.extend(_collect_leaf_nodes(root))

    leaves.sort(key=lambda node: (node.label, node.name, node.smarts))
    entries = tuple(
        FunctionalGroupVocabularyEntry(
            index=index,
            name=node.name,
            label=node.label,
            smarts=node.smarts,
            pattern=node.pattern,
        )
        for index, node in enumerate(leaves)
    )
    return FunctionalGroupVocabulary(entries=entries)


def count_leaf_functional_groups(
    mol: Chem.Mol,
    vocabulary: FunctionalGroupVocabulary | None = None,
) -> FunctionalGroupCountResult:
    """
    Count leaf-level functional groups for one molecule.

    Matches are deduplicated per functional-group label by unique atom-index set,
    so automorphisms or equivalent match reorderings are counted only once.
    Overlap across different functional-group labels is allowed.
    """
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("count_leaf_functional_groups expects a non-empty RDKit molecule.")

    active_vocabulary = vocabulary or build_leaf_functional_group_vocabulary()
    counts = np.zeros(active_vocabulary.size, dtype=np.int32)
    count_dict: dict[str, int] = {}
    matched_atom_sets: dict[str, tuple[tuple[int, ...], ...]] = {}
    params = Chem.SubstructMatchParameters()
    params.uniquify = False
    params.maxMatches = MAX_SUBSTRUCT_MATCHES
    params.useChirality = False

    for entry in active_vocabulary.entries:
        raw_matches = mol.GetSubstructMatches(entry.pattern, params)
        unique_matches = sorted(
            {tuple(sorted(match)) for match in raw_matches},
            key=lambda atom_tuple: (len(atom_tuple), atom_tuple),
        )
        counts[entry.index] = len(unique_matches)
        if unique_matches:
            count_dict[entry.label] = len(unique_matches)
            matched_atom_sets[entry.label] = tuple(unique_matches)

    return FunctionalGroupCountResult(
        counts=counts,
        count_dict=count_dict,
        matched_atom_sets=matched_atom_sets,
    )


def build_fg_count_matrix(
    mols: list[Chem.Mol] | tuple[Chem.Mol, ...],
    vocabulary: FunctionalGroupVocabulary | None = None,
    dtype: np.dtype = np.int32,
) -> tuple[np.ndarray, FunctionalGroupVocabulary]:
    """
    Convert a sequence of molecules into a shared leaf-level FG count matrix.

    The output matrix has shape `(n_molecules, vocabulary_size)` and uses the
    fixed vocabulary order returned alongside it.
    """
    active_vocabulary = vocabulary or build_leaf_functional_group_vocabulary()
    count_rows = [
        count_leaf_functional_groups(mol, vocabulary=active_vocabulary).counts
        for mol in mols
    ]
    if not count_rows:
        matrix = np.zeros((0, active_vocabulary.size), dtype=dtype)
    else:
        matrix = np.asarray(count_rows, dtype=dtype)
    return matrix, active_vocabulary


def _collect_leaf_nodes(node: FunctionalGroups.FGHierarchyNode) -> list[FunctionalGroups.FGHierarchyNode]:
    if not node.children:
        return [node]
    leaves = []
    for child in sorted(node.children, key=lambda item: (item.label, item.name, item.smarts)):
        leaves.extend(_collect_leaf_nodes(child))
    return leaves

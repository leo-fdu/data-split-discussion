from __future__ import annotations

from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import FunctionalGroups

from .config import MAX_SUBSTRUCT_MATCHES
from .data_types import (
    FunctionalGroupCountResult,
    FunctionalGroupVocabulary,
    FunctionalGroupVocabularyEntry,
)


@lru_cache(maxsize=1)
def build_leaf_functional_group_vocabulary() -> FunctionalGroupVocabulary:
    """
    Build a fixed-order leaf-level RDKit functional-group vocabulary.

    Only leaf nodes from the RDKit FunctionalGroups hierarchy are retained.
    Entries are ordered deterministically by `(label, name, smarts)`. Invalid
    leaf nodes with missing pattern, SMARTS, or label information are skipped.
    """
    hierarchy = FunctionalGroups.BuildFuncGroupHierarchy()
    leaves = []
    for root in hierarchy:
        leaves.extend(_collect_valid_leaf_nodes(root))

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


def validate_functional_group_vocabulary(
    vocabulary: FunctionalGroupVocabulary,
    mols: list[Chem.Mol] | tuple[Chem.Mol, ...] | None = None,
) -> dict[str, object]:
    """
    Return a lightweight deterministic summary of one FG vocabulary.

    When molecules are provided, the summary also reports how many molecules hit
    each FG dimension, which FG labels never hit, and the top-hit entries.
    """
    summary: dict[str, object] = {
        "vocabulary_size": vocabulary.size,
        "labels": [entry.label for entry in vocabulary.entries],
        "names": [entry.name for entry in vocabulary.entries],
        "smarts": [entry.smarts for entry in vocabulary.entries],
    }
    if mols is None:
        return summary

    count_matrix, _ = build_fg_count_matrix(mols, vocabulary=vocabulary)
    molecule_hit_counts = np.count_nonzero(count_matrix > 0, axis=0).astype(int)

    hits_per_fg = {
        entry.label: int(molecule_hit_counts[entry.index])
        for entry in vocabulary.entries
    }
    never_hit_labels = [
        entry.label
        for entry in vocabulary.entries
        if molecule_hit_counts[entry.index] == 0
    ]
    top_hit_entries = sorted(
        (
            {
                "label": entry.label,
                "name": entry.name,
                "smarts": entry.smarts,
                "num_molecules_hit": int(molecule_hit_counts[entry.index]),
            }
            for entry in vocabulary.entries
        ),
        key=lambda item: (-item["num_molecules_hit"], item["label"], item["name"], item["smarts"]),
    )[:10]

    summary.update(
        {
            "num_molecules": len(mols),
            "molecule_hit_counts": hits_per_fg,
            "never_hit_labels": never_hit_labels,
            "top_hit_entries": top_hit_entries,
        }
    )
    return summary


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


def _collect_valid_leaf_nodes(
    node: FunctionalGroups.FGHierarchyNode,
) -> list[FunctionalGroups.FGHierarchyNode]:
    if not node.children:
        return [node] if _is_valid_leaf_node(node) else []
    leaves = []
    for child in sorted(node.children, key=lambda item: (item.label, item.name, item.smarts)):
        leaves.extend(_collect_valid_leaf_nodes(child))
    return leaves


def _is_valid_leaf_node(node: FunctionalGroups.FGHierarchyNode) -> bool:
    pattern = node.pattern
    return bool(
        pattern is not None
        and node.smarts
        and node.label
        and pattern.GetNumAtoms() > 0
    )

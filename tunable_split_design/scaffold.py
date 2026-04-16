from __future__ import annotations

import heapq
from collections import deque

from rdkit import Chem
from rdkit.Chem import rdFMCS

from .config import DEFAULT_MAX_MCS_ROUNDS, MAX_SUBSTRUCT_MATCHES
from .types import (
    ScaffoldExtractionResult,
    ScaffoldMatchResult,
    ScaffoldMatchRound,
)


def extract_expanded_scaffold(mol: Chem.Mol) -> ScaffoldExtractionResult:
    """
    Extract the deterministic expanded scaffold subgraph for one molecule.

    The scaffold keeps all carbon atoms, all ring atoms and ring bonds, all
    multiple bonds with their endpoint atoms, and then greedily connects the
    resulting retained components with deterministic shortest paths. When
    multiple equal-length shortest paths exist, the path with the
    lexicographically smallest atom-index sequence is selected.
    """
    _validate_input_mol(mol)
    fragment_atom_sets = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    if len(fragment_atom_sets) > 1:
        raise ValueError(
            "Expanded scaffold extraction expects a single connected molecule; "
            f"received {len(fragment_atom_sets)} disconnected fragments."
        )

    required_atoms = _find_required_atoms(mol)
    if not required_atoms:
        empty_mol = Chem.Mol()
        return ScaffoldExtractionResult(
            scaffold_mol=empty_mol,
            scaffold_smiles="",
            scaffold_atom_indices=(),
            scaffold_bond_indices=(),
            num_atoms=0,
            num_bonds=0,
        )

    required_bonds = _find_required_bonds(mol, required_atoms)
    selected_atoms = set(required_atoms)
    selected_bonds = set(required_bonds)

    while True:
        components = _selected_components(selected_atoms, selected_bonds, mol)
        if len(components) <= 1:
            break
        path = _choose_next_component_path(mol, components)
        if not path:
            raise ValueError("Failed to connect retained scaffold components.")
        selected_atoms.update(path)
        selected_bonds.update(_path_to_bond_indices(mol, path))

    scaffold_atom_indices = tuple(sorted(selected_atoms))
    scaffold_bond_indices = tuple(sorted(selected_bonds))
    scaffold_smiles = Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=list(scaffold_atom_indices),
        bondsToUse=list(scaffold_bond_indices),
        canonical=True,
    )
    scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
    if scaffold_mol is None:
        raise ValueError(
            "RDKit failed to rebuild the extracted expanded scaffold from fragment SMILES."
        )

    return ScaffoldExtractionResult(
        scaffold_mol=scaffold_mol,
        scaffold_smiles=scaffold_smiles,
        scaffold_atom_indices=scaffold_atom_indices,
        scaffold_bond_indices=scaffold_bond_indices,
        num_atoms=len(scaffold_atom_indices),
        num_bonds=len(scaffold_bond_indices),
    )


def compute_scaffold_similarity(
    scaffold_a: Chem.Mol,
    scaffold_b: Chem.Mol,
    max_rounds: int = DEFAULT_MAX_MCS_ROUNDS,
) -> ScaffoldMatchResult:
    """
    Compute iterative bond-based MCS scaffold similarity for two scaffold graphs.

    Each round runs exact atom-type and exact bond-order MCS on the currently
    unmasked scaffold subgraphs, chooses one deterministic match instance in each
    scaffold, and masks all matched atoms and bonds before the next round. Match
    candidates are deduplicated by unique atom-index and bond-index sets, then
    the lexicographically smallest bond-index tuple is selected.
    """
    if max_rounds <= 0:
        raise ValueError("max_rounds must be a positive integer.")
    if scaffold_a is None or scaffold_b is None:
        raise ValueError("compute_scaffold_similarity expects non-None RDKit molecules.")

    num_bonds_a = scaffold_a.GetNumBonds()
    num_bonds_b = scaffold_b.GetNumBonds()
    if num_bonds_a == 0 and num_bonds_b == 0:
        return ScaffoldMatchResult(
            round_bond_counts=(),
            matched_bond_total=0,
            similarity=1.0,
            distance=0.0,
            status="empty_scaffolds",
            rounds=(),
        )
    if num_bonds_a == 0 or num_bonds_b == 0:
        return ScaffoldMatchResult(
            round_bond_counts=(),
            matched_bond_total=0,
            similarity=0.0,
            distance=1.0,
            status="one_scaffold_has_no_bonds",
            rounds=(),
        )

    working_a = _kekulized_copy(scaffold_a)
    working_b = _kekulized_copy(scaffold_b)
    masked_atoms_a: set[int] = set()
    masked_atoms_b: set[int] = set()
    masked_bonds_a: set[int] = set()
    masked_bonds_b: set[int] = set()
    round_counts: list[int] = []
    rounds: list[ScaffoldMatchRound] = []

    for round_index in range(max_rounds):
        active_a = _build_active_submol(working_a, masked_atoms_a, masked_bonds_a)
        active_b = _build_active_submol(working_b, masked_atoms_b, masked_bonds_b)
        if active_a is None or active_b is None:
            break

        mcs_result = _run_exact_bond_mcs(active_a[0], active_b[0])
        if mcs_result.numBonds <= 0:
            break
        query = Chem.MolFromSmarts(mcs_result.smartsString)
        if query is None or query.GetNumBonds() <= 0:
            break

        selected_a = _select_deterministic_match(query, active_a[0], active_a[1], active_a[2])
        selected_b = _select_deterministic_match(query, active_b[0], active_b[1], active_b[2])
        if selected_a is None or selected_b is None:
            matched_bond_total = int(sum(round_counts))
            denominator = max(num_bonds_a, num_bonds_b)
            similarity = 0.0 if denominator == 0 else matched_bond_total / denominator
            similarity = float(min(1.0, max(0.0, similarity)))
            return ScaffoldMatchResult(
                round_bond_counts=tuple(round_counts),
                matched_bond_total=matched_bond_total,
                similarity=similarity,
                distance=float(min(1.0, max(0.0, 1.0 - similarity))),
                status="failed_to_select_match",
                rounds=tuple(rounds),
            )

        round_counts.append(len(selected_a[1]))
        rounds.append(
            ScaffoldMatchRound(
                round_index=round_index,
                bond_count=len(selected_a[1]),
                atom_indices_a=selected_a[0],
                bond_indices_a=selected_a[1],
                atom_indices_b=selected_b[0],
                bond_indices_b=selected_b[1],
            )
        )
        masked_atoms_a.update(selected_a[0])
        masked_bonds_a.update(selected_a[1])
        masked_atoms_b.update(selected_b[0])
        masked_bonds_b.update(selected_b[1])

    matched_bond_total = int(sum(round_counts))
    denominator = max(num_bonds_a, num_bonds_b)
    similarity = 0.0 if denominator == 0 else matched_bond_total / denominator
    similarity = float(min(1.0, max(0.0, similarity)))
    distance = float(min(1.0, max(0.0, 1.0 - similarity)))
    status = "ok" if round_counts else "no_common_bonds"

    return ScaffoldMatchResult(
        round_bond_counts=tuple(round_counts),
        matched_bond_total=matched_bond_total,
        similarity=similarity,
        distance=distance,
        status=status,
        rounds=tuple(rounds),
    )


def _validate_input_mol(mol: Chem.Mol) -> None:
    if mol is None:
        raise ValueError("Received None instead of an RDKit molecule.")
    if mol.GetNumAtoms() == 0:
        raise ValueError("Received an empty RDKit molecule.")


def _find_required_atoms(mol: Chem.Mol) -> set[int]:
    required_atoms: set[int] = set()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6 or atom.IsInRing():
            required_atoms.add(atom.GetIdx())
    for bond in mol.GetBonds():
        if bond.GetBondTypeAsDouble() > 1.0:
            required_atoms.add(bond.GetBeginAtomIdx())
            required_atoms.add(bond.GetEndAtomIdx())
    return required_atoms


def _find_required_bonds(mol: Chem.Mol, required_atoms: set[int]) -> set[int]:
    required_bonds: set[int] = set()
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondTypeAsDouble() > 1.0:
            required_bonds.add(bond.GetIdx())
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx in required_atoms and end_idx in required_atoms:
            required_bonds.add(bond.GetIdx())
    return required_bonds


def _selected_components(
    selected_atoms: set[int],
    selected_bonds: set[int],
    mol: Chem.Mol,
) -> list[tuple[int, ...]]:
    adjacency = {atom_idx: set() for atom_idx in selected_atoms}
    for bond_idx in selected_bonds:
        bond = mol.GetBondWithIdx(bond_idx)
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx in adjacency and end_idx in adjacency:
            adjacency[begin_idx].add(end_idx)
            adjacency[end_idx].add(begin_idx)

    components: list[tuple[int, ...]] = []
    visited: set[int] = set()
    for start_idx in sorted(selected_atoms):
        if start_idx in visited:
            continue
        queue = deque([start_idx])
        visited.add(start_idx)
        component: list[int] = []
        while queue:
            atom_idx = queue.popleft()
            component.append(atom_idx)
            for neighbor_idx in sorted(adjacency[atom_idx]):
                if neighbor_idx in visited:
                    continue
                visited.add(neighbor_idx)
                queue.append(neighbor_idx)
        components.append(tuple(sorted(component)))
    return components


def _choose_next_component_path(
    mol: Chem.Mol,
    components: list[tuple[int, ...]],
) -> tuple[int, ...]:
    best_path: tuple[int, ...] | None = None
    for left_index in range(len(components)):
        for right_index in range(left_index + 1, len(components)):
            for start_idx in components[left_index]:
                for end_idx in components[right_index]:
                    path = _deterministic_shortest_path(mol, start_idx, end_idx)
                    if path is None:
                        continue
                    if best_path is None or (len(path), path) < (len(best_path), best_path):
                        best_path = path
    if best_path is None:
        raise ValueError("No connecting path exists between retained scaffold components.")
    return best_path


def _deterministic_shortest_path(
    mol: Chem.Mol,
    start_idx: int,
    end_idx: int,
) -> tuple[int, ...] | None:
    if start_idx == end_idx:
        return (start_idx,)

    heap: list[tuple[int, tuple[int, ...], int]] = [(0, (start_idx,), start_idx)]
    best_paths: dict[int, tuple[int, tuple[int, ...]]] = {start_idx: (0, (start_idx,))}

    while heap:
        distance, path, atom_idx = heapq.heappop(heap)
        best_distance, best_path = best_paths[atom_idx]
        if (distance, path) != (best_distance, best_path):
            continue
        if atom_idx == end_idx:
            return path
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in sorted(atom.GetNeighbors(), key=lambda item: item.GetIdx()):
            neighbor_idx = neighbor.GetIdx()
            candidate = (distance + 1, path + (neighbor_idx,))
            current = best_paths.get(neighbor_idx)
            if current is None or candidate < current:
                best_paths[neighbor_idx] = candidate
                heapq.heappush(heap, (candidate[0], candidate[1], neighbor_idx))
    return None


def _path_to_bond_indices(mol: Chem.Mol, atom_path: tuple[int, ...]) -> tuple[int, ...]:
    if len(atom_path) <= 1:
        return ()
    bond_indices: list[int] = []
    for left_idx, right_idx in zip(atom_path[:-1], atom_path[1:]):
        bond = mol.GetBondBetweenAtoms(left_idx, right_idx)
        if bond is None:
            raise ValueError(
                f"Atoms {left_idx} and {right_idx} are not directly bonded in the molecule."
            )
        bond_indices.append(bond.GetIdx())
    return tuple(bond_indices)


def _kekulized_copy(mol: Chem.Mol) -> Chem.Mol:
    copied_mol = Chem.Mol(mol)
    Chem.Kekulize(copied_mol, clearAromaticFlags=True)
    return copied_mol


def _build_active_submol(
    mol: Chem.Mol,
    masked_atoms: set[int],
    masked_bonds: set[int],
) -> tuple[Chem.Mol, dict[int, int], dict[int, int]] | None:
    active_bond_indices = [
        bond.GetIdx()
        for bond in mol.GetBonds()
        if bond.GetIdx() not in masked_bonds
        and bond.GetBeginAtomIdx() not in masked_atoms
        and bond.GetEndAtomIdx() not in masked_atoms
    ]
    if not active_bond_indices:
        return None

    active_atom_indices = sorted(
        {
            atom_idx
            for bond_idx in active_bond_indices
            for atom_idx in (
                mol.GetBondWithIdx(bond_idx).GetBeginAtomIdx(),
                mol.GetBondWithIdx(bond_idx).GetEndAtomIdx(),
            )
        }
    )
    old_to_new_atom: dict[int, int] = {}
    new_to_old_atom: dict[int, int] = {}
    new_to_old_bond: dict[int, int] = {}
    rw_mol = Chem.RWMol()

    for old_atom_idx in active_atom_indices:
        atom = mol.GetAtomWithIdx(old_atom_idx)
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetChiralTag(atom.GetChiralTag())
        new_atom.SetNoImplicit(atom.GetNoImplicit())
        new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
        new_idx = rw_mol.AddAtom(new_atom)
        old_to_new_atom[old_atom_idx] = new_idx
        new_to_old_atom[new_idx] = old_atom_idx

    for old_bond_idx in sorted(active_bond_indices):
        bond = mol.GetBondWithIdx(old_bond_idx)
        begin_idx = old_to_new_atom[bond.GetBeginAtomIdx()]
        end_idx = old_to_new_atom[bond.GetEndAtomIdx()]
        rw_mol.AddBond(begin_idx, end_idx, bond.GetBondType())
        new_bond = rw_mol.GetBondBetweenAtoms(begin_idx, end_idx)
        if new_bond is None:
            raise ValueError("Failed to rebuild an active scaffold bond.")
        new_bond.SetStereo(bond.GetStereo())
        new_to_old_bond[new_bond.GetIdx()] = old_bond_idx

    active_mol = rw_mol.GetMol()
    Chem.SanitizeMol(active_mol)
    return active_mol, new_to_old_atom, new_to_old_bond


def _run_exact_bond_mcs(mol_a: Chem.Mol, mol_b: Chem.Mol) -> rdFMCS.MCSResult:
    params = rdFMCS.MCSParameters()
    params.MaximizeBonds = True
    params.Timeout = 0
    params.AtomTyper = rdFMCS.AtomCompare.CompareElements
    params.BondTyper = rdFMCS.BondCompare.CompareOrderExact
    return rdFMCS.FindMCS([mol_a, mol_b], params)


def _select_deterministic_match(
    query: Chem.Mol,
    mol: Chem.Mol,
    new_to_old_atom: dict[int, int],
    new_to_old_bond: dict[int, int],
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    params = Chem.SubstructMatchParameters()
    params.uniquify = False
    params.maxMatches = MAX_SUBSTRUCT_MATCHES
    params.useChirality = False
    raw_matches = mol.GetSubstructMatches(query, params)

    deduplicated: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for atom_match in raw_matches:
        old_atom_indices = tuple(sorted(new_to_old_atom[new_idx] for new_idx in atom_match))
        old_bond_indices = []
        for query_bond in query.GetBonds():
            begin_idx = atom_match[query_bond.GetBeginAtomIdx()]
            end_idx = atom_match[query_bond.GetEndAtomIdx()]
            matched_bond = mol.GetBondBetweenAtoms(begin_idx, end_idx)
            if matched_bond is None:
                old_bond_indices = []
                break
            old_bond_indices.append(new_to_old_bond[matched_bond.GetIdx()])
        if not old_bond_indices:
            continue
        bond_tuple = tuple(sorted(old_bond_indices))
        key = (old_atom_indices, bond_tuple)
        deduplicated.setdefault(key, key)

    if not deduplicated:
        return None
    selected_key = min(deduplicated.values(), key=lambda item: (item[1], item[0]))
    return selected_key

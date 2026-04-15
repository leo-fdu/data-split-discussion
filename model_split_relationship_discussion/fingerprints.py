from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator


FINGERPRINT_TYPES = ["morgan", "maccs", "atompair"]

MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
    includeChirality=True,
)
ATOMPAIR_GENERATOR = rdFingerprintGenerator.GetAtomPairGenerator(
    fpSize=2048,
    includeChirality=True,
)


def _build_fingerprint(mol: Chem.Mol, fingerprint_type: str):
    if fingerprint_type == "morgan":
        return MORGAN_GENERATOR.GetFingerprint(mol)
    if fingerprint_type == "maccs":
        return MACCSkeys.GenMACCSKeys(mol)
    if fingerprint_type == "atompair":
        return ATOMPAIR_GENERATOR.GetFingerprint(mol)
    raise ValueError(f"Unsupported fingerprint type: {fingerprint_type}")


def build_fingerprint_matrix(smiles_list, fingerprint_type: str) -> np.ndarray:
    invalid = []
    fingerprints = []

    for row_idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            invalid.append((row_idx, smiles))
            continue
        fingerprints.append(_build_fingerprint(mol, fingerprint_type))

    if invalid:
        examples = "\n".join(
            f"row {row_idx}: {smiles}" for row_idx, smiles in invalid[:5]
        )
        raise ValueError(
            f"Failed to parse SMILES while building {fingerprint_type} fingerprints.\n"
            f"Examples:\n{examples}"
        )

    if not fingerprints:
        raise ValueError("No fingerprints were generated.")

    n_bits = fingerprints[0].GetNumBits()
    matrix = np.zeros((len(fingerprints), n_bits), dtype=np.uint8)

    for row_idx, fingerprint in enumerate(fingerprints):
        DataStructs.ConvertToNumpyArray(fingerprint, matrix[row_idx])

    return matrix

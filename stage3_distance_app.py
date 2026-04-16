from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

try:
    import streamlit as st
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "Streamlit is required to run this app. Install it in your environment "
        "and run `streamlit run stage3_distance_app.py`."
    ) from exc

from tunable_split_design import (
    FunctionalGroupCountResult,
    FunctionalGroupVocabulary,
    MoleculeTable,
    ScaffoldExtractionResult,
    combine_distance_pair,
    compute_fg_distance,
    compute_scaffold_similarity,
    count_leaf_functional_groups,
    extract_expanded_scaffold,
    load_molecule_table,
    validate_functional_group_vocabulary,
)
from tunable_split_design.fg_features import build_leaf_functional_group_vocabulary


ROOT_DIR = Path(__file__).resolve().parent
SMILESONLY_DIR = ROOT_DIR / "data" / "smilesonly"
SMILES_COLUMN = "smiles"
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 320
PAIR_STATE_KEY = "stage3_pair_indices"
DATASET_STATE_KEY = "stage3_selected_dataset"
SEED_STATE_KEY = "stage3_seed_value"
SEED_CURSOR_KEY = "stage3_seed_cursor"


@dataclass(frozen=True)
class DatasetBundle:
    """Cached dataset resources needed by the pairwise explorer."""

    dataset_name: str
    csv_path: str
    file_signature: str
    original_row_count: int
    dropped_invalid_count: int
    dropped_invalid_indices: tuple[int, ...]
    dropped_multifragment_count: int
    dropped_multifragment_indices: tuple[int, ...]
    dropped_scaffold_failed_count: int
    dropped_scaffold_failed_indices: tuple[int, ...]
    table: MoleculeTable
    scaffolds: tuple[ScaffoldExtractionResult, ...]
    fg_results: tuple[FunctionalGroupCountResult, ...]
    fg_count_matrix: np.ndarray
    vocabulary: FunctionalGroupVocabulary
    vocabulary_summary: dict[str, object]


@st.cache_data(show_spinner=False)
def list_smilesonly_datasets(data_dir: str) -> tuple[str, ...]:
    """Return sorted dataset CSV paths under `data/smilesonly`."""
    directory = Path(data_dir)
    if not directory.exists():
        raise FileNotFoundError(f"SMILES dataset directory does not exist: {directory}")
    csv_paths = sorted(directory.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files were found under: {directory}")
    return tuple(str(path.resolve()) for path in csv_paths)


def file_signature(path: Path) -> str:
    """Return a lightweight file signature for cache invalidation."""
    stat = path.stat()
    return f"{stat.st_mtime_ns}-{stat.st_size}"


@st.cache_resource(show_spinner=False, max_entries=2)
def load_dataset_bundle(csv_path: str, signature: str) -> DatasetBundle:
    """Load one dataset and precompute scaffolds plus FG features."""
    del signature
    path = Path(csv_path)
    filtered_df, original_row_count, dropped_invalid_indices = filter_valid_smiles_dataframe(path)
    parsed_table = load_molecule_table(
        filtered_df,
        smiles_column=SMILES_COLUMN,
        invalid_smiles="raise",
    )
    if parsed_table.size == 0:
        raise ValueError(f"No valid molecules were retained from {path.name}.")

    vocabulary = build_leaf_functional_group_vocabulary()
    usable_records: list[dict[str, object]] = []
    usable_mols: list[Chem.Mol] = []
    usable_canonical_smiles: list[str] = []
    scaffolds: list[ScaffoldExtractionResult] = []
    fg_results: list[FunctionalGroupCountResult] = []
    fg_count_rows: list[np.ndarray] = []
    dropped_multifragment_indices: list[int] = []
    dropped_scaffold_failed_indices: list[int] = []

    for index, mol in enumerate(parsed_table.mols):
        row = parsed_table.dataframe.iloc[index]
        smiles = parsed_table.canonical_smiles[index]
        dataset_row_index = int(row.get("original_dataset_row_index", row["source_row_index"]))
        fragment_atom_sets = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
        if len(fragment_atom_sets) > 1:
            dropped_multifragment_indices.append(dataset_row_index)
            continue

        try:
            scaffold_result = extract_expanded_scaffold(mol)
        except Exception as exc:
            dropped_scaffold_failed_indices.append(dataset_row_index)
            continue
        scaffolds.append(scaffold_result)

        try:
            fg_result = count_leaf_functional_groups(mol, vocabulary=vocabulary)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to count leaf functional groups for molecule {index} ({smiles}): {exc}"
            ) from exc
        fg_results.append(fg_result)
        fg_count_rows.append(np.asarray(fg_result.counts, dtype=np.int32))
        usable_records.append(row.to_dict())
        usable_mols.append(mol)
        usable_canonical_smiles.append(smiles)

    if not usable_records:
        raise ValueError(
            f"No usable molecules remained after filtering invalid SMILES, multi-fragment "
            f"molecules, and scaffold extraction failures in {path.name}."
        )

    table = MoleculeTable(
        dataframe=pd.DataFrame(usable_records).reset_index(drop=True),
        smiles_column=parsed_table.smiles_column,
        canonical_smiles=tuple(usable_canonical_smiles),
        mols=tuple(usable_mols),
        source=parsed_table.source,
        dropped_invalid_indices=parsed_table.dropped_invalid_indices,
    )

    fg_count_matrix = (
        np.asarray(fg_count_rows, dtype=np.int32)
        if fg_count_rows
        else np.zeros((0, vocabulary.size), dtype=np.int32)
    )
    vocabulary_summary = validate_functional_group_vocabulary(vocabulary, mols=table.mols)

    return DatasetBundle(
        dataset_name=path.name,
        csv_path=str(path.resolve()),
        file_signature=file_signature(path),
        original_row_count=original_row_count,
        dropped_invalid_count=len(dropped_invalid_indices),
        dropped_invalid_indices=dropped_invalid_indices,
        dropped_multifragment_count=len(dropped_multifragment_indices),
        dropped_multifragment_indices=tuple(dropped_multifragment_indices),
        dropped_scaffold_failed_count=len(dropped_scaffold_failed_indices),
        dropped_scaffold_failed_indices=tuple(dropped_scaffold_failed_indices),
        table=table,
        scaffolds=tuple(scaffolds),
        fg_results=tuple(fg_results),
        fg_count_matrix=fg_count_matrix,
        vocabulary=vocabulary,
        vocabulary_summary=vocabulary_summary,
    )


def filter_valid_smiles_dataframe(
    csv_path: str | Path,
    smiles_column: str = SMILES_COLUMN,
) -> tuple[pd.DataFrame, int, tuple[int, ...]]:
    """Read one CSV and drop rows whose SMILES cannot be parsed by RDKit."""
    path = Path(csv_path)
    raw_df = pd.read_csv(path)
    if smiles_column not in raw_df.columns:
        raise ValueError(f"Input data is missing required column {smiles_column!r}.")

    valid_row_indices: list[int] = []
    dropped_invalid_indices: list[int] = []
    for row_index, raw_value in enumerate(raw_df[smiles_column].tolist()):
        smiles = str(raw_value).strip()
        if not smiles or smiles.lower() in {"nan", "none", "null"}:
            dropped_invalid_indices.append(row_index)
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            dropped_invalid_indices.append(row_index)
            continue
        valid_row_indices.append(row_index)

    filtered_df = raw_df.iloc[valid_row_indices].copy()
    filtered_df["original_dataset_row_index"] = valid_row_indices
    return filtered_df, len(raw_df), tuple(dropped_invalid_indices)


@st.cache_data(show_spinner=False, max_entries=512)
def render_molecule_image(
    smiles: str,
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT,
) -> bytes | None:
    """Render one SMILES string with a black-and-white RDKit style."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return render_molecule_image_from_mol(mol, width=width, height=height)


def render_molecule_image_from_mol(
    mol: Chem.Mol,
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT,
) -> bytes | None:
    """Render one RDKit molecule as PNG bytes for `st.image(...)`."""
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    mol_to_draw = Chem.Mol(mol)
    mol_to_draw = Chem.RemoveHs(mol_to_draw)
    if mol_to_draw.GetNumAtoms() == 0:
        return None

    rdDepictor.Compute2DCoords(mol_to_draw)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.useBWAtomPalette()
    options.bondLineWidth = 2.5
    options.padding = 0.05
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol_to_draw)
    drawer.FinishDrawing()
    image_bytes = drawer.GetDrawingText()
    return bytes(image_bytes)


@st.cache_data(show_spinner=False, max_entries=1024)
def compute_pair_metrics(
    csv_path: str,
    signature: str,
    index_a: int,
    index_b: int,
) -> dict[str, object]:
    """Compute pairwise scaffold and FG metrics for one molecule pair."""
    bundle = load_dataset_bundle(csv_path, signature)
    counts_a = np.asarray(bundle.fg_count_matrix[index_a], dtype=np.int32)
    counts_b = np.asarray(bundle.fg_count_matrix[index_b], dtype=np.int32)
    scaffold_a = bundle.scaffolds[index_a]
    scaffold_b = bundle.scaffolds[index_b]

    try:
        scaffold_match = compute_scaffold_similarity(
            scaffold_a.scaffold_mol,
            scaffold_b.scaffold_mol,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to compute scaffold similarity for pair ({index_a}, {index_b}): {exc}"
        ) from exc

    d_fg = compute_fg_distance(counts_a, counts_b)
    active_fg_rows = build_active_fg_rows(bundle.vocabulary, counts_a, counts_b)

    return {
        "d_scaffold": float(scaffold_match.distance),
        "d_fg": float(d_fg),
        "scaffold_match_status": scaffold_match.status,
        "round_bond_counts": list(scaffold_match.round_bond_counts),
        "matched_bond_total": int(scaffold_match.matched_bond_total),
        "scaffold_similarity": float(scaffold_match.similarity),
        "fg_active_dimension_count": len(active_fg_rows),
        "active_fg_rows": active_fg_rows,
        "fg_count_dict_a": dict(bundle.fg_results[index_a].count_dict),
        "fg_count_dict_b": dict(bundle.fg_results[index_b].count_dict),
        "scaffold_smiles_a": scaffold_a.scaffold_smiles,
        "scaffold_smiles_b": scaffold_b.scaffold_smiles,
    }


def build_active_fg_rows(
    vocabulary: FunctionalGroupVocabulary,
    counts_a: np.ndarray,
    counts_b: np.ndarray,
) -> list[dict[str, object]]:
    """Build a debug-friendly table for active FG dimensions in one pair."""
    rows: list[dict[str, object]] = []
    for entry in vocabulary.entries:
        count_a = int(counts_a[entry.index])
        count_b = int(counts_b[entry.index])
        if count_a + count_b == 0:
            continue
        if count_a > 0 and count_b > 0:
            relation = "shared"
        elif count_a > 0:
            relation = "A only"
        else:
            relation = "B only"
        rows.append(
            {
                "label": entry.label,
                "A_count": count_a,
                "B_count": count_b,
                "relation": relation,
                "normalized_gap": abs(count_a - count_b) / max(count_a, count_b),
            }
        )
    return rows


def sample_pair_random(num_items: int) -> tuple[int, int]:
    """Sample two distinct molecule indices uniformly at random."""
    if num_items < 2:
        raise ValueError("At least two molecules are required to sample a pair.")
    first, second = random.SystemRandom().sample(range(num_items), 2)
    return tuple(sorted((first, second)))


def sample_pair_with_seed(dataset_key: str, seed_value: int, cursor: int, num_items: int) -> tuple[int, int]:
    """Sample a reproducible pair given dataset name, seed value, and cursor."""
    if num_items < 2:
        raise ValueError("At least two molecules are required to sample a pair.")
    rng = random.Random(f"{dataset_key}|{seed_value}|{cursor}")
    first, second = rng.sample(range(num_items), 2)
    return tuple(sorted((first, second)))


def ensure_session_pair(dataset_key: str, seed_value: int, num_items: int) -> None:
    """Keep pair sampling stable across reruns and lambda changes."""
    dataset_changed = st.session_state.get(DATASET_STATE_KEY) != dataset_key
    seed_changed = st.session_state.get(SEED_STATE_KEY) != seed_value
    missing_pair = PAIR_STATE_KEY not in st.session_state

    if dataset_changed:
        st.session_state[DATASET_STATE_KEY] = dataset_key
        st.session_state[SEED_STATE_KEY] = seed_value
        st.session_state[SEED_CURSOR_KEY] = 0
        apply_seed_pair(dataset_key, seed_value, num_items)
        return

    if seed_changed:
        st.session_state[SEED_STATE_KEY] = seed_value
        st.session_state[SEED_CURSOR_KEY] = 0

    if missing_pair:
        apply_seed_pair(dataset_key, seed_value, num_items)


def apply_seed_pair(dataset_key: str, seed_value: int, num_items: int) -> None:
    """Advance the deterministic seeded sampler and store the new pair."""
    cursor = int(st.session_state.get(SEED_CURSOR_KEY, 0))
    st.session_state[PAIR_STATE_KEY] = sample_pair_with_seed(
        dataset_key=dataset_key,
        seed_value=seed_value,
        cursor=cursor,
        num_items=num_items,
    )
    st.session_state[SEED_CURSOR_KEY] = cursor + 1


def to_fg_dataframe(count_dict: dict[str, int]) -> pd.DataFrame:
    """Convert one FG count dictionary into a display table."""
    if not count_dict:
        return pd.DataFrame(columns=["functional_group", "count"])
    rows = [
        {"functional_group": label, "count": int(count)}
        for label, count in sorted(count_dict.items())
    ]
    return pd.DataFrame(rows)


def describe_pair_member(bundle: DatasetBundle, index: int) -> dict[str, object]:
    """Return one molecule's display metadata."""
    row = bundle.table.dataframe.iloc[index]
    original_smiles = str(row[bundle.table.smiles_column])
    canonical_smiles = bundle.table.canonical_smiles[index]
    source_row_index = int(row["source_row_index"])
    dataset_row_index = int(row.get("original_dataset_row_index", source_row_index))
    scaffold = bundle.scaffolds[index]
    fg_result = bundle.fg_results[index]
    return {
        "clean_index": index,
        "source_row_index": source_row_index,
        "dataset_row_index": dataset_row_index,
        "original_smiles": original_smiles,
        "canonical_smiles": canonical_smiles,
        "scaffold_smiles": scaffold.scaffold_smiles,
        "fg_count_dict": dict(fg_result.count_dict),
        "molecule_image": render_molecule_image(canonical_smiles),
        "scaffold_image": render_molecule_image(scaffold.scaffold_smiles),
    }


def render_fg_block(title: str, count_dict: dict[str, int]) -> None:
    """Render a compact FG count block."""
    st.markdown(title)
    fg_df = to_fg_dataframe(count_dict)
    if fg_df.empty:
        st.caption("No leaf-level FG hit.")
    else:
        st.dataframe(fg_df, use_container_width=True, hide_index=True)


def main() -> None:
    """Run the Stage 3 pairwise distance explorer app."""
    st.set_page_config(page_title="Stage 3 Pairwise Distance Explorer", layout="wide")
    st.title("Pairwise Distance Explorer")
    st.caption(
        "Interactively inspect one molecule pair at a time and see how "
        "scaffold distance, FG distance, and lambda combine into total distance."
    )

    try:
        dataset_paths = list_smilesonly_datasets(str(SMILESONLY_DIR))
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.sidebar.header("Controls")
    selected_dataset_path = st.sidebar.selectbox(
        "Dataset CSV",
        options=dataset_paths,
        format_func=lambda path: Path(path).name,
    )
    lambda_value = st.sidebar.slider(
        "lambda",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )
    seed_value = int(
        st.sidebar.number_input(
            "Seed",
            value=0,
            step=1,
            help="Used by the deterministic resample button and dataset-change auto sampling.",
        )
    )
    random_button = st.sidebar.button("Randomly sample two molecules", use_container_width=True)
    seeded_button = st.sidebar.button("Resample (using current seed)", use_container_width=True)
    show_debug = st.sidebar.checkbox("Show detailed debug info", value=False)
    st.sidebar.caption(
        "First load of a large dataset may take time because all molecules are preprocessed "
        "into scaffolds and FG counts, then cached."
    )

    selected_path = Path(selected_dataset_path)
    signature = file_signature(selected_path)
    with st.spinner(f"Preprocessing {selected_path.name} ..."):
        try:
            bundle = load_dataset_bundle(str(selected_path.resolve()), signature)
        except Exception as exc:
            st.error(f"Failed to preprocess dataset {selected_path.name}: {exc}")
            st.stop()

    if bundle.table.size < 2:
        st.error(
            f"{bundle.dataset_name} contains only {bundle.table.size} usable molecule(s); "
            "at least 2 are required."
        )
        st.stop()

    st.info(
        f"Loaded {bundle.table.size} usable molecules from {bundle.original_row_count} rows. "
        f"Dropped {bundle.dropped_invalid_count} invalid SMILES, "
        f"{bundle.dropped_multifragment_count} multi-fragment molecules, and "
        f"{bundle.dropped_scaffold_failed_count} scaffold-extraction failures."
    )

    try:
        ensure_session_pair(bundle.csv_path, seed_value, bundle.table.size)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    if random_button:
        try:
            st.session_state[PAIR_STATE_KEY] = sample_pair_random(bundle.table.size)
        except Exception as exc:
            st.error(str(exc))
            st.stop()
    elif seeded_button:
        try:
            apply_seed_pair(bundle.csv_path, seed_value, bundle.table.size)
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    index_a, index_b = st.session_state[PAIR_STATE_KEY]
    try:
        with st.spinner("Computing pairwise distances ..."):
            pair_metrics = compute_pair_metrics(bundle.csv_path, signature, index_a, index_b)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    d_total = combine_distance_pair(
        pair_metrics["d_scaffold"],
        pair_metrics["d_fg"],
        lambda_value,
    )
    molecule_a = describe_pair_member(bundle, index_a)
    molecule_b = describe_pair_member(bundle, index_b)

    st.subheader("Current Pair")
    overview_left, overview_right = st.columns(2)
    with overview_left:
        st.markdown(f"**Dataset:** `{bundle.dataset_name}`")
        st.markdown(f"**Current lambda:** `{lambda_value:.2f}`")
        st.markdown(f"**Pair indices:** `{index_a}` and `{index_b}`")
        st.markdown(
            f"**Dataset rows:** `{molecule_a['dataset_row_index']}` and `{molecule_b['dataset_row_index']}`"
        )
    with overview_right:
        st.markdown(f"**Molecule A canonical SMILES:** `{molecule_a['canonical_smiles']}`")
        if molecule_a["original_smiles"] != molecule_a["canonical_smiles"]:
            st.markdown(f"**Molecule A original SMILES:** `{molecule_a['original_smiles']}`")
        st.markdown(f"**Molecule B canonical SMILES:** `{molecule_b['canonical_smiles']}`")
        if molecule_b["original_smiles"] != molecule_b["canonical_smiles"]:
            st.markdown(f"**Molecule B original SMILES:** `{molecule_b['original_smiles']}`")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Molecule A")
        if molecule_a["molecule_image"] is not None:
            st.image(molecule_a["molecule_image"], width=IMAGE_WIDTH)
        else:
            st.warning("Molecule A image could not be rendered.")
        st.markdown(f"**Scaffold SMILES:** `{molecule_a['scaffold_smiles'] or '<empty>'}`")
        if molecule_a["scaffold_image"] is not None:
            st.image(molecule_a["scaffold_image"], width=IMAGE_WIDTH)
        else:
            st.caption("Scaffold image unavailable.")
        render_fg_block("**Non-zero FG counts**", molecule_a["fg_count_dict"])

    with col_b:
        st.subheader("Molecule B")
        if molecule_b["molecule_image"] is not None:
            st.image(molecule_b["molecule_image"], width=IMAGE_WIDTH)
        else:
            st.warning("Molecule B image could not be rendered.")
        st.markdown(f"**Scaffold SMILES:** `{molecule_b['scaffold_smiles'] or '<empty>'}`")
        if molecule_b["scaffold_image"] is not None:
            st.image(molecule_b["scaffold_image"], width=IMAGE_WIDTH)
        else:
            st.caption("Scaffold image unavailable.")
        render_fg_block("**Non-zero FG counts**", molecule_b["fg_count_dict"])

    st.divider()
    st.subheader("Distance Results")
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("D_scaffold", f"{pair_metrics['d_scaffold']:.4f}")
    metric_b.metric("D_FG", f"{pair_metrics['d_fg']:.4f}")
    metric_c.metric("D_total", f"{d_total:.4f}")
    st.latex(r"D_{\mathrm{total}} = \lambda \cdot D_{\mathrm{scaffold}} + (1 - \lambda) \cdot D_{\mathrm{FG}}")
    st.markdown(
        f"`D_total = {lambda_value:.2f} * {pair_metrics['d_scaffold']:.4f} + "
        f"(1 - {lambda_value:.2f}) * {pair_metrics['d_fg']:.4f} = {d_total:.4f}`"
    )

    if show_debug:
        st.divider()
        st.subheader("Debug Details")
        st.markdown(f"**Vocabulary size:** `{bundle.vocabulary.size}`")
        st.markdown(f"**Dataset source:** `{bundle.csv_path}`")
        st.markdown(f"**Scaffold match status:** `{pair_metrics['scaffold_match_status']}`")
        st.markdown(f"**Round bond counts:** `{pair_metrics['round_bond_counts']}`")
        st.markdown(f"**Matched bond total:** `{pair_metrics['matched_bond_total']}`")
        st.markdown(f"**Scaffold similarity:** `{pair_metrics['scaffold_similarity']:.4f}`")
        st.markdown(f"**Scaffold distance:** `{pair_metrics['d_scaffold']:.4f}`")
        st.markdown(f"**FG active dimensions:** `{pair_metrics['fg_active_dimension_count']}`")
        st.markdown(f"**Dropped invalid SMILES:** `{bundle.dropped_invalid_count}`")
        st.markdown(f"**Dropped multi-fragment molecules:** `{bundle.dropped_multifragment_count}`")
        st.markdown(f"**Dropped scaffold failures:** `{bundle.dropped_scaffold_failed_count}`")
        st.markdown(
            f"**Never-hit vocabulary entries in this dataset:** "
            f"`{len(bundle.vocabulary_summary.get('never_hit_labels', []))}`"
        )

        debug_left, debug_right = st.columns(2)
        with debug_left:
            st.markdown("**FG count dict: Molecule A**")
            st.json(pair_metrics["fg_count_dict_a"])
        with debug_right:
            st.markdown("**FG count dict: Molecule B**")
            st.json(pair_metrics["fg_count_dict_b"])

        active_fg_df = pd.DataFrame(pair_metrics["active_fg_rows"])
        if active_fg_df.empty:
            st.caption("No active FG dimensions for this pair.")
        else:
            st.markdown("**Active FG dimensions for this pair**")
            st.dataframe(active_fg_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

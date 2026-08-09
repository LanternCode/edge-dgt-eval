import random
import torch
from pipeline.EdgeClassification import TaskHooks, ProvidedSplitsTask, TNNTrainConfig, run_pipeline_for_task
from pipeline.GraphBenchmark import GraphBenchmark
from pipeline.GNNBridge import run_gnn_suite, GNNTrainConfig
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from rdkit import Chem
from rdkit.Chem import rdchem


@dataclass(frozen=True)
class MolDatasetConfig:
    num_graphs: int = 1000
    train_count: int = 800
    val_count: int = 100
    test_count: int = 100
    min_atoms: int = 6
    max_atoms: int = 140
    max_mol_attempts: int = 10000
    max_stitch_attempts: int = 200


def _default_fragments() -> List[str]:
    """
    Returns a list of SMILES strings used as building blocks for molecule generation.

    Returns:
        List[str]: A hardcoded list of SMILES strings (e.g., "CCO", "c1ccccc1").
    """
    return [
        "CCO", "c1ccccc1", "CC=O", "C#N", "CCN", "CCCl", "CCBr",
        "C=CC=C", "c1ccncc1", "C1CCCCC1", "O=C=O", "CC(=O)O", "C#CC",
        "c1ccccc1O", "CC(C)O", "CC(C)N",
        "C=C", "C#C", "C1=CC=CC=C1", "CC#N", "C1CC1", "C=N", "CC#CC",
        "C1=CC=CN=C1", "CC(N)=O", "CC(C)=O", "c1ccco1", "C1=CN=CN1",
        "CC=CC", "C1CCOC1", "C1=COC=C1", "CN(C)C", "c1cccs1", "C(=O)N",
    ]


def _electronegativity() -> Dict[int, float]:
    """
    Provides a mapping of atomic numbers to their Pauling electronegativity values.

    Returns:
        Dict[int, float]: A dictionary where keys are atomic numbers (e.g., 6 for Carbon)
        and values are electronegativity floats (e.g., 2.55).
    """
    return {
        1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
        15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66,
    }


def _atom_rich_features(atom, eneg: Dict[int, float]) -> Dict[str, float]:
    """
    Computes specific numerical descriptors for a single atom using RDKit and a provided electronegativity map.

    Args:
        atom: The atom object to analyze.
        eneg (Dict[int, float]): A dictionary mapping atomic numbers to electronegativity values.

    Returns:
        Dict[str, float]: A dictionary of features.
    """
    atomic_num = atom.GetAtomicNum()
    return {
        "atomic_num": float(atomic_num),
        "total_degree": float(atom.GetTotalDegree()),
        "formal_charge": float(atom.GetFormalCharge()),
        "is_aromatic": float(int(atom.GetIsAromatic())),
        "is_in_ring": float(int(atom.IsInRing())),
        "electronegativity": float(eneg.get(atomic_num, 0.0)),
    }


def atom_feature_keys_for_mode(feature_mode: str) -> List[str]:
    """
    Modes:
      - "default"  : rich RDKit-derived atom descriptors.
      - "ablation" : minimal atom descriptors (prevents valence arithmetic leak).
    """
    rich = [
        "atomic_num", "total_degree", "formal_charge",
        "electronegativity", "is_in_ring", "is_aromatic",
    ]
    ablation = [
        "atomic_num", "formal_charge", "electronegativity",
    ]

    m = str(feature_mode).strip().lower()
    if m == "default":
        return rich
    if m == "ablation":
        return ablation
    raise ValueError(f"Unknown feature_mode={feature_mode!r}. Use 'default' or 'ablation'.")


def _bond_type_to_class(bond) -> Optional[int]:
    """
    Maps an RDKit bond object to an integer class label for the learning task.

    Args:
        bond: The bond object to classify.

    Returns:
        Optional[int]: The integer label (0: Single, 1: Double, 2: Triple, 3: Aromatic)
        or None if the bond type is not supported.
    """
    bt = bond.GetBondType()
    if bt == rdchem.BondType.SINGLE:
        return 0
    if bt == rdchem.BondType.DOUBLE:
        return 1
    if bt == rdchem.BondType.TRIPLE:
        return 2
    if bt == rdchem.BondType.AROMATIC:
        return 3
    return None


def _candidate_attach_atoms(mol: "Chem.Mol") -> List[int]:
    """
    Identifies atoms in a molecule that are suitable for forming new bonds during stitching.

    Args:
        mol (rdkit.Chem.Mol): The molecule to scan.

    Returns:
        List[int]: A list of atom indices representing Carbons or non-aromatic Nitrogens that have
        at least one implicit hydrogen available.
    """
    idxs: List[int] = []
    for a in mol.GetAtoms():
        z = a.GetAtomicNum()
        h = int(a.GetNumImplicitHs())
        if h <= 0: continue
        if z == 6:
            idxs.append(a.GetIdx())
        elif z == 7 and (not a.GetIsAromatic()):
            idxs.append(a.GetIdx())
    return idxs


def _try_stitch(mol_a, mol_b: "Chem.Mol", rng: random.Random) -> Optional["Chem.Mol"]:
    """
    Attempts to create a single bond between two molecules at randomly selected valid attachment points.

    Args:
        mol_a: The first molecule.
        mol_b (rdkit.Chem.Mol): The second molecule (fragment) to attach.
        rng (random.Random): The random number generator instance.

    Returns:
        Optional[rdkit.Chem.Mol]: The combined, sanitized molecule if successful,
        or None if no valid attachment points exist or sanitization fails.
    """
    na = mol_a.GetNumAtoms()
    cand_a = _candidate_attach_atoms(mol_a)
    cand_b = _candidate_attach_atoms(mol_b)
    if not cand_a or not cand_b:
        return None

    ia = rng.choice(cand_a)
    ib = rng.choice(cand_b) + na

    combo = Chem.CombineMols(mol_a, mol_b)
    rw = Chem.RWMol(combo)
    rw.AddBond(int(ia), int(ib), rdchem.BondType.SINGLE)

    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return out


def _build_random_molecule(fragment_mols, rng, min_atoms, max_atoms, max_stitch_attempts) -> Optional["Chem.Mol"]:
    """
    Iteratively builds a random molecule by stitching fragments until a target size is reached.

    Args:
        fragment_mols: A list of available RDKit molecule fragments.
        rng (random.Random): The random number generator.
        min_atoms (int): The minimum allowable number of atoms in the final molecule.
        max_atoms (int): The maximum allowable number of atoms.
        max_stitch_attempts (int): The maximum number of times to try connecting a specific fragment before giving up.

    Returns:
        Optional[rdkit.Chem.Mol]: A generated RDKit molecule within the size constraints, or None if generation failed.
    """
    target = rng.randint(min_atoms, max_atoms)
    starters = [m for m in fragment_mols if m.GetNumAtoms() <= target] or fragment_mols[:]
    mol = Chem.Mol(starters[rng.randrange(len(starters))])

    while mol.GetNumAtoms() < target:
        cur = mol.GetNumAtoms()
        remaining = target - cur
        candidates = [m for m in fragment_mols if m.GetNumAtoms() <= remaining]
        if not candidates:
            candidates = [m for m in fragment_mols if (cur + m.GetNumAtoms()) <= max_atoms]
        if not candidates:
            break

        frag = Chem.Mol(candidates[rng.randrange(len(candidates))])
        stitched = None
        for _ in range(max_stitch_attempts):
            stitched = _try_stitch(mol, frag, rng)
            if stitched is not None:
                break
        if stitched is None:
            return None

        mol = stitched
        if mol.GetNumAtoms() > max_atoms:
            return None

    if mol.GetNumAtoms() < min_atoms or mol.GetNumAtoms() > max_atoms:
        return None
    try:
        smi = Chem.MolToSmiles(mol, canonical=True)
        mol2 = Chem.MolFromSmiles(smi)
        if mol2 is None:
            return None
        Chem.SanitizeMol(mol2)
        return mol2
    except Exception:
        return None


def _mol_to_graph_tensors(
        mol,
        eneg: Dict[int, float],
        atom_feature_keys: Sequence[str]
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Converts an RDKit molecule into the specific tensor format required for the graph neural network.

    Args:
        mol: The source molecule.
        eneg (Dict[int, float]): The electronegativity mapping for feature extraction.
        atom_feature_keys (Sequence[str]): A list of feature names to extract for every atom.

    Returns:
        Tuple[Tensor, Dict, Tensor, Tensor]: Returns the Adjacency matrix (A), Feature dictionary (feats),
        Label matrix (L), and Edge Mask (mask).
    """
    n = mol.GetNumAtoms()
    A = torch.zeros((n, n), dtype=torch.float32)
    L = torch.zeros((n, n), dtype=torch.int64)

    feat_acc: Dict[str, List[float]] = {k: [] for k in atom_feature_keys}
    for atom in mol.GetAtoms():
        f = _atom_rich_features(atom, eneg)
        for k in atom_feature_keys:
            feat_acc[k].append(float(f[k]))

    feats = {k: torch.tensor(v, dtype=torch.float32) for k, v in feat_acc.items()}

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        cls = _bond_type_to_class(bond)
        if cls is None:
            continue

        A[i, j] = A[j, i] = 1.0
        L[i, j] = L[j, i] = int(cls)

    mask = torch.gt(A, 0.5)
    mask.fill_diagonal_(False)

    return A, feats, L, mask


def _generate_items(fragments, cfg, atom_feature_keys, seed=None):
    """
    Driver function that generates the full dataset of graph tensors by building and converting random molecules.

    Args:
        fragments (List[str]): A list of SMILES strings to use as fragments.
        cfg (MolDatasetConfig): The configuration object containing counts and limits.
        atom_feature_keys (Sequence[str]): The list of features to extract for the atoms.

    Returns:
        Tuple[List, Dict]: A list of generated graph items (tuples of tensors)
        and a metadata dictionary containing generation stats.
    """
    rng = random.Random(seed)
    eneg = _electronegativity()
    fragment_mols = []
    for s in fragments:
        m = Chem.MolFromSmiles(s)
        if m:
            try:
                Chem.SanitizeMol(m); fragment_mols.append(m)
            except:
                pass

    items = []
    smiles_seen = set()
    attempts = 0
    while len(items) < cfg.num_graphs and attempts < cfg.max_mol_attempts:
        attempts += 1
        mol = _build_random_molecule(fragment_mols, rng, cfg.min_atoms, cfg.max_atoms, cfg.max_stitch_attempts)
        if not mol:
            continue

        smi = Chem.MolToSmiles(mol, canonical=True)
        if smi in smiles_seen:
            continue
        smiles_seen.add(smi)

        A, feats, L, mask = _mol_to_graph_tensors(mol, eneg, atom_feature_keys)

        if mask.any():
            items.append((A, feats, L, mask))

        if len(items) % 200 == 0:
            print(f"[data] generated {len(items)}/{cfg.num_graphs} molecules...")

    if len(items) < cfg.num_graphs:
        raise RuntimeError(f"Only generated {len(items)}/{cfg.num_graphs} graphs.")
    return items, {"generated": len(items), "attempts": attempts}


class MolBenchmark(GraphBenchmark):
    def __init__(self, ds_cfg, fragments, atom_feature_keys):
        super().__init__()
        self.ds_cfg = ds_cfg
        self.fragments = fragments
        self.atom_feature_keys = atom_feature_keys
        self.metadata = {}

    def generate_dataset(self, specs, hooks=None, prepackage=True, directed=False, seed=None):
        items, meta = _generate_items(self.fragments, self.ds_cfg, self.atom_feature_keys, seed=seed)
        self.metadata.update(meta)
        return items


# Choose the feature mode and prepare the hooks
feature_mode = "default"  # Switch to "ablation" for limited feature range

# Declare custom atom features alongside canonical structural features
atom_keys = atom_feature_keys_for_mode(feature_mode)
canonical_structural = ["degree", "twohop", "triangles"]
hooks = TaskHooks(feature_set=[(k, "node") for k in atom_keys] + canonical_structural)

# Create the task object using an 80/10/10 split across 1000 task-specific graphs with 6-140 nodes each
ds_cfg = MolDatasetConfig(num_graphs=1000,
                          train_count=800, val_count=100, test_count=100,
                          min_atoms=6, max_atoms=140)

total = float(ds_cfg.num_graphs)
bench = MolBenchmark(ds_cfg, _default_fragments(), atom_keys)
task = ProvidedSplitsTask(
    name="mol_bond_inference",
    directed=False,
    hooks=hooks,
    num_classes=4,
    eval_on_existing_edges_only=True,
    bench=bench,
    ratios=(ds_cfg.train_count / total, ds_cfg.val_count / total, ds_cfg.test_count / total)
)

# Define the config used by the TNN pipeline
dense_cfg = TNNTrainConfig(
    batch_size=32,
    epochs=80,
    lr=1e-3,
    weight_decay=1e-4,
    grad_clip=1.0,
    tx_force_adj_channel=False,
    early_stop_patience=10
)

# Run the TNN pipeline on all five models
MODELS = ["mlp", "deep_mlp", "cnn", "transformer", "rf"]
results = run_pipeline_for_task(task, MODELS, dense_cfg)

# Define the config used by the GNN pipeline
gnn_cfg = GNNTrainConfig(
    epochs=60,
    batch_size=dense_cfg.batch_size
)

# Run the GNN pipeline on all four models
run_gnn_suite(
    task=task,
    encoders=["gcn", "sage", "gin", "edge_tx", "gps"],
    cfg=gnn_cfg
)

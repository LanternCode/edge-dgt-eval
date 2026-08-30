import contextlib
import json
import math
import os
import random
import h5py
import numpy as np
import scipy.sparse as sp
import torch
from pipeline.EdgeClassification import (ProvidedSplitsTask, TNNTrainConfig,
                                TaskHooks, effective_mask, run_pipeline_for_task)
from pipeline.GraphBenchmark import GraphBenchmark
from pipeline.GNNBridge import GNNTrainConfig, run_gnn_suite
from scipy.sparse import linalg as spla
from torch_geometric.data import Data
from dataclasses import dataclass, field
from hashlib import sha1
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ACMKnobs:
    """
    Configuration for the Algebraic Connectivity Maximisation (ACM) task.

    Parameters
    ----------
    data_dir: Root directory containing the PowerGraph datasets (each graph has a `raw/blist.mat`).
    k_add: Number of candidate edges to add when evaluating a model's predicted solution quality.
    k_train: Number of "positive" edges used to create the supervision labels for training graphs.
    candidate_limit: Maximum number of candidate non-edges sampled per graph.
    mc_min_combos: If C(n, k_add) is larger than this threshold, skip exhaustive search and use Monte Carlo.
    beam_size: Beam size for the approximate optimal search.
    beam_candidates_per_step: How many top candidate edges to consider at each beam step.
    enumerate_cut: Maximum number of combinations allowed for exhaustive enumeration (hard cutoff).
    cache_path: Path to a JSON cache file storing previously computed optimal values.
    """
    data_dir: str = "data/powergraph"
    k_add: int = 5
    k_train: int = 20
    candidate_limit: int = 1000
    mc_min_combos: int = 1_000_000
    beam_size: int = 8192
    beam_candidates_per_step: int = 64
    enumerate_cut: int = 200_000
    cache_path: str = "data/powergraph/acm_opt_cache.json"


KNOBS = ACMKnobs()


# =====================================================================
# 1. Graph math helpers
# =====================================================================
def _laplacian_lambda2(A: np.ndarray) -> float:
    """
    Compute the algebraic connectivity λ2 (2nd-smallest Laplacian eigenvalue).

    Parameters
    ----------
    A: Dense adjacency matrix of shape (N, N). Values are treated as edge weights,
        but this task typically uses binary 0/1 adjacency.

    Returns
    -------
    float
        The second-smallest eigenvalue of the (combinatorial) Laplacian L = D - A.
        Returns 0.0 if the computation fails or N < 2.
    """
    n = int(A.shape[0])
    if n < 2:
        return 0.0

    deg = np.asarray(A, dtype=np.float64).sum(axis=1)
    L = sp.diags(deg, 0, shape=(n, n)) - sp.csr_matrix(A.astype(np.float64))

    try:
        vals = spla.eigsh(L, k=2, which="SM", return_eigenvectors=False)
        vals = np.sort(np.asarray(vals, dtype=np.float64))
        return float(vals[1]) if vals.size >= 2 else 0.0
    except Exception:
        return 0.0


def _node_features_degree_fiedler(A: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute node-level features used by this task (degree and Fiedler vector proxy).

    Parameters
    ----------
    A: Dense adjacency matrix of shape (N, N).

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with:
          - "degree": float32 vector of shape (N,)
          - "fiedler": float32 vector of shape (N,)
        If the Fiedler computation fails, "fiedler" falls back to zeros.
    """
    n = int(A.shape[0])
    deg = np.asarray(A, dtype=np.float64).sum(axis=1).astype(np.float32)

    fiedler = np.zeros((n,), dtype=np.float32)
    if n >= 2:
        d = np.asarray(A, dtype=np.float64).sum(axis=1)
        L = sp.diags(d, 0, shape=(n, n)) - sp.csr_matrix(A.astype(np.float64))
        try:
            vals, vecs = spla.eigsh(L, k=2, which="SM")
            order = np.argsort(vals)
            f = np.asarray(vecs[:, order[1]], dtype=np.float32).reshape(-1)
            fiedler = f
        except Exception:
            pass

    return {"degree": deg, "fiedler": fiedler}


def _apply_edges(A: np.ndarray, edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    Return a new adjacency with additional undirected edges applied.

    Parameters
    ----------
    A: Base dense adjacency matrix of shape (N, N).
    edges: List of edges (u, v) with 0 <= u, v < N. Edges are treated as undirected.

    Returns
    -------
    np.ndarray
        A copy of A (float32) with A[u, v] and A[v, u] set to 1.0 for all provided edges.
    """
    out = np.array(A, copy=True, dtype=np.float32)
    for u, v in edges:
        out[u, v] = 1.0
        out[v, u] = 1.0
    return out


# =====================================================================
# 2. "Optimal" search helpers (used for reporting upper bound)
# =====================================================================
def _exhaustive_best(A: np.ndarray, C: List[Tuple[int, int]], k_add: int) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Compute the optimal λ2 by exhaustive enumeration over candidate edges.

    Parameters
    ----------
    A: Base adjacency matrix (N, N).
    C: Candidate edge list (non-edges) as (u, v) pairs.
    k_add: Number of edges to add.

    Returns
    -------
    Tuple[float, List[Tuple[int, int]]]
        Best λ2 achieved and the corresponding set of k_add edges.
    """
    best_val = -1.0
    best_set: List[Tuple[int, int]] = []
    for subset in combinations(C, k_add):
        val = _laplacian_lambda2(_apply_edges(A, list(subset)))
        if val > best_val:
            best_val = val
            best_set = list(subset)
    return float(best_val), best_set


def _monte_carlo_best(
    A: np.ndarray,
    C: List[Tuple[int, int]],
    k_add: int,
    n_samples: int,
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Approximate the best λ2 by Monte Carlo sampling of edge subsets.

    Parameters
    ----------
    A: Base adjacency matrix (N, N).
    C: Candidate edge list.
    k_add: Number of edges to add.
    n_samples: Number of random subsets to evaluate.

    Returns
    -------
    Tuple[float, List[Tuple[int, int]]]
        Best λ2 found among sampled subsets and the corresponding edge subset.
    """
    best_val = -1.0
    best_set: List[Tuple[int, int]] = []
    if k_add <= 0 or len(C) < k_add:
        return 0.0, []

    for _ in range(int(n_samples)):
        subset = random.sample(C, k_add)
        val = _laplacian_lambda2(_apply_edges(A, subset))
        if val > best_val:
            best_val = val
            best_set = list(subset)
    return float(best_val), best_set


def _beam_search_best(
    A: np.ndarray,
    C: List[Tuple[int, int]],
    k_add: int,
    beam_size: int,
    per_step_candidates: int,
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Approximate the best λ2 using a simple beam search.

    Parameters
    ----------
    A: Base adjacency matrix (N, N).
    C: Candidate edge list.
    k_add: Number of edges to add.
    beam_size: Maximum number of partial solutions retained at each step.
    per_step_candidates: Number of candidate edges considered for expansion at each step.

    Returns
    -------
    Tuple[float, List[Tuple[int, int]]]
        Best λ2 achieved by the beam search and the corresponding edge subset.
    """
    if k_add <= 0 or len(C) < k_add:
        return 0.0, []

    base = _laplacian_lambda2(A)
    gains: List[Tuple[float, Tuple[int, int]]] = []
    for (i, j) in C:
        gain = _laplacian_lambda2(_apply_edges(A, [(i, j)])) - base
        gains.append((float(gain), (i, j)))
    gains.sort(key=lambda x: x[0], reverse=True)
    top = [e for _, e in gains[: max(1, int(per_step_candidates))]]

    beams: List[Tuple[float, List[Tuple[int, int]]]] = [(base, [])]
    for _step in range(int(k_add)):
        next_beams: List[Tuple[float, List[Tuple[int, int]]]] = []
        for _score, chosen in beams:
            chosen_set = set(chosen)
            for e in top:
                if e in chosen_set:
                    continue
                new = chosen + [e]
                val = _laplacian_lambda2(_apply_edges(A, new))
                next_beams.append((float(val), new))

        next_beams.sort(key=lambda x: x[0], reverse=True)
        beams = next_beams[: max(1, int(beam_size))]

    return float(beams[0][0]), list(beams[0][1])


@dataclass
class OptCache:
    """
    A JSON-backed cache that automatically saves changes when used as a context manager.

    Parameters
    ----------
    path: Path to the JSON cache file on disk.
    """
    path: Path
    data: Dict[str, Any] = field(default_factory=dict)
    dirty: bool = False

    def __post_init__(self) -> None:
        if self.path.exists():
            with contextlib.suppress(Exception):
                self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, key: str) -> Optional[Any]:
        return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.dirty = True

    def flush(self) -> None:
        if self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
            self.dirty = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()


def _hash_state(A: np.ndarray, C: List[Tuple[int, int]], k_add: int) -> str:
    """
    Create a stable hash key for a particular optimization instance.

    Parameters
    ----------
    A: Base adjacency matrix (N, N).
    C: Candidate edge list.
    k_add: Number of edges to add.

    Returns
    -------
    str
        SHA1 digest string representing this instance.
    """
    h = sha1()
    h.update(np.asarray(A, dtype=np.uint8).tobytes())
    h.update(str(int(k_add)).encode("utf-8"))
    for u, v in C:
        h.update(f"{int(u)}-{int(v)};".encode("utf-8"))
    return h.hexdigest()


def _run_optimal_with_cache(
    graph_name: str,
    A: np.ndarray,
    C: List[Tuple[int, int]],
    k_add: int,
    cache: OptCache,
) -> Tuple[float, List[Tuple[int, int]], str]:
    """
    Compute (or load) the "optimal" λ2 for reporting, using a cache.

    Parameters
    ----------
    graph_name:
        Human-readable graph identifier (used only for logging/caching metadata).
    A: Base adjacency matrix (N, N).
    C: Candidate edge list.
    k_add: Number of edges to add.
    cache: OptCache instance used to memoize results.

    Returns
    -------
    Tuple[float, List[Tuple[int, int]], str]
        (best_lambda2, best_edges, method_name), where method_name is one of:
        {"exhaustive", "monte_carlo", "beam"}.
    """
    key = _hash_state(A, C, k_add)
    hit = cache.get(key)

    if isinstance(hit, dict) and "best_l2" in hit and "edges" in hit and "method" in hit:
        return float(hit["best_l2"]), [tuple(x) for x in hit["edges"]], str(hit["method"])

    combos = math.comb(len(C), k_add) if (0 <= k_add <= len(C)) else 0
    if 0 < combos <= KNOBS.enumerate_cut:
        best_l2, best_edges = _exhaustive_best(A, C, k_add)
        method = "exhaustive"
    elif combos >= KNOBS.mc_min_combos:
        best_l2, best_edges = _monte_carlo_best(A, C, k_add, n_samples=50000)
        method = "monte_carlo"
    else:
        best_l2, best_edges = _beam_search_best(
            A,
            C,
            k_add,
            beam_size=KNOBS.beam_size,
            per_step_candidates=KNOBS.beam_candidates_per_step,
        )
        method = "beam"

    cache.set(key,{"graph": graph_name, "best_l2": float(best_l2), "edges": [list(e) for e in best_edges], "method": method})
    return float(best_l2), list(best_edges), method


# =====================================================================
# 3. Data loading & task construction
# =====================================================================
def load_graph_from_blist(path: str) -> Data:
    """
    Load a graph from an HDF5 `.mat` file containing the `bList` edge list.

    Parameters
    ----------
    path:
        File path to `blist.mat` (HDF5-backed). The dataset key is expected to be `bList`
        and edges are expected to be 1-indexed.

    Returns
    -------
    torch_geometric.data.Data
        PyG Data object with:
          - edge_index: LongTensor of shape (2, E) in 0-indexed node IDs
          - num_nodes: int inferred from the maximum node ID in edge_index
    """
    with h5py.File(path, "r") as f:
        edge_list = f["bList"][:]
    edge_index = torch.tensor(edge_list, dtype=torch.long) - 1
    if edge_index.dim() == 2 and edge_index.size(0) != 2:
        edge_index = edge_index.t().contiguous()
    num_nodes = int(edge_index.max().item()) + 1
    return Data(edge_index=edge_index, num_nodes=num_nodes)


def get_candidates(data: Data, limit: int) -> List[Tuple[int, int]]:
    """
    Sample candidate non-edges for a graph.

    Parameters
    ----------
    data: PyG Data with `edge_index` and `num_nodes`.
    limit: Maximum number of candidate pairs to return.

    Returns
    -------
    List[Tuple[int, int]]
        List of unique undirected node pairs (u, v) with u < v, sampled uniformly at random
        from non-edges. If sampling can't find enough unique pairs, returns fewer than `limit`.
    """
    n = int(data.num_nodes)
    edges = set(map(tuple, data.edge_index.t().tolist()))
    cands: List[Tuple[int, int]] = []
    attempts = 0
    while len(cands) < int(limit) and attempts < int(limit) * 20:
        u, v = random.randint(0, n - 1), random.randint(0, n - 1)
        if u != v and (u, v) not in edges and (v, u) not in edges:
            if u > v:
                u, v = v, u
            pair = (u, v)
            if pair not in cands:
                cands.append(pair)
        attempts += 1
    return cands


def _label_matrix_top_m(A: np.ndarray, C: List[Tuple[int, int]], top_m: int) -> np.ndarray:
    """
    Create a label matrix by marking the top-M candidates that most increase λ2.

    Parameters
    ----------
    A: Base adjacency matrix (N, N).
    C: Candidate edge list.
    top_m: Number of positives to mark in the label matrix.

    Returns
    -------
    np.ndarray
        Integer label matrix L of shape (N, N) with L[u, v] = L[v, u] = 1 for the
        top-M candidates (by single-edge λ2 gain). All other entries are 0.
    """
    base = _laplacian_lambda2(A)
    scores: List[Tuple[float, Tuple[int, int]]] = []
    for (i, j) in C:
        gain = _laplacian_lambda2(_apply_edges(A, [(i, j)])) - base
        scores.append((float(gain), (i, j)))
    scores.sort(key=lambda x: x[0], reverse=True)

    L = np.zeros_like(A, dtype=np.int64)
    for _, (i, j) in scores[: int(top_m)]:
        L[i, j] = 1
        L[j, i] = 1
    return L


def _dense_adj_from_pyg(data: Data) -> np.ndarray:
    """
    Convert a PyG edge_index graph into a dense undirected adjacency.

    Parameters
    ----------
    data: PyG Data with `edge_index` and `num_nodes`.

    Returns
    -------
    np.ndarray
        Dense float32 adjacency of shape (N, N) with symmetric 0/1 entries.
    """
    n = int(data.num_nodes)
    A = np.zeros((n, n), dtype=np.float32)
    ei = data.edge_index.detach().cpu().numpy()
    A[ei[0], ei[1]] = 1.0
    A[ei[1], ei[0]] = 1.0
    return A


def _candidate_mask(n: int, C: List[Tuple[int, int]]) -> np.ndarray:
    """
    Build a symmetric boolean mask for candidate edges.

    Parameters
    ----------
    n: Number of nodes.
    C: Candidate edge list as (u, v) with u < v.

    Returns
    -------
    np.ndarray
        Boolean mask M of shape (n, n) where M[u, v] and M[v, u] are True for candidates.
    """
    mask = np.zeros((int(n), int(n)), dtype=bool)
    for u, v in C:
        mask[u, v] = True
        mask[v, u] = True
    return mask


def _make_sample(
    graph: Data,
    candidates: List[Tuple[int, int]],
    k_train: int,
    is_test: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Create a single pipeline sample tuple (A, feats, L, mask) for one graph.

    Parameters
    ----------
    graph: PyG Data object representing the graph topology.
    candidates: Candidate non-edge list used to define the supervision/evaluation mask.
    k_train: Number of positives to assign in L for non-test graphs.
    is_test: If True, produces an all-zero label matrix (to match the original script behavior).

    Returns
    -------
    Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]
        - A: float32 Tensor (N, N) dense adjacency
        - feats: dict of float32 node features (each Tensor shape (N,))
        - L: float32 Tensor (N, N) label matrix
        - mask: bool Tensor (N, N) specifying which (i, j) pairs are supervised/evaluated
    """
    A = _dense_adj_from_pyg(graph)
    feats_np = _node_features_degree_fiedler(A)
    feats_t = {k: torch.from_numpy(v).float() for k, v in feats_np.items()}

    mask_np = _candidate_mask(A.shape[0], candidates)

    L_np = np.zeros((A.shape[0], A.shape[0]), dtype=np.float32)
    if not bool(is_test):
        L_np = _label_matrix_top_m(A, candidates, top_m=int(k_train)).astype(np.float32)

    return torch.from_numpy(A), feats_t, torch.from_numpy(L_np), torch.from_numpy(mask_np)


class ACMBenchmark(GraphBenchmark):
    def __init__(self, knobs: ACMKnobs = KNOBS):
        super().__init__()
        graph_names = ["ieee24", "ieee39", "ieee118", "uk"]
        paths = {n: os.path.join(knobs.data_dir, n, n, "raw", "blist.mat") for n in graph_names}

        graphs: Dict[str, Data] = {}
        candidates: Dict[str, List[Tuple[int, int]]] = {}
        for name, p in paths.items():
            if os.path.exists(p):
                graphs[name] = load_graph_from_blist(p)
                candidates[name] = get_candidates(graphs[name], knobs.candidate_limit)

        if not graphs:
            raise FileNotFoundError(
                f"No graphs found under {knobs.data_dir!r}. Expected subfolders like ieee24/.../raw/blist.mat."
            )

        test_name = "uk" if "uk" in graphs else list(graphs.keys())[-1]
        train_names = [n for n in graphs if n != test_name]

        train_samples = [
            _make_sample(graphs[n], candidates[n], knobs.k_train, is_test=False)
            for n in train_names
        ]
        val_samples   = [train_samples[-1]]
        train_samples = train_samples[:-1]
        test_samples  = [_make_sample(graphs[test_name], candidates[test_name], knobs.k_train, is_test=False)]

        self.splits = {"train": train_samples, "val": val_samples, "test": test_samples}

        A_test = _dense_adj_from_pyg(graphs[test_name])
        self.eval_bundle = {
            "test_name": test_name,
            "A_test":    A_test,
            "C_test":    candidates[test_name],
            "k_add":     int(knobs.k_add),
        }


# =====================================================================
# 4. Post-hoc evaluation: λ2 after adding top-k predicted edges
# =====================================================================
def compute_model_lambda2(bundle: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute final λ2 for each model by adding its top-k predicted candidate edges.

    This uses `_prob` from each model's test output and selects the top `k_add` edges
    among the candidate mask entries.

    Parameters
    ----------
    bundle: The evaluation bundle dict containing keys {"A_test", "C_test", "k_add"}.
    results: Dictionary mapping model_name -> result bundle (as returned by the pipeline/GNN suite).
        Each result should contain `result["test"]["_prob"]`.

    Returns
    -------
    Dict[str, float]
        Mapping model_name -> λ2 of the graph after adding the top-k predicted edges.
        Models without probabilities get a score of 0.0.
    """
    A_base = np.asarray(bundle["A_test"], dtype=np.float32)
    C_test = list(bundle["C_test"])
    k_add = int(bundle["k_add"])
    final_scores: Dict[str, float] = {}

    for model_name, res in results.items():
        test_res = res.get("test", {}) if isinstance(res, dict) else {}
        probs = test_res.get("_prob")

        if not probs:
            final_scores[str(model_name)] = 0.0
            continue

        probs_arr = np.asarray(probs, dtype=np.float64)

        mask_t = torch.from_numpy(_candidate_mask(A_base.shape[0], C_test))
        A_t = torch.from_numpy(A_base).float()
        m = effective_mask(mask_t.unsqueeze(0), A_t.unsqueeze(0), directed=False).squeeze(0).cpu().numpy()

        rows, cols = np.where(m)
        flat_indices = list(zip(rows.tolist(), cols.tolist()))

        edge_scores: Dict[Tuple[int, int], float] = {}
        if len(probs_arr) == len(flat_indices):
            for i, (r, c) in enumerate(flat_indices):
                u, v = (r, c) if r < c else (c, r)
                edge_scores[(u, v)] = max(edge_scores.get((u, v), -1.0), float(probs_arr[i]))

        cands_with_scores: List[Tuple[float, Tuple[int, int]]] = []
        for u, v in C_test:
            pair = (u, v) if u < v else (v, u)
            if pair in edge_scores:
                cands_with_scores.append((edge_scores[pair], pair))

        cands_with_scores.sort(key=lambda x: x[0], reverse=True)
        top_k = [x[1] for x in cands_with_scores[: int(k_add)]]

        A_final = _apply_edges(A_base, top_k)
        final_scores[str(model_name)] = _laplacian_lambda2(A_final)

    return final_scores


def print_lambda2_summary(l2_scores: Dict[str, float], base_l2: float, opt_l2: float, test_name: str) -> None:
    """
    Pretty-print the final ACM comparison table.

    Parameters
    ----------
    l2_scores: Mapping from model key to final λ2 value.
    base_l2: λ2 of the base test graph (no added edges).
    opt_l2: λ2 of the (approx/true) optimal selection of k edges (upper bound for gain).
    test_name: Name of the held-out test graph used in the report.

    Returns
    -------
    None
        Prints a table to stdout.
    """
    model_display_map = {
        "mlp": "MLP",
        "deep_mlp": "Deep MLP",
        "sage": "GraphSAGE",
        "rf": "Random Forest",
        "gin": "GIN",
        "cnn": "CNN",
        "gcn": "GCN",
        "transformer": "Transformer",
        "edge_tx": "Graph Transformer",
        "gps": "GPS",
        "greedy": "Greedy Algorithm*"
    }

    order = ["mlp", "deep_mlp", "sage", "rf", "gin", "cnn", "gcn", "transformer", "edge_tx", "gps", "greedy"]

    print("\n" + "=" * 80)
    print("FINAL SUMMARY: Algebraic Connectivity Maximisation")
    print(f"Test Graph: {test_name} | Base λ2: {base_l2:.4f} | Optimal λ2: {opt_l2:.4f}")
    print("=" * 80)
    print(f"{'Model':<20} {'Base λ2':<10} {'Final λ2':<10} {'Optimal λ2':<10} {'Model Gain %':<12}")
    print("-" * 70)

    for key in order:
        if key not in l2_scores:
            continue
        display_name = model_display_map.get(key, key)
        final_val = float(l2_scores[key])
        gain = (final_val - base_l2) / (opt_l2 - base_l2) * 100.0 if opt_l2 > base_l2 else 0.0
        print(f"{display_name:<20} {base_l2:<10.4f} {final_val:<10.4f} {opt_l2:<10.4f} {gain:<12.1f}%")
    print("=" * 80)


print("[1/5] Building Task and Datasets...")
bench = ACMBenchmark(KNOBS)
bundle = bench.eval_bundle

hooks = TaskHooks(
    label_fn=None,
    feature_set=[
        "degree", ("fiedler", "node"), "endpoint_degree",
        "deg_diff", "clustering_coeff", "cn",
        "jaccard", "adamic_adar"
    ],
    allow_adj_channel=True
)
task = ProvidedSplitsTask(name="ACM_Fix_V2", directed=False, hooks=hooks, bench=bench)
task.batch_size = 16

base_l2 = _laplacian_lambda2(bundle["A_test"])

print(f"Running Optimal Search (Beam Search k={KNOBS.k_add})...")
with OptCache(Path(KNOBS.cache_path)) as cache:
    opt_l2, _, method = _run_optimal_with_cache(
        bundle["test_name"], bundle["A_test"], bundle["C_test"], KNOBS.k_add, cache
    )
print(f"Found Optimal λ2: {opt_l2:.4f} (method: {method})")

print("Computing Greedy Baseline...")
greedy_l2, _ = _monte_carlo_best(bundle["A_test"], bundle["C_test"], KNOBS.k_add, n_samples=5000)

# Define the config used by the TNN pipeline
print("[2/5] Running Matrix Models Pipeline...")
matrix_cfg = TNNTrainConfig(
    epochs=30,
    lr=1e-3,
    use_mask_channel=True
)

# Run the TNN pipeline on all five models
#matrix_models = ["mlp", "deep_mlp", "cnn", "transformer", "rf"]
matrix_models = []
pipeline_out = run_pipeline_for_task(task, matrix_models, matrix_cfg)

# Define the config used by the GNN pipeline
print("\n[3/5] Running GNN Suite...")
gnn_cfg = GNNTrainConfig(
    epochs=30,
    lr=5e-3,
    hidden=64
)

# Run the GNN pipeline on all five models
#gnn_encoders = ["gcn", "sage", "gin", "edge_tx"]
gnn_encoders = ["gps"]
gnn_out = run_gnn_suite(task, gnn_encoders, gnn_cfg)

print("\n[4/5] Computing Final Metrics...")
all_results: Dict[str, Any] = {}
if isinstance(pipeline_out, dict) and "results" in pipeline_out:
    all_results.update(pipeline_out["results"])
if isinstance(gnn_out, dict) and "results" in gnn_out:
    all_results.update(gnn_out["results"])

l2_scores = compute_model_lambda2(bundle, all_results)
l2_scores["greedy"] = float(greedy_l2)

print_lambda2_summary(l2_scores, float(base_l2), float(opt_l2), str(bundle["test_name"]))

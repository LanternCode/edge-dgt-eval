import os
import gzip
import urllib.request
import numpy as np
import networkx as nx
import argparse
from pipeline.EdgeClassification import (TaskHooks, TNNTrainConfig, run_pipeline_for_task,
                                FeatureRegistry, ProvidedSplitsTask)
from pipeline.GNNBridge import GNNTrainConfig, run_gnn_edges_suite
from pipeline.GraphBenchmark import GraphBenchmark
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from torch.utils.data import Dataset

HIGGS_URL = "https://snap.stanford.edu/data/higgs-retweet_network.edgelist.gz"
HIGGS_DIR = os.path.join("data", "higgs_retweet")
HIGGS_GZ = os.path.join(HIGGS_DIR, "higgs-retweet_network.edgelist.gz")
HIGGS_TXT = os.path.join(HIGGS_DIR, "higgs-retweet_network.edgelist")


def _ensure_higgs_local() -> str:
    """Download & unzip HIGGS if needed; return path to the edgelist text file."""
    os.makedirs(HIGGS_DIR, exist_ok=True)
    if os.path.exists(HIGGS_TXT):
        return HIGGS_TXT
    if not os.path.exists(HIGGS_GZ):
        print(f"[HIGGS] downloading {HIGGS_URL} …")
        urllib.request.urlretrieve(HIGGS_URL, HIGGS_GZ)
    print("[HIGGS] decompressing …")
    with gzip.open(HIGGS_GZ, "rb") as fin, open(HIGGS_TXT, "wb") as fout:
        fout.write(fin.read())
    os.remove(HIGGS_GZ)
    return HIGGS_TXT


def _load_directed_graph(path: str) -> nx.DiGraph:
    """Read u v lines into a directed graph; drop self-loops."""
    G = nx.DiGraph()
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u != v:
                G.add_edge(u, v)
    return G


@dataclass
class HiggsRetweetEdgesTask:
    """
    Full-graph (no NODE_CAP) sparse task for edge mini-batch training.
    """
    name: str = "HIGGS_Retweet_SparseEdges"

    # Fixed parameter for negative sampling ratio (negatives per positive)
    neg_pos_ratio: float = 1.0

    def __post_init__(self) -> None:
        txt = _ensure_higgs_local()
        G = _load_directed_graph(txt)

        # --- Nodes and reindexing ---
        nodes = sorted(G.nodes())
        n2i = {n: i for i, n in enumerate(nodes)}
        N = len(nodes)
        if N == 0 or G.number_of_edges() == 0:
            raise RuntimeError("Empty graph or no edges.")

        # --- Positive edges (directed) ---
        pos = np.array([(n2i[u], n2i[v]) for (u, v) in G.edges() if u != v], dtype=np.int64)
        if pos.size == 0:
            raise RuntimeError("Graph has no positive edges after reindexing.")

        # --- Sparse COO representation ---
        rows = pos[:, 0].copy()
        cols = pos[:, 1].copy()

        # --- CSR-like row index for fast window extraction ---
        order = np.lexsort((cols, rows))  # sort by row, then col
        rows_s = rows[order]
        cols_s = cols[order]

        row_ptr = np.zeros(N + 1, dtype=np.int64)
        np.add.at(row_ptr, rows_s + 1, 1)
        np.cumsum(row_ptr, out=row_ptr)

        self._csr_row_ptr = row_ptr
        self._csr_cols = cols_s

        # --- Negative sampling (global), ratio controlled by env ---
        neg_ratio = float(self.neg_pos_ratio)
        rng = np.random.default_rng()
        need_neg = int(max(1, round(neg_ratio * pos.shape[0])))
        pos_set = set((int(i), int(j)) for i, j in pos)
        neg = []
        B = max(100_000, 5 * need_neg // 3)
        while len(neg) < need_neg:
            ui = rng.integers(0, N, size=B, dtype=np.int64)
            vi = rng.integers(0, N, size=B, dtype=np.int64)
            keep = (ui != vi)
            ui, vi = ui[keep], vi[keep]
            for a, b in zip(ui, vi):
                if len(neg) >= need_neg:
                    break
                p = (int(a), int(b))
                if p not in pos_set:
                    neg.append(p)
        neg = np.asarray(neg, dtype=np.int64)

        # --- Splits: 60/20/20 for pos; mirror proportions for neg ---
        def _split_pairs(P: np.ndarray):
            rng = np.random.default_rng()
            idx = rng.permutation(P.shape[0])
            p_tr = int(round(0.6 * len(idx)))
            p_va = int(round(0.2 * len(idx)))
            tr = P[idx[:p_tr]]
            va = P[idx[p_tr:p_tr + p_va]]
            te = P[idx[p_tr + p_va:]]
            return tr, va, te

        P_tr, P_va, P_te = _split_pairs(pos)
        N_tr, N_va, N_te = _split_pairs(neg)

        # --- Cache for gnn_bridge ---
        self._N = int(N)
        self._rows = rows
        self._cols = cols
        self._edges_tr = {"pos": P_tr, "neg": N_tr}
        self._edges_va = {"pos": P_va, "neg": N_va}
        self._edges_te = {"pos": P_te, "neg": N_te}

        print(
            f"[HIGGS-SPARSE] nodes: {N:,} | pos: {pos.shape[0]:,} | neg (ratio={self.neg_pos_ratio}): {neg.shape[0]:,}\n"
            f"  splits (pos): {len(P_tr):,}/{len(P_va):,}/{len(P_te):,} | (neg): {len(N_tr):,}/{len(N_va):,}/{len(N_te):,}"
        )

    def node_feature_dict(self, A_repr) -> Dict[str, np.ndarray]:
        """Cheap global node features: out/in degree as 1D arrays (N,)."""
        N = self._N
        rows = self._rows
        cols = self._cols
        out_deg = np.bincount(rows, minlength=N).astype(np.float32)
        in_deg = np.bincount(cols, minlength=N).astype(np.float32)
        return {"out_degree": out_deg, "in_degree": in_deg}


class ECTileDataset(Dataset):
    """
    Builds a dense P×P adjacency tile around a (u,v) pair from a large sparse directed graph.

    Uses CSR-style row pointers to avoid scanning all edges per sample.
    Supervision is restricted to the centre cell via the mask.
    """
    def __init__(
            self,
            N: int,
            row_ptr: np.ndarray,
            cols_sorted: np.ndarray,
            edges: Dict[str, np.ndarray],
            P: int = 384,
            node_feats: Optional[Dict[str, np.ndarray]] = None,
            node_feat_keys: Optional[Tuple[str, ...]] = None,
    ):
        self.N = int(N)
        self.row_ptr = row_ptr.astype(np.int64, copy=False)
        self.cols_sorted = cols_sorted.astype(np.int64, copy=False)
        self.P = int(P)

        pos = edges["pos"].astype(np.int64, copy=False)
        neg = edges["neg"].astype(np.int64, copy=False)
        self.pairs = np.concatenate([pos, neg], axis=0)

        self.node_feats = node_feats or {}
        self.node_feat_keys = node_feat_keys if node_feat_keys is not None else tuple(self.node_feats.keys())

    def __len__(self) -> int:
        return int(self.pairs.shape[0])

    def __getitem__(self, i):
        u, v = int(self.pairs[i, 0]), int(self.pairs[i, 1])
        N, P = self.N, self.P

        if N <= P:
            r0 = 0
            c0 = 0
        else:
            r0 = min(max(0, u - P // 2), N - P)
            c0 = min(max(0, v - P // 2), N - P)

        r1, c1 = r0 + P, c0 + P

        # Densify window using CSR row slices (equivalent to boolean-scanning + scatter)
        A = np.zeros((P, P), dtype=np.float32)
        for r in range(r0, r1):
            s = int(self.row_ptr[r])
            e = int(self.row_ptr[r + 1])
            if e <= s:
                continue

            cs = self.cols_sorted[s:e]  # sorted cols in this row
            lo = np.searchsorted(cs, c0, side="left")
            hi = np.searchsorted(cs, c1, side="left")
            if hi > lo:
                A[r - r0, cs[lo:hi] - c0] = 1.0

        # Labels = current observed edges in the tile; mask supervises only the centre pair
        L = (A > 0).astype(np.float32)
        M = np.zeros((P, P), dtype=bool)
        M[u - r0, v - c0] = True

        feats = {}
        for k in self.node_feat_keys:
            x = self.node_feats[k][r0:r1]
            x = np.asarray(x)
            if x.ndim == 2 and x.shape[1] == 1:
                x = x[:, 0]
            if x.ndim != 1:
                raise RuntimeError(f"Node feature {k} must be 1D (N,), got shape {x.shape}")
            feats[k] = x.astype(np.float32, copy=False)

        return A, feats, L, M


class HiggsRetweetTileBench(GraphBenchmark):
    def __init__(self, task: HiggsRetweetEdgesTask, tile_size: int = 384):
        super().__init__()
        node_feats = task.node_feature_dict(None)
        self.node_feat_keys = tuple(node_feats.keys())
        FeatureRegistry.register_custom_features(list(self.node_feat_keys))
        self.splits = {
            k: ECTileDataset(task._N, task._csr_row_ptr, task._csr_cols, edges,
                             P=tile_size, node_feats=node_feats, node_feat_keys=self.node_feat_keys)
            for k, edges in [("train", task._edges_tr), ("val", task._edges_va), ("test", task._edges_te)]
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HIGGS edge classification tasks.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["mlp", "deep_mlp", "cnn", "transformer", "rf", "gcn", "gin", "edge_tx", "sage"],
        help="Which model architecture to run."
    )

    args = parser.parse_args()
    task = HiggsRetweetEdgesTask(neg_pos_ratio=5.0)

    bench = HiggsRetweetTileBench(task)

    hooks = TaskHooks(
        label_fn=None,
        feature_set=[
            "deg_row", "deg_col", "deg_diff",
            "jaccard", "adamic_adar",
            "transpose", "power_2", "power_3"
        ]
    )

    ec_task = ProvidedSplitsTask(
        name="HIGGS_Retweet_Tiles",
        directed=True,
        hooks=hooks,
        bench=bench,
        num_workers=4
    )

    if args.model in ["gcn", "gin", "edge_tx", "sage"]:
        print(f"--- Running GNN Pipeline for model: {args.model} ---")

        gnn_cfg = GNNTrainConfig(
            epochs=50,
            weight_decay=1e-2,
            hidden=256,
            layers=4,
            dropedge_p=0.2,
            pairwise_on_demand=True,
            gnn_zero_supervised=True,
            batch_size=1
        )

        run_gnn_edges_suite(
            task=ec_task,
            encoders=[args.model],
            cfg=gnn_cfg
        )
    else:
        print(f"--- Running EC Pipeline for model: {args.model} ---")

        ec_cfg = TNNTrainConfig(
            epochs=30,
            batch_size=8,
            weight_decay=1e-4,
            tx_force_adj_channel=False,
            early_stop_patience=3,
            rf_neg_pos_ratio=5.0
        )

        # Threshold Metric = "bacc" is required for: CNN, RF, MLP and Deep MLP
        if args.model in ["cnn", "rf", "mlp", "deep_mlp"]:
            print(f"-> Applying config: threshold_metric='bacc' (for {args.model})")
            ec_cfg.threshold_metric = "bacc"

        # Select By Metric = "bacc" is required for: RF, MLP and Deep MLP
        if args.model in ["rf", "mlp", "deep_mlp"]:
            print(f"-> Applying config: select_by='bacc' (for {args.model})")
            ec_cfg.select_by = "bacc"

        bundle = run_pipeline_for_task(task=ec_task, models=[args.model], cfg=ec_cfg)

import os
import gzip
import urllib.request
import numpy as np
import networkx as nx
import argparse
from pipeline.EdgeClassification import TaskHooks, TNNTrainConfig, run_pipeline_for_task, ProvidedSplitsTask
from pipeline.GNNBridge import GNNTrainConfig, run_gnn_edges_suite
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


def _build_csr(N: int, rows: np.ndarray, cols: np.ndarray):
    order = np.lexsort((cols, rows))
    rows_s = rows[order]
    cols_s = cols[order]

    row_ptr = np.zeros(N + 1, dtype=np.int64)
    np.add.at(row_ptr, rows_s + 1, 1)
    np.cumsum(row_ptr, out=row_ptr)
    return row_ptr, cols_s


class HiggsRetweetEdgesTask(ProvidedSplitsTask):
    """
    Full-graph sparse task using local induced subgraphs for edge mini-batch training.
    """
    def __init__(self, neg_pos_ratio: float = 5.0, tile_size: int = 384):
        """Initialise the HIGGS task, target splits, observable graph, and tile benchmark."""
        hooks = TaskHooks(
            label_fn=None,
            feature_set=[
                "deg_row", "deg_col", "cn", "adamic_adar",
                "transpose", ("out_degree", "node"), ("in_degree", "node")
            ]
        )

        super().__init__(
            name="HIGGS_Retweet_Tiles",
            directed=True,
            hooks=hooks,
            num_workers=4
        )

        self.neg_pos_ratio = float(neg_pos_ratio)
        self.tile_size = int(tile_size)

        txt = _ensure_higgs_local()
        G = _load_directed_graph(txt)

        # Nodes and reindexing
        nodes = sorted(G.nodes())
        n2i = {n: i for i, n in enumerate(nodes)}
        N = len(nodes)
        if N == 0 or G.number_of_edges() == 0:
            raise RuntimeError("Empty graph or no edges.")

        # Positive edges
        pos = np.array([(n2i[u], n2i[v]) for (u, v) in G.edges() if u != v], dtype=np.int64)
        if pos.size == 0:
            raise RuntimeError("Graph has no positive edges after reindexing.")

        rng = np.random.default_rng(self.seed)

        # Target splits
        def _split_pairs(P: np.ndarray):
            idx = rng.permutation(P.shape[0])
            p_tr = int(round(0.6 * len(idx)))
            p_va = int(round(0.2 * len(idx)))
            tr = P[idx[:p_tr]]
            va = P[idx[p_tr:p_tr + p_va]]
            te = P[idx[p_tr + p_va:]]
            return tr, va, te

        P_tr, P_va, P_te = _split_pairs(pos)

        # Unique negative candidates
        neg_ratio = float(self.neg_pos_ratio)
        need_neg = int(max(1, round(neg_ratio * pos.shape[0])))
        pos_keys = set((pos[:, 0] * np.int64(N) + pos[:, 1]).tolist())
        neg_keys = []
        neg_set = set()
        B = max(100_000, 5 * need_neg // 3)
        while len(neg_keys) < need_neg:
            ui = rng.integers(0, N, size=B, dtype=np.int64)
            vi = rng.integers(0, N, size=B, dtype=np.int64)
            keep = (ui != vi)
            ui, vi = ui[keep], vi[keep]
            for a, b in zip(ui, vi):
                if len(neg_keys) >= need_neg:
                    break
                key = int(a) * N + int(b)
                if key not in pos_keys and key not in neg_set:
                    neg_set.add(key)
                    neg_keys.append(key)

        neg_keys = np.asarray(neg_keys, dtype=np.int64)
        neg = np.column_stack((neg_keys // N, neg_keys % N)).astype(np.int64, copy=False)
        N_tr, N_va, N_te = _split_pairs(neg)

        # Observable training graph
        train_rows = P_tr[:, 0].copy()
        train_cols = P_tr[:, 1].copy()
        row_ptr, cols_s = _build_csr(N, train_rows, train_cols)
        col_ptr, rows_s = _build_csr(N, train_cols, train_rows)

        self._N = int(N)
        self._csr_row_ptr = row_ptr
        self._csr_cols = cols_s
        self._csc_col_ptr = col_ptr
        self._csc_rows = rows_s
        self._out_degree = np.diff(row_ptr).astype(np.float32)
        self._in_degree = np.diff(col_ptr).astype(np.float32)
        self._edges_tr = {"pos": P_tr, "neg": N_tr}
        self._edges_va = {"pos": P_va, "neg": N_va}
        self._edges_te = {"pos": P_te, "neg": N_te}

        print(
            f"[HIGGS-SPARSE] nodes: {N:,} | pos: {pos.shape[0]:,} | neg (ratio={self.neg_pos_ratio}): {neg.shape[0]:,}\n"
            f"  splits (pos): {len(P_tr):,}/{len(P_va):,}/{len(P_te):,} | (neg): {len(N_tr):,}/{len(N_va):,}/{len(N_te):,}"
        )

        self._bench_instance = HiggsRetweetTileBench(self, tile_size=self.tile_size)

    def node_feature_dict(self, A_repr) -> Dict[str, np.ndarray]:
        """Global node features from the observable training graph."""
        return {
            "out_degree": self._out_degree,
            "in_degree": self._in_degree
        }


class ECTileDataset(Dataset):
    """
    Builds a local induced subgraph around a (u,v) pair from the observable directed graph.

    Supervision is restricted to the target pair via the mask.
    """
    def __init__(
            self,
            N: int,
            row_ptr: np.ndarray,
            cols_sorted: np.ndarray,
            col_ptr: np.ndarray,
            rows_sorted: np.ndarray,
            edges: Dict[str, np.ndarray],
            P: int = 384,
            node_feats: Optional[Dict[str, np.ndarray]] = None,
            node_feat_keys: Optional[Tuple[str, ...]] = None,
    ):
        """Initialise candidate pairs, sparse graph indices, tile size, and node features."""
        self.N = int(N)
        self.row_ptr = row_ptr.astype(np.int64, copy=False)
        self.cols_sorted = cols_sorted.astype(np.int64, copy=False)
        self.col_ptr = col_ptr.astype(np.int64, copy=False)
        self.rows_sorted = rows_sorted.astype(np.int64, copy=False)
        self.P = int(P)

        pos = edges["pos"].astype(np.int64, copy=False)
        neg = edges["neg"].astype(np.int64, copy=False)
        self.pairs = np.concatenate([pos, neg], axis=0)
        self.labels = np.concatenate([
            np.ones(pos.shape[0], dtype=np.float32),
            np.zeros(neg.shape[0], dtype=np.float32)
        ])

        self.node_feats = node_feats or {}
        self.node_feat_keys = node_feat_keys if node_feat_keys is not None else tuple(self.node_feats.keys())

    def __len__(self) -> int:
        """Return the number of supervised candidate pairs."""
        return int(self.pairs.shape[0])

    def _has_edge(self, u: int, v: int) -> bool:
        """Return whether the observable training graph contains the directed edge u -> v."""
        s = int(self.row_ptr[u])
        e = int(self.row_ptr[u + 1])
        cs = self.cols_sorted[s:e]
        j = int(np.searchsorted(cs, v, side="left"))
        return j < cs.shape[0] and int(cs[j]) == v

    def _sample_nodes(self, u: int, v: int, target_present: bool) -> np.ndarray:
        """Build a neighbourhood-first local node set without using the target edge."""
        nodes = [u, v]
        seen = {u, v}
        frontier = [u, v]

        while frontier and len(nodes) < self.P:
            remaining = self.P - len(nodes)
            quota = max(1, (remaining + len(frontier) - 1) // len(frontier))
            next_frontier = []

            for x in frontier:
                if len(nodes) >= self.P:
                    break

                rs = int(self.row_ptr[x])
                re = int(self.row_ptr[x + 1])
                outgoing = self.cols_sorted[rs:re]

                cs = int(self.col_ptr[x])
                ce = int(self.col_ptr[x + 1])
                incoming = self.rows_sorted[cs:ce]

                oi = 0
                ii = 0
                added = 0
                use_outgoing = True

                while added < quota and len(nodes) < self.P and (
                        oi < outgoing.shape[0] or ii < incoming.shape[0]):
                    n = None
                    from_outgoing = False

                    if use_outgoing and oi < outgoing.shape[0]:
                        n = int(outgoing[oi])
                        oi += 1
                        from_outgoing = True
                    elif ii < incoming.shape[0]:
                        n = int(incoming[ii])
                        ii += 1
                    elif oi < outgoing.shape[0]:
                        n = int(outgoing[oi])
                        oi += 1
                        from_outgoing = True

                    use_outgoing = not use_outgoing
                    if n is None:
                        break

                    if target_present:
                        if from_outgoing and x == u and n == v:
                            continue
                        if not from_outgoing and x == v and n == u:
                            continue

                    if n in seen:
                        continue

                    seen.add(n)
                    nodes.append(n)
                    next_frontier.append(n)
                    added += 1

            frontier = next_frontier

        return np.asarray(nodes, dtype=np.int64)

    def __getitem__(self, i):
        """Build and return one leakage-safe induced subgraph sample for a candidate pair."""
        u, v = int(self.pairs[i, 0]), int(self.pairs[i, 1])
        y = float(self.labels[i])
        target_present = self._has_edge(u, v)
        nodes = self._sample_nodes(u, v, target_present)
        n = int(nodes.shape[0])
        local = {int(node): j for j, node in enumerate(nodes)}

        # Induced adjacency with the target edge removed
        A = np.zeros((n, n), dtype=np.float32)
        for j, r in enumerate(nodes):
            s = int(self.row_ptr[r])
            e = int(self.row_ptr[r + 1])
            for c in self.cols_sorted[s:e]:
                c = int(c)
                if target_present and int(r) == u and c == v:
                    continue
                k = local.get(c)
                if k is not None:
                    A[j, k] = 1.0

        # Target label and supervision mask
        L = np.zeros((n, n), dtype=np.float32)
        M = np.zeros((n, n), dtype=bool)
        L[0, 1] = y
        M[0, 1] = True

        # Global node features with the target contribution removed
        feats = {}
        for k in self.node_feat_keys:
            x = np.asarray(self.node_feats[k][nodes], dtype=np.float32).copy()
            if target_present and k == "out_degree":
                x[0] = max(0.0, x[0] - 1.0)
            if target_present and k == "in_degree":
                x[1] = max(0.0, x[1] - 1.0)
            feats[k] = x

        return A, feats, L, M


class HiggsRetweetTileBench:
    """Expose the pre-divided HIGGS tile datasets through the framework benchmark interface."""

    def __init__(self, task: HiggsRetweetEdgesTask, tile_size: int = 384):
        """Create train, validation, and test tile datasets with task-declared custom node features."""
        node_feats = task.node_feature_dict(None)
        self.node_feat_keys = tuple(node_feats.keys())
        self.splits = {
            k: ECTileDataset(task._N, task._csr_row_ptr, task._csr_cols,
                             task._csc_col_ptr, task._csc_rows, edges,
                             P=tile_size, node_feats=node_feats, node_feat_keys=self.node_feat_keys)
            for k, edges in [("train", task._edges_tr), ("val", task._edges_va), ("test", task._edges_te)]
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HIGGS edge classification tasks.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["mlp", "deep_mlp", "cnn", "transformer", "rf", "gcn", "gin", "edge_tx", "sage", "gps"],
        help="Which model architecture to run."
    )

    args = parser.parse_args()
    task = HiggsRetweetEdgesTask(neg_pos_ratio=5.0)

    if args.model in ["gcn", "gin", "edge_tx", "sage", "gps"]:
        print(f"--- Running GNN Pipeline for model: {args.model} ---")

        gnn_cfg = GNNTrainConfig(
            epochs=50,
            weight_decay=1e-2,
            hidden=256,
            layers=4,
            dropedge_p=0.2,
            gnn_zero_supervised=True,
            neg_pos_ratio=1.0
        )

        run_gnn_edges_suite(
            task=task,
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

        bundle = run_pipeline_for_task(task=task, models=[args.model], cfg=ec_cfg)
        
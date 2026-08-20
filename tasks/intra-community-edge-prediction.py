import gzip
import shutil
import urllib.request
import community as community_louvain
import networkx as nx
import numpy as np
from pipeline.EdgeClassification import TaskHooks, ProvidedSplitsTask, TNNTrainConfig, run_pipeline_for_task
from pipeline.GraphBenchmark import GraphBenchmark
from pipeline.GNNBridge import GNNTrainConfig, run_gnn_edges_suite
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from typing import Dict, List, Tuple


def _build_csr(N: int, rows: np.ndarray, cols: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort (rows, cols) into CSR form with columns ascending within each row."""
    order = np.lexsort((cols, rows))
    rows_s, cols_s = rows[order], cols[order]
    row_ptr = np.zeros(N + 1, dtype=np.int64)
    np.add.at(row_ptr, rows_s + 1, 1)
    np.cumsum(row_ptr, out=row_ptr)
    return row_ptr, cols_s


def _bfs_nodes(row_ptr: np.ndarray, cols: np.ndarray, seeds: List[int], budget: int) -> set:
    """Breadth-first neighbourhood around `seeds`, capped at `budget` nodes."""
    out, frontier = set(seeds), list(seeds)
    while frontier and len(out) < budget:
        nxt = []
        for x in frontier:
            for c in cols[row_ptr[x]:row_ptr[x + 1]]:
                c = int(c)
                if c not in out:
                    out.add(c)
                    nxt.append(c)
                    if len(out) >= budget:
                        return out
        frontier = nxt
    return out


class ICEPTileDataset(Dataset):
    """
    Local induced subgraphs of the Facebook graph, each supervising a disjoint
    subset of one split's edges.

    Tile adjacency contains every observed edge, matching the untiled task, where
    edge existence is input rather than label. Supervision is restricted by the mask.
    Tiles are grown from multiple scattered seeds so a single tile spans several
    communities and carries both labels.
    """

    def __init__(
            self,
            N: int,
            row_ptr: np.ndarray,
            cols: np.ndarray,
            comm: np.ndarray,
            split_edges: np.ndarray,
            tile_size: int = 512,
            seeds_per_tile: int = 4,
            seed: int = 0
    ):
        self.row_ptr = row_ptr
        self.cols = cols
        self.comm = comm
        self.edges = split_edges.astype(np.int64, copy=False)
        self.tiles: List[Tuple[np.ndarray, np.ndarray]] = []

        if self.edges.shape[0] == 0:
            return

        rng = np.random.default_rng(seed)
        E = self.edges.shape[0]
        assigned = np.zeros(E, dtype=bool)
        order = rng.permutation(E)
        ptr = 0

        while not assigned.all():
            seeds: List[int] = []
            while ptr < E and len(seeds) < seeds_per_tile:
                e = int(order[ptr])
                ptr += 1
                if not assigned[e]:
                    seeds.append(e)
            if not seeds:
                seeds = np.flatnonzero(~assigned)[:seeds_per_tile].tolist()

            budget = max(2, tile_size // len(seeds))
            node_set: set = set()
            for e in seeds:
                u, v = int(self.edges[e, 0]), int(self.edges[e, 1])
                node_set |= _bfs_nodes(row_ptr, cols, [u, v], budget)

            nodes = np.fromiter(sorted(node_set), dtype=np.int64, count=len(node_set))
            inside = np.zeros(N, dtype=bool)
            inside[nodes] = True

            claim = (~assigned) & inside[self.edges[:, 0]] & inside[self.edges[:, 1]]
            assigned |= claim
            self.tiles.append((nodes, np.flatnonzero(claim)))

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, i):
        nodes, edge_ids = self.tiles[i]
        n = int(nodes.shape[0])
        local = {int(g): j for j, g in enumerate(nodes)}

        A = np.zeros((n, n), dtype=np.float32)
        for j, r in enumerate(nodes):
            for c in self.cols[self.row_ptr[r]:self.row_ptr[r + 1]]:
                k = local.get(int(c))
                if k is not None:
                    A[j, k] = 1.0

        c_local = self.comm[nodes]
        L = ((c_local[:, None] == c_local[None, :]) & (A > 0.5)).astype(np.float32)

        M = np.zeros((n, n), dtype=bool)
        for e in edge_ids:
            a, b = local[int(self.edges[e, 0])], local[int(self.edges[e, 1])]
            M[a, b] = M[b, a] = True

        return A, {}, L, M


class FacebookLouvainTileBench(GraphBenchmark):
    """
    Pre-divided tile datasets for the Facebook / Louvain task.

    Subclasses GraphBenchmark so the pipeline owns canonical feature derivation:
    each tile's requested features are derived from that tile's adjacency.
    """

    def __init__(self, cfg, tile_size: int = 512, seeds_per_tile: int = 4, seed: int = 0):
        super().__init__()
        self.cfg = cfg

        path = Path(getattr(cfg, "data_dir", "."))
        path.mkdir(parents=True, exist_ok=True)
        txt = path / "facebook_combined.txt"
        if not txt.exists():
            gz = path / "facebook_combined.txt.gz"
            if not gz.exists():
                url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
                print(f"Downloading {url}...")
                urllib.request.urlretrieve(url, gz)
            with gzip.open(gz, "rb") as f_in, open(txt, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            gz.unlink()

        G = nx.read_edgelist(txt, nodetype=int)
        G.remove_edges_from(nx.selfloop_edges(G))

        if getattr(cfg, "test_mode", False):
            keep = sorted(G.nodes())[:int(getattr(cfg, "test_num_nodes", 256))]
            G = G.subgraph(keep).copy()

        nodes = sorted(G.nodes())
        node_map = {g: i for i, g in enumerate(nodes)}
        N = len(nodes)

        part = community_louvain.best_partition(G, resolution=getattr(cfg, "louvain_resolution", 1.0))
        comm = np.array([part[g] for g in nodes], dtype=np.int64)

        e = np.array([(node_map[u], node_map[v]) for u, v in G.edges()], dtype=np.int64)
        undirected = np.concatenate([e, e[:, ::-1]], axis=0)
        row_ptr, cols = _build_csr(N, undirected[:, 0], undirected[:, 1])

        idx = np.arange(e.shape[0])
        idx_tr, idx_te = train_test_split(idx, test_size=cfg.test_size, random_state=seed)
        idx_tr, idx_va = train_test_split(
            idx_tr, test_size=cfg.val_size / (1.0 - cfg.test_size), random_state=seed
        )

        self.splits: Dict[str, Dataset] = {
            name: ICEPTileDataset(
                N, row_ptr, cols, comm, e[sel],
                tile_size=tile_size, seeds_per_tile=seeds_per_tile, seed=seed
            )
            for name, sel in (("train", idx_tr), ("val", idx_va), ("test", idx_te))
        }

        pos = int(((comm[e[:, 0]] == comm[e[:, 1]])).sum())
        print(
            f"[ICEP-TILES] nodes: {N:,} | edges: {e.shape[0]:,} | intra-community: {pos:,} "
            f"({100.0 * pos / e.shape[0]:.1f}%)\n"
            f"  tiles (train/val/test): {len(self.splits['train'])}/"
            f"{len(self.splits['val'])}/{len(self.splits['test'])} | tile_size={tile_size}"
        )


class FacebookLouvainTask(ProvidedSplitsTask):
    def __init__(self, seed=None, tile_size: int = 512, seeds_per_tile: int = 4):
        hooks = TaskHooks(
            label_fn=None,
            feature_set=[
                "degree", "deg_diff",
                "triangles", "clustering_coeff",
                "jaccard", "adamic_adar",
                "powers", "shortest_path"
            ],
            allow_adj_channel=True
        )

        super().__init__(
            name="facebook_intra_community",
            directed=False,
            hooks=hooks,
            eval_on_existing_edges_only=True,
            num_workers=4,
            seed=seed
        )

        self.test_mode = False
        self.test_num_nodes = 256
        self.test_size = 0.20
        self.val_size = 0.10
        self.louvain_resolution = 1.0
        self.data_dir = "data/facebook_intra_community"
        self._bench_instance = FacebookLouvainTileBench(
            self, tile_size=tile_size, seeds_per_tile=seeds_per_tile, seed=self.seed
        )


task = FacebookLouvainTask()

dense_cfg = TNNTrainConfig(
    epochs=10 if task.test_mode else 50,
    batch_size=4,
    tx_token_budget=1024,
    tx_token_policy="from_mask",
    supervised_redaction_policy="none",
    threshold_metric="bacc",
    select_by="bacc",
    early_stop_patience=5
)

MODELS = ["mlp", "deep_mlp", "cnn", "transformer", "rf"]
run_pipeline_for_task(task, MODELS, dense_cfg)

gnn_cfg = GNNTrainConfig(
    epochs=10 if task.test_mode else 100,
    batch_size=1
)

run_gnn_edges_suite(
    task=task,
    encoders=("gcn", "sage", "gin", "edge_tx", "gps"),
    cfg=gnn_cfg
)

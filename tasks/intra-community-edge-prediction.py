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


# Download and load the graph. Prepare a functional supervision mask.
class FacebookLouvainBenchmark(GraphBenchmark):
    def __init__(self, cfg):
        self.cfg = cfg
        self._loaded_data = None
        super().__init__(num_graphs=1)

    def _ensure_data(self):
        if self.__dict__.get("_loaded_data") is not None:
            return self._loaded_data

        path = Path(getattr(self.cfg, "data_dir", "."))
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

        if getattr(self.cfg, "test_mode", False):
            nodes = sorted(G.nodes())[:int(getattr(self.cfg, "test_num_nodes", 256))]
            G = G.subgraph(nodes).copy()

        N = G.number_of_nodes()
        nodes = sorted(G.nodes())
        node_map = {n: i for i, n in enumerate(nodes)}

        part = community_louvain.best_partition(G, resolution=getattr(self.cfg, "louvain_resolution", 1.0))
        comm = np.array([part[n] for n in nodes])

        A = np.zeros((N, N), dtype=np.float32)
        for u, v in G.edges():
            i, j = node_map[u], node_map[v]
            A[i, j] = A[j, i] = 1.0

        same_comm = (comm[:, None] == comm[None, :])
        Y = (same_comm & (A > 0.5)).astype(np.float32)

        iu, ju = np.triu_indices(N, k=1)
        edge_mask = (A[iu, ju] > 0.5)
        edge_idx = np.where(edge_mask)[0]

        idx_train, idx_test = train_test_split(edge_idx, test_size=self.cfg.test_size)
        idx_train, idx_val = train_test_split(idx_train, test_size=self.cfg.val_size / (1.0 - self.cfg.test_size))

        def make_mask(indices):
            m = np.zeros((N, N), dtype=bool)
            m[iu[indices], ju[indices]] = True
            return m | m.T

        train_m = make_mask(idx_train)
        val_m = make_mask(idx_val)
        test_m = make_mask(idx_test)

        self._loaded_data = {
            "A": A,
            "Y": Y,
            "splits": {
                "train": train_m,
                "val": val_m,
                "test": test_m
            },
            "directed": False,
        }
        return self._loaded_data

    def __getattr__(self, key):
        if key not in {"A", "Y", "splits", "pos_weight", "directed"}:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            )

        data = self._ensure_data()
        return data[key]


# Define the task-specific configuration object
class FacebookLouvainTask(ProvidedSplitsTask):
    def __init__(self, seed=None):
        hooks = TaskHooks(
            label_fn=lambda *_: self.bench.Y,
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
            seed=seed
        )

        self.test_mode = False
        self.test_num_nodes = 256
        self.test_size = 0.20
        self.val_size = 0.10
        self.louvain_resolution = 1.0
        self.data_dir = "data/facebook_intra_community"
        self.bench = FacebookLouvainBenchmark(self)
        self._bench_instance = self.bench


# Create the task object using the provided graph splits
task = FacebookLouvainTask()

# Define the config used by the TNN pipeline
dense_cfg = TNNTrainConfig(
    epochs=10 if task.test_mode else 50,
    batch_size=1,
    tx_token_budget=1024 if task.test_mode else 16384,
    tx_token_policy="from_mask",
    supervised_redaction_policy="none",
    threshold_metric="bacc",
    select_by="bacc",
    early_stop_patience=5
)

# Run the TNN pipeline on all five models
MODELS = ["mlp", "deep_mlp", "cnn", "transformer", "rf"]
run_pipeline_for_task(task, MODELS, dense_cfg)

# Define the config used by the GNN pipeline
gnn_cfg = GNNTrainConfig(
    epochs=10 if task.test_mode else 100,
    batch_size=1
)

# Run the GNN pipeline on all five models using Scalable Mode
run_gnn_edges_suite(
    task=task,
    encoders=("gcn", "sage", "gin", "edge_tx", "gps"),
    cfg=gnn_cfg
)

import json
import os
import torch
import torch.nn.functional as F
import numpy as np
from pipeline.EdgeClassification import TNNTrainConfig, TaskHooks, ProvidedSplitsTask, run_pipeline_for_task
from pipeline.GraphBenchmark import GraphBenchmark
from pipeline.GNNBridge import run_gnn_suite, GNNTrainConfig
from typing import List, Tuple, Optional


class MinimalFillBenchmark(GraphBenchmark):
    """
    Custom Benchmark loader for the Minimum Fill-In (Chordal Completion) task.

    This class loads:
      1. Graph topology and 'fill' edges (targets) from a JSONL file.
      2. Pre-computed node/edge features from a corresponding .pt file.

    It standardizes all graphs to a global maximum node size (N_MAX) via zero-padding,
    which allows for efficient dense batching in the downstream pipeline.
    """
    def __init__(self, dataset_path: str, features_path: Optional[str] = None):
        super().__init__()
        # Default feature path convention: dataset.jsonl -> dataset.jsonl.features.current.pt
        features_path = features_path or (dataset_path + ".features.current.pt")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Missing dataset: {dataset_path}")

        # Load graph topology records (Edges, Fill Edges, Split info)
        records = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if s := line.strip(): records.append(json.loads(s))

        # Load features dictionary: GraphID -> FeatureDict
        # Map to CPU immediately to avoid pinning GPU memory during simple data loading
        payload = torch.load(features_path, map_location="cpu", weights_only=False)
        feat_map = {int(k): v for k, v in payload.get("features", {}).items()}

        # Determine the maximum number of nodes across the dataset.
        # All dense tensors (A, L, Mask, Features) will be padded to this size (N_MAX)
        self.global_max_n = max(int(r["num_nodes"]) for r in records)
        self.records = records
        self.feat_map = feat_map

        # Process splits immediately into memory
        self.splits = {
            "train": self._process_split("train"),
            "val": self._process_split("val"),
            "test": self._process_split("test")
        }

    def _process_split(self, split_name) -> List[Tuple]:
        """
        Converts raw JSON records into the (A, L, Feats, Mask) tuple format
        expected by the pipeline's collate_fn_pad.
        """
        data = []
        for rec in self.records:
            if rec.get("split") != split_name: continue

            gid, n = int(rec["graph_id"]), int(rec["num_nodes"])

            # 1. Construct Input Adjacency (A) - the initial graph structure - undirected
            a = np.zeros((n, n), dtype=np.float32)
            for u, v in rec["edges"]:
                a[u, v] = a[v, u] = 1.0

            # 2. Construct Labels (L) - the set of "fill edges" required to make the graph chordal
            l = np.zeros((n, n), dtype=np.float32)
            for u, v in rec["fill_edges"]:
                l[u, v] = l[v, u] = 1.0

            # 3. Construct Evaluation Mask (M)
            # Critical for Link Prediction/Completion: evaluate/train on non-edge pairs
            # Logic: Select (A == 0) within the valid submatrix [:n, :n]
            mask = (a <= 0.5)
            np.fill_diagonal(mask, False)

            # 4. Process Features (F) - retrieve pre-computed features and keep them aligned to n
            feats = {}
            raw_feats = self.feat_map.get(gid, {})
            for k, v in raw_feats.items():
                t = v.clone() if isinstance(v, torch.Tensor) else torch.tensor(v).float()

                # Handle 2D features
                if t.ndim == 2:
                    h, w = t.shape
                    if h < n or w < n:
                        t = F.pad(t, (0, max(0, n - w), 0, max(0, n - h)), value=0.0)
                    t = t[:n, :n]

                # Handle 1D features
                elif t.ndim == 1:
                    # Pad end to reach (N_MAX,)
                    if t.shape[0] < n:
                        t = F.pad(t, (0, n - int(t.shape[0])), value=0.0)
                    t = t[:n]

                feats[k] = t

            data.append((torch.from_numpy(a), torch.from_numpy(l), feats, torch.from_numpy(mask)))

        return data


bench = MinimalFillBenchmark(
    dataset_path="data/minfill_true_1000/min_fill_dataset.jsonl",
    features_path="data/minfill_true_1000/min_fill_dataset.jsonl.features.current.pt"
)

hooks = TaskHooks(
    label_fn=None,
    feature_set=[
        "degree", "triangles", "clustering_coeff",
        "powers", "jaccard", "adamic_adar", "shortest_path"
    ],
    allow_adj_channel=True
)

# Create the task object and provide the path to the true minimal fillins and features
task = ProvidedSplitsTask(
    name="true_min_fill_chordal_completion",
    directed=False,
    hooks=hooks,
    bench=bench
)

# Define the config used by the TNN pipeline
dense_cfg = TNNTrainConfig(
    epochs=50,
    early_stop_patience=5
)

# Run the TNN pipeline on all five models
run_pipeline_for_task(
    task=task,
    models=["rf", "mlp", "deep_mlp", "cnn", "transformer"],
    cfg=dense_cfg
)

# Define the config used by the GNN pipeline
gnn_cfg = GNNTrainConfig(
    epochs=60,
    gnn_zero_supervised=True,
    lap_pe_k=8,
    neg_pos_ratio=4.0
)

# Run the GNN pipeline on all five models
run_gnn_suite(
    task=task,
    encoders=["gcn", "sage", "gin", "edge_tx", "gps"],
    cfg=gnn_cfg
)

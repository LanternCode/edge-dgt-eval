import numpy as np
import networkx as nx
from pipeline.EdgeClassification import TaskHooks, TNNTrainConfig, run_pipeline_for_task, ProvidedSplitsTask
from pipeline.GraphBenchmark import GraphBenchmark
from pipeline.GNNBridge import GNNTrainConfig, run_gnn_suite


# GraphBenchmark provides task-specific shape graphs
def label_shape_completion(G: nx.Graph) -> np.ndarray:
    """
    Return the stored complete cycle adjacency for this graph.
    Requires GraphBenchmark to have populated 'complete_adj'.
    No self-loops allowed.
    """
    L_full = G.graph.get("complete_adj", None)
    if L_full is None:
        raise ValueError("Graph missing 'complete_adj'. Ensure graph_type is 'shape_cycle'.")

    L_full = np.asarray(L_full, dtype=np.float32)
    np.fill_diagonal(L_full, 0.0)
    return L_full


# Request all features minus transpose (the graph is symmetric)
hooks = TaskHooks(
    label_fn=label_shape_completion,
    feature_set=[
        "degree", "deg_row", "deg_col", "deg_diff",
        "triangles", "clustering_coeff", "powers",
        "jaccard", "adamic_adar", "shortest_path"
    ],
    allow_adj_channel=True
)

# Use only the 'shape_cycle' graph family
bench = GraphBenchmark(
    graph_types=["shape_cycle"],
    shape_cycle_shapes=["triangle", "square", "pentagon", "hexagon"],
    shape_cycle_removal_prob=0.3
)

# Create the task object using the 70/15/15 split across 1000 graphs with 6-140 nodes each
task = ProvidedSplitsTask(
    name="shape_completion",
    directed=False,
    hooks=hooks,
    num_graphs=1000,
    min_nodes=6,
    max_nodes=140,
    ratios=(0.70, 0.15, 0.15),
    bench=bench
)

# Define the config used by the TNN pipeline
cfg = TNNTrainConfig(
    epochs=20,
    batch_size=32,
    weight_decay=1e-2,
    supervised_redaction_policy="none",
    tx_token_budget=2048,
    select_by="bacc"
)

# Run the TNN pipeline on all five models
MODELS = ["mlp", "deep_mlp", "cnn", "transformer", "rf"]
results = run_pipeline_for_task(task, MODELS, cfg)

# Define the config used by the GNN pipeline
gnn_cfg = GNNTrainConfig(
    epochs=cfg.epochs,
    lr=cfg.lr,
    batch_size=cfg.batch_size
)

# Run the GNN pipeline on all four models
run_gnn_suite(
    task=task,
    encoders=("gcn", "sage", "gin", "edge_tx", "gps"),
    cfg=gnn_cfg
)

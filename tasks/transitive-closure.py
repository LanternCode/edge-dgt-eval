import numpy as np
from pipeline.EdgeClassification import TaskHooks, ProvidedSplitsTask, TNNTrainConfig, run_pipeline_for_task
from pipeline.GNNBridge import GNNTrainConfig, run_gnn_suite


# The label function defines a true transitive closure for the input X graph
def label_transitive_closure_unbounded(A_obs: np.ndarray) -> np.ndarray:
    """
    Unbounded closure-only labels: 1 if i reaches j in ANY number of hops
    AND there is no direct edge A[i,j]. No self-loops allowed.
    """
    A = (A_obs > 0)
    reach = A.copy()

    # Boolean matrix multiplication to flood-fill the graph
    while True:
        next_reach = reach | (reach @ A > 0)
        if np.array_equal(next_reach, reach):
            break  # Stop when no new reachable pairs are found
        reach = next_reach

    closure_only = reach & (~A)  # drop direct edges
    L = closure_only.astype(np.float32)
    np.fill_diagonal(L, 0.0)
    return L


# Define the task-specific configuration object
hooks = TaskHooks(
    label_fn=label_transitive_closure_unbounded,
    feature_set=[
        "powers", "deg_row", "deg_col",
        "triangles", "clustering_coeff",
        "transpose"
    ],
    orientation="dag",
    allow_adj_channel=True
)

# Create the task object using the default split across 1000 graphs with 6-140 nodes each
task = ProvidedSplitsTask(
    name="transitive_closure",
    directed=True,
    hooks=hooks,
    num_graphs=1000,
    min_nodes=6, max_nodes=140,
    mask_policy="non_edges"
)

# Define the config used by the TNN pipeline
cfg = TNNTrainConfig(
    epochs=40,
    batch_size=32,
    lr=1e-3,
    early_stop_patience=10
)

# Run the TNN pipeline on all five models
MODELS = ["mlp", "deep_mlp", "cnn", "transformer", "rf"]
results = run_pipeline_for_task(task, MODELS, cfg)

# Define the config used by the GNN pipeline
gnn_cfg = GNNTrainConfig(
    lr=cfg.lr,
    batch_size=cfg.batch_size
)

# Run the GNN pipeline on all five models
run_gnn_suite(
    task=task,
    encoders=("gcn", "sage", "gin", "edge_tx", "gps"),
    cfg=gnn_cfg
)

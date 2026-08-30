import numpy as np
from pipeline.EdgeClassification import TaskHooks, ProvidedSplitsTask, TNNTrainConfig, run_pipeline_for_task
from pipeline.GNNBridge import GNNTrainConfig, run_gnn_suite


# ---------------------------------------------------------------------
# Labeler: mark “shortcuts” = existing edges that also have an alt path
# of length in {2..K_max}. No self-loops allowed.
# ---------------------------------------------------------------------
def label_shortcut_elimination(A_obs: np.ndarray, K_max: int = 2) -> np.ndarray:
    A = (A_obs > 0)
    if K_max < 2:
        L = np.zeros_like(A, dtype=np.float32)
        np.fill_diagonal(L, 0.0)
        return L

    cur = A.copy()  # paths of length 1
    reach_alt = np.zeros_like(A, dtype=bool)
    for _ in range(2, int(K_max) + 1):
        cur = (cur @ A) > 0  # boolean matmul via numeric @ + threshold
        reach_alt |= cur

    L = (A & reach_alt).astype(np.float32)  # existing edges with an alternate path
    np.fill_diagonal(L, 0.0)
    return L


# Orient to DAGs and exclude power_2 as it's key to the solution
hooks = TaskHooks(
    label_fn=lambda A: label_shortcut_elimination(A, K_max=2),
    feature_set=[
    "power_3", "power_4", "power_5",
    "degree", "deg_row", "deg_col", "deg_diff",
    "triangles", "clustering_coeff"
    ],
    orientation="dag",
    ensure_connected=True,
    allow_adj_channel=True
)

# Create the task object using the 70/20/10 split across 1000 graphs with 6-140 nodes each
task = ProvidedSplitsTask(
    name="shortcut_elimination",
    directed=True,
    hooks=hooks,
    num_graphs=1000,
    min_nodes=6,
    max_nodes=140,
    ratios=(0.7, 0.2, 0.1),
    eval_on_existing_edges_only=True
)

# Define the config used by the TNN pipeline
cfg = TNNTrainConfig(
    epochs=30,
    batch_size=32,
    tx_token_budget=2048,
    supervised_redaction_policy="none"
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

# Run the GNN pipeline on all five models
run_gnn_suite(
    task=task,
    encoders=("gcn", "sage", "gin", "edge_tx", "gps"),
    cfg=gnn_cfg,
    display_decimals=6, display_truncate=False
)

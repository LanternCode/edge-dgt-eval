import numpy as np
from pipeline.EdgeClassification import TaskHooks, ProvidedSplitsTask, TNNTrainConfig, run_pipeline_for_task
from pipeline.GNNBridge import GNNTrainConfig, run_gnn_suite


# The label function defines the symmetric closure of the input graph
def label_symmetric_closure(A_obs: np.ndarray) -> np.ndarray:
    """
    Directed recognition of symmetric closure:
    L[i, j] = 1 if A_obs[i, j] == 1 or A_obs[j, i] == 1. No self-loops allowed.
    """
    L = ((A_obs + A_obs.T) > 0).astype(np.float32)
    np.fill_diagonal(L, 0.0)
    return L


# Define the task-specific configuration object
hooks_sym = TaskHooks(
    label_fn=label_symmetric_closure,
    feature_set=[
        "powers", "deg_row", "deg_col",
        "triangles", "clustering_coeff", "cn",
        "jaccard", "adamic_adar", "transpose"
    ],
    allow_adj_channel=True
)

# Create the task object using a 70/20/10 split across 400 graphs with 6-140 nodes each
task_sym = ProvidedSplitsTask(
    name="symmetric_closure",
    directed=True,
    hooks=hooks_sym,
    num_graphs=400,
    min_nodes=6, max_nodes=140,
    ratios=(0.7, 0.2, 0.1)
)

# Define the config used by the TNN pipeline
cfg = TNNTrainConfig(
    lr=3e-4, weight_decay=1e-2, epochs=30, early_stop_patience=5,
    tx_dmodel=384, tx_layers=6, tx_heads=6, tx_dropout=0.1,
    supervised_redaction_policy="none", use_mask_channel=True
)

# Run the TNN pipeline on all five models
MODELS = ["mlp", "deep_mlp", "cnn", "transformer", "rf"]
results = run_pipeline_for_task(task_sym, MODELS, cfg)

# Define the config used by the GNN pipeline
gnn_cfg = GNNTrainConfig(
    epochs=cfg.epochs,
    lr=cfg.lr
)

# Run the GNN pipeline on all four models
run_gnn_suite(
    task=task_sym,
    encoders=("gcn", "sage", "gin", "edge_tx", "gps"),
    cfg=gnn_cfg
)

import numpy as np
import networkx as nx
from pipeline.GraphBenchmark import GraphBenchmark
from pipeline.GNNBridge import GNNTrainConfig, run_gnn_suite
from pipeline.EdgeClassification import ProvidedSplitsTask, TaskHooks, TNNTrainConfig, run_pipeline_for_task


# ---------------------------------------------------------------------
# 1) Canonical labels: spanning tree edges (unique by BFS with tie-breaks)
# ---------------------------------------------------------------------
def bfs_spanning_tree_labels(A: np.ndarray, G: nx.Graph) -> np.ndarray:
    """
    Deterministic spanning tree:
      - choose the root as the smallest index node in the largest connected component
      - BFS traversal with neighbor order sorted by node index
    Output: (N,N) symmetric 0/1 matrix with tree edges == 1
    """
    N = A.shape[0]
    Gu = nx.Graph(G)  # always treat as undirected for spanning trees

    # choose component with most nodes, root = min index in that component
    comps = sorted(nx.connected_components(Gu), key=lambda c: (-len(c), min(c)))
    root = min(comps[0])

    # BFS with deterministic order (sorted neighbors).
    # This reliance on sort means the label depends on Node IDs.
    tree = nx.bfs_tree(Gu, root, sort_neighbors=lambda it: iter(sorted(it)))

    # Convert to undirected tree edges (symmetrize)
    L = np.zeros((N, N), dtype=np.float32)
    for u, v in tree.edges():
        L[u, v] = 1.0
        L[v, u] = 1.0

    # ensure no self-loops
    np.fill_diagonal(L, 0.0)
    return L


# Define the task-specific configuration object
hooks = TaskHooks(
    label_fn=bfs_spanning_tree_labels,
    ensure_connected=True,
    feature_set=[
        "powers", "degree", "deg_diff",
        "triangles", "clustering_coeff",
        "cn", "jaccard", "adamic_adar"
    ]
)

# Exclude shape_cycle due ti conflicting objectives
bench = GraphBenchmark(
    graph_types=[
        "erdos_renyi", "barabasi_albert", "watts_strogatz",
        "random_regular", "stochastic_block", "powerlaw_cluster",
        "random_geometric", "balanced_tree", "tree_plus_chords"
    ]
)

# Create the task object using the 70/20/10 split across 1000 graphs with 6-140 nodes each
task = ProvidedSplitsTask(
    name="spanning_tree",
    directed=False,
    hooks=hooks,
    num_graphs=1000,
    min_nodes=6,
    max_nodes=140,
    ratios=(0.7, 0.2, 0.1),
    eval_on_existing_edges_only=True,
    mask_policy="edges",
    bench=bench
)

# Define the config used by the TNN pipeline
ec_cfg = TNNTrainConfig(
    epochs=50,
    weight_decay=1e-4,
    early_stop_patience=5,
    threshold_metric="bacc", select_by="bacc",
    deep_mlp_dropout=0.15, tx_dropout=0.15, cnn_dropout=0.15,
    deep_mlp_layers=3, deep_mlp_hidden=128,
    rf_neg_pos_ratio=5.0
)

# Run the TNN pipeline on all five models
run_pipeline_for_task(
    task=task,
    models=["cnn", "mlp", "deep_mlp", "transformer", "rf"],
    cfg=ec_cfg
)

# Define the config used by the GNN pipeline
gnn_cfg = GNNTrainConfig(
    epochs=ec_cfg.epochs,
    dropout=0.15,
    lap_pe_k=8,
    use_tree_aux_loss=True,
    neg_pos_ratio=5.0
)

# Run the GNN pipeline on all four models
gnn_encoders = ["gcn", "sage", "gin", "edge_tx", "gps"]
run_gnn_suite(
    task=task,
    encoders=gnn_encoders,
    cfg=gnn_cfg
)

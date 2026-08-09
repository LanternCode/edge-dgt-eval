import numpy as np
import networkx as nx
from pipeline.EdgeClassification import TaskHooks, TNNTrainConfig, run_pipeline_for_task, ProvidedSplitsTask
from pipeline.GraphBenchmark import GraphBenchmark
from pipeline.GNNBridge import GNNTrainConfig, run_gnn_suite


def label_feedback_edge_set(G: nx.Graph) -> np.ndarray:
    """
    Binary labels: 1.0 for edges in the task's canonical minimum feedback edge set.

    For an undirected unweighted graph, deleting E \\ T for any spanning forest T
    removes all cycles with minimum cardinality. Because the optimum is generally
    non-unique, this task intentionally uses NetworkX's deterministic spanning-tree
    choice to define one supervised optimum rather than accepting any valid optimum.
    """
    N = G.number_of_nodes()
    L = np.zeros((N, N), dtype=np.float32)

    # Spanning forest edges are kept (label 0); back-edges are deleted (label 1)
    spanning_forest = nx.minimum_spanning_tree(G)
    tree_edges = {tuple(sorted(e)) for e in spanning_forest.edges()}

    for u, v in G.edges():
        if tuple(sorted((u, v))) not in tree_edges:
            L[u, v] = L[v, u] = 1.0

    np.fill_diagonal(L, 0.0)
    return L


hooks = TaskHooks(
    label_fn=label_feedback_edge_set,
    feature_set=[
        "powers", "degree", "endpoint_degree",
        "triangles", "clustering_coeff",
        "jaccard", "adamic_adar"
    ],
    allow_adj_channel=True
)

# Exclude the shape_cycle graph family because it has completion-specific hidden ground truth
bench = GraphBenchmark(
    graph_types=[
        "erdos_renyi", "barabasi_albert", "watts_strogatz",
        "random_regular", "stochastic_block", "powerlaw_cluster",
        "random_geometric", "balanced_tree", "tree_plus_chords"
    ]
)

# Create the task object using the default split across 1000 task-specific graphs with 6-140 nodes each
task_fes = ProvidedSplitsTask(
    name="feedback_edge_set",
    directed=False,
    hooks=hooks,
    num_graphs=1000,
    min_nodes=6,
    max_nodes=140,
    ratios=(0.70, 0.20, 0.10),
    bench=bench,
    eval_on_existing_edges_only=True
)

# Define the config used by the TNN pipeline
cfg = TNNTrainConfig(
    epochs=30,
    weight_decay=1e-3,
    tx_dmodel=128,
    supervised_redaction_policy="none",
    rf_neg_pos_ratio=1.0,
    threshold_metric="bacc",
    select_by="bacc"
)

# Run the TNN pipeline on all five models
MODELS = ["mlp", "deep_mlp", "cnn", "transformer", "rf"]
results = run_pipeline_for_task(task_fes, MODELS, cfg)

# Define the config used by the GNN pipeline
gnn_cfg = GNNTrainConfig(
    epochs=cfg.epochs,
    batch_size=32,
    neg_pos_ratio=1.0
)

# Run the GNN pipeline on all four models
run_gnn_suite(
    task=task_fes,
    encoders=("gcn", "sage", "gin", "edge_tx", "gps"),
    cfg=gnn_cfg
)

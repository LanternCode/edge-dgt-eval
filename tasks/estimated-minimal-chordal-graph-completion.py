import random
import numpy as np
import networkx as nx
import math
from pipeline.EdgeClassification import TaskHooks, TNNTrainConfig, run_pipeline_for_task, ProvidedSplitsTask
from pipeline.GraphBenchmark import GraphBenchmark
from pipeline.GNNBridge import GNNTrainConfig, run_gnn_suite


# ---------------------------------------------------------------------
# Labeler: the estimated minimal chordal graph is generated using
# networkx's complete_to_chordal_graph greedy algorithm.
# ---------------------------------------------------------------------
def label_minimal_chordal_completion(A: np.ndarray) -> np.ndarray:
    """
    Computes the chordal completion of the graph represented by adjacency A.
    Label L[i,j] = 1 if edge (i,j) must be added to make G chordal, else 0.
    """
    # Convert numpy adjacency to NetworkX graph
    G = nx.from_numpy_array(A)

    # Optimization: if already chordal, return empty labels (all zeros)
    if nx.is_chordal(G):
        return np.zeros_like(A, dtype=np.float32)

    # Compute chordal completion; returns (H, alpha), we only need H
    H, _ = nx.complete_to_chordal_graph(G)

    # Convert completion back to adjacency
    A_full = nx.to_numpy_array(H)

    # The labels are the new edges: (Full - Observed)
    # Using > 0.5 for stable boolean subtraction on floats
    L = ((A_full - A) > 0.5).astype(np.float32)
    np.fill_diagonal(L, 0.0)
    return L


class ChordalCompletionBenchmark(GraphBenchmark):
    def __init__(self, num_graphs, min_nodes, max_nodes, graph_types, **kwargs):
        self.valid_types = graph_types
        self.node_range = (min_nodes, max_nodes)

        super().__init__(
            num_graphs=num_graphs,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            graph_types=graph_types,
            **kwargs
        )

    def generate_graph(self, graph_type: str, num_nodes: int, directed: bool = False, rng=random) -> nx.Graph:
        """
        Generates a graph of the specified type and size, enforcing the non-chordal constraint via rejection sampling.

        If the requested configuration (type + size) fails to produce a non-chordal graph after `max_retries` attempts
        (e.g., a small cycle or grid which is mathematically impossible to be non-chordal), this method abandons the
        current configuration and recursively attempts a new random type and size from the benchmark's allowed pool.

        Graph Families & Sources:
          - 'cycle': Generated locally (simple cycle graph).
          - 'grid':  Generated locally (2D grid graph).
          - 'tree':  Delegated to GraphBenchmark's 'tree_plus_chords'.
                     (Generates a random tree and adds extra edges to likely create cycles).
          - 'er':    Delegated to GraphBenchmark's 'erdos_renyi'.
          - 'ba':    Delegated to GraphBenchmark's 'barabasi_albert'.

        Args:
            graph_type (str): The specific graph family to generate.
            num_nodes (int): The number of nodes in the graph.

        Returns:
            nx.Graph: A guaranteed non-chordal graph.
        """
        max_retries = 400

        def _ensure_n(g_in):
            g_in = nx.convert_node_labels_to_integers(g_in, first_label=0)
            m = g_in.number_of_nodes()
            if m < num_nodes:
                g_in.add_nodes_from(range(m, num_nodes))
            return g_in

        # Attempt to generate the requested configuration
        for _ in range(max_retries):
            G = None
            if graph_type == "cycle":
                if num_nodes < 4:
                    pass  # Impossible to be non-chordal
                else:
                    G = nx.cycle_graph(num_nodes)
            elif graph_type == "grid":
                side = int(math.floor(math.sqrt(num_nodes)))
                if side < 2:
                    pass
                else:
                    width = int(math.ceil(num_nodes / side))
                    G = nx.grid_2d_graph(side, width)
                    G = nx.convert_node_labels_to_integers(G, first_label=0)
                    G = G.subgraph(range(num_nodes)).copy()
            elif graph_type == "tree":
                # Delegate to benchmark's tree_plus_chords
                G = super().generate_graph("tree_plus_chords", num_nodes, directed=directed, rng=rng)
            elif graph_type == "er":
                G = super().generate_graph("erdos_renyi", num_nodes, directed=directed, rng=rng)
            elif graph_type == "ba":
                G = super().generate_graph("barabasi_albert", num_nodes, directed=directed, rng=rng)
            else:
                G = super().generate_graph(graph_type, num_nodes, directed=directed, rng=rng)

            if G is not None:
                G = _ensure_n(G)
                if not nx.is_chordal(G):
                    return G

        # If we fail after max_retries, drop this config and recurse with a new one
        new_type = rng.choice(self.valid_types)
        new_size = rng.randint(self.node_range[0], self.node_range[1])
        return self.generate_graph(new_type, new_size, directed=directed, rng=rng)


# Inform the pipeline about task-specific graph families
bench = ChordalCompletionBenchmark(
    num_graphs=1000,
    min_nodes=6,
    max_nodes=140,
    graph_types=["cycle", "grid", "tree", "er", "ba"]
)

hooks = TaskHooks(
    label_fn=label_minimal_chordal_completion,
    feature_set=[
        "powers", "degree", "endpoint_degree",
        "deg_diff", "triangles", "clustering_coeff",
        "cn", "jaccard", "adamic_adar",
        "shortest_path"
    ],
    allow_adj_channel=True
)

# Create the task object using the 80/10/10 split across 1000 non-chordal graphs with 6-140 nodes each
task = ProvidedSplitsTask(
    name="estimated_minimal_chordal_completion",
    directed=False,
    hooks=hooks,
    ratios=(0.8, 0.1, 0.1),
    bench=bench,
    mask_policy="non_edges"
)

# Define the config used by the TNN pipeline
cfg = TNNTrainConfig(epochs=50)

# Run the TNN pipeline on all five models
models = ["rf", "mlp", "deep_mlp", "cnn", "transformer"]
results = run_pipeline_for_task(task=task, models=models, cfg=cfg)

# Define the config used by the GNN pipeline
gnn_encoders = ["gcn", "sage", "gin", "edge_tx", "gps"]
gnn_cfg = GNNTrainConfig(
    epochs=120,
    weight_decay=1e-3,
    lap_pe_k=8
)

# Run the GNN pipeline on all four models
run_gnn_suite(task, gnn_encoders, gnn_cfg)

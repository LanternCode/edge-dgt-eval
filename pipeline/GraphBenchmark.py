# SPDX-License-Identifier: CC-BY-SA-4.0

import heapq
import inspect
import torch
import random
import networkx as nx
import numpy as np
from ._utils.features import pairwise_batch_from_adj
from collections import deque
from time import perf_counter
from collections.abc import Sized
from typing import List, Sequence, Tuple, Dict, Optional, cast
from torch.utils.data import DataLoader, Dataset, Subset
from decimal import Decimal


class GraphBenchmark:
    """
    GraphBenchmark: generates random graphs, builds task-ready datasets,
    provides optional feature extraction and flexible splitting with *optional*
    packaging into PyTorch DataLoaders.

    Public methods we rely on:
    - generate_graph(graph_type, num_nodes, directed=False)
    - extract_features(adj, feature_set)
    - sample_specs(num_graphs, min_nodes, max_nodes)
    - generate_dataset(specs, hooks, directed=False, seed=None)
    - make_loaders(dataset, batch_size, ratios=(0.7, 0.2, 0.1), collate_fn=None, seed=None, pin_memory=None)

    When using existing split definitions, ProvidedSplitsTask reads bench.splits and will either:
    (a) use split masks for a single graph, or
    (b) preserve pre-divided train/val/test sample collections.

    New minimal helpers (single responsibility):
      - sample_specs(num_graphs, min_nodes, max_nodes)
      - generate_dataset(specs, hooks)
      - make_loaders(dataset, batch_size, ratios=(...), collate_fn=None)
    """
    def __init__(
        self,
        show_progress: bool = False,
        graph_types: Optional[List[str]] = None,
        shape_cycle_shapes: Optional[List[str]] = None,
        shape_cycle_removal_prob: float = 0.30
    ):
        self.show_progress = show_progress
        self.graph_types = graph_types or [
            'erdos_renyi', 'barabasi_albert', 'watts_strogatz',
            'random_regular', 'stochastic_block', 'powerlaw_cluster',
            'random_geometric', 'balanced_tree', 'tree_plus_chords',
            'shape_cycle',
        ]
        
        # Shape completion settings (used when graph_type == 'shape_cycle')
        self.shape_cycle_shapes = shape_cycle_shapes or ["triangle", "square", "pentagon", "hexagon"]
        self.shape_cycle_removal_prob = float(shape_cycle_removal_prob)

        # Warning state tracking
        self._warned_shapes = set()
        self._warned_generation_errors = set()

    # ------------------------------------------------------------------ #
    # 1) Graph generators (undirected by default; some families support native directed generation, with optional post-generation orientation via hooks)
    # ------------------------------------------------------------------ #
    def generate_graph(self, graph_type: str, num_nodes: int, directed: bool = False, rng=random) -> nx.Graph:
        """
        Generates a base topology for a given graph family.

        1. erdos_renyi:
        - Base: Probability dynamically scaled as p = 3.5 / N.
        - Directed: Natively supported via NetworkX.

        2. barabasi_albert:
        - Base: Preferential attachment with m=2 to preserve long tails/hubs.
        - Directed: Uses Bollobás et al. scale-free model (nx.scale_free_graph)
        to preserve massive directed hubs (random orientation would flatten the distribution).

        3. watts_strogatz:
        - Base: Small-world graph with low average degree.
          For the undirected case, if the sampled `k` is odd, NetworkX uses `k - 1` neighbours.
        - Directed: Directed analogue of the same construction with mostly local edges and some rewired long-range shortcuts.

        4. random_regular:
        - Base: Strict 4-regular graph for num_nodes > 4; falls back to a cycle graph for very small graphs.
        - Directed: Eulerian circuit orientation. Traces a continuous path through an
        undirected 4-regular base to guarantee exact in-degree=2 and out-degree=2.

        5. stochastic_block:
        - Base: 2-block partition (50/50 split) with p_in > p_out.
        - Directed: Natively supported via NetworkX.

        6. powerlaw_cluster:
        - Base: Holme-Kim model (m=2) combining scale-free hubs with high clustering.
        - Directed: Custom Directed Triad Formation. Standard directed preferential
        attachment combined with probabilistic backward links to close directed triangles.

        7. random_geometric:
        - Base: Spatial transmission radius scaled by area: r = sqrt(3.5 / (N * pi)).
        - Directed: Spatial K-Nearest Neighbours (K=3). Nodes project directed edges
        to their closest geographic neighbours (fixed out-degree, variable in-degree).

        8. balanced_tree:
        - Base: Full tree with random branching factor r in [2, 4].
        - Directed: Natively supported. Edges strictly point outward from the root.

        9. tree_plus_chords:
        - Base: Random labelled tree + up to 0.75 * N random chords (subject to valid-pair availability and the retry cap).
        - Directed: Base tree is directed via BFS outward from root 0; chords are
        injected as non-reciprocal directed pairs (u -> v), strictly 
        preventing bidirectional edges if v -> u already exists.

        10. shape_cycle:
        - Base: Generates multiple incomplete ring topologies (shapes) occupying ~50% of the nodes.
        The remaining ~50% of nodes form a cycle-free background forest attached to the shapes.
        - Directed: Natively supported. Shape edges strictly point forward u -> (u+1) % L.
        Background forest edges are randomly oriented.

        Args:
            graph_type (str): The family of the graph to generate.
            num_nodes (int): Target number of nodes (N).
            directed (bool): Whether to construct a mathematically sound directed topology.
            rng (random.Random): The random number generator instance to use for isolated topology generation.
            Defaults to the global `random` module.

        Returns:
            nx.Graph or nx.DiGraph: The generated base topology.
        """
        if graph_type == 'erdos_renyi':
            p = min(1.0, 3.5 / num_nodes)
            return nx.erdos_renyi_graph(num_nodes, p, directed=directed, seed=rng)

        elif graph_type == 'barabasi_albert':
            if directed:
                G = nx.DiGraph(nx.scale_free_graph(num_nodes, seed=rng))
                G.remove_edges_from(list(nx.selfloop_edges(G)))
                return G
            m = max(1, min(2, num_nodes - 1))
            return nx.barabasi_albert_graph(num_nodes, m, seed=rng)

        elif graph_type == 'watts_strogatz':
            if directed:
                G = nx.DiGraph()
                G.add_nodes_from(range(num_nodes))
                k, p = max(1, min(2, num_nodes - 1)), rng.uniform(0.1, 0.5)
                for u in range(num_nodes):
                    for offset in range(1, k + 1):
                        v = (u + offset) % num_nodes
                        if rng.random() < p:
                            choices = [n for n in range(num_nodes) if n != u and not G.has_edge(u, n)]
                            if choices: v = rng.choice(choices)
                        G.add_edge(u, v)
                return G
            k = rng.randint(2, min(num_nodes - 1, 6))
            p = rng.uniform(0.1, 0.5)
            return nx.watts_strogatz_graph(num_nodes, k, p, seed=rng)

        elif graph_type == 'random_regular':
            if num_nodes <= 4:
                G_base = nx.cycle_graph(num_nodes)
                return nx.DiGraph(G_base) if directed else G_base
            
            if directed:
                G_und = nx.random_regular_graph(4, num_nodes, seed=rng)
                G_dir = nx.DiGraph()
                G_dir.add_nodes_from(range(num_nodes))
                for c in nx.connected_components(G_und):
                    sub = G_und.subgraph(c)
                    G_dir.add_edges_from(nx.eulerian_circuit(sub))
                return G_dir
            return nx.random_regular_graph(4, num_nodes, seed=rng)

        elif graph_type == 'stochastic_block':
            sizes = [num_nodes // 2, num_nodes - num_nodes // 2]
            p_in, p_out = rng.uniform(0.1, 0.5), rng.uniform(0.01, 0.1)
            probs = [[p_in, p_out], [p_out, p_in]]
            return nx.stochastic_block_model(sizes, probs, directed=directed, seed=rng)

        elif graph_type == 'powerlaw_cluster':
            if directed:
                G = nx.DiGraph(nx.scale_free_graph(num_nodes, seed=rng))
                G.remove_edges_from(list(nx.selfloop_edges(G)))
                p_triad, new_edges = rng.uniform(0.1, 0.5), []
                for u, v in list(G.edges()):
                    if rng.random() < p_triad:
                        in_neighbors = [w for w in G.predecessors(v) if w != u and not G.has_edge(u, w)]
                        if in_neighbors:
                            new_edges.append((u, rng.choice(in_neighbors)))
                G.add_edges_from(new_edges)
                return G
            m, p = max(1, min(2, num_nodes - 1)), rng.uniform(0.1, 0.5)
            return nx.powerlaw_cluster_graph(num_nodes, m, p, seed=rng)

        elif graph_type == 'random_geometric':
            if directed:
                G = nx.DiGraph()
                G.add_nodes_from(range(num_nodes))
                pos = {i: (rng.random(), rng.random()) for i in range(num_nodes)}
                nx.set_node_attributes(G, pos, 'pos')
                for i in range(num_nodes):
                    dists = [(j, (pos[i][0] - pos[j][0]) ** 2 + (pos[i][1] - pos[j][1]) ** 2) for j in range(num_nodes) if i != j]
                    for j, _ in heapq.nsmallest(3, dists, key=lambda x: x[1]): 
                        G.add_edge(i, j)
                return G
            radius = np.sqrt(3.5 / (num_nodes * np.pi))
            return nx.random_geometric_graph(num_nodes, radius, seed=rng)

        elif graph_type == 'balanced_tree':
            r = rng.randint(2, 4)
            h = 0
            while (r**(h+1) - 1) // (r - 1) < num_nodes:
                h += 1
            G = nx.balanced_tree(r, h, create_using=nx.DiGraph if directed else nx.Graph)
            return G.subgraph(list(G.nodes)[:num_nodes]).copy()

        elif graph_type == 'tree_plus_chords':
            T = nx.random_labeled_tree(num_nodes, seed=rng)
            G = nx.DiGraph() if directed else nx.Graph()
            G.add_nodes_from(range(num_nodes))
            if directed:
                G.add_edges_from(nx.bfs_edges(T, 0))
            else:
                G.add_edges_from(T.edges())

            k, added, attempts = int(round(0.75 * num_nodes)), 0, 0
            while added < k and attempts < k * 10:
                u, v = rng.randint(0, num_nodes - 1), rng.randint(0, num_nodes - 1)
                if u != v and not G.has_edge(u, v) and (not directed or not G.has_edge(v, u)):
                    G.add_edge(u, v)
                    added += 1
                attempts += 1
            return G

        elif graph_type == 'shape_cycle':
            shape_to_len = {'triangle': 3, 'square': 4, 'pentagon': 5, 'hexagon': 6}

            # 1. Validation & Selection
            valid_shapes = []
            for s in self.shape_cycle_shapes:
                if shape_to_len.get(s, 999) <= num_nodes:
                    valid_shapes.append(s)
                else:
                    if s not in self._warned_shapes:
                        if s in shape_to_len:
                            print(f"[WARN] '{s}' requires {shape_to_len[s]} nodes but N={num_nodes}; skipping this shape.")
                        else:
                            print(f"[WARN] '{s}' is an undefined shape; skipping.")
                        self._warned_shapes.add(s)

            if not valid_shapes:
                raise ValueError(
                    f"[DATASET GENERATOR] Cannot generate 'shape_cycle' with N={num_nodes}. "
                    f"The smallest configured shape requires more nodes."
                )
            else:
                shape = rng.choice(valid_shapes)
                L = shape_to_len[shape]

            # 2. Allocation (~50% shape nodes, ~50% background nodes)
            num_shapes = max(1, (num_nodes // 2) // L)
            nodes = list(range(num_nodes))
            rng.shuffle(nodes)

            shape_nodes_list = nodes[:num_shapes * L]
            bg_nodes = nodes[num_shapes * L:]

            A_full = np.zeros((num_nodes, num_nodes), dtype=np.uint8)
            A_obs = np.zeros((num_nodes, num_nodes), dtype=np.uint8)

            def add_edge(mat, u, v):
                mat[u, v] = 1
                if not directed:
                    mat[v, u] = 1

            # 3. Build Core Shapes (Guaranteed at least one dropped edge per shape)
            for i in range(num_shapes):
                cyc = shape_nodes_list[i * L : (i + 1) * L]
                
                # Write perfect shape to A_full
                for j in range(L):
                    add_edge(A_full, cyc[j], cyc[(j + 1) % L])

                # Determine drops for A_obs
                edges = [(cyc[j], cyc[(j + 1) % L]) for j in range(L)]
                drop_idx = rng.randrange(L)
                
                kept_edges = []
                for j, (u, v) in enumerate(edges):
                    if j == drop_idx:
                        continue  # Dropped
                    if rng.random() >= self.shape_cycle_removal_prob:
                        kept_edges.append((u, v))

                # Ensure at least one edge survives so the shape is not wiped
                if not kept_edges:
                    safe_idx = (drop_idx + 1) % L
                    kept_edges.append(edges[safe_idx])

                for u, v in kept_edges:
                    add_edge(A_obs, u, v)

            # 4. Build Background Forest (Cycle-free attachment)
            active_pool = shape_nodes_list.copy()
            for u in bg_nodes:
                if rng.random() < 0.04:
                    continue  # Small chance to remain completely isolated

                v = rng.choice(active_pool)
                # Randomly assign direction of the single connecting edge
                if directed and rng.random() < 0.5:
                    A_obs[u, v] = 1
                else:
                    A_obs[v, u] = 1
                    if not directed:
                        A_obs[u, v] = 1

                active_pool.append(u)  # Now available for other bg nodes to attach to

            # 5. Compile Metadata
            G = nx.DiGraph() if directed else nx.Graph()
            G.add_nodes_from(range(num_nodes))
            I, J = np.where((A_obs if directed else np.triu(A_obs, 1)) > 0)
            G.add_edges_from(zip(I.tolist(), J.tolist()))

            G.graph['complete_adj'] = A_full.astype(np.float32)
            G.graph['shape_nodes'] = sorted(shape_nodes_list)
            G.graph['shape_type'] = shape
            G.graph['skip_organic_mutation'] = True

            return G

        else:
            raise ValueError(f"Unknown graph type: {graph_type}")

    # ------------------------------------------------------------------ #
    # 2) Sampling, dataset building, optional packaging (no redundancy)
    # ------------------------------------------------------------------ #
    def sample_specs(
        self,
        num_graphs: int,
        min_nodes: int,
        max_nodes: int
    ) -> List[Tuple[str, int]]:
        sizes = np.random.randint(min_nodes, max_nodes + 1, size=num_graphs)
        return [(random.choice(self.graph_types), int(N)) for N in sizes]

    def generate_dataset(
        self,
        specs: List[Tuple[str, int]],     # e.g. produced by bench.sample_specs(...)
        hooks,
        directed: bool = False,
        seed: Optional[int] = None
    ) -> Dataset:
        graphs, labels, feats = [], [], []
        fast_label_fn = self._compile_label_fn(getattr(hooks, "label_fn", None))
        local_rng = random.Random(seed) if seed is not None else random

        orientation_mode = getattr(hooks, "orientation", None)
        if orientation_mode == "dag" and not bool(directed):
            raise ValueError(
                f"[DATASET GENERATOR] DAG means Directed Acyclic Graph. hooks.orientation={orientation_mode!r} requires directed=True."
            )

        # DAG orientation takes precedence over connectivity enforcement
        ensure_connected = bool(getattr(hooks, "ensure_connected", False))
        if ensure_connected and orientation_mode == "dag":
            ensure_connected = False
            print(
                "[DATASET GENERATOR] hooks.ensure_connected=True is ignored when hooks.orientation='dag'. "
                "DAG orientation discards lower-triangle edges, so generated graphs may be weakly disconnected.",
                flush=True,
            )

        # Deque
        total = len(specs)
        recent = deque(maxlen=5) if self.show_progress else None
        t0 = perf_counter() if self.show_progress else None
        if self.show_progress:
            print(
                f"[generate_dataset] 0/{total} | avg/graph=0.0000s | elapsed=0.00s | recent={list(recent)}",
                end="\r",
                flush=True
            )

        for i, (original_gtype, N) in enumerate(specs, start=1):
            t_graph = perf_counter() if self.show_progress else None

            # 1) Generate a graph belonging to one of the allowed graph families
            is_dir = bool(directed)
            G = None
            gtype = original_gtype

            # Safeguard to prevent infinite loops
            attempts = 0
            max_retries = 20  
            while attempts < max_retries:
                try:
                    candidate = self.generate_graph(gtype, int(N), directed=is_dir, rng=local_rng)
                    if candidate.number_of_nodes() != int(N):
                        raise ValueError(
                            f"[DATASET GENERATION ERROR] Generator returned {candidate.number_of_nodes()} nodes for requested N={int(N)}. "
                            "Try adjusting the requested graph families, num_nodes, and min- and max-nodes settings."
                        )
                    G = candidate
                    break
                except Exception as e:
                    # Log the exact error only once per graph family
                    if gtype not in self._warned_generation_errors:
                        print(f"[WARN] Failed to generate '{gtype}' (N={N}): {e}. Re-rolling graph family.")
                        self._warned_generation_errors.add(gtype)
                    
                    attempts += 1
                    gtype = local_rng.choice(
                        [t for t in self.graph_types if t != gtype] or self.graph_types
                    )

            if G is None:
                raise RuntimeError(
                    f"[DATASET GENERATOR] Aborting: Failed to generate a graph with N={N} "
                    f"after {max_retries} attempts. Please ensure your `min_nodes` is compatible "
                    f"with your chosen `graph_types`."
                )

            # 2) Organic mutation - Drop 10-15% of edges to break algorithmic perfection.
            # Small edge counts round down and may realise a lower fraction.
            # Graph families that implement internal mutation bypass this via `skip_organic_mutation`.
            if not G.graph.get('skip_organic_mutation', False):
                edges = list(G.edges())
                drop_count = int(len(edges) * local_rng.uniform(0.10, 0.15))
                if drop_count > 0:
                    G.remove_edges_from(local_rng.sample(edges, drop_count))

            # 3) Bottleneck-free connectivity (Proportional Multi-Stitching)
            if ensure_connected:
                Gu = nx.Graph(G)
                if not nx.is_connected(Gu):
                    # Sort components by size descending to always attach to the largest body
                    comps = sorted(list(nx.connected_components(Gu)), key=len, reverse=True)
                    main_body = list(comps[0])

                    for island in comps[1:]:
                        island_nodes = list(island)
                        # Scale bridge edges: 1 connection per 5 nodes in the island (capped at 3)
                        k_bridges = max(1, min(3, len(island_nodes) // 5 + 1))

                        for _ in range(k_bridges):
                            u = local_rng.choice(main_body)
                            v = local_rng.choice(island_nodes)
                            # Randomise direction if the graph is directed to avoid 1-way choke points
                            if is_dir and local_rng.random() < 0.5:
                                G.add_edge(v, u)
                            else:
                                G.add_edge(u, v)

                        # Merge the island into the main body for the next iterations
                        main_body.extend(island_nodes)

            # Lock in Ground Truth (Output of Phase 3)
            G.remove_edges_from(list(nx.selfloop_edges(G)))
            if 'complete_adj' in G.graph:
                A_obs = self._sanitise_adj(nx.to_numpy_array(G, dtype=np.float32), nx.is_directed(G))
                # Background-forest and stitched edges live only in G
                A_true = np.maximum(G.graph['complete_adj'], A_obs)
                G_true = nx.DiGraph() if is_dir else nx.Graph()
                G_true.add_nodes_from(G.nodes(data=True))
                G_true.graph.update(G.graph)
                I, J = np.where(A_true > 0)
                G_true.add_edges_from(zip(I.tolist(), J.tolist()))
                A_true = self._sanitise_adj(A_true, nx.is_directed(G_true))
            else:
                G_true = G.copy()
                A_true = self._sanitise_adj(
                    nx.to_numpy_array(G_true, dtype=np.float32), nx.is_directed(G_true)
                )
                A_obs = A_true.copy()

            if orientation_mode:
                A_obs = self._orient_to_directed(A_obs, mode=orientation_mode)

            if getattr(hooks, "feature_set", False):
                F_extra = self.extract_features(A_obs, feature_set=hooks.feature_set, directed=is_dir)
            else:
                F_extra = {}

            F = {k: (v.astype(np.float32) if v is not None else None) for k, v in F_extra.items()}
            L = fast_label_fn(A_obs, A_true, G_true).astype(np.float32, copy=False)
            if L.shape != A_obs.shape:
                raise ValueError(
                    f"[LABEL SHAPE] hooks.label_fn returned {L.shape} for graph {i}/{total} "
                    f"(type '{gtype}', adjacency {A_obs.shape}). Label matrices must match the "
                    f"observed adjacency exactly."
                )

            graphs.append(A_obs.astype(np.float32, copy=False))
            labels.append(L)
            feats.append(F)

            if self.show_progress:
                dt_graph = perf_counter() - t_graph
                recent.append((gtype, int(N), round(dt_graph, 4)))
                elapsed = perf_counter() - t0
                avg = elapsed / i
                print(
                    f"[Dataset generation] {i}/{total} | avg/graph={avg:.4f}s | elapsed={elapsed:.2f}s | recent={list(recent)}",
                    end="\n" if i == total else "\r",
                    flush=True,
                )

        class _GBBuilt(Dataset):
            def __init__(self, Gs, Ls, Fs):
                self.Gs = list(Gs)
                self.Ls = [np.array(l, copy=True) for l in Ls]
                self.Fs = list(Fs)

            def __len__(self):
                return len(self.Gs)

            def __getitem__(self, i):
                return self.Gs[i], self.Ls[i], self.Fs[i]

        return _GBBuilt(graphs, labels, feats)

    def make_loaders(
        self,
        dataset: Dataset,
        batch_size: int,
        ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        collate_fn = None,
        seed: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Split a dataset into train/val/test loaders according to the requested ratios.

        A zero ratio produces an empty split. Every positive-ratio split receives at least
        one sample when enough samples exist to populate all requested splits.
        """
        n = len(cast(Sized, dataset))
        idx = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

        r0, r1, r2 = (Decimal(str(float(r))) for r in ratios)
        if min(r0, r1, r2) < 0 or abs(r0 + r1 + r2 - 1) > Decimal("1e-6"):
            raise ValueError(f"[PIPELINE SPLIT CONFIG] ratios must be non-negative and sum to 1.0; got {tuple(ratios)}.")

        requested = (r0, r1, r2)
        if n < sum(r > 0 for r in requested):
            raise ValueError(
                f"[PIPELINE SPLIT CONFIG] {n} sample(s) cannot populate every split requested "
                f"by ratios={tuple(ratios)}. Each positive-ratio split requires at least one sample."
            )

        counts = [1 if r > 0 else 0 for r in requested]
        while sum(counts) < n:
            i = max(
                range(3),
                key=lambda j: requested[j] * n - counts[j]
            )
            counts[i] += 1

        a = counts[0]
        b = a + counts[1]
        tr, va, te = idx[:a], idx[a:b], idx[b:]

        train_ds = Subset(dataset, tr)
        val_ds   = Subset(dataset, va)
        test_ds  = Subset(dataset, te)

        # Fall back to cuda check if not explicitly provided
        pin = torch.cuda.is_available() if pin_memory is None else pin_memory
        g = None
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(int(seed))

        train_shuffle = len(train_ds) > 0
        persistent = bool(num_workers > 0 and getattr(dataset, "_pipeline_persistent_workers", False))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=train_shuffle, collate_fn=collate_fn, pin_memory=pin, generator=g, num_workers=num_workers, persistent_workers=persistent)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=pin, num_workers=num_workers, persistent_workers=persistent)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=pin, num_workers=num_workers, persistent_workers=persistent)
        return train_loader, val_loader, test_loader

    # ------------------------------------------------------------------ #
    # 3) Feature extraction (A_obs → features). Non-2D features are allowed.
    #    Downstream consumers may use 1D node features and/or 2D edge mats.
    # ------------------------------------------------------------------ #
    def extract_features(
            self,
            adj: np.ndarray,
            feature_set: Optional[Sequence[str]] = None,
            directed: bool = False
    ) -> Dict[str, np.ndarray]:
        if not feature_set:
            return {}

        feature_list = list(feature_set)
        deg = adj.sum(axis=1).astype(np.float32, copy=False) if (
            'degree' in feature_list or 'clustering_coeff' in feature_list
        ) else None

        feats: Dict[str, np.ndarray] = {}
        adj_f = adj.astype(np.float32, copy=False)
        if 'transpose' in feature_list:
            feats['transpose'] = adj_f.T

        # Handle explicit requests, e.g., 'power_2', without pulling in all powers
        need_a3 = 'triangles' in feature_list or 'clustering_coeff' in feature_list
        requested_powers = {k for k in (2, 3, 4, 5) if f'power_{k}' in feature_list}
        power_cache: Dict[int, np.ndarray] = {}
        if requested_powers or need_a3:
            power_cache[2] = adj_f @ adj_f
            if 2 in requested_powers:
                feats['power_2'] = power_cache[2]

            if 3 in requested_powers or need_a3:
                power_cache[3] = power_cache[2] @ adj_f
                if 3 in requested_powers:
                    feats['power_3'] = power_cache[3]

            if 4 in requested_powers or 5 in requested_powers:
                power_cache[4] = power_cache[2] @ power_cache[2]
                if 4 in requested_powers:
                    feats['power_4'] = power_cache[4]

            if 5 in requested_powers:
                feats['power_5'] = power_cache[4] @ adj_f

        if 'degree' in feature_list:
            feats['degree'] = deg  # 1D node-wise

        if 'triangles' in feature_list or 'clustering_coeff' in feature_list:
            A3 = power_cache[3]
            tri = np.diag(A3).copy()
            if not directed:
                tri /= 2
            if 'triangles' in feature_list:
                feats['triangles'] = tri  # 1D node-wise
            if 'clustering_coeff' in feature_list:
                if directed:
                    in_deg = adj_f.sum(axis=0)
                    reciprocal_deg = (adj_f * adj_f.T).sum(axis=1)
                    possible = deg * in_deg - reciprocal_deg
                else:
                    possible = deg * (deg - 1) / 2
                with np.errstate(divide='ignore', invalid='ignore'):
                    feats['clustering_coeff'] = np.where(possible > 0, tri / possible, 0.0)

        # Unique nodes at exactly distance 2. Outward reachability when directed.
        if 'twohop' in feature_list:
            A01 = (adj > 0).astype(np.float32)
            two = (A01 @ A01) > 0
            # Exclude the source and direct neighbours
            np.fill_diagonal(two, False)
            two &= ~(A01 > 0)
            feats['twohop'] = two.sum(axis=1).astype(np.float32)  # 1D node-wise

        # Pairwise link-prediction features (all 2D)
        pairwise_keys = [
            k for k in ('cn', 'jaccard', 'adamic_adar', 'deg_diff', 'deg_row', 'deg_col')
            if k in feature_list
        ]
        if pairwise_keys:
            A_t = torch.as_tensor(adj, dtype=torch.float32)
            batch = pairwise_batch_from_adj(A_t, pairwise_keys, is_directed=directed)
            for k, v in batch.items():
                feats[k] = v.cpu().numpy().astype(np.float32)

        return feats

    @staticmethod
    def _orient_to_directed(adj_matrix: np.ndarray, mode: str = "dag") -> np.ndarray:
        """
        Orients an adjacency matrix to a directed state.
        - 'dag': Enforces acyclicity by retaining only the upper triangle (lower ID to higher ID).
        """
        if mode == "dag":
            return np.triu(adj_matrix, 1)
        else:
            raise ValueError(f"Unknown orientation mode: {mode}")

    @staticmethod
    def _compile_label_fn(label_fn):
        if label_fn is None:
            return lambda A_obs, *_: np.zeros_like(A_obs, dtype=np.float32)

        sig = inspect.signature(label_fn)
        params = list(sig.parameters.values())
        param_names = [p.name for p in params]

        if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
            return label_fn

        mapping = []
        for p in param_names:
            p_lower = p.lower()
            if p_lower in {'a_obs', 'a', 'a_np', 'adj', 'adj_obs'}:
                mapping.append('A_obs')
            elif p_lower in {'a_true', 'adj_true'}:
                mapping.append('A_true')
            elif p_lower in {'g_true', 'g', '_g', 'graph'}:
                mapping.append('G_true')
            else:
                raise ValueError(f"Unknown parameter '{p}' in label_fn. "
                                 f"Allowed aliases must map to A_obs, A_true, or G_true.")

        slots = tuple({'A_obs': 0, 'A_true': 1, 'G_true': 2}[m] for m in mapping)

        def fast_label_fn(A_obs, A_true, G_true):
            available = (A_obs, A_true, G_true)
            return label_fn(*(available[s] for s in slots))

        return fast_label_fn
    
    @staticmethod
    def _sanitise_adj(mat: np.ndarray, is_directed: bool) -> np.ndarray:
        mat = (mat > 0).astype(np.float32)
        np.fill_diagonal(mat, 0.0)
        if not is_directed:
            mat = np.maximum(mat, mat.T)
        return mat
    
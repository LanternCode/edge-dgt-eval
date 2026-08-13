# SPDX-License-Identifier: CC-BY-SA-4.0

"""
gnn_bridge.py

Thin adapter to map raw graph tuples (A, node_feats, edge_feats, meta) into
model-ready tensors with consistent shape conventions, per-graph
standardisation via FeatureRegistry helpers, and shared utils.features helpers for
pairwise edge features. This module intentionally delegates:

  - Pairwise features -> utils.features.pairwise_batch_from_adj / utils.features.pairwise_for_pairs
  - Self-loops        -> Adjacency A is used as given for feature assembly and propagation.
                         The Edge-Masked Transformer adds temporary self-loops only to its
                         internal attention allow-mask. Supervision/evaluation policy is
                         handled by the shared masking logic.
  - Standardisation   -> per-graph z-scoring via FeatureRegistry.zscore_nodes_per_graph/zscore_edges_per_graph

This avoids duplicated math and keeps feature semantics aligned across models while sharing core preprocessing utilities.

Fe/Fn conventions:
  - X (N, Fn) stacks node features.
  - E (N, N, Fe) stacks edge features.
  - By default, 1D inputs and all non-square 2D inputs are treated as node features.
    Node feature matrices are assembled in (N, F) form; (F, N) inputs are transposed internally,
    where N is always a specific graph's own node count before batch padding - never the
    (possibly larger) padded batch size other graphs in the same batch may require.
    Undersized features are zero-padded to N, but providing features larger than N will raise an error.
    Square (N, N) inputs are treated as edge features only for canonical keys or custom keys declared as edge features.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from .EdgeClassification import (
    _format_duration, _normalise_feature_spec, _pick_probe_loader, _pos_weight, _reset_pipeline_runtime_state,
    effective_mask, FeatureRegistry, _task_to_meta_dict, finalise_summary,
    _resolve_loaders, get_optimal_threshold, save_pipeline_checkpoint
)
from ._utils.features import pairwise_for_pairs, pairwise_batch_from_adj, shortest_path_from_adj
from ._utils.run_lifecycle import begin_or_attach_run, install_boundary_hooks
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
    accuracy_score, balanced_accuracy_score
)
from typing import Dict, List, Optional, Sequence, Tuple, Any, Union
from dataclasses import dataclass
from types import SimpleNamespace
from torch.amp import GradScaler
from torch.utils.data import DataLoader


def _canonicalise_encoders(encoders: Sequence[str]) -> List[str]:
    """
    Map encoder aliases to canonical keys and drop duplicates while preserving order.
    Note: "transformer", "tx", and related aliases resolve to the GNN edge-masked
    transformer canonical key "edge_tx", not the dense PatchTransformer.
    """
    if isinstance(encoders, str):
        encoders = [encoders]

    alias_map = {
        "gcn": {"gcn", "graphconv", "graph_conv"},
        "sage": {"sage", "graphsage", "graph_sage", "gs"},
        "gin": {"gin"},
        "edge_tx": {"edge_tx", "edge-transformer", "edge_transformer", "transformer", "tx", "edge-tx"},
        "gps": {"gps", "graph_gps", "graph-gps"}
    }

    seen = set()
    canon: List[str] = []
    for name in encoders:
        n = str(name).strip().lower()
        resolved = None
        for key, aliases in alias_map.items():
            if n in aliases:
                resolved = key
                break

        if resolved is None:
            raise ValueError(f"Unknown encoder {name!r}; allowed: {sorted(alias_map.keys())}.")

        if resolved not in seen:
            seen.add(resolved)
            canon.append(resolved)
            
    return canon


def _coerce_train_cfg(cfg: Any):
    """
    Ensure cfg exposes GNNTrainConfig defaults while preserving any provided overrides.
    """
    # Prevent cross-run mutations
    if isinstance(cfg, dict):
        cfg = SimpleNamespace(**cfg)
    else:
        cfg = copy.copy(cfg)
        
    if not isinstance(cfg, GNNTrainConfig):
        base = GNNTrainConfig()
        for k, v in base.__dict__.items():
            if not hasattr(cfg, k):
                setattr(cfg, k, v)

    return cfg


def _random_walk_structural_encoding(
    A: torch.Tensor,
    node_mask: torch.Tensor,
    steps: int,
    is_directed: bool
) -> torch.Tensor:
    """
    Compute RWSE columns diag(P^1), ..., diag(P^k) from the valid-node subgraph.

    This is intentionally not a canonical task feature. Callers opt in by passing
    rwse_steps > 0, allowing GPS to own the encoding without changing feature_set=True.
    """
    steps = int(steps)
    N = int(A.size(0))
    dev = A.device

    if steps <= 0:
        return torch.zeros((N, 0), device=dev, dtype=torch.float32)

    rwse = torch.zeros((N, steps), device=dev, dtype=torch.float32)
    idx = torch.nonzero(node_mask.to(device=dev, dtype=torch.bool), as_tuple=False).view(-1)
    if int(idx.numel()) == 0:
        return rwse

    A_sub = (A[idx][:, idx].to(torch.float32) > 0.5).to(torch.float32)
    if not is_directed:
        A_sub = torch.maximum(A_sub, A_sub.t())

    with torch.amp.autocast(device_type=dev.type, enabled=False):
        deg = A_sub.sum(dim=-1, keepdim=True).clamp_min(1.0)
        P = (A_sub / deg).to(torch.float32)
        P_power = P

        for s in range(steps):
            rwse[idx, s] = torch.diagonal(P_power, 0).to(dtype=rwse.dtype)
            if s + 1 < steps:
                P_power = torch.matmul(P_power, P).to(torch.float32)

    return rwse


def _resolve_node_matrix_orientation(rows: int, cols: int, N_unpadded: int) -> Tuple[bool, int]:
    """
    Decide whether a 2D node feature tensor is already node-major (its first
    dimension is this graph's node count before batch padding) or arrived as
    (F, N_unpadded) and needs transposing.

    Adjacency matrices are padded to a shared batch size so graphs of different
    sizes can be stacked together; node feature tensors are not padded the same
    way at collation time. The padded adjacency size and N_unpadded are the same
    number only when this graph is the largest one in its batch, so this check
    needs N_unpadded specifically, not the padded size.

    Returns (needs_transpose, feature_count).
    """
    if rows != N_unpadded and cols == N_unpadded:
        return True, rows
    return False, cols


def _assemble_features_for_graph(
        A: torch.Tensor,
        feats: Dict[str, torch.Tensor],
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        schema: Optional[Dict[str, Any]] = None,
        lap_pe_k: int = 0,
        lap_pe_sign_flip: bool = False,
        rwse_steps: int = 0,
        build_edge_mats: bool = True,
        append_pairwise: bool = True,
        is_directed: bool = False,
        extra_edge_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build node features X and edge/pairwise features E for a single (possibly padded) graph

    Semantics:
      - `node_mask` indicates which nodes are real/valid (True) versus padding (False).
        Must be a 1D boolean tensor of length N (pre-normalised by the caller)
      - `edge_mask` indicates which (i, j) positions are "active" for downstream masking
        and per-graph edge z-scoring. In some tasks it corresponds to valid padded cells,
        and in others it corresponds to supervised/candidate edges. This function treats
        it consistently as an "active-edge" mask of shape (N, N) and:
          * masks E outside active positions, and
          * uses it as the mask for per-graph edge z-scoring

    Laplacian positional encodings:
      If `lap_pe_k > 0`, the function computes up to the top-k non-trivial eigenvectors
      of the normalised Laplacian on the subgraph induced by valid nodes. If fewer than
      k non-trivial vectors exist, the remainder is padded with zeros, ensuring X gains
      exactly `lap_pe_k` columns

    Random-walk structural encodings:
      If `rwse_steps > 0`, the function appends diag(P^1), ..., diag(P^k), where P is
      the row-stochastic random-walk matrix on the valid-node subgraph

    Args:
        A:
            Adjacency matrix for a single graph, shape (N, N). Values are typically 0/1
            (or probabilities). This function uses `A > 0.5` as the edge indicator when
            needed for Laplacian PE
        feats:
            Feature dictionary for this graph. Entries may include:
              - node features shaped (N, F) or (N,),
              - edge/pairwise features shaped (N, N),
              - scalars (treated as constants),
            and possibly other tensors; non-tensor / non-ndarray entries are ignored.
            If `schema` is provided, it controls which keys are used
        node_mask:
            Node validity mask, shape (N,) boolean. True = real node, False = padding.
            Must be pre-normalised by the caller (_prepare_features_batch)
        edge_mask:
            Active edge mask, shape (N, N) boolean. Used for adjacency redaction,
            E masking, and edge z-scoring. Must be pre-normalised by the caller
        schema:
            Optional schema containing "node_keys" and/or "edge_keys" lists that define
            which feature keys to include and in what order
        lap_pe_k:
            Number of Laplacian positional encoding eigenvectors to append to X
        lap_pe_sign_flip:
            If True, randomly flip each Laplacian eigenvector's sign before z-scoring
        rwse_steps:
            Number of random-walk structural encoding columns to append to X
        build_edge_mats:
            If False, the returned E will have shape (N, N, 0) and edge feature assembly
            is skipped
        append_pairwise:
            If True, append derived pairwise features (FeatureRegistry.HEAVY_PAIRWISE_KEYS)
            as supported by `pairwise_batch_from_adj`

    Returns:
        X:
            Node feature matrix of shape (N, F_node [+ lap_pe_k]), sanitised, masked on
            padded nodes, and per-graph z-scored (using `node_mask`)
        E:
            Edge/pairwise feature cube of shape (N, N, F_edge), sanitised, masked on
            inactive edges, and per-graph z-scored (using `edge_mask`). If
            `build_edge_mats` is False, shape is (N, N, 0)
    """
    N = int(A.size(0))
    dev = A.device

    # ----------------------------
    # 0) Masks (pre-normalised by _prepare_features_batch)
    # ----------------------------
    nm_vec = node_mask.to(device=dev, dtype=torch.bool)
    em_bool = edge_mask.to(device=dev, dtype=torch.bool)

    # This graph's node count before batch padding (N above may be the padded, batch-shared size)
    N_unpadded = int(nm_vec.sum().item())

    # Float forms used for multiplication
    em_f = em_bool.float()

    # ----------------------------
    # 1) Build Node Features (X)
    # ----------------------------
    def _as_node_cols(v: Any) -> torch.Tensor:
        """Coerce v into a (N, Fv) float32 tensor on device"""
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(v)
        elif torch.is_tensor(v):
            t = v
        else:
            t = torch.tensor(v)
        t = t.to(device=dev, dtype=torch.float32)

        if t.numel() == 1:
            return torch.full((N, 1), float(t.item()), device=dev, dtype=torch.float32)

        if t.dim() == 1:
            if t.shape[0] < N:
                t = torch.nn.functional.pad(t, (0, N - int(t.shape[0])))
            elif t.shape[0] > N:
                raise ValueError(
                    f"[FEATURE SHAPE] 1D Node feature has more elements ({t.shape[0]}) than the graph size N ({N}). "
                    f"Silent cropping is unsupported as it can cause undocumented side effects. "
                    f"If this is intended, please crop the feature manually before passing it to the pipeline."
                )
            return t.view(N, 1)

        if t.dim() > 2:
            t = t.view(t.shape[0], -1)

        # Custom node features may arrive as (F, N_unpadded)
        rows, feat = int(t.shape[0]), int(t.shape[1])
        needs_transpose, feat = _resolve_node_matrix_orientation(rows, feat, N_unpadded)
        if needs_transpose:
            t = t.transpose(0, 1).contiguous()
            rows = int(t.shape[0])

        if rows < N:
            t = torch.nn.functional.pad(t, (0, 0, 0, N - rows))
        elif rows > N:
            raise ValueError(
                f"[FEATURE SHAPE] 2D Node feature matrix has more rows ({rows}) than the graph size N ({N}). "
                f"Silent cropping is unsupported as it can cause undocumented side effects. "
                f"If this is intended, please crop the feature manually before passing it to the pipeline."
            )
        return t.to(torch.float32)

    node_cols: List[torch.Tensor] = []
    for k in schema["node_keys"]:
        v = feats.get(k, None)
        if v is None:
            f_dim = schema.get("node_dims", {}).get(k, 1)
            node_cols.append(torch.zeros((N, f_dim), device=dev, dtype=torch.float32))
        else:
            node_cols.append(_as_node_cols(v))

    X = torch.cat(node_cols, dim=1) if node_cols else torch.zeros((N, 0), device=dev, dtype=torch.float32)

    # ----------------------------
    # 2) Build Edge Features (E)
    # ----------------------------
    if not build_edge_mats:
        E = torch.zeros((N, N, 0), device=dev, dtype=torch.float32)
    else:
        def _as_edge_mat(v: Any) -> torch.Tensor:
            """Coerce v into an (N, N) float32 tensor on device, padding as needed."""
            if v is None:
                return torch.zeros((N, N), device=dev, dtype=torch.float32)

            t = torch.as_tensor(v, device=dev, dtype=torch.float32)
            
            rows, cols = int(t.shape[0]), int(t.shape[1])
            if rows > N or cols > N:
                raise ValueError(
                    f"[FEATURE SHAPE] 2D Edge feature matrix has shape ({rows}, {cols}) which exceeds the graph size ({N}, {N}). "
                    f"Silent cropping is unsupported as it can cause undocumented side effects. "
                    f"If this is intended, please crop the feature manually before passing it to the pipeline."
                )
            out = torch.zeros((N, N), device=dev, dtype=torch.float32)
            out[:rows, :cols] = t
            return out

        edge_keys = schema["edge_keys"]
        edge_mats: List[torch.Tensor] = []
        for k in edge_keys:
            if k == "shortest_path":
                edge_mats.append(shortest_path_from_adj(A, is_directed=is_directed))
                continue

            v = feats.get(k, None)
            edge_mats.append(_as_edge_mat(v))

        if append_pairwise:
            # Filter out heavy pairwise keys that are already provided in the schema
            needed_keys = [
                k for k in FeatureRegistry.HEAVY_PAIRWISE_KEYS
                if k not in edge_keys
            ]
            if needed_keys:
                for _v in pairwise_batch_from_adj(
                    A, needed_keys, is_directed=is_directed
                ).values():
                    edge_mats.append(_v.to(device=dev, dtype=torch.float32))

        E = torch.stack(edge_mats, dim=-1) if edge_mats else torch.zeros((N, N, 0), device=dev, dtype=torch.float32)

    # ----------------------------
    # 3) Positional / structural encodings
    # ----------------------------
    if lap_pe_k > 0:
        pe_fixed = torch.zeros((N, int(lap_pe_k)), device=dev, dtype=torch.float32)
        idx = torch.nonzero(nm_vec, as_tuple=False).view(-1)
        if int(idx.numel()) >= 2:
            A_sub = A[idx][:, idx].to(torch.float32)
            A_sym = torch.maximum(A_sub, A_sub.t())
            deg = A_sym.sum(-1)
            d_inv_sqrt = torch.pow(torch.clamp(deg, min=1e-8), -0.5)
            L = torch.eye(idx.numel(), device=dev, dtype=torch.float32) - (
                    d_inv_sqrt[:, None] * A_sym * d_inv_sqrt[None, :])
            _, evecs = torch.linalg.eigh(L)

            n_avail = max(0, int(evecs.shape[1]) - 1)
            n_keep = min(int(lap_pe_k), n_avail)
            if n_keep > 0:
                pe_fixed[idx, :n_keep] = evecs[:, 1: 1 + n_keep]

        if lap_pe_sign_flip:
            signs = torch.randint(0, 2, (1, int(lap_pe_k)), device=dev).to(torch.float32)
            signs = signs.mul_(2.0).sub_(1.0)
            pe_fixed = pe_fixed * signs

        X = torch.cat([X, pe_fixed], dim=1)

    if rwse_steps > 0:
        rwse = _random_walk_structural_encoding(
            A, nm_vec, steps=int(rwse_steps), is_directed=bool(is_directed)
        )
        X = torch.cat([X, rwse], dim=1)

    # ----------------------------
    # 4) Sanitise, mask, and per-graph z-score
    # ----------------------------
    X = torch.nan_to_num(X, nan=0.0)

    # Per-graph z-scoring (node-wise)
    X_b = X.unsqueeze(0)
    X_mask_b = nm_vec.view(1, N)
    X = FeatureRegistry.zscore_nodes_per_graph(X_b, mask=X_mask_b).squeeze(0)
    X = X * nm_vec.view(N, 1).to(X.dtype)

    # Sanitise and per-graph z-score (edge-wise) combined
    E_extra = None
    if E.numel():
        E = torch.nan_to_num(E, nan=0.0)
        E_b = E.unsqueeze(0)

        if extra_edge_mask is not None:
            xm_bool = extra_edge_mask.to(device=E.device, dtype=torch.bool)
            xm_f = xm_bool.to(E.dtype)
            E_extra = FeatureRegistry.zscore_edges_per_graph(E_b, mask=xm_bool.unsqueeze(0)).squeeze(0)
            E_extra = E_extra * xm_f.unsqueeze(-1)

        E_mask_b = em_bool.unsqueeze(0)
        E = FeatureRegistry.zscore_edges_per_graph(E_b, mask=E_mask_b).squeeze(0)
        E = E * em_f.unsqueeze(-1)
    elif extra_edge_mask is not None:
        E_extra = E

    if extra_edge_mask is not None:
        return X, (E, E_extra)
    return X, E


def _redact_supervised_edges(
    A_in: torch.Tensor,
    mask_in: torch.Tensor,
    zero_supervised: bool
) -> torch.Tensor:
    """Zeroes out supervised/candidate edges in the adjacency matrix before encoding."""
    if not zero_supervised:
        return A_in
    
    m = mask_in
    if m.dtype != torch.bool: 
        m = m > 0.5

    return A_in.masked_fill(m, 0.0)


# ------------------------------- GCN Encoder ----------------------------------
class _GCNEncoder(torch.nn.Module):
    """
    GCN-style encoder with residuals, per-layer normalisation, and Jumping Knowledge.
    Output dimension is ALWAYS `hidden` (even with JK concatenation).
    """
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        depth: int,
        dropout: float = 0.0,
        use_residual: bool = True,
        directed: bool = False,
        learnable_layer_norm: bool = True
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.dropout = float(dropout)
        self.use_residual = bool(use_residual)
        self.directed = bool(directed)

        self.in_lin = torch.nn.Linear(in_dim, hidden)
        self.convs  = torch.nn.ModuleList([torch.nn.Linear(hidden, hidden) for _ in range(depth)])
        self.norms = torch.nn.ModuleList([
            nn.LayerNorm(hidden, elementwise_affine=learnable_layer_norm) for _ in range(depth)
        ])
        self.drop   = torch.nn.Dropout(p=self.dropout)
        self.jk_proj = torch.nn.Linear((depth + 1) * hidden, hidden)

    @staticmethod
    def _norm_adj(
        A: torch.Tensor,
        node_mask: torch.Tensor,
        directed: bool
    ) -> torch.Tensor:
        """
        Dense normalisation:
        - directed: row-stochastic normalisation of A
        - undirected: D^{-1/2} A D^{-1/2}

        A is binarised with (A > 0.5), masked to valid nodes, and then normalised as provided
        """
        N = A.size(0)
        A = torch.gt(A, 0.5).to(dtype=torch.float32)
        m = node_mask.float().view(N, 1)
        A = A * (m @ m.t())

        if directed:
            deg = A.sum(-1, keepdim=True).clamp_min(1.0)
            return A / deg
        else:
            deg = A.sum(-1)
            d_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
            return d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)

    def forward_one(self, A: torch.Tensor, X: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """
        A:(N,N) (dense or sparse), X:(N,in_dim), node_mask:(N,)
        Returns Z:(N, hidden) — fixed width regardless of JK mode.
        """
        # Build propagation
        P = self._norm_adj(A, node_mask, self.directed)
        m_float = node_mask.unsqueeze(1).to(dtype=X.dtype)

        # layer 0
        H = F.relu(self.in_lin(X))
        H = self.drop(H)
        H = H * m_float

        states = [H]
        for lin, norm in zip(self.convs, self.norms):
            H_agg = torch.matmul(P, H.float()).to(H.dtype)
            H_new = lin(H_agg)
            if self.use_residual:
                H_new = H_new + H
            H_new = norm(H_new)
            H_new = F.relu(H_new)
            H_new = self.drop(H_new)
            H_new = H_new * m_float
            H = H_new
            states.append(H)

        # JK aggregation
        H_cat = torch.cat(states, dim=1)
        H_jk = self.jk_proj(H_cat) * m_float
        return H_jk

    def forward(self, A: torch.Tensor, X_batch: List[torch.Tensor], node_mask: torch.Tensor) -> List[torch.Tensor]:
        return [self.forward_one(A[b], X_batch[b], node_mask[b]) for b in range(A.size(0))]


# ------------------------------ GraphSAGE (mean) ------------------------------
class _SAGEEncoder(nn.Module):
    """
    Mean-aggregator GraphSAGE.
    For directed adjacency, uses the pipeline's row-wise convention:
    node i aggregates the mean of outgoing neighbors j where A[i, j] > 0.
    """
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        depth: int,
        out_dim: int,
        dropout: float = 0.1,
        learnable_layer_norm: bool = True
    ):
        super().__init__()
        self.self_lins  = nn.ModuleList([nn.Linear(in_dim if i == 0 else hidden, hidden) for i in range(depth)])
        self.neigh_lins = nn.ModuleList([nn.Linear(in_dim if i == 0 else hidden, hidden) for i in range(depth)])
        self.out = nn.Linear(hidden, out_dim)
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden, elementwise_affine=learnable_layer_norm) for _ in range(depth)
        ])
        self.drop = nn.Dropout(dropout)

    def forward_one(self, A: torch.Tensor, X: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        # Precompute normalised sparse propagation matrix before the loop
        P_sp = _norm_adj_sparse(A, node_mask)
        
        # Pre-compute expanded mask
        m_float = node_mask.unsqueeze(1).to(dtype=X.dtype)
        
        H = X
        for ls, ln, norm in zip(self.self_lins, self.neigh_lins, self.norms):
            # Sparse mean aggregation (fp32 spMM)
            with torch.amp.autocast(device_type="cuda", enabled=False):
                neigh = torch.sparse.mm(P_sp, H.to(torch.float32)).to(H.dtype)

            # SAGE combine + light stabilisation
            H = F.relu(ls(H) + ln(neigh))
            H = torch.nan_to_num(H)
            H = norm(H)
            H = self.drop(H)
            H = H * m_float

        return self.out(H) * m_float

    def forward(self, A: Union[torch.Tensor, List[torch.Tensor]], X_batch: List[torch.Tensor], node_mask: torch.Tensor) -> List[torch.Tensor]:
        return [self.forward_one(A[b], X_batch[b], node_mask[b]) for b in range(len(X_batch))]


# ------------------------------ GIN (sum aggregator) --------------------------
class _GINEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        depth: int,
        out_dim: int,
        eps_init: float = 0.0,
        dropout: float = 0.1,
        learnable_layer_norm: bool = True
    ):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps_init))
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim if i == 0 else hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden)
            ) for i in range(depth)
        ])
        self.out = nn.Linear(hidden, out_dim)
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden, elementwise_affine=learnable_layer_norm) for _ in range(depth)
        ])
        self.drop = nn.Dropout(dropout)

    def forward_one(self, A: torch.Tensor, X: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        A_bin = _masked_binary_sparse(A, node_mask)
        
        # Pre-compute expanded mask
        m_float = node_mask.unsqueeze(1).to(dtype=X.dtype)
        
        H = X
        for mlp, norm in zip(self.mlps, self.norms):
            with torch.amp.autocast(device_type="cuda", enabled=False):
                agg = torch.sparse.mm(A_bin, H.to(torch.float32))

            agg = agg.to(H.dtype)  # SUM over row-neighbors; for directed A, this is outgoing neighbors
            H = mlp((1.0 + self.eps) * H + agg)
            H = torch.nan_to_num(H)
            H = norm(H)
            H = self.drop(H)
            H = H * m_float

        return self.out(H) * m_float

    def forward(self, A: Union[torch.Tensor, List[torch.Tensor]], X_batch: List[torch.Tensor], node_mask: torch.Tensor) -> List[torch.Tensor]:
        return [self.forward_one(A[b], X_batch[b], node_mask[b]) for b in range(len(X_batch))]


# -------------------------- Edge-Masked Transformer ---------------------------
class _EdgeMaskedTransformer(nn.Module):
    """
    Transformer encoder over nodes with attention restricted to neighbors in (A+I).
    Strictly uses the dense path to maintain attention denominator equivalence.
    """
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        depth: int,
        heads: int,
        out_dim: int,
        dropout: float = 0.1,
        learnable_layer_norm: bool = True
    ):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        enc_layer.norm1 = nn.LayerNorm(hidden, elementwise_affine=learnable_layer_norm)
        enc_layer.norm2 = nn.LayerNorm(hidden, elementwise_affine=learnable_layer_norm)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.out = nn.Linear(hidden, out_dim)
        self.register_buffer("_eye_cache", torch.empty(0, 0, dtype=torch.bool), persistent=False)

    def forward_one(self, A: torch.Tensor, X: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        # Add self-loops for the attention allow-mask so every valid node can attend to itself.
        # This avoids degenerate all-masked attention rows (e.g., isolated nodes after masking) and
        # mirrors the common A+I convention used in graph message passing.
        N = A.shape[0]
        if self._eye_cache.device != A.device or self._eye_cache.size(0) < N:
            self._eye_cache = torch.eye(N, device=A.device, dtype=torch.bool)
        eye = self._eye_cache[:N, :N]
        valid = node_mask.to(dtype=torch.bool)

        allow = ((A > 0) | eye) & valid[:, None] & valid[None, :]
        padding_self = eye & (~valid)[:, None] & (~valid)[None, :]
        allow = allow | padding_self

        attn_block = ~allow
        H = self.inp(X).unsqueeze(0)  # (1, seq_len, H)
        H = self.enc(H, mask=attn_block)
        Z = self.out(H[0])
        Z = torch.nan_to_num(Z, nan=0.0)
        return Z * valid.unsqueeze(1).to(Z.dtype)

    def forward(self, A: Union[torch.Tensor, List[torch.Tensor]], X_batch: List[torch.Tensor], node_mask: torch.Tensor) -> List[torch.Tensor]:
        return [self.forward_one(A[b], X_batch[b], node_mask[b]) for b in range(len(X_batch))]


# ----------------------------- GPS Layer ------------------------------------
class _GPSLayer(nn.Module):
    """
    Single GPS layer: parallel local MPNN (GINE-style when edge features are available) and
    global multi-head self-attention, combined with pre-norm residual
    connections and a feed-forward network.
    """
    def __init__(
        self,
        hidden: int,
        heads: int,
        dropout: float = 0.1,
        learnable_layer_norm: bool = True,
        edge_dim: int = 0
    ):
        super().__init__()
        # Pre-norms for each branch
        self.norm_local = nn.LayerNorm(hidden, elementwise_affine=learnable_layer_norm)
        self.norm_attn = nn.LayerNorm(hidden, elementwise_affine=learnable_layer_norm)
        self.norm_ffn = nn.LayerNorm(hidden, elementwise_affine=learnable_layer_norm)
        self.edge_dim = int(edge_dim)

        # Local MPNN branch: GINE-style when edge features are available
        self.eps = nn.Parameter(torch.tensor(0.0))
        self.local_nn = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.local_edge_lin = nn.Linear(self.edge_dim, hidden) if self.edge_dim > 0 else None

        # Global multi-head self-attention branch
        self.attn = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
            nn.Dropout(dropout)
        )

        self.drop_local = nn.Dropout(dropout)
        self.drop_attn = nn.Dropout(dropout)

    def forward(
        self,
        H: torch.Tensor,
        A_sparse: torch.Tensor,
        node_mask: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            H: (N, hidden) node representations.
            A_sparse: sparse (N, N) binary adjacency for local message passing.
            node_mask: (N,) boolean validity mask.
            edge_attr: optional (E, Fe) edge features aligned with A_sparse.indices().
        Returns:
            Updated node representations (N, hidden).
        """
        m_float = node_mask.unsqueeze(1).to(H.dtype)

        # --- Local MPNN branch (GIN-style, pre-norm) ---
        H_ln = self.norm_local(H)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            A_coo = A_sparse.coalesce()
            if self.local_edge_lin is not None and edge_attr is not None and edge_attr.numel():
                row, col = A_coo.indices()
                edge_msg = self.local_edge_lin(
                    edge_attr.to(device=H.device, dtype=self.local_edge_lin.weight.dtype)
                ).to(H_ln.dtype)
                msg = F.relu(H_ln[col] + edge_msg)
                agg = torch.zeros_like(H_ln, dtype=torch.float32)
                agg.index_add_(0, row, msg.to(torch.float32))
                agg = agg.to(H.dtype)
            else:
                agg = torch.sparse.mm(A_coo, H_ln.to(torch.float32)).to(H.dtype)
        h_local = self.local_nn((1.0 + self.eps) * H_ln + agg)
        h_local = torch.nan_to_num(h_local)
        h_local = self.drop_local(h_local) * m_float

        # --- Global attention branch (pre-norm) ---
        H_ln2 = self.norm_attn(H)
        H_seq = H_ln2.unsqueeze(0)                      # (1, N, D) for batch_first
        key_pad = (~node_mask).unsqueeze(0)              # (1, N), True=ignore
        h_global, _ = self.attn(
            H_seq, H_seq, H_seq,
            key_padding_mask=key_pad,
            need_weights=False
        )
        h_global = torch.nan_to_num(h_global.squeeze(0))
        h_global = self.drop_attn(h_global) * m_float

        # --- Residual combine ---
        H = H + h_local + h_global

        # --- FFN (pre-norm) ---
        H = H + self.ffn(self.norm_ffn(H)) * m_float

        return H * m_float


# ---------------------- GPS (Graph Transformer) Encoder ---------------------
class _GPSEncoder(nn.Module):
    """
    GPS (General, Powerful, Scalable) Graph Transformer encoder
    (Rampasek et al., NeurIPS 2022).

    Each layer combines:
      1. A local GINE-style MPNN over graph neighbours when edge features are available,
      2. Global multi-head self-attention over all valid nodes,
      3. A feed-forward network,
    all with pre-norm residual connections.

    The local branch provides structural inductive bias while the global
    branch gives every node an unrestricted receptive field in a single
    layer. Sparse adjacency routing follows the SAGE/GIN convention.
    """
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        depth: int,
        heads: int,
        out_dim: int,
        dropout: float = 0.1,
        learnable_layer_norm: bool = True,
        edge_dim: int = 0
    ):
        super().__init__()
        self.edge_dim = int(edge_dim)
        self.in_proj = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([
            _GPSLayer(hidden, heads, dropout, learnable_layer_norm, edge_dim=self.edge_dim)
            for _ in range(depth)
        ])
        self.out_proj = nn.Linear(hidden, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward_one(
        self,
        A: torch.Tensor,
        X: torch.Tensor,
        node_mask: torch.Tensor,
        E: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            A: (N, N) sparse binary adjacency (routed by _prepare_adj_for_encoder).
            X: (N, in_dim) per-graph z-scored node features.
            node_mask: (N,) boolean validity mask.
            E: optional (N, N, Fe) edge features for local GPS message passing.
        Returns:
            Z: (N, out_dim) node embeddings.
        """
        A_bin = _masked_binary_sparse(A, node_mask)
        m_float = node_mask.unsqueeze(1).to(X.dtype)
        edge_attr = None
        if E is not None and self.edge_dim > 0 and E.numel():
            row, col = A_bin.indices()
            edge_attr = E[row, col]

        H = F.relu(self.in_proj(X))
        H = self.drop(H) * m_float

        for layer in self.layers:
            H = layer(H, A_bin, node_mask, edge_attr=edge_attr)

        return self.out_proj(H) * m_float

    def forward(
        self,
        A: Union[torch.Tensor, List[torch.Tensor]],
        X_batch: List[torch.Tensor],
        node_mask: torch.Tensor,
        E_batch: Optional[List[torch.Tensor]] = None
    ) -> List[torch.Tensor]:
        return [
            self.forward_one(A[b], X_batch[b], node_mask[b], None if E_batch is None else E_batch[b])
            for b in range(len(X_batch))
        ]


def _masked_binary_sparse(A_sp: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Coalesce, mask by valid nodes, binarise. Returns sparse COO."""
    A_coo = A_sp.coalesce()
    r, c = A_coo.indices()
    N = int(A_coo.size(0))

    keep = node_mask[r] & node_mask[c]
    r, c = r[keep], c[keep]

    idx = torch.stack([r, c], dim=0)
    ones = torch.ones(idx.size(1), device=A_coo.device, dtype=torch.float32)
    return torch.sparse_coo_tensor(
        idx, ones, size=(N, N), device=A_coo.device, dtype=torch.float32
    ).coalesce()


def _norm_adj_sparse(A_sp, node_mask):
    """
    Row-stochastic sparse normalisation of binary adjacency.

    Directed convention:
      - A[i, j] = 1 means an edge i -> j.
      - Row i is normalised by out-degree(i).
      - Therefore sparse.mm(P, H)[i] is the mean over node i's outgoing
        neighbors / successors j with A[i, j] > 0.

    This matches the directed GCN path's row-stochastic convention.
    """
    A_bin = _masked_binary_sparse(A_sp, node_mask)
    r, c = A_bin.indices()
    v = A_bin.values()
    N = int(A_bin.size(0))

    deg = torch.zeros(N, device=A_bin.device, dtype=torch.float32)
    deg.index_add_(0, r, v)
    deg = deg.clamp_min(1.0)
    v = v / deg[r]

    return torch.sparse_coo_tensor(
        torch.stack([r, c], dim=0), v, size=(N, N),
        device=A_bin.device, dtype=torch.float32
    ).coalesce()


# ------------------------------- Edge Decoder ---------------------------------
class _EdgeFeatureDecoder(nn.Module):
    """
    Decodes edge scores from node embeddings and edge features.

    The scoring function concatenates:
      [z_i, z_j, |z_i - z_j|, z_i * z_j, E_ij]
    and passes this vector through an MLP.
    """
    def __init__(self, dim: int, edge_dim: int, hidden: int = 128, directed: bool = False, num_classes: int = 1):
        """
        Args:
            dim: Dimension of node embeddings (d).
            edge_dim: Dimension of edge features (Fe).
            hidden: Hidden dimension for the scoring MLP.
            directed: If False, the output logits are symmetrised (s_ij + s_ji)/2.
            num_classes: Number of output classes. If >1, output dim is K. If <=1, output dim is 1.
        """
        super().__init__()
        self.directed = directed
        self.num_classes = num_classes

        # Input: 4 raw combinations of node embeddings + original edge features
        in_dim = 4 * dim + edge_dim
        out_dim = num_classes if num_classes > 1 else 1

        self.scorer = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, Z_batch: List[torch.Tensor], node_mask: torch.Tensor, E_batch: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            Z_batch: List of (N, d) node embeddings per graph.
            node_mask: (B, N) bool mask indicating valid nodes.
            E_batch: List of (N, N, Fe) edge feature matrices per graph.

        Returns:
            Logits tensor of shape (B, N, N) if binary, or (B, N, N, K) if multiclass.
        """
        if not self.directed and len(Z_batch) > 0:
            N_max = Z_batch[0].shape[0]
            if getattr(self, "_cached_tril_mask", None) is None or self._cached_tril_mask.shape[0] != N_max:
                self._cached_tril_mask = torch.tril(torch.ones(N_max, N_max, dtype=torch.bool, device=Z_batch[0].device), diagonal=-1)
            tril_mask = self._cached_tril_mask
        else:
            tril_mask = None

        outs = []
        for b, Z in enumerate(Z_batch):
            N, _ = Z.shape
            nm_bool = node_mask[b].bool()
            mask_2d = nm_bool[:, None] & nm_bool[None, :]

            if not self.directed:
                mask_2d = torch.triu(mask_2d, diagonal=1)

            # Get indices of only the valid (i, j) pairs
            row_idx, col_idx = torch.nonzero(mask_2d, as_tuple=True)
            if row_idx.numel() == 0:
                outs.append(torch.zeros((N, N) if self.num_classes <= 1 else (N, N, self.num_classes), 
                                        device=Z.device, dtype=Z.dtype))
                continue

            # Extract only valid node embeddings
            zi = Z[row_idx]
            zj = Z[col_idx]

            # Extract only valid edge features
            if not self.directed:
                E_valid = E_batch[b][row_idx, col_idx]
                E_valid_rev = E_batch[b][col_idx, row_idx]
            else:
                E_valid = E_batch[b][row_idx, col_idx]

            # Concatenate as a flattened 2D tensor (V, features)
            diff_abs = (zi - zj).abs()
            prod = zi * zj
            phi_valid = torch.cat([zi, zj, diff_abs, prod, E_valid], dim=-1)

            # Run the MLP only on valid pairs
            s_valid = self.scorer(phi_valid)

            # Evaluate reverse orientation mathematically identically to score_pairs
            if not self.directed:
                phi_rev = torch.cat([zj, zi, diff_abs, prod, E_valid_rev], dim=-1)
                s_valid = 0.5 * (s_valid + self.scorer(phi_rev))

            # Scatter back into a dense N x N grid
            out_dim = s_valid.size(-1)
            s = torch.zeros((N, N, out_dim), device=Z.device, dtype=s_valid.dtype)
            s[row_idx, col_idx] = s_valid

            # Enforce symmetry for undirected graphs
            if not self.directed:
                s_T = s.transpose(0, 1)
                s[tril_mask] = s_T[tril_mask]

            # Remove singleton dimension for binary tasks -> (N, N)
            if self.num_classes <= 1:
                s = s.squeeze(-1)

            outs.append(s)

        # Stack batch -> (B, N, N) or (B, N, N, K)
        return torch.stack(outs, dim=0)
    
    def score_pairs(self, z_u, z_v, edge_feats, edge_feats_rev=None):
        """Score M specific pairs without materialising N×N."""
        diff_abs = (z_u - z_v).abs()
        prod = z_u * z_v
        phi = torch.cat([z_u, z_v, diff_abs, prod, edge_feats], dim=-1)
        s = self.scorer(phi)
        if not self.directed:
            ef_rev = edge_feats_rev if edge_feats_rev is not None else edge_feats
            phi_rev = torch.cat([z_v, z_u, diff_abs, prod, ef_rev], dim=-1)
            s = 0.5 * (s + self.scorer(phi_rev))

        return s.squeeze(-1)


def _prepare_features_batch(
    A: torch.Tensor,
    feats: List[Dict[str, torch.Tensor]],
    mask: torch.Tensor,
    schema: Dict[str, Any],
    lap_pe_k: int = 0,
    lap_pe_sign_flip: bool = False,
    rwse_steps: int = 0,
    build_edge_mats: bool = True,
    append_pairwise: bool = True,
    is_directed: bool = False,
    build_local_edge_mats: bool = False
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """
    Assemble per-graph node/edge feature tensors for a batched input.

    This function converts a padded batch (A, feats, mask) into lists of per-graph
    tensors expected by the GNN encoder, and it returns a per-graph node validity mask

    A key subtlety is that `mask` is used across tasks with different semantics:
      - In many tasks, `mask` acts like a padding/validity mask over (i, j) pairs.
      - In some tasks (e.g. "predict only missing edges"), `mask` is a supervision /
        candidate-edge mask and can legitimately be all-False for an existing node's
        entire row/column (e.g., a fully-connected node has no missing edges)
    Due to this, node validity  comes from the `_N` metadata recorded by `collate_fn_pad`
    for every sample, which is exact regardless of mask semantics.

    Args:
        A:
            Batched adjacency tensor of shape (B, N, N). Values are typically 0/1
            (or probabilities). Non-zero entries indicate observed edges
        feats:
            A list of length B containing per-graph feature dictionaries.

            Feature dict values may include node features shaped (N, F), scalar node
            features shaped (N,) and/or edge/pairwise features shaped (N, N).
            This function does not assume a fixed schema; `schema` controls how
            `_assemble_features_for_graph` interprets entries
        mask:
            Batched edge mask tensor of shape (B, N, N). Semantics depend on the task:
              - If used as a validity mask, True entries indicate valid (i, j) positions.
              - If used as a supervision/candidate mask, True entries indicate which
                (i, j) pairs are eligible for loss/eval.
        schema:
            Schema describing which features exist and how to assemble them
            into node and edge feature tensors. Passed through to
            `_assemble_features_for_graph`
        lap_pe_k:
            If > 0, include Laplacian positional encodings with k eigenvectors
            (as supported by `_assemble_features_for_graph`)
        lap_pe_sign_flip:
            If True, randomly flip each Laplacian eigenvector's sign before z-scoring
        rwse_steps:
            Number of random-walk structural encoding columns to append to each Xb
        build_edge_mats:
            If True, construct edge/pairwise feature matrices Eb for each graph.
            If False, Eb may be returned as an empty/placeholder tensor depending on
            `_assemble_features_for_graph`
        append_pairwise:
            If True, append any derived pairwise features (e.g., from adjacency) as
            supported by `_assemble_features_for_graph`
        is_directed:
            Whether the graphs are directed. Affects pairwise feature assembly

    Returns:
        X_batch:
            List of length B. Each element is the per-graph node-feature tensor Xb
            assembled by `_assemble_features_for_graph` (typically shape (N, Fn))
        E_batch:
            List of length B. Each element is the per-graph edge/pairwise-feature tensor
            Eb assembled by `_assemble_features_for_graph` (often shape (N, N, Fe) or
            another encoder-specific format)
        node_mask_batch:
            Boolean tensor of shape (B, N) indicating which nodes are real/valid (True)
            versus padding (False). Importantly, a node can be valid even if it has zero
            masked pairs in `mask` (e.g., fully-connected under a "non-edges only" mask)
    """
    B, N, _ = A.shape
    dev = A.device

    X_batch: List[torch.Tensor] = []
    E_batch: List[torch.Tensor] = []
    node_mask_batch = torch.zeros((B, N), dtype=torch.bool, device=dev)
    seq_idx = torch.arange(N, device=dev)
    mask_bool = mask.to(device=dev, dtype=torch.bool)
    if not is_directed:
        # effective_mask() treats (i, j) and (j, i) as one supervised pair, so masks differing
        # only in orientation must yield identical edge features and z-score statistics.
        # The decoder also reads E[j, i] for its reverse term.
        mask_bool = mask_bool | mask_bool.transpose(1, 2)

    for b in range(B):
        fdict = feats[b]
        edge_mask_b = mask_bool[b]
        Ab = A[b]

        # collate_fn_pad records the unpadded node count for every sample and pads A with zeros and mask with False
        true_node_count = int(fdict["_N"]) if isinstance(fdict, dict) else 0
        node_mask_b = (seq_idx < true_node_count)

        local_mask_b = None
        if build_local_edge_mats:
            local_mask_b = (Ab > 0.5) & node_mask_b[:, None] & node_mask_b[None, :]

        Xb, Eb = _assemble_features_for_graph(
            Ab, fdict, node_mask_b, edge_mask_b,
            schema=schema,
            lap_pe_k=int(lap_pe_k),
            lap_pe_sign_flip=bool(lap_pe_sign_flip),
            rwse_steps=int(rwse_steps),
            build_edge_mats=bool(build_edge_mats),
            append_pairwise=bool(append_pairwise),
            is_directed=bool(is_directed),
            extra_edge_mask=local_mask_b
        )

        X_batch.append(Xb)
        E_batch.append(Eb)
        node_mask_batch[b] = node_mask_b

    return X_batch, E_batch, node_mask_batch


# --------------------------- Unified Bridge Wrapper ---------------------------
class GraphEdgeClassifier(nn.Module):
    """
    Internal bridge model that wraps a node encoder and an edge decoder.
    This class is orchestrated exclusively by the GNN suite runners 
    (`train_one_gnn`, `train_one_gnn_edges`). It relies on dynamically injected 
    feature schemas and cannot be instantiated directly by the user.

    Flow:
      1. Prepare features (standardise, mask).
      2. Encode nodes -> Z (List of N x d).
      3. Decode edges -> Logits (B x N x N [x K]).
    """
    def __init__(
        self,
        encoder_type: str = "gcn",
        hidden: int = 64,
        depth: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        directed: bool = False,
        edge_hidden: int = 128,
        num_classes: int = 1,
        learnable_layer_norm: bool = True
    ):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        self.hidden = hidden
        self.depth = depth
        self.heads = heads
        self.dropout = dropout
        self.directed = directed
        self.edge_hidden = edge_hidden
        self.num_classes = num_classes
        self.learnable_layer_norm = bool(learnable_layer_norm)
        self.lap_pe_k = 0
        self.gps_lap_pe_k = 16
        self.gps_lap_pe_sign_flip = True
        self.gps_rwse_steps = 16

        # Initialised lazily based on input feature dimensions
        self.enc: Optional[nn.Module] = None
        self.dec: Optional[_EdgeFeatureDecoder] = None

    def _build_encoder(self, in_dim: int, edge_dim: int = 0) -> torch.nn.Module:
        """Factory for the node encoder based on canonical 'encoder_type'."""
        et = self.encoder_type

        if et == "gcn":
            return _GCNEncoder(
                in_dim=int(in_dim),
                hidden=int(self.hidden),
                depth=int(self.depth),
                dropout=float(self.dropout),
                use_residual=True,
                directed=bool(getattr(self, "directed", False)),
                learnable_layer_norm=bool(getattr(self, "learnable_layer_norm", True))
            )

        elif et == "sage":
            return _SAGEEncoder(
                in_dim=int(in_dim),
                hidden=int(self.hidden),
                depth=int(self.depth),
                out_dim=int(self.hidden),
                dropout=float(self.dropout),
                learnable_layer_norm=bool(getattr(self, "learnable_layer_norm", True))
            )

        elif et == "gin":
            return _GINEncoder(
                in_dim=int(in_dim),
                hidden=int(self.hidden),
                depth=int(self.depth),
                out_dim=int(self.hidden),
                dropout=float(self.dropout),
                learnable_layer_norm=bool(getattr(self, "learnable_layer_norm", True))
            )

        elif et == "edge_tx":
            return _EdgeMaskedTransformer(
                in_dim=int(in_dim),
                hidden=int(self.hidden),
                depth=int(self.depth),
                heads=int(self.heads),
                out_dim=int(self.hidden),
                dropout=float(self.dropout),
                learnable_layer_norm=bool(getattr(self, "learnable_layer_norm", True))
            )

        elif et == "gps":
            return _GPSEncoder(
                in_dim=int(in_dim),
                hidden=int(self.hidden),
                depth=int(self.depth),
                heads=int(self.heads),
                out_dim=int(self.hidden),
                dropout=float(self.dropout),
                learnable_layer_norm=bool(getattr(self, "learnable_layer_norm", True)),
                edge_dim=int(edge_dim)
            )

        else:
            raise ValueError(f"Unknown encoder_type={et!r}")

    def _build_decoder(self, node_dim: int, edge_dim: int) -> _EdgeFeatureDecoder:
        """Factory for the edge decoder using measured embedding dimensions."""
        return _EdgeFeatureDecoder(
            dim=node_dim,
            edge_dim=edge_dim,
            hidden=self.edge_hidden,
            directed=self.directed,
            num_classes=self.num_classes
        )

    def _effective_lap_pe_k(self) -> int:
        """Resolve the universal LapPE width, with GPS enforcing a model-owned minimum."""
        lap_pe_k = int(getattr(self, "lap_pe_k", 0))
        if self.encoder_type == "gps":
            lap_pe_k = max(lap_pe_k, int(getattr(self, "gps_lap_pe_k", 16)))
        return lap_pe_k

    def _effective_rwse_steps(self) -> int:
        """Resolve the GPS-owned RWSE width without affecting other encoders."""
        if self.encoder_type == "gps":
            return int(getattr(self, "gps_rwse_steps", 0))
        return 0

    def _lap_pe_sign_flip_enabled(self) -> bool:
        """Apply sign augmentation only to GPS-owned LapPE during training."""
        return (
            self.training
            and self.encoder_type == "gps"
            and bool(getattr(self, "gps_lap_pe_sign_flip", True))
        )

    def _prepare_adj_for_encoder(
            self,
            A: torch.Tensor,
            *,
            as_list: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Apply training-time DropEdge and encoder-specific adjacency routing."""
        p_drop = float(getattr(self, "dropedge_p", 0.0))
        A_for_enc = _drop_edges(A, p_drop, directed=bool(self.directed)) if self.training and p_drop > 0 else A

        if self.encoder_type in ("sage", "gin", "gps"):
            if as_list:
                if A_for_enc.dim() == 3:
                    return [A_for_enc[b] if A_for_enc[b].is_sparse else A_for_enc[b].to_sparse_coo()
                            for b in range(A_for_enc.size(0))]
            elif not A_for_enc.is_sparse:
                return A_for_enc.to_sparse_coo()

        return A_for_enc
        
    def _encode_batch(
            self,
            A: torch.Tensor,
            feats: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
            mask: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Shared encode pipeline: schema check, feature assembly, lazy encoder/decoder init, encode.

        Returns:
            Z_batch: List of (N, d) node embeddings per graph.
            E_batch: List of (N, N, Fe) edge features per graph.
            node_mask: (B, N) bool validity mask.
        """
        schema = getattr(self, "feature_schema", None)
        if schema is None:
            raise RuntimeError(
                "[PIPELINE] GNN execution requires a resolved feature_schema from the pipeline. "
                "Schema-less structural feature inference is unsupported."
            )

        is_gps = self.encoder_type == "gps"
        X_batch, E_batch, node_mask = _prepare_features_batch(
            A, feats, mask, schema=schema,
            lap_pe_k=self._effective_lap_pe_k(),
            lap_pe_sign_flip=self._lap_pe_sign_flip_enabled(),
            rwse_steps=self._effective_rwse_steps(),
            build_edge_mats=True,
            append_pairwise=True,
            is_directed=bool(self.directed),
            build_local_edge_mats=is_gps
        )

        E_local_batch = None
        if is_gps:
            E_local_batch = [pair[1] for pair in E_batch]
            E_batch = [pair[0] for pair in E_batch]

        dev = A.device
        Fn = int(X_batch[0].size(1))
        Fe = int(E_batch[0].size(-1)) if (E_batch and E_batch[0].numel()) else 0
        if self.enc is None:
            self.enc = self._build_encoder(Fn, edge_dim=Fe).to(dev)

        A_for_enc = self._prepare_adj_for_encoder(
            A, as_list=self.encoder_type in ("sage", "gin", "gps")
        )

        if self.encoder_type == "gps":
            Z_batch = self.enc(A_for_enc, X_batch, node_mask, E_local_batch)
        else:
            Z_batch = self.enc(A_for_enc, X_batch, node_mask)

        d = int(Z_batch[0].size(1))
        for Eb in E_batch:
            if Eb.numel() and int(Eb.size(-1)) != Fe:
                raise ValueError(
                    f"[MODEL ERROR] Inconsistent edge feature width across batch. "
                    f"The full-matrix GNN decoder expects a single schema-fixed Fe per batch; "
                    f"expected Fe={Fe}, got Fe={int(Eb.size(-1))}."
                )
        if self.dec is None:
            self.dec = self._build_decoder(d, Fe).to(dev)

        return Z_batch, E_batch, node_mask

    def forward(
            self,
            A: torch.Tensor,
            feats: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
            mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Executes the full encode-decode pass for the edge classifier.

        Adjacency routing note:
            For SAGE, GIN, and GPS, adjacency is routed to sparse COO before encoder use.
            This may look like an optimisation target but it is not treated as a
            planned change by itself. Any attempt to alter this path should be
            justified by profiling on a supported code path and must not
            change DropEdge, masking, or training semantics.

        Args:
            A: (B, N, N) Adjacency matrix.
            feats: List of feature dicts per graph.
            mask: (B, N, N) Validity mask.

        Returns:
            Logits: (B, N, N) or (B, N, N, K).
        """
        Z_batch, E_batch, node_mask = self._encode_batch(A, feats, mask)
        return self.dec(Z_batch, node_mask, E_batch)

    def score_pairs_selected(
        self,
        A: torch.Tensor,
        feats: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
        mask: torch.Tensor,
        idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes logits only for the selected (b, i, j) pairs using the full-matrix
        feature pipeline, without scattering back into a dense N×N grid.

        Returns:
            Logits: (M,) if binary, or (M, K) if multiclass.
        """
        Z_batch, E_batch, _ = self._encode_batch(A, feats, mask)
        dev = A.device

        parts = []
        for b, Z in enumerate(Z_batch):
            keep = idx[:, 0] == b
            if not torch.any(keep):
                continue

            src = idx[keep, 1]
            dst = idx[keep, 2]
            z_u = Z[src]
            z_v = Z[dst]
            edge_feats = E_batch[b][src, dst]
            edge_feats_rev = E_batch[b][dst, src] if not self.directed else None
            parts.append(self.dec.score_pairs(z_u, z_v, edge_feats, edge_feats_rev))

        if not parts:
            if self.num_classes > 1:
                return torch.empty((0, self.num_classes), device=dev, dtype=Z_batch[0].dtype)
            return torch.empty((0,), device=dev, dtype=Z_batch[0].dtype)

        return torch.cat(parts, dim=0)

    def score_pairs_on_demand(
            self,
            Z: torch.Tensor,
            A: torch.Tensor,
            src: torch.Tensor,
            dst: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes logits only for specific pairs (src, dst).
        Used for memory efficiency on large graphs.

        Returns:
            Logits: (M,) if binary, or (M, K) if multiclass.
        """
        dev = Z.device
        M = src.numel()

        # Compute degrees
        row_deg = (A > 0.5).to(Z.dtype).sum(dim=1)
        col_deg = (A > 0.5).to(Z.dtype).sum(dim=0) if getattr(self, "directed", False) else row_deg
        d_u = row_deg[src]
        d_v = col_deg[dst]
        deg_diff = (d_u - d_v).abs().view(-1, 1)

        # Compute structural features
        pairwise_keys = list(FeatureRegistry.HEAVY_PAIRWISE_KEYS)
        pairwise_feats = pairwise_for_pairs(
            A, src, dst, pairwise_keys,
            is_directed=getattr(self, "directed", False),
            row_deg=row_deg.to(torch.float32),
            col_deg=col_deg.to(torch.float32)
        )
        cn = pairwise_feats["cn"]
        jacc = pairwise_feats["jaccard"]
        aa = pairwise_feats["adamic_adar"]

        # Fetch node embeddings
        z_u = Z[src]
        z_v = Z[dst]

        # Gather structural features as the edge feature vector
        edge_feats = torch.cat([
            d_u.view(-1, 1), d_v.view(-1, 1), deg_diff,
            cn.view(-1, 1), jacc.view(-1, 1), aa.view(-1, 1)
        ], dim=1)

        # Build reverse edge features with swapped per-node slots (d_u <-> d_v)
        edge_feats_rev = None
        if not self.directed:
            edge_feats_rev = torch.cat([
                d_v.view(-1, 1), d_u.view(-1, 1), deg_diff,
                cn.view(-1, 1), jacc.view(-1, 1), aa.view(-1, 1)
            ], dim=1)

        return self.dec.score_pairs(z_u, z_v, edge_feats, edge_feats_rev)

    def encode_only(self, A: torch.Tensor, X_batch, node_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns node embeddings Z for a single graph or batch size 1.
        Useful for running the encoder once before scoring many pairs on demand.
        """
        # Normalise inputs to single graph format (N, N)
        if A.dim() == 3:
            A = A[0]

        if node_mask.dim() == 2:
            node_mask = node_mask[0]

        X_batch = list(X_batch)
        if len(X_batch) == 0:
            raise ValueError("encode_only expects non-empty X_batch.")

        # Extract the single graph's feature matrix directly
        dev = A.device
        X = X_batch[0]
        if X.dim() == 1:
            X = X.view(-1, 1)

        # Lazy init encoder
        Fn = int(X.size(1))
        if self.enc is None:
            self.enc = self._build_encoder(Fn).to(dev)

        # Apply training-time adjacency preparation for the encoder
        A_for_enc = self._prepare_adj_for_encoder(A)

        Z = self.enc.forward_one(A_for_enc, X, node_mask)
        
        # Explicitly initialise the decoder during warmup when pairwise_on_demand=True so the optimiser tracks it
        if self.dec is None and getattr(self, "pairwise_on_demand", False):
            # In on-demand mode, decoder edge features are fixed: [d_u, d_v, deg_diff, cn, jaccard, adamic_adar]
            self.dec = self._build_decoder(node_dim=Z.size(1), edge_dim=6).to(dev)

        return Z


# ------------------------------ Training Utilities ----------------------------
# GNN suites use a fixed validation-F1 policy:
#   - binary threshold tuning is always done on validation F1
#   - best-checkpoint selection is always done on validation F1
# Dense-pipeline knobs such as cfg.threshold_metric / cfg.select_by do not apply here.
@dataclass
class GNNTrainConfig:
    """
    Class-imbalance policy:
        _pos_weight is the default for GNN binary BCE training. When
        neg_pos_ratio is set, it replaces _pos_weight for the BCE term with
        target-ratio negative balancing plus unweighted BCE. Auxiliary losses,
        when enabled, may still operate on the full supervised logits.
    """
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    batch_size: int = 16

    # Model sizes
    hidden: int = 128
    layers: int = 3
    heads: int = 4
    dropout: float = 0.10

    # Task-agnostic knobs
    dropedge_p: float = 0.10  # training-time DropEdge strength for encoder message passing - 0.0 disables it
    lap_pe_k: int = 0  # 0 disables universal Laplacian positional encodings
    gps_lap_pe_k: int = 16  # GPS uses at least this many Laplacian PE columns in full-matrix mode
    gps_lap_pe_sign_flip: bool = True  # training-time sign augmentation for GPS-owned LapPE
    gps_rwse_steps: int = 16  # optional GPS-owned RWSE columns; 0 disables RWSE
    gnn_zero_supervised: bool = False  # if True, zero supervised cells in A before message passing
    learnable_layer_norm: bool = True  # if False, use non-affine LayerNorm across all GNN encoders

    # Optimiser/scheduler
    scheduler: str = "cosine"  # ["none", "cosine"]

    # Sampling
    neg_pos_ratio: Optional[float] = None

    # Tree-specific auxiliary losses (used only in spanning-tree-style tasks)
    use_tree_aux_loss: bool = False

    # Saving
    save_dir: Optional[str] = "saved_checkpoints"  # None to skip saving


def _supervised_indices(
    mask: torch.Tensor, A: torch.Tensor, is_directed: bool, edges_only: bool
) -> torch.Tensor:
    """Build the effective supervision mask and return nonzero (B, i, j) indices."""
    m = effective_mask(mask, A, is_directed)
    if edges_only:
        m = m & (A > 0.5)
    return torch.nonzero(m, as_tuple=False)


def _balance_binary_negpos(y_sel: torch.Tensor, ratio: float) -> torch.Tensor:
    """
    Compute indices that rebalance binary labels to the target neg:pos ratio.
    Subsamples negatives when there are too many; oversamples with replacement when too few.
    """
    pos_idx = torch.nonzero(y_sel > 0.5, as_tuple=False).squeeze(-1)
    neg_idx = torch.nonzero(y_sel <= 0.5, as_tuple=False).squeeze(-1)

    # One-class batch: nothing to balance against, so keep everything
    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        return torch.cat([pos_idx, neg_idx])

    target_neg = int(pos_idx.numel() * ratio)

    if target_neg < neg_idx.numel():
        perm = torch.randperm(neg_idx.numel(), device=y_sel.device)[:target_neg]
        neg_idx = neg_idx[perm]
    elif target_neg > neg_idx.numel() and neg_idx.numel() > 0:
        deficit = target_neg - neg_idx.numel()
        repl = neg_idx[torch.randint(0, neg_idx.numel(), (deficit,), device=y_sel.device)]
        neg_idx = torch.cat([neg_idx, repl])

    return torch.cat([pos_idx, neg_idx])


def _tree_count_penalty(
    logits: torch.Tensor,
    mask: torch.Tensor,
    A: torch.Tensor,
    *,
    directed: bool = False,
    edges_only: bool = True
) -> torch.Tensor:
    """
    Encourage the total predicted edge count to match the tree edge count (N - 1).
    Uses L1 loss normalised by N to maintain O(1) scale with the primary BCE loss.
    """
    Z = logits
    m = effective_mask(mask, A, directed)

    if edges_only:
        m = m & (A > 0.5)

    probs = torch.sigmoid(Z)
    pred_edge_count = (probs * m).sum(dim=(1, 2))

    node_mask = mask.any(dim=-1) | mask.any(dim=-2)
    node_mask = node_mask | (A > 0.5).any(dim=-1) | (A > 0.5).any(dim=-2)

    n_valid = node_mask.sum(dim=1).to(pred_edge_count.dtype)
    target_edge_count = (n_valid - 1).clamp_min(0)

    # L1 penalty normalised by the graph size
    loss = torch.abs(pred_edge_count - target_edge_count) / n_valid.clamp_min(1.0)

    return loss.mean()


def _strip_adj_from_edge_keys(feature_schema: Dict[str, Any]) -> Dict[str, Any]:
    if "adj" not in feature_schema.get("edge_keys", []):
        return feature_schema

    return {
        **feature_schema,
        "edge_keys": [k for k in feature_schema["edge_keys"] if k != "adj"]
    }


# -------------------------------- Model factory --------------------------------
def _build_gnn_model(
    encoder: str,
    *,
    hidden: int = 128,
    layers: int = 3,
    heads: int = 4,
    dropout: float = 0.1,
    device: torch.device,
    directed: bool = False,
    num_classes: int = 1,
    learnable_layer_norm: bool = True
) -> nn.Module:
    """
    Factory method to create a GraphEdgeClassifier with the specified encoder.
    """
    model = GraphEdgeClassifier(
        encoder_type=encoder,
        hidden=hidden,
        depth=layers,
        heads=heads,
        dropout=dropout,
        directed=directed,
        edge_hidden=hidden,
        num_classes=num_classes,
        learnable_layer_norm=learnable_layer_norm
    )
    model = model.to(device)
    return model


# -------------------------------- Train/Eval -----------------------------------
@torch.no_grad()
def _gnn_eval_split(
        model: nn.Module,
        loader: DataLoader,
        criterion: Optional[nn.Module],
        device: torch.device,
        num_classes: int,
        is_directed: bool,
        edges_only: bool,
        zero_supervised: bool,
        fixed_thr: Optional[float] = None,
        fallback_thr: float = 0.5
):
    """Shared evaluation logic for both full-batch and single-graph on-demand GNN trainers."""
    model.eval()
    all_labels = []

    if num_classes > 1:
        all_preds, all_probs = [], []
        tot_loss, tot_cnt = 0.0, 0
    else:
        all_logits = []

    for A, feats, L, mask in loader:
        A = A.to(device)
        mask = mask.to(device)
        A_enc = _redact_supervised_edges(A, mask, zero_supervised)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, enabled=torch.cuda.is_available()):
            # Path A: On-demand
            if getattr(model, "pairwise_on_demand", False):
                X_batch, _, node_mask = _prepare_features_batch(
                    A_enc, feats, mask, schema=getattr(model, "feature_schema", None),
                    lap_pe_k=model._effective_lap_pe_k(),
                    lap_pe_sign_flip=model._lap_pe_sign_flip_enabled(),
                    rwse_steps=model._effective_rwse_steps(),
                    build_edge_mats=False, append_pairwise=False,
                    is_directed=bool(is_directed)
                )
                Z = model.encode_only(A_enc, X_batch, node_mask)

                idx = _supervised_indices(mask, A, is_directed, edges_only)
                if idx.numel() == 0:
                    continue

                i_idx, j_idx = idx[:, 1], idx[:, 2]
                idx_cpu = idx.cpu()
                if num_classes > 1:
                    y_sel = L[idx_cpu[:, 0], idx_cpu[:, 1], idx_cpu[:, 2]].long().to(device)
                else:
                    y_sel = (L[idx_cpu[:, 0], idx_cpu[:, 1], idx_cpu[:, 2]] > 0.5).to(torch.float32)

                A_single = A_enc[0] if A_enc.dim() == 3 else A_enc
                z_sel = torch.nan_to_num(
                    model.score_pairs_on_demand(Z, A_single, i_idx, j_idx),
                    nan=0.0, posinf=30.0, neginf=-30.0
                )

            # Path B: Full Matrix
            else:
                idx = _supervised_indices(mask, A, is_directed, edges_only)
                if idx.numel() == 0:
                    continue

                z_sel = torch.nan_to_num(
                    model.score_pairs_selected(A_enc, feats, mask, idx),
                    nan=0.0, posinf=30.0, neginf=-30.0
                )

                idx_cpu = idx.cpu()
                if num_classes > 1:
                    y_sel = L[idx_cpu[:, 0], idx_cpu[:, 1], idx_cpu[:, 2]].long().to(device)
                else:
                    y_sel = (L[idx_cpu[:, 0], idx_cpu[:, 1], idx_cpu[:, 2]] > 0.5).to(torch.float32)


        # Store Results
        if num_classes > 1:
            loss = criterion(z_sel.float(), y_sel)
            tot_loss += float(loss.cpu()) * y_sel.numel()
            tot_cnt += y_sel.numel()

            # Cast to float32 before softmax to guarantee NumPy compatibility downstream
            probs = torch.softmax(z_sel.float(), dim=-1)

            all_probs.append(probs.cpu())
            all_preds.append(probs.argmax(dim=-1).cpu())
            all_labels.append(y_sel.cpu())
        else:
            all_logits.append(z_sel.detach().to(torch.float32).cpu())
            all_labels.append(y_sel.cpu())

    metrics = {"loss/edge": (tot_loss / tot_cnt) if tot_cnt > 0 else float("nan")} if num_classes > 1 else {}

    if num_classes > 1:
        if not all_preds:
            metrics.update({"accuracy": float("nan"), "F1_macro": float("nan"), "P_macro": float("nan"), "R_macro": float("nan"),
                            "AUC_macro": float("nan"), "AUPRC_macro": float("nan"), "BAcc_macro": float("nan")})
            metrics["f1"] = float("nan")
            return metrics, None

        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()
        y_prob = torch.cat(all_probs).numpy() if all_probs else None

        # Filter out unexpected negative labels before sklearn evaluation; padding is excluded by mask
        valid_mask = (y_true >= 0)
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        if y_prob is not None:
            y_prob = y_prob[valid_mask]

        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["F1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["P_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["R_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["BAcc_macro"] = balanced_accuracy_score(y_true, y_pred) if len(np.unique(y_true)) >= 2 else float("nan")

        if y_prob is not None:
            aucs, aprs = [], []
            for k in range(num_classes):
                y_bin = (y_true == k)
                
                # Skip degenerate classes
                if not y_bin.any() or y_bin.all():
                    continue
                
                pk = y_prob[:, k]
                
                try:
                    aucs.append(roc_auc_score(y_bin, pk))
                except Exception:
                    pass
                
                try:
                    aprs.append(average_precision_score(y_bin, pk))
                except Exception:
                    pass
            
            # Average the valid classes
            metrics["AUC_macro"] = float(np.mean(aucs)) if aucs else float("nan")
            metrics["AUPRC_macro"] = float(np.mean(aprs)) if aprs else float("nan")
        else:
            metrics["AUC_macro"], metrics["AUPRC_macro"] = float("nan"), float("nan")

        metrics["f1"] = metrics["F1_macro"]
        if y_prob is not None:
            metrics["_prob"] = y_prob.tolist()
        metrics["_y"] = y_true.astype(int).tolist()
        return metrics, None
    else:
        if not all_logits:
            return {
                "loss/edge": float("nan"),
                "accuracy": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "bacc": float("nan"),
                "auroc": float("nan"),
                "auprc": float("nan"),
            }, None

        z_tensor = torch.nan_to_num(torch.cat(all_logits), nan=0.0, posinf=30.0, neginf=-30.0)
        y_tensor = torch.cat(all_labels)

        pw = _pos_weight(y_tensor)
        loss = nn.BCEWithLogitsLoss(pos_weight=pw)(z_tensor, y_tensor)
        metrics["loss/edge"] = float(loss.cpu())

        p = torch.sigmoid(z_tensor).cpu().numpy()
        y = y_tensor.cpu().numpy()

        thr = fixed_thr
        if thr is None:
            thr = get_optimal_threshold(y, p, "f1", fallback_thr)

        yp = (p >= thr).astype(np.int32)
        y_i = y.astype(np.int32)

        TP = int(np.count_nonzero((y_i == 1) & (yp == 1)))
        TN = int(np.count_nonzero((y_i == 0) & (yp == 0)))
        FP = int(np.count_nonzero((y_i == 0) & (yp == 1)))
        FN = int(np.count_nonzero((y_i == 1) & (yp == 0)))

        acc = (TP + TN) / max(1, TP + TN + FP + FN)
        prec = TP / max(1, TP + FP)
        rec = TP / max(1, TP + FN)
        f1 = (2.0 * prec * rec / max(1e-12, prec + rec)) if (prec + rec) > 0 else 0.0

        # BAcc averages TPR and TNR; one is undefined when a class is absent
        _both = bool(np.any(y_i == 1)) and bool(np.any(y_i == 0))
        bacc = 0.5 * (rec + (TN / max(1, TN + FP))) if _both else float("nan")

        metrics.update({"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "bacc": float(bacc)})

        if bool(np.any(y_i == 1)) and bool(np.any(y_i == 0)):
            metrics["auroc"] = roc_auc_score(y_i, p)
            metrics["auprc"] = average_precision_score(y_i, p)
        else:
            metrics["auroc"], metrics["auprc"] = float("nan"), float("nan")

        metrics["_prob"] = p.tolist()
        metrics["_y"] = y_i.astype(int).tolist()
        return metrics, float(thr)


def train_one_gnn(
        encoder: str,
        task,
        cfg: Union[GNNTrainConfig, SimpleNamespace, dict],
        *,
        loaders: Tuple[DataLoader, DataLoader, DataLoader],
        feature_schema: Dict[str, Any]
) -> Dict[str, Dict[str, float]]:
    """
    Standard Full-Batch GNN Training Loop.
    Supports Binary (BCE) and Multiclass (CrossEntropy).

    Loaders and the feature schema are runner-owned, set by run_gnn_suite before
    loader resolution and always passed in.
    """
    _t_model_start = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _coerce_train_cfg(cfg)

    is_directed = bool(getattr(task, "directed", False))
    edges_only = bool(getattr(task, "eval_on_existing_edges_only", False))
    num_classes = int(getattr(task, "num_classes", 1))

    train_loader, val_loader, test_loader = loaders
    probe_loader = _pick_probe_loader(train_loader, val_loader, test_loader)

    # Remove 'adj' from edge features
    feature_schema = _strip_adj_from_edge_keys(feature_schema)

    # Build Model
    A0, feats0, _, M0 = next(iter(probe_loader))
    model = _build_gnn_model(
        encoder=encoder,
        hidden=int(cfg.hidden), layers=int(cfg.layers),
        heads=int(cfg.heads), dropout=float(cfg.dropout),
        device=device, directed=is_directed,
        num_classes=num_classes,
        learnable_layer_norm=bool(getattr(cfg, "learnable_layer_norm", True))
    )

    model.dropedge_p = float(getattr(cfg, "dropedge_p", 0.10))
    model.lap_pe_k = int(getattr(cfg, "lap_pe_k", 0))
    model.gps_lap_pe_k = int(getattr(cfg, "gps_lap_pe_k", 16))
    model.gps_lap_pe_sign_flip = bool(getattr(cfg, "gps_lap_pe_sign_flip", True))
    model.gps_rwse_steps = int(getattr(cfg, "gps_rwse_steps", 16))
    model.feature_schema = feature_schema

    if num_classes > 1 and bool(getattr(cfg, "use_tree_aux_loss", False)):
        print(
            "[WARN] use_tree_aux_loss=True is unsupported for multiclass tasks; "
            "disabling tree auxiliary loss for this run.",
            flush=True
        )
        cfg.use_tree_aux_loss = False

    zero_supervised = bool(getattr(cfg, "gnn_zero_supervised", False))

    # Warmup
    model.eval()
    with torch.no_grad():
        A0_dev = A0.to(device)
        M0_dev = M0.to(device)
        A0_enc = _redact_supervised_edges(A0_dev, M0_dev, zero_supervised)
        _ = model(A0_enc, feats0, M0_dev)

    # Optimiser, scaler and scheduler
    optimiser = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())
    scheduler_type = str(getattr(cfg, "scheduler", "none")).lower()
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=int(cfg.epochs))
    else:
        scheduler = None

    # Loss Selection
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05) if num_classes > 1 else None
    disp_dec = int(getattr(cfg, "display_decimals", 4))

    # Eval Loop Closure
    def _eval_split(loader, fixed_thr, fallback_thr=0.5):
        return _gnn_eval_split(
            model=model, loader=loader, criterion=criterion, device=device,
            num_classes=num_classes, is_directed=is_directed, edges_only=edges_only,
            zero_supervised=zero_supervised, fixed_thr=fixed_thr, fallback_thr=fallback_thr
        )

    # Training Loop
    best_val, best_thr, best_state = -1.0, (None if num_classes > 1 else 0.5), None
    best_val_metrics = None
    last_known_thr = 0.5

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        epoch_loss, epoch_count = 0.0, 0

        for A, feats, L, mask in train_loader:
            A = A.to(device)
            mask = mask.to(device)
            A_enc = _redact_supervised_edges(A, mask, zero_supervised)
            optimiser.zero_grad(set_to_none=True)
            logits = None
            prebalanced_binary = False

            with torch.amp.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, enabled=torch.cuda.is_available()):
                idx = _supervised_indices(mask, A, is_directed, edges_only)
                if idx.numel() == 0:
                    continue

                idx_cpu = idx.cpu()
                if num_classes > 1:
                    y_sel = L[idx_cpu[:, 0], idx_cpu[:, 1], idx_cpu[:, 2]].long().to(device)
                else:
                    y_sel = (L[idx_cpu[:, 0], idx_cpu[:, 1], idx_cpu[:, 2]] > 0.5).to(device=device, dtype=torch.float32)

                    if not bool(getattr(cfg, "use_tree_aux_loss", False)):
                        ratio_raw = getattr(cfg, "neg_pos_ratio", None)
                        ratio = None if ratio_raw is None else float(ratio_raw)
                        if ratio is not None and ratio > 0:
                            keep = _balance_binary_negpos(y_sel, ratio)
                            idx = idx[keep]
                            y_sel = y_sel[keep]
                            prebalanced_binary = True

                if bool(getattr(cfg, "use_tree_aux_loss", False)):
                    logits = model(A_enc, feats, mask)
                    if num_classes > 1:
                        z_sel = logits[idx[:, 0], idx[:, 1], idx[:, 2], :]
                    else:
                        z_sel = logits[idx[:, 0], idx[:, 1], idx[:, 2]]
                else:
                    z_sel = model.score_pairs_selected(A_enc, feats, mask, idx)

                if num_classes > 1:
                    loss = criterion(z_sel, y_sel)
                    bs = int(y_sel.numel())
                else:
                    ratio_raw = getattr(cfg, "neg_pos_ratio", None)
                    ratio = None if ratio_raw is None else float(ratio_raw)
                    if ratio is not None and ratio > 0:
                        if prebalanced_binary:
                            loss = nn.BCEWithLogitsLoss()(z_sel, y_sel)
                            bs = int(y_sel.numel())
                        else:
                            keep = _balance_binary_negpos(y_sel, ratio)
                            z_bal, y_bal = z_sel[keep], y_sel[keep]
                            loss = nn.BCEWithLogitsLoss()(z_bal, y_bal)
                            bs = int(y_bal.numel())
                    else:
                        # Default: _pos_weight
                        pw = _pos_weight(y_sel)
                        loss = nn.BCEWithLogitsLoss(pos_weight=pw)(z_sel, y_sel)
                        bs = int(y_sel.numel())

            # This penalises deviation from expected tree density (N-1 edges)
            if bool(getattr(cfg, "use_tree_aux_loss", False)) and logits is not None:
                aux = _tree_count_penalty(
                    logits, mask, A=A,
                    directed=is_directed, edges_only=edges_only
                )
                loss = loss + aux

            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()

            if float(cfg.grad_clip) > 0:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))

            scaler.step(optimiser)
            scaler.update()

            epoch_loss += loss.item() * bs
            epoch_count += bs

        if scheduler is not None:
            scheduler.step()

        val_metrics, thr = _eval_split(val_loader, None, fallback_thr=last_known_thr)
        if thr is not None:
            last_known_thr = thr

        # Switch the print label dynamically to match the dense pipeline
        label = "F1_macro" if "F1_macro" in val_metrics else "f1"
        val_f1 = val_metrics.get(label, 0.0)
        avg_loss = (epoch_loss / epoch_count) if epoch_count else float("nan")
        print(f"[{encoder.upper()}] Ep {epoch} Train loss/edge: {avg_loss:.{disp_dec}f} | Val {label}: {val_f1:.{disp_dec}f}")
        sel_val = float(val_metrics.get("f1", float("nan")))
        if np.isfinite(sel_val) and sel_val > best_val:
            best_val = sel_val
            best_thr = thr if thr is not None else (None if num_classes > 1 else last_known_thr)
            best_val_metrics = dict(val_metrics)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    test_metrics, _ = _eval_split(test_loader, best_thr)
    print(f"[{encoder.upper()}] Test f1: {test_metrics.get('f1', 0.0):.{disp_dec}f}")

    meta = {
        "model_key": encoder,
        "best_val_threshold": best_thr,
        "directed": is_directed,
        "task": _task_to_meta_dict(task),
        "feature_schema": feature_schema,
        "pairwise_on_demand": getattr(model, "pairwise_on_demand", False),
        "cfg": dict(
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
            epochs=int(cfg.epochs),
            batch_size=1 if getattr(model, "pairwise_on_demand", False) else int(cfg.batch_size)
        ),
        "elapsed_seconds": round(time.monotonic() - _t_model_start, 3)
    }
    state_to_save = best_state if best_state is not None else {k: v.cpu() for k, v in model.state_dict().items()}
    ckpt_path = save_pipeline_checkpoint(encoder, state_to_save, task, cfg, meta)

    return {
        "elapsed_seconds": meta["elapsed_seconds"],
        "val": best_val_metrics if best_val_metrics is not None else {
            "f1": float("nan")},
        "val_best_metric": best_val if best_val_metrics is not None else float("nan"),
        "thr": best_thr,
        "test": test_metrics,
        "ckpt": ckpt_path
    }


def train_one_gnn_edges(
        encoder: str,
        task,
        cfg: Union[GNNTrainConfig, SimpleNamespace, dict],
        *,
        loaders: Tuple[DataLoader, DataLoader, DataLoader],
        feature_schema: Dict[str, Any]
) -> Dict[str, Dict[str, float]]:
    """
    On-demand GNN training, one graph at a time. Supports Binary (BCE) and Multiclass (CrossEntropy).

    Loaders and the feature schema are runner-owned, set by run_gnn_edges_suite before
    loader resolution and always passed in.
    """
    _t_model_start = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _coerce_train_cfg(cfg)

    is_directed = bool(getattr(task, "directed", False))
    edges_only = bool(getattr(task, "eval_on_existing_edges_only", False))
    num_classes = int(getattr(task, "num_classes", 1))
    cfg.batch_size = 1

    train_loader, val_loader, test_loader = loaders
    probe_loader = _pick_probe_loader(train_loader, val_loader, test_loader)
    feature_schema = _strip_adj_from_edge_keys(feature_schema)

    # Build Model
    A0, feats0, _, M0 = next(iter(probe_loader))
    model = _build_gnn_model(
        encoder=encoder,
        hidden=int(cfg.hidden), layers=int(cfg.layers),
        heads=int(cfg.heads), dropout=float(cfg.dropout),
        device=device, directed=is_directed,
        num_classes=num_classes,
        learnable_layer_norm=bool(getattr(cfg, "learnable_layer_norm", True))
    )

    model.dropedge_p = float(getattr(cfg, "dropedge_p", 0.10))
    model.feature_schema = feature_schema
    model.pairwise_on_demand = True

    # Feature guards
    if bool(getattr(cfg, "use_tree_aux_loss", False)):
        print(
            "[WARN] use_tree_aux_loss=True is unsupported when pairwise_on_demand=True; "
            "disabling tree auxiliary loss for this run.",
            flush=True
        )
        cfg.use_tree_aux_loss = False

    model.lap_pe_k = int(getattr(cfg, "lap_pe_k", 0))
    model.gps_lap_pe_k = int(getattr(cfg, "gps_lap_pe_k", 16))
    model.gps_lap_pe_sign_flip = bool(getattr(cfg, "gps_lap_pe_sign_flip", True))
    model.gps_rwse_steps = int(getattr(cfg, "gps_rwse_steps", 16))
    zero_supervised = bool(getattr(cfg, "gnn_zero_supervised", False))

    # Warmup
    model.eval()
    with torch.no_grad():
        A0_dev = A0.to(device)
        M0_dev = M0.to(device)
        A0_enc = _redact_supervised_edges(A0_dev, M0_dev, zero_supervised)
        X_batch, _, node_mask = _prepare_features_batch(
            A0_enc, feats0, M0_dev, schema=feature_schema,
            lap_pe_k=model._effective_lap_pe_k(),
            lap_pe_sign_flip=model._lap_pe_sign_flip_enabled(),
            rwse_steps=model._effective_rwse_steps(),
            build_edge_mats=False, append_pairwise=False,
            is_directed=bool(is_directed)
        )
        Z = model.encode_only(A0_enc, X_batch, node_mask)

    # Optimiser, scaler and scheduler
    optimiser = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())
    scheduler_type = str(getattr(cfg, "scheduler", "none")).lower()
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=int(cfg.epochs))
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05) if num_classes > 1 else None
    disp_dec = int(getattr(cfg, "display_decimals", 4))

    # Eval Loop Closure
    def _eval_split(loader, fixed_thr, fallback_thr=0.5):
        return _gnn_eval_split(
            model=model, loader=loader, criterion=criterion, device=device,
            num_classes=num_classes, is_directed=is_directed, edges_only=edges_only,
            zero_supervised=zero_supervised, fixed_thr=fixed_thr, fallback_thr=fallback_thr
        )

    # Training Loop
    best_val, best_thr, best_state = -1.0, (None if num_classes > 1 else 0.5), None
    best_val_metrics = None
    last_known_thr = 0.5

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        epoch_loss, epoch_count = 0.0, 0

        for A, feats, L, mask in train_loader:
            A = A.to(device)
            mask = mask.to(device)
            A_enc = _redact_supervised_edges(A, mask, zero_supervised)
            optimiser.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, enabled=torch.cuda.is_available()):
                # 1. Encode
                X_batch, _, node_mask = _prepare_features_batch(
                    A_enc, feats, mask, schema=feature_schema,
                    lap_pe_k=model._effective_lap_pe_k(),
                    lap_pe_sign_flip=model._lap_pe_sign_flip_enabled(),
                    rwse_steps=model._effective_rwse_steps(),
                    build_edge_mats=False, append_pairwise=False,
                    is_directed=bool(is_directed)
                )
                Z = model.encode_only(A_enc, X_batch, node_mask)

                idx = _supervised_indices(mask, A, is_directed, edges_only)
                if idx.numel() == 0:
                    continue
                i_idx, j_idx = idx[:, 1], idx[:, 2]
                idx_cpu = idx.cpu()
                y_sel = L[idx_cpu[:, 0], idx_cpu[:, 1], idx_cpu[:, 2]].to(device)

                # 3. Negatives (Binary Only)
                ratio_raw = getattr(cfg, "neg_pos_ratio", None)
                ratio = None if ratio_raw is None else float(ratio_raw)

                if (num_classes == 1) and (ratio is not None) and (ratio > 0):
                    keep = _balance_binary_negpos(y_sel, ratio)
                    i_idx = i_idx[keep]
                    j_idx = j_idx[keep]
                    y_sel = y_sel[keep]

                if i_idx.numel() == 0:
                    continue

                A_single = A_enc[0] if A_enc.dim() == 3 else A_enc
                scores = model.score_pairs_on_demand(Z, A_single, i_idx, j_idx)

                # 4. Loss
                if num_classes > 1:
                    y_target = y_sel.long()
                    if y_target.numel() == 0:
                        continue
                    loss = criterion(scores, y_target)
                    bs = int(y_target.numel())
                else:
                    y_target = (y_sel > 0.5).float()
                    if y_target.numel() == 0:
                        continue

                    if ratio is not None and ratio > 0:
                        loss = nn.BCEWithLogitsLoss()(scores, y_target)
                    else:
                        pw = _pos_weight(y_target)
                        loss = nn.BCEWithLogitsLoss(pos_weight=pw)(scores, y_target)

                    bs = int(y_target.numel())

            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()

            if float(cfg.grad_clip) > 0:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))

            scaler.step(optimiser)
            scaler.update()

            epoch_loss += loss.item() * bs
            epoch_count += bs

        if scheduler is not None:
            scheduler.step()

        val_metrics, thr = _eval_split(val_loader, None, fallback_thr=last_known_thr)
        if thr is not None:
            last_known_thr = thr

        # Switch the print label dynamically to match the dense pipeline
        label = "F1_macro" if "F1_macro" in val_metrics else "f1"
        val_f1 = val_metrics.get(label, 0.0)
        avg_loss = (epoch_loss / epoch_count) if epoch_count else float("nan")
        print(f"[{encoder.upper()}] Ep {epoch} Train loss/edge: {avg_loss:.{disp_dec}f} | Val {label}: {val_f1:.{disp_dec}f}")
        sel_val = float(val_metrics.get("f1", float("nan")))
        if np.isfinite(sel_val) and sel_val > best_val:
            best_val = sel_val
            best_thr = thr if thr is not None else (None if num_classes > 1 else last_known_thr)
            best_val_metrics = dict(val_metrics)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    test_metrics, _ = _eval_split(test_loader, best_thr)
    print(f"[{encoder.upper()}] Test f1: {test_metrics.get('f1', 0.0):.{disp_dec}f}")

    meta = {
        "model_key": encoder,
        "best_val_threshold": best_thr,
        "directed": is_directed,
        "task": _task_to_meta_dict(task),
        "feature_schema": feature_schema,
        "pairwise_on_demand": getattr(model, "pairwise_on_demand", False),
        "cfg": dict(
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
            epochs=int(cfg.epochs),
            batch_size=1 if getattr(model, "pairwise_on_demand", False) else int(cfg.batch_size)
        ),
        "elapsed_seconds": round(time.monotonic() - _t_model_start, 3)
    }
    state_to_save = best_state if best_state is not None else {k: v.cpu() for k, v in model.state_dict().items()}
    ckpt_path = save_pipeline_checkpoint(encoder, state_to_save, task, cfg, meta)

    return {
        "elapsed_seconds": meta["elapsed_seconds"],
        "val": best_val_metrics if best_val_metrics is not None else {
            "f1": float("nan")},
        "val_best_metric": best_val if best_val_metrics is not None else float("nan"),
        "thr": best_thr,
        "test": test_metrics,
        "ckpt": ckpt_path,
    }


def run_gnn_suite(
    task,
    encoders: Sequence[str],
    cfg: Any,
    *,
    quiet: bool = False,
    display_decimals: int = 4,
    display_truncate: bool = False
) -> Dict[str, Any]:
    """
    Bridge helper to train/evaluate multiple full-batch GNN encoders.

    Run lifecycle:
        - This call starts a new run unless it attaches to an already-open dense run
          created by `run_pipeline_for_task(...)` on the same task in the same script/cell.
        - When attached after the dense runner, the dense and GNN stages are treated as
          one combined run and share a single merged results bundle / final summary table.
        - Otherwise, this call is itself a complete standalone run.

    Contract:
        - Returns the active run bundle in the standard
          {"results": ..., "metadata": ...} format.
        - `quiet=True` suppresses the final summary table only.
          It does not affect run finalisation, cache clearing, or feature reset.
          
    Feature schema:
        Uses the user-specified feature set from task.hooks.feature_set.
    """
    canon = _canonicalise_encoders(encoders)
    cfg = _coerce_train_cfg(cfg)
    cfg.display_decimals = int(display_decimals)
    cfg.display_truncate = bool(display_truncate)

    install_boundary_hooks()
    base_bundle = begin_or_attach_run(
        task_key=(id(task), str(getattr(task, "name", "task"))),
        stage="gnn_full",
        bundle_factory=lambda: {"results": {}, "metadata": {}},
        can_attach=lambda active_stage, next_stage, active_task_key, next_task_key: (
            active_stage == ["tnn"] and next_stage in {"gnn_full", "gnn_edges"} and active_task_key == next_task_key
        ),
        reset_cb=lambda: _reset_pipeline_runtime_state(task),
        summary_cb=lambda b, quiet, dd, dt: (
            None if quiet else finalise_summary(b, task, display_decimals=dd, display_truncate=dt)
        ),
        quiet=quiet,
        display_decimals=display_decimals,
        display_truncate=display_truncate,
    )
    results_target = base_bundle["results"]

    hooks = getattr(task, "hooks", None)
    if hooks is None:
        requested_features, custom_types = None, {}
    else:
        requested_features, custom_types = _normalise_feature_spec(
            getattr(hooks, "feature_set", []),
            directed=task.directed
        )

    shared_loaders = _resolve_loaders(task, cfg)
    shared_feature_schema = _infer_feature_schema(
        list(shared_loaders),
        requested_keys=requested_features,
        custom_types=custom_types
    )

    for enc in canon:
        print("\n" + "=" * 80)
        print(f"Running GNN encoder: {enc.upper()}")
        print("=" * 80)

        out = train_one_gnn(enc, task, cfg, loaders=shared_loaders, feature_schema=shared_feature_schema)
        entry = {
            "val": out.get("val", {}),
            "test": out.get("test", {}),
            "thr": out.get("thr"),
            "ckpt": out.get("ckpt"),
            "raw": out,
        }
        results_target[enc] = entry
        _elapsed = out.get("elapsed_seconds")
        if _elapsed is not None:
            print(f"[TIME] Training total: {_format_duration(_elapsed)}")

    meta = {
        "task": _task_to_meta_dict(task),
        "encoders": canon,
        "suite": "gnn_full_batch",
    }
    base_bundle.setdefault("metadata", {}).update(meta)
    task._latest_results = base_bundle

    return base_bundle


def run_gnn_edges_suite(
        task,
        encoders: Sequence[str],
        cfg: Any,
        *,
        quiet: bool = False,
        display_decimals: int = 4,
        display_truncate: bool = False,
) -> Dict[str, Any]:
    """
    Bridge helper for single-graph on-demand GNN training via `train_one_gnn_edges(...)`.

    Run lifecycle:
        - This call starts a new run unless it attaches to an already-open dense run
          created by `run_pipeline_for_task(...)` on the same task in the same script/cell.
        - When attached after the dense runner, the dense and single-graph on-demand GNN stages
          are treated as one combined run and share a single merged results bundle /
          final summary table.
        - Otherwise, this call is itself a complete standalone run.

    Contract:
        - Returns the active run bundle in the standard
          {"results": ..., "metadata": ...} format.
        - `quiet=True` suppresses the final summary table only.
          It does not affect run finalisation, cache clearing, or feature reset.

    Feature schema:
        - Discovers all features present in the sample feature dicts via untargeted
          schema inference. Typed custom declarations in task.hooks.feature_set provide
          node/edge metadata without narrowing discovery. Node features flow through to
          the encoder. The decoder's structural pair features are computed on the fly
          from the adjacency.
    """
    canon = _canonicalise_encoders(encoders)
    cfg = _coerce_train_cfg(cfg)
    cfg.display_decimals = int(display_decimals)
    cfg.display_truncate = bool(display_truncate)

    install_boundary_hooks()
    base_bundle = begin_or_attach_run(
        task_key=(id(task), str(getattr(task, "name", "task"))),
        stage="gnn_edges",
        bundle_factory=lambda: {"results": {}, "metadata": {}},
        can_attach=lambda active_stage, next_stage, active_task_key, next_task_key: (
            active_stage == ["tnn"] and next_stage in {"gnn_full", "gnn_edges"} and active_task_key == next_task_key
        ),
        reset_cb=lambda: _reset_pipeline_runtime_state(task),
        summary_cb=lambda b, quiet, dd, dt: (
            None if quiet else finalise_summary(b, task, display_decimals=dd, display_truncate=dt)
        ),
        quiet=quiet,
        display_decimals=display_decimals,
        display_truncate=display_truncate,
    )
    results_target = base_bundle["results"]

    hooks = getattr(task, "hooks", None)
    if hooks is None:
        custom_types = {}
    else:
        _, custom_types = _normalise_feature_spec(
            getattr(hooks, "feature_set", []),
            directed=task.directed
        )

    if int(getattr(cfg, "batch_size", 16)) != 1:
        print("[WARN] Scalable Mode processes one graph at a time; the configured batch_size is ignored.", flush=True)
    cfg.batch_size = 1

    shared_loaders = _resolve_loaders(task, cfg, batch_size=1)
    shared_feature_schema = _infer_feature_schema(
        list(shared_loaders), custom_types=custom_types
    )

    for enc in canon:
        print("\n" + "=" * 80)
        print(f"Running GNN encoder: {enc.upper()} (single-graph on-demand)")
        print("=" * 80)

        out = train_one_gnn_edges(
            enc, task, cfg,
            loaders=shared_loaders,
            feature_schema=shared_feature_schema
        )
        entry = {
            "val": out.get("val", {}),
            "test": out.get("test", {}),
            "thr": out.get("thr"),
            "ckpt": out.get("ckpt"),
            "raw": out
        }

        results_target[enc] = entry
        _elapsed = out.get("elapsed_seconds")
        if _elapsed is not None:
            print(f"[TIME] Training total: {_format_duration(_elapsed)}")

    meta = {
        "task": _task_to_meta_dict(task),
        "encoders": canon,
        "suite": "gnn_scalability_mode",
    }
    base_bundle.setdefault("metadata", {}).update(meta)
    task._latest_results = base_bundle

    return base_bundle


def _drop_edges(A: torch.Tensor, p: float, directed: bool) -> torch.Tensor:
    """
    Apply training-time DropEdge to the adjacency used for encoder message passing.
    - p controls regularisation strength - p <= 0 disables DropEdge;
    - Supports dense (2D or (1,N,N)) tensors used by the pipeline;
    - For undirected graphs, symmetry is preserved in the returned adjacency;
    - Always keeps self-loops.
    """
    if p <= 0.0:
        return A

    # Dense path (2D or (1,N,N))
    orig_dim = A.dim()
    if orig_dim == 2:
        A = A.unsqueeze(0)
    B, N, _ = A.shape

    dev = A.device
    keep = (torch.rand((B, N, N), device=dev) > p).to(dtype=A.dtype)
    if not directed:
        keep = torch.triu(keep, diagonal=1)
        keep = (keep + keep.transpose(-1, -2) > 0).to(dtype=A.dtype)

    eye = torch.eye(N, device=dev, dtype=A.dtype).unsqueeze(0)
    out = A * keep
    out = out * (1.0 - eye) + A * eye  # preserve diagonal from A
    return out if orig_dim == 3 else out.squeeze(0)


def _infer_feature_schema(
    loaders: Union[DataLoader, Sequence[DataLoader]],
    requested_keys: Optional[Sequence[str]] = None,
    custom_types: Optional[Dict[str, str]] = None,
    max_batches: int = 64
) -> Dict[str, Any]:
    """
    Inspect batches across one or more loaders and build a fixed schema:
      - node_keys: ordered list of keys that should produce one (N,1) column each
      - edge_keys: ordered list of keys that should produce one (N,N) channel each

    When `requested_keys` is provided, the scanner maintains a shrinking set of
    keys still being sought. Each batch is inspected only for those outstanding
    keys, and scanning stops as soon as all requested keys have been located
    (or all loaders are exhausted). This guarantees that a feature present in
    any batch of any loader will be discovered, while minimising redundant work.

    When `requested_keys` is None (no hooks / open-ended discovery), scanning
    is capped at `max_batches` total across all loaders to avoid a full dataset
    traversal.
    """
    if isinstance(loaders, DataLoader):
        loaders = [loaders]

    custom_types = dict(custom_types or {})
    node_keys: List[str] = []
    edge_keys: List[str] = []
    node_dims: Dict[str, int] = {}

    seen_nodes: set = set()
    seen_edges: set = set()
    requested = None if requested_keys is None else set(requested_keys)

    # Targeted scan: shrinking set, exhaustive across all loaders.
    # Untargeted scan: cap total batches to avoid full dataset traversal.
    seeking = set(requested) - {"adj", "shortest_path"} if requested is not None else None

    # Keep the caller's ordering. Set iteration order varies per process under Python's randomised string hashing
    requested_order = [] if requested_keys is None else [k for k in requested_keys if k not in ("adj", "shortest_path")]
    total_batches = 0

    def _classify_and_register(k: str, t: torch.Tensor, N_unpadded: int):
        """Classify a discovered key using declared custom type first, then canonical shape rules."""
        explicit_type = custom_types.get(k)

        if explicit_type == "node":
            if t.dim() == 0 or (t.dim() == 2 and t.shape == (1, 1)) or t.dim() == 1:
                if k not in seen_nodes:
                    seen_nodes.add(k)
                    node_keys.append(k)
                    node_dims[k] = 1
                return
            if t.dim() == 2:
                r, c = int(t.shape[0]), int(t.shape[1])
                if k not in seen_nodes:
                    seen_nodes.add(k)
                    node_keys.append(k)
                    _, node_dims[k] = _resolve_node_matrix_orientation(r, c, N_unpadded)
            return

        if explicit_type == "edge":
            if t.dim() == 2 and k not in seen_edges:
                seen_edges.add(k)
                edge_keys.append(k)
            return

        if t.dim() == 0 or (t.dim() == 2 and t.shape == (1, 1)) or t.dim() == 1:
            if k not in seen_nodes:
                seen_nodes.add(k)
                node_keys.append(k)
                node_dims[k] = 1
            return

        if t.dim() != 2:
            return

        r, c = int(t.shape[0]), int(t.shape[1])
        if r == c and (k in FeatureRegistry.CANONICAL or k == "adj"):
            if k not in seen_edges:
                seen_edges.add(k)
                edge_keys.append(k)
        elif r == c:
            raise RuntimeError(
                f"[CUSTOM FEATURE] Ambiguous {r}x{c} feature '{k}'. "
                f"Declare it in TaskHooks.feature_set as ('{k}', 'node') or ('{k}', 'edge')."
            )
        else:
            if k not in seen_nodes:
                seen_nodes.add(k)
                node_keys.append(k)
                _, node_dims[k] = _resolve_node_matrix_orientation(r, c, N_unpadded)

    done = False
    for loader in loaders:
        if done:
            break
        for A, feats, *_ in loader:
            if done:
                break
            B = A.size(0)
            for b in range(B):
                fdict = feats[b] if isinstance(feats, (list, tuple)) else feats
                N_unpadded = int(fdict["_N"])

                # Only inspect the keys we are still looking for
                if seeking is not None:
                    keys_to_check = [k for k in requested_order if k in seeking and k in fdict]
                else:
                    keys_to_check = list(fdict.keys())

                for k in keys_to_check:
                    v = fdict[k]
                    if not torch.is_tensor(v):
                        continue
                    _classify_and_register(k, v, N_unpadded)
                    if seeking is not None:
                        seeking.discard(k)

                # All requested keys found — stop scanning entirely
                if seeking is not None and len(seeking) == 0:
                    done = True
                    break

            total_batches += 1
            if seeking is None and total_batches >= max_batches:
                done = True

    if requested is not None and "shortest_path" in requested:
        seen_edges.add("shortest_path")
        edge_keys.append("shortest_path")

    # Warn about requested features absent from all loaders.
    # Unlike the dense pipeline (which zero-fills missing channels at a known shape),
    # the GNN schema cannot infer dimensionality for features it has never observed.
    if seeking:
        print(
            f"[WARN] Requested feature(s) {sorted(seeking)} were not found in any loader "
            f"batch and will be absent from the GNN feature schema. If these features are "
            f"expected, ensure they are present in the dataset's feature dictionaries.",
            flush=True
        )

    schema: Dict[str, Any] = {
        "node_keys": node_keys,
        "node_dims": node_dims,
        "edge_keys": edge_keys
    }

    return schema

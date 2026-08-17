# SPDX-License-Identifier: CC-BY-SA-4.0

"""
Edge-Classification Pipeline

=============================================================================
⚠️ UNSUPPORTED USAGE WARNING ⚠️

Please note that direct instantiation or invocation of internal classes and 
methods (typically prefixed with an underscore, such as `_GCNEncoder`, 
`_SAGEEncoder`, or `_eval_split`) is completely unsupported. 

These internal components are designed for high throughput and assume the 
data has already been sanitised by the pipeline. They do not contain 
defensive guards for:
  - Unexpected input shapes or missing dimensions
  - Sparse vs. dense tensor mismatches
  - Missing or misaligned validity masks

Only the core pipeline entry paths are supported. Bypassing the intended
pipeline runners may result in crashes, silent tensor broadcasting errors,
repeated loader construction, stale or mismatched runtime state assumptions,
or mathematically incorrect results.
=============================================================================

Core abstractions
-----------------
- TaskSpec: tiny adapter every dataset implements to provide batches of
  (A_obs, feature_dict, L, mask), plus split loaders.

- FeatureRegistry: stacks the features into BHWC/BCHW, computes per-channel
  train-only mean/std, standardises, and records the exact channel manifest.

- Model Zoo: Standard MLP, Deep MLP, CNN, Transformer — all consume pre-stacked
  BCHW tensors (adj/mask/features) produced by this file's helpers, and the
  transformer optionally accepts a task mask for token gating.

- Trainer/Evaluator: shared loop across models; binary mode reports Acc/P/R/F1,
  AUROC/AUPRC, BAcc (with tuned threshold on Val utilising state momentum for 
  degenerate splits), and multiclass reports Acc/P_macro/R_macro/F1_macro/AUC_macro/AUPRC_macro/BAcc_macro.

- Saving: each trained model is saved to .pth with state_dict + meta (channel stats,
  manifest, flags, best threshold, and task info).

Usage
-----
- Supported task entry paths are:
  (1) ProvidedSplitsTask for generated datasets or pre-split datasets, or
  (2) a single-graph task exposing `task.bench` + `task.hooks`.
- Direct task-provided `train_dataloader` / `val_dataloader` / `test_dataloader`
  entry points are unsupported.
- Requires `GraphBenchmark` importable on sys.path.
- Works on CPU or CUDA. In Colab with GPU enabled, it will use cuda automatically.

Notes
-----
- Directed/undirected behaviour comes from the TaskSpec (`task.directed`), not cfg.

Self-loops:
    Adjacency matrices are used as provided by the task/loaders. Dense models and GNN
    encoders consume A as given; this pipeline does not add I to A (with the strict 
    exception of the Edge-Masked Transformer, which temporarily adds self-loops to its 
    internal attention mask to prevent NaN attention scores on isolated nodes).

    Supervision/evaluation masking is handled separately from adjacency. For undirected
    tasks, only strict upper-triangle off-diagonal pairs are evaluated once. Diagonal
    (i,i) entries are ignored in the undirected mask. For directed tasks, diagonal
    (i,i) entries are included only when a self-loop is present in A.

Normalisation:
    The dense path uses train-fitted dataset-level channel statistics,
    while the GNN path uses per-graph normalisation.

    Dense / image-style path:
        FeatureRegistry fits per-channel train statistics on the masked training
        positions selected by effective_mask(mask, A, directed) after channel
        assembly/redaction, and applies those same channel mean/std values to
        val/test via standardise_bchw.

    GNN path:
        The full-matrix GNN path standardises node and edge features per graph using
        FeatureRegistry.zscore_nodes_per_graph/zscore_edges_per_graph.
        The on-demand edge evaluation path does not z-score its decoder-side structural
        features; it derives them directly from the adjacency passed into `score_pairs_on_demand`.

Pairwise features:
    Dense models derive pairwise channels during channel stacking from the adjacency channel after any 
    configured redaction. `degree` is a 1D node feature. `endpoint_degree` expands to the pairwise
    endpoint channels `deg_row` and `deg_col`; `deg_diff` is their absolute difference.
    If `allow_adj_channel=False` and no dense features are requested, `run_pipeline_for_task(...)`
    temporarily uses a default structural feature set during loader resolution, then restores the
    caller's original `task.hooks.feature_set`.
    In dense models, 1D node features are broadcast to row/col channels internally.
    In GNN models, 1D inputs are node features and square canonical or typed custom `(N, N)` inputs are edge features.

Directed GNN convention:
    For GNN encoders, adjacency is interpreted row-wise: A[i, j] = 1 means i -> j.
    Directed propagation/attention follows row i, so node i aggregates or attends over
    its outgoing neighbors unless a model states otherwise.
"""
import copy
import inspect
import gc
import random
import warnings
import math
import dataclasses as _dc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import time
from ._utils.features import CANONICAL_FEATURES, DIRECTED_AUTO_FEATURES, UNDIRECTED_AUTO_FEATURES, pairwise_batch_from_adj, shortest_path_from_adj
from ._utils.run_lifecycle import begin_or_attach_run, get_active_run_checkpoint_timestamp
from .GraphBenchmark import GraphBenchmark
from types import SimpleNamespace
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable, Union, Sequence, ClassVar, FrozenSet, Set, Literal, Any
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import RandomForestClassifier
from functools import partial
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score, roc_curve, precision_recall_curve
)


# ============================================================
# 0) Reproducibility
# ============================================================
def seed_everything(seed: int):
    """Locks down all random states for pipeline reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# 1) Collate, masks, stacking, channel statistics
# ============================================================
def collate_fn_pad(batch, *, num_classes: int = 1):
    """
    Collates a batch of graph samples into padded dense tensors.

    Supported input tuple structures:
      - (A, L, Fdict)       : Standard generation pipeline (e.g., GraphBenchmark)
      - (A, Fdict, L, mask) : Extended pipeline with explicit evaluation masks

    Expected input types:
      - A, L, mask: np.ndarray or torch.Tensor (shape N x N)
      - Fdict: dict[str, np.ndarray | torch.Tensor]

    Returns:
        A_batch (Tensor): Padded adjacency matrices, shape (B, N_max, N_max), float32.
        F_list (List[Dict[str, Tensor]]): Unpadded node/edge features per graph.
        L_batch (Tensor): Padded label matrices, shape (B, N_max, N_max).
        M_batch (Tensor): Padded boolean evaluation masks, shape (B, N_max, N_max).
    """
    def to_tensor_2d(x, dtype=torch.float32):
        if isinstance(x, torch.Tensor):
            t = x
        else:
            t = torch.from_numpy(np.asarray(x))

        if t.ndim != 2 or t.shape[0] != t.shape[1]:
            raise ValueError("First element must be an NxN adjacency/label matrix.")
        return t.to(dtype=dtype)

    A_list, L_list, F_list, M_list = [], [], [], []
    for item in batch:
        comps = list(item)

        # Extract and format the adjacency matrix (A)
        A = comps[0]
        A = to_tensor_2d(A, dtype=torch.float32)
        N = A.shape[0]

        # Enforce binary contract
        if not torch.all((A == 0.0) | (A == 1.0)):
            raise ValueError("The pipeline strictly requires a binary adjacency matrix.")

        # Extract the feature dictionary
        feats = next((c for c in comps if isinstance(c, dict)), {})

        # Convert feature arrays to float32 tensors
        feats_t = {}
        feats_t["_N"] = N
        for k, v in feats.items():
            if k == "_N":
                raise ValueError("[RESERVED FEATURE] '_N' is pipeline-owned node-count metadata and cannot be supplied by a sample.")
            if v is None:
                continue

            # Safely pass other internal metadata through as raw Python types
            if isinstance(k, str) and k.startswith("_"):
                feats_t[k] = v
                continue

            if isinstance(v, torch.Tensor):
                feats_t[k] = v.float()
            else:
                vv = np.asarray(v)
                if np.issubdtype(vv.dtype, np.number) or np.issubdtype(vv.dtype, np.bool_):
                    # Only convert to a PyTorch tensor if the data is actually numeric
                    feats_t[k] = torch.from_numpy(vv).float()
                else:
                    # Safely pass strings, dicts, or custom objects through untouched
                    feats_t[k] = v

        # Extract remaining N x N matrices for labels (L) and masks (M)
        matrices = []
        for c in comps[1:]:
            if isinstance(c, dict):
                continue
            arr = torch.from_numpy(np.asarray(c)) if not isinstance(c, torch.Tensor) else c
            if arr.ndim != 2 or tuple(arr.shape) != (N, N):
                raise ValueError(
                    f"[SAMPLE SHAPE] Sample contains a matrix component of shape {tuple(arr.shape)}; label "
                    f"and evaluation mask matrices must match the adjacency at ({N}, {N}). "
                    f"Roles are assigned in order, so a mis-shaped matrix would shift the remaining "
                    f"components into the wrong slots."
                )
            matrices.append(arr)

        # Roles are positional: the first non-dict matrix becomes L and the second becomes M
        L = matrices[0] if len(matrices) > 0 else None
        M = matrices[1] if len(matrices) > 1 else None

        # Standardise label matrix
        if L is None:
            L = torch.zeros((N, N), dtype=torch.float32)
        elif L.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            L = L.float()

        if num_classes > 1:
            valid = torch.isfinite(L) & (L == L.round()) & (L >= 0) & (L < num_classes)
            if not bool(valid.all()):
                raise ValueError(
                    f"[INVALID MULTICLASS LABEL] Multiclass labels must be integer class IDs in [0, {num_classes - 1}]."
                )

        # Standardise evaluation mask
        if M is None:
            M = torch.ones((N, N), dtype=torch.bool)
        else:
            M = M.bool()

        A_list.append(A.float())
        L_list.append(L)
        F_list.append(feats_t)
        M_list.append(M)

    # Pad batch tensors to the maximum graph size (N_max)
    N_max = max(A.shape[0] for A in A_list)

    def pad_2d(t, fill=0):
        pad = (0, N_max - t.shape[1], 0, N_max - t.shape[0])
        return torch.nn.functional.pad(t, pad, value=fill)

    A_batch = torch.stack([pad_2d(A, 0) for A in A_list], dim=0)
    L_batch = torch.stack([pad_2d(L, 0) for L in L_list], dim=0)
    M_batch = torch.stack([pad_2d(M, 0) for M in M_list], dim=0)

    # Feature dictionaries remain unpadded; the FeatureRegistry handles broadcasting
    return A_batch, F_list, L_batch, M_batch


def effective_mask(
    mask: torch.Tensor,
    A: torch.Tensor,
    directed: bool
) -> torch.Tensor:
    """
    Build the boolean mask used for loss/metrics.

    Rules:
      - Off-diagonal candidates come from the provided `mask`.
      - For undirected tasks, only strict upper-triangle off-diagonal entries are kept
        so each unordered pair is evaluated once.
      - Diagonal entries are ignored for undirected tasks.
      - For directed tasks, diagonal entries are included only where a self-loop is
        actually present in `A`.

    Shapes:
      mask: (B, N, N) or (N, N)
      A:    (B, N, N) or (N, N)
      ->    (B, N, N) bool
    """
    # Ensure boolean
    if mask.dtype != torch.bool:
        mask = mask > 0.5

    if A.dtype != torch.bool:
        A_bool = A > 0.5
    else:
        A_bool = A

    m = mask

    if m.dim() == 2:
        m = m.unsqueeze(0)

    if not directed and not torch.equal(m, m.transpose(-1, -2)):
        raise ValueError("[INVALID MASK] Undirected task masks must be symmetric.")

    if A_bool.dim() == 2:
        A_bool = A_bool.unsqueeze(0)

    _, N, _ = m.shape
    eye = torch.eye(N, dtype=torch.bool, device=m.device).unsqueeze(0)

    # Off-diagonal candidates come from the supervision mask
    offdiag = m & (~eye)

    # Undirected -> evaluate each off-diagonal edge only once
    if not directed:
        return torch.triu(offdiag, diagonal=1)

    # Directed only: include diagonal candidates where a self-loop is actually present in A
    diag_present = m & A_bool & eye
    return offdiag | diag_present


def _build_meta(
    model_key: str,
    task,
    registry,
    cfg,
    runtime,
    best_thr: float | None
) -> dict:
    """
    Canonical metadata snapshot saved with checkpoints and summaries.
    """
    allow_adj_task = bool(getattr(getattr(task, "hooks", object()), "allow_adj_channel", False))
    tx_force = bool(getattr(cfg, "tx_force_adj_channel", True))

    meta = dict(
        # model
        model_key=model_key,
        best_val_threshold=(float(best_thr) if best_thr is not None else None),

        # registry / features (effective)
        manifest=list(getattr(registry, "manifest", [])),
        use_mask_channel=bool(getattr(registry, "use_mask_channel", False)),
        directed=bool(getattr(registry, "directed", True)),
        supervised_redaction_policy=str(getattr(registry, "supervised_redaction_policy", "adj_only")),

        # task semantics
        task=_task_to_meta_dict(task),
        edges_only=bool(getattr(task, "eval_on_existing_edges_only", False)),
        allow_adj_channel_task=allow_adj_task,
        tx_force_adj_channel=tx_force,
        seed=getattr(task, "seed", None),

        # pipeline-resolved inputs (useful for reproducibility)
        feature_keys=getattr(runtime, "feature_keys", None),
        keep_idx=getattr(runtime, "keep_idx", None),
        eff_in_ch=getattr(runtime, "eff_in_ch", None),

        # training config (subset)
        cfg=dict(
            lr=float(getattr(cfg, "lr", 0.0)),
            weight_decay=float(getattr(cfg, "weight_decay", 0.0)),
            epochs=int(getattr(cfg, "epochs", 0)),
            batch_size=int(getattr(cfg, "batch_size", 0))
        )
    )

    return meta


def save_pipeline_checkpoint(model_key: str, state_dict: dict, task: Any, cfg: Any, meta: dict) -> Optional[str]:
    """Save a model checkpoint under the configured checkpoint root, task name, and run timestamp."""
    save_dir = getattr(cfg, "save_dir", "saved_checkpoints")
    if save_dir is None:
        return None

    save_root = Path(save_dir)
    if not save_root.is_absolute():
        save_root = Path(__file__).resolve().parents[1] / save_root
    
    timestamp = get_active_run_checkpoint_timestamp()
    out_dir = save_root / str(getattr(task, "name", "task")) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{model_key}.pth"
    
    torch.save({"state_dict": state_dict, "meta": meta}, ckpt_path)
    print(f"[{model_key.upper()}] Saved checkpoint → {ckpt_path}")
    return str(ckpt_path)


def _pos_weight(y: torch.Tensor) -> torch.Tensor:
    pos = y.sum().to(dtype=torch.float32)
    neg = float(y.numel()) - pos
    w = neg / pos.clamp_min(1.0)
    return torch.clamp(w, min=1e-3, max=1e3)  # Clamp to prevent gradient explosion


# ============================================================
# 2) FeatureRegistry
# ============================================================
class FeatureRegistry:
    """
    Single source of truth for:
      - channel order / expansion (adj, optional mask, expanded features),
      - computing dataset statistics on train (mean/std),
      - standardising BCHW tensors,
      - stacking BCHW/BHWC consistently across heads.

    Notes on structural features:
      - In the dense/image-style path, `deg_row`, `deg_col`, and `deg_diff` are canonical 
        adjacency-derived pairwise channels. `deg_diff` explicitly provides the absolute difference 
        (|row - col|) as a non-linear inductive bias to aid structural symmetry detection.
      - `twohop` is a 1D node feature: the number of unique nodes at exactly graph distance 2,
        excluding the source and its direct neighbours. Directed graphs use outward reachability.

    Policies:
        supervised_redaction_policy:
            "all"      → zero every channel at supervised (i,j)
            "adj_only" → zero only the adjacency input channel during channel assembly [default].
                         Adjacency-derived channels are then computed from that redacted
                         adjacency, while precomputed non-adjacency feature tensors remain visible.
            "none"     → no redaction; all channels remain visible at supervised (i,j)
            
        Channel statistics are always computed over positions selected by
        effective_mask(mask, A, directed) after channel assembly/redaction.

    """
    CANONICAL: ClassVar[Set[str]] = set(CANONICAL_FEATURES)

    DERIVABLE_FROM_ADJ: ClassVar[FrozenSet[str]] = frozenset({
        "cn", "jaccard", "adamic_adar", "deg_row", "deg_col", "deg_diff"
    })

    # Subset of DERIVABLE_FROM_ADJ commonly appended when append_pairwise=True
    HEAVY_PAIRWISE_KEYS: ClassVar[FrozenSet[str]] = frozenset({
        "cn", "jaccard", "adamic_adar"
    })

    # Keep a list of dynamically padded features to avoid clogging the IO stream with duplicate warnings
    _warned_pad_features: ClassVar[Set[str]] = set()
    _stats_cache: ClassVar[Dict] = {}

    def __init__(
        self,
        use_mask_channel: Optional[bool] = None,
        directed: bool = False,
        supervised_redaction_policy: str = "adj_only",
        custom_feature_types: Optional[Dict[str, str]] = None,
    ):
        # keep tri-state so the training loop can decide per model before fit()
        self.use_mask_channel = use_mask_channel
        self.directed = bool(directed)
        self.custom_feature_types = dict(custom_feature_types or {})

        self.manifest: List[str] = []
        self.mean: Optional[torch.Tensor] = None    # (C,)
        self.std:  Optional[torch.Tensor] = None    # (C,)

        # Redaction policy at the supervised pixel (i,j) for BCHW inputs.
        # "all"      → zero every channel at (i,j)
        # "adj_only" → zero only the 'adj' channel at (i,j) [default]
        # "none"     → no redaction; all channels remain visible at (i,j)
        self.supervised_redaction_policy = str(supervised_redaction_policy)

    # ----- Stacking (BCHW/BHWC) -----
    def stack_channels_BCHW(
            self,
            A: torch.Tensor,
            feats: List[Dict[str, torch.Tensor]],
            mask: Optional[torch.Tensor],
            feature_keys: Sequence[str],
            *, include_adj: bool = True
    ) -> torch.Tensor:
        """
        Assemble input channels in BCHW order for image-style heads (CNN/Transformer).

        Channel order (when present):
          [0] 'adj'     → always first (observed adjacency, 0/1 float), with supervised cells redacted
          [1] 'mask'    → optional; appended only if self.use_mask_channel == True
          [2..] features from `feature_keys` (expanded; e.g., deg_row/deg_col, cn, ...)

        Redaction is controlled by `self.supervised_redaction_policy`:
        - "adj_only": redact the 'adj' channel at supervised cells before adjacency-derived
          channels are computed. As a result, channels derived from adjacency inside this
          function also reflect that redaction, while precomputed non-adjacency features do not.
        - "all": redact all 2D per-edge channels (both derived and precomputed).

        Returns:
            x_bchw : (B, C, N, N)
        """
        B, N, _ = A.shape
        dtype = A.dtype
        X: List[torch.Tensor] = []

        # ----- [0] adjacency channel (with supervised entries zeroed) -----
        adj = A.to(dtype)

        # Build the redaction mask directly from the task-provided mask
        if mask is not None:
            mb = mask.to(A.device, non_blocking=True)
            if mb.dtype != torch.bool:
                mb = mb > 0.5
        else:
            mb = None

        # Redact adjacency exactly at the task-mask coordinates
        if mb is not None and getattr(self, "supervised_redaction_policy", "adj_only") != "none":
            adj = adj.clone()
            adj[mb] = 0

        if include_adj:
            X.append(adj.unsqueeze(1))  # (B,1,N,N)

        # ----- [1] optional mask-as-input channel (separate from loss/eval mask) -----
        if self.use_mask_channel:
            if mask is None:
                m = torch.ones((B, N, N), dtype=torch.bool, device=A.device)
            else:
                m = mask.to(A.device, non_blocking=True)
                if m.dtype != torch.bool:
                    m = m > 0.5
            X.append(m.float().unsqueeze(1))  # (B,1,N,N)

        # Redacted adjacency per-graph for feature derivations
        A_base = adj

        # Pre-compute pairwise features for the full 3D batch
        _has_power = any(k.startswith("power_") for k in feature_keys)
        _pw_keys = [k for k in feature_keys if k in FeatureRegistry.DERIVABLE_FROM_ADJ]
        if _has_power and "jaccard" in _pw_keys and "cn" not in _pw_keys:
            _pw_keys.append("cn")
        _pw_cache_batch = pairwise_batch_from_adj(A_base, _pw_keys, is_directed=self.directed) if _pw_keys else {}

        # Pre-compute batched matrix transforms of adjacency
        _mat_cache_batch = {}
        _power_memo = {}
        if _has_power and "cn" in _pw_cache_batch and (
            self.directed or torch.equal(A_base, A_base.transpose(-1, -2))
        ):
            _power_memo[2] = _pw_cache_batch["cn"].to(dtype)
        
        for k in feature_keys:
            if k == "transpose":
                _mat_cache_batch[k] = A_base.transpose(1, 2).contiguous()
            elif k == "shortest_path":
                _mat_cache_batch[k] = torch.stack([
                    shortest_path_from_adj(A_base[b], is_directed=self.directed)
                    for b in range(B)
                ], dim=0).to(dtype)
            elif k.startswith("power_"):
                p = int(k.split("_", 1)[1])

                if p == 2 and 2 not in _power_memo:
                    _power_memo[2] = (A_base @ A_base).to(dtype)
                elif p == 3 and 3 not in _power_memo:
                    if 2 not in _power_memo:
                        _power_memo[2] = (A_base @ A_base).to(dtype)
                    _power_memo[3] = (_power_memo[2] @ A_base).to(dtype)
                elif p in (4, 5) and p not in _power_memo:
                    if 2 not in _power_memo:
                        _power_memo[2] = (A_base @ A_base).to(dtype)
                    if 4 not in _power_memo:
                        _power_memo[4] = (_power_memo[2] @ _power_memo[2]).to(dtype)
                    if p == 5:
                        _power_memo[5] = (A_base @ _power_memo[4]).to(dtype)

                _mat_cache_batch[k] = _power_memo[p]

        # [2..] feature channels
        ordered_feat_parts: List[torch.Tensor] = []
        for k in feature_keys:
            if k in FeatureRegistry.DERIVABLE_FROM_ADJ:
                ordered_feat_parts.append(_pw_cache_batch[k].to(dtype).unsqueeze(1))
                continue

            if k == "transpose" or k == "shortest_path" or k.startswith("power_"):
                ordered_feat_parts.append(_mat_cache_batch[k].unsqueeze(1))
                continue

            per_graph_parts: List[torch.Tensor] = []
            for b in range(B):
                fdict: Dict[str, torch.Tensor] = feats[b] if isinstance(feats[b], dict) else {}
                M: Optional[torch.Tensor] = None

                # Precomputed features passed in `feats[b][k]`
                v = fdict.get(k, None)
                if isinstance(v, torch.Tensor):
                    if v.dim() == 2:
                        # assume (N,N)
                        M = v.to(A.device, dtype=dtype, non_blocking=True)

                        # Assert padding for 2D features
                        if M.shape[0] != N or M.shape[1] != N:
                            if M.shape[0] > N or M.shape[1] > N:
                                raise ValueError(
                                    f"[FEATURE SHAPE] 2D custom feature '{k}' has shape {tuple(M.shape)} which exceeds the padded graph size ({N}, {N}). "
                                    f"Silent cropping is unsupported as it can cause undocumented side effects. "
                                    f"If this is intended, please crop the feature manually before passing it to the pipeline."
                                )
                            if (
                                k not in FeatureRegistry.CANONICAL
                                and k not in self.custom_feature_types
                                and k not in FeatureRegistry._warned_pad_features
                            ):
                                print(
                                    f"[WARN] Dimension mismatch for custom 2D feature '{k}'. "
                                    f"Got {tuple(M.shape)} but expected ({N}, {N}) to match padded batch. "
                                    f"Padding with zeros to avoid a crash."
                                )
                                FeatureRegistry._warned_pad_features.add(k)

                            pad_r = max(0, N - M.shape[0])
                            pad_c = max(0, N - M.shape[1])
                            M = torch.nn.functional.pad(M, (0, pad_c, 0, pad_r), value=0.0)

                    elif v.dim() == 1:
                        # 1D feature -> emit two channels (row/col)
                        rows = v.shape[0]
                        if rows > N:
                            raise ValueError(
                                f"[FEATURE SHAPE] 1D custom feature '{k}' has length {rows} which exceeds the padded graph size N ({N}). "
                                f"Silent cropping is unsupported as it can cause undocumented side effects. "
                                f"If this is intended, please crop the feature manually before passing it to the pipeline."
                            )
                        tmp = torch.zeros(N, dtype=dtype, device=A.device)
                        if rows > 0:
                            tmp[:rows] = v.to(dtype, non_blocking=True)

                        # row and col expansions (become two separate channels)
                        Mr = tmp.view(N, 1).expand(N, N)
                        Mc = tmp.view(1, N).expand(N, N)
                        per_graph_parts.append(torch.stack([Mr, Mc], dim=0).unsqueeze(0))
                        continue
                    else:
                        # unsupported rank -> let default zeros handle it
                        M = None
                else:
                    # not a tensor / missing -> let default zeros handle it
                    M = None

                # Default zeros if still None
                if M is None:
                    M = torch.zeros(N, N, dtype=dtype, device=A.device)

                    # Handle missing 1D features that require two channels (row/col)
                    if getattr(self, "manifest", None) and f"{k}_row" in self.manifest:
                        per_graph_parts.append(torch.stack([M, M], dim=0).unsqueeze(0))
                        continue

                per_graph_parts.append(M.unsqueeze(0).unsqueeze(0))

            if per_graph_parts:
                ordered_feat_parts.append(torch.cat(per_graph_parts, dim=0))
            else:
                ordered_feat_parts.append(torch.zeros((B, 0, N, N), dtype=dtype, device=A.device))

        # Concatenate along channel: per-batch base channels first, then feature channels
        base = torch.cat(X, dim=1) if X else torch.zeros((B, 0, N, N), dtype=dtype, device=A.device)
        if ordered_feat_parts:
            feat_cat = torch.cat(ordered_feat_parts, dim=1)  # (B, C_f, N, N)
            out = torch.cat([base, feat_cat], dim=1)
        else:
            out = base

        # "all" covers every model-input channel, including the optional 'mask' channel
        if mb is not None and getattr(self, "supervised_redaction_policy", "adj_only") == "all":
            out = out.masked_fill(mb.unsqueeze(1), 0)
        return out

    # ----- Stats / standardisation -----
    @torch.no_grad()
    def compute_channel_stats(
            self,
            loader: torch.utils.data.DataLoader,
            feature_keys: Sequence[str],
            max_batches: int = 1024,
            *, include_adj: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mean/std per channel from a bounded train-loader pass over at most `max_batches` batches.
        """
        sum_c, sumsq_c, count, C_seen = None, None, 0, None
        for i, (A, feats, _, mask) in enumerate(loader):
            # The Central Limit Theorem guarantees convergence well before 1,024 batches
            if i >= max_batches:
                break

            x = self.stack_channels_BCHW(A, feats, mask, feature_keys, include_adj=include_adj)  # (B,C,N,N)
            if C_seen is None:
                C_seen = int(x.shape[1])

            # Compute statistics exclusively on the redacted supervised edges (effective_mask).
            m = effective_mask(mask, A, self.directed)
            flat = x.permute(0, 2, 3, 1)[m].to(torch.float64)
            if flat.numel() == 0:
                continue
            s = flat.sum(dim=0)
            q = (flat * flat).sum(dim=0)
            if sum_c is None:
                sum_c, sumsq_c = s, q
            else:
                sum_c += s
                sumsq_c += q
            count += flat.shape[0]

        if not count:
            # Fallback: zero mean, unit std. 
            # C_seen is only None if the dataloader yielded exactly 0 batches.
            # In that case trust the pre-resolved manifest length.
            C = C_seen if C_seen is not None else len(self.manifest)
            return (
                torch.zeros(C, dtype=torch.float32),
                torch.ones(C, dtype=torch.float32)
            )

        mean = (sum_c / count).to(torch.float32)
        var = (sumsq_c / count) - (mean.to(torch.float64) ** 2)
        std = torch.sqrt(torch.clamp(var.to(torch.float32), min=1e-8))
        return mean, std

    def fit(
            self,
            train_loader: torch.utils.data.DataLoader,
            feature_keys: Sequence[str],
            *, include_adj: bool = True
    ) -> None:
        """
        Establish stable manifest and per-channel (mean,std) from train split.
        Manifest order: ['adj', ('mask'), expanded(features...)].

        MLP and Deep MLP reuse the same manifest and per-channel (mean,std), therefore the cache.
        """
        cache_key = (
            id(getattr(train_loader, "dataset", train_loader)),
            tuple(feature_keys),
            self.use_mask_channel,
            include_adj,
            self.directed,
            self.supervised_redaction_policy,
            tuple(sorted(self.custom_feature_types.items()))
        )

        if cache_key in self.__class__._stats_cache:
            cached = self.__class__._stats_cache[cache_key]
            self.manifest = list(cached['manifest'])
            self.mean = cached['mean'].clone() if cached['mean'] is not None else None
            self.std = cached['std'].clone() if cached['std'] is not None else None
            return
        
        keys = list(feature_keys)
        names: List[str] = ["adj"] if include_adj else []
        channel_sources: Dict[str, str] = {"adj": "adj"} if include_adj else {}
        if self.use_mask_channel:
            names.append("mask")
            channel_sources["mask"] = "mask"

        canonical_1d_keys = {"degree", "triangles", "clustering_coeff", "twohop"}
        for k in keys:
            explicit_type = self.custom_feature_types.get(k)
            emitted_names = (
                [f"{k}_row", f"{k}_col"]
                if k in canonical_1d_keys or explicit_type == "node"
                else [k]
            )
            for channel_name in emitted_names:
                previous_source = channel_sources.get(channel_name)
                if previous_source is not None and previous_source != k:
                    raise ValueError(
                        f"[CUSTOM FEATURE REGISTRATION] Feature '{k}' produces dense channel '{channel_name}', "
                        f"which is already produced by '{previous_source}'. Rename the custom feature "
                        f"to avoid colliding with a generated dense channel name."
                    )
                if previous_source is None:
                    names.append(channel_name)
                    channel_sources[channel_name] = k

        self.manifest = names
        self.mean, self.std = self.compute_channel_stats(train_loader, feature_keys, include_adj=include_adj)
        self.__class__._stats_cache[cache_key] = {
            'manifest': list(self.manifest),
            'mean': self.mean.clone() if self.mean is not None else None,
            'std': self.std.clone() if self.std is not None else None
        }

    def standardise_bchw(self, x: torch.Tensor, keep_idx: Optional[List[int]] = None) -> torch.Tensor:
        """
        x: (B, C_eff, H, W)
        Uses dataset means/stds; guards against tiny std and extreme values.
        """
        if self.mean is None or self.std is None:
            return x

        mean_sliced = self.mean[keep_idx] if keep_idx is not None else self.mean
        std_sliced = self.std[keep_idx] if keep_idx is not None else self.std

        mu = mean_sliced.view(1, -1, 1, 1).to(x.device, non_blocking=True)
        sig = std_sliced.view(1, -1, 1, 1).to(x.device, non_blocking=True)

        # Guard against tiny std -> huge magnitudes -> exploding logits/loss
        x_std = (x - mu) / sig.clamp_min(1e-3)

        # Keep dynamic range bounded for stability
        x_std.clamp_(-6.0, 6.0)

        return x_std

    @staticmethod
    def zscore_edges_per_graph(E: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Per-graph, per-channel z-score for edge tensors.
        Args:
            E: (B, N, N, C).
            mask: optional (B, N, N) boolean mask; True=valid. Broadcast over channels.
        Returns:
            Tensor with same shape as E, z-scored per graph per channel.
        """
        B, H, W, C = E.shape
        flat = E.reshape(B, H * W, C)

        if mask is not None:
            # Accept (B, N, N) or (B, 1, N, N); broadcast over channels.
            if mask.dim() == 4 and mask.size(1) == 1:
                mask = mask.squeeze(1)
            # Reshape explicitly to match H*W to avoid accidental mismatches.
            m = mask.reshape(B, H * W, 1).to(device=E.device, dtype=E.dtype)

            # avoid empty division
            count = torch.clamp(m.sum(dim=1, keepdim=True), min=1.0)
            mean = (flat * m).sum(dim=1, keepdim=True) / count
            var = ((flat - mean) ** 2 * m).sum(dim=1, keepdim=True) / count
        else:
            mean = flat.mean(dim=1, keepdim=True)
            var = flat.var(dim=1, unbiased=False, keepdim=True)

        std = torch.clamp(var.sqrt(), min=1e-8)
        return ((flat - mean) / std).reshape(B, H, W, C)

    @staticmethod
    def zscore_nodes_per_graph(X: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Per-graph, per-feature z-score for node tensors.

        Args:
            X: (B, N, F).
            mask: optional (B, N) boolean mask. True=valid.

        Returns:
            Tensor with same shape as X, z-scored per graph per feature.
        """
        if X.dim() != 3:
            raise ValueError("Expected node tensor with shape (B, N, F)")

        B, N, n_feat = X.shape

        # Short-circuit the z-scoring if the feature dimension is zero
        if n_feat == 0:
            return X

        if mask is not None:
            m0 = mask.to(X.device, non_blocking=True)
            if m0.dtype != torch.bool:
                m0 = m0 > 0.5

            m = m0.view(B, N, 1).expand_as(X)

            count = torch.clamp(m.sum(dim=1, keepdim=True).to(X.dtype), min=1.0)
            mean = (X * m).sum(dim=1, keepdim=True) / count
            var = ((X - mean) ** 2 * m).sum(dim=1, keepdim=True) / count
        else:
            mean = X.mean(dim=1, keepdim=True)
            var = X.var(dim=1, unbiased=False, keepdim=True)

        std = torch.clamp(var.sqrt(), min=1e-8)
        out = ((X - mean) / std)

        return out


# ============================================================
# 3) TaskSpec + example task adapters
# ============================================================
@dataclass
class TaskSpec:
    """
    Minimal interface each task implements.
    The pipeline is agnostic to how data is produced internally.
    """
    name: str
    directed: bool


@dataclass
class TaskHooks:
    """
    Lightweight configuration for building a task dataset from GraphBenchmark.

    - label_fn: Optional callable for computing the label matrix L. Supported
    signatures are:
        * None -> returns an all-zero label matrix.
        * A callable whose parameter names map to A_obs, A_true, and/or G_true.
        * A callable with *args, which receives (A_obs, A_true, G_true).

    Parameter matching is alias-based and invocation is positional. Keyword-only
    parameters and arbitrary **kwargs are not supported.

    - feature_set: True for all canonical features, False for none, or a list containing
      canonical feature names/macros and typed custom declarations `(name, "node"|"edge")`.
    - orientation: None | "dag"; passed to *_orient_to_directed on A_obs.
    - ensure_connected: If True, connect components using proportional multi-stitching.
    - allow_adj_channel: If True, include adjacency as an explicit numerical feature.
    """
    label_fn: Optional[Union[
        Callable[[np.ndarray, nx.Graph], np.ndarray],
        Callable[[np.ndarray], np.ndarray]
    ]] = None
    feature_set: Union[bool, List[Union[str, Tuple[str, Literal["node", "edge"]]]]] = False
    orientation: Optional[str] = None
    ensure_connected: bool = False
    allow_adj_channel: bool = False


class ProvidedSplitsTask(TaskSpec):
    """
    Public task adapter for supported dataset-backed workflows.

    Use this for:
      - generated multi-graph datasets with ratio-based splitting, or
      - pre-split datasets exposed through `bench.splits`.

    Direct task-provided dataloader methods are unsupported.

    Args:
        name: Task name for logging/checkpoint directories.
        directed: Whether evaluation is over full matrix (directed) or upper-triangle (undirected).
        hooks: TaskHooks
            - label_fn is optional (None yields all-zero labels), feature_set controls canonical/custom features, and allow_adj_channel gates 'adj' as input.
        num_graphs: How many graphs to sample. Passed through to GraphBenchmark.
        min_nodes / max_nodes: Node count bounds for sampling.
        ratios: Train/val/test fractions for GraphBenchmark.make_loaders.
            Zero-sized splits are allowed. Splits are formed via integer cut points,
            so on small datasets a positive fraction may still round down to an empty split.
        num_classes: Optional override for multi-class tasks (default 1).
        eval_on_existing_edges_only: If True, loss/metrics are evaluated only on observed edges (mask ∩ A).
        show_progress: Whether to show GraphBenchmark sampling progress.
    """
    def __init__(
        self,
        name: str,
        directed: bool,
        hooks: TaskHooks,
        num_graphs: int = 400,
        min_nodes: int = 6,
        max_nodes: int = 140,
        ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        num_classes: int | None = None,
        eval_on_existing_edges_only: bool = False,
        show_progress: bool = False,
        mask_policy: Optional[str] = None,
        bench: Optional[GraphBenchmark] = None,
        pin_memory: Optional[bool] = None,
        num_workers: int = 0,
        seed: Optional[int] = None
    ):
        self.hooks = hooks
        self.num_graphs = num_graphs
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.ratios = ratios
        self.show_progress = bool(show_progress)
        self._bench_instance = bench
        self._owns_bench_instance = False
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.mask_policy = mask_policy  # Optional per-task mask builder. None keeps mask=ones in collate_fn_pad
        self.eval_on_existing_edges_only = bool(eval_on_existing_edges_only)
        self.num_classes = int(num_classes) if num_classes is not None else 1

        if seed is None:
            seed = random.randint(0, 2 ** 32 - 1)
            print(f"\n{'=' * 80}\n[SEED] No seed provided. Generated random seed: {seed}\n{'=' * 80}\n")
        else:
            print(f"\n{'=' * 80}\n[SEED] Using provided seed: {seed}\n{'=' * 80}\n")

        self.seed = int(seed)
        seed_everything(self.seed)

        super().__init__(
            name=name,
            directed=bool(directed)
        )


    def _build_loaders(self, batch_size: int = None) -> tuple:
        """
        Build and return (train_loader, val_loader, test_loader).

        Behaviour is inferred from `bench.splits`:
        - If `bench.splits` contains (N,N) split masks, use the single-graph loader builder.
        - If `bench.splits` contains pre-divided split collections (lists/tuples/Datasets of samples),
          preserve split membership and build three loaders directly.
        """
        bench = self._bench_instance
        if bench is None:
            bench = GraphBenchmark(show_progress=bool(getattr(self, "show_progress", False)))
            self._bench_instance = bench
            self._owns_bench_instance = True

        bs = int(batch_size if batch_size is not None else getattr(self, "batch_size", 16))
        nw = getattr(self, "num_workers", 0)
        pin = _resolve_loader_pin_memory(self)
        collate = partial(collate_fn_pad, num_classes=self.num_classes)

        # Validate and cache mask policy before any dataset access
        _cached_mask_policy = str(self.mask_policy).lower() if self.mask_policy else ""
        if _cached_mask_policy not in {"", "edges", "non_edges", "all"}:
            raise ValueError(f"Unknown mask_policy: {self.mask_policy!r}")

        def _is_square_matrix(x) -> bool:
            if isinstance(x, (list, tuple, dict, Dataset)):
                return False
            try:
                t = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))
            except Exception:
                return False
            return (t.ndim == 2) and (t.shape[0] == t.shape[1])

        def _is_sample_record(item) -> bool:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                return False
            if not _is_square_matrix(item[0]):
                return False
            return any(isinstance(c, dict) for c in item)

        def _split_is_predivided(split_obj) -> bool:
            if isinstance(split_obj, Dataset):
                return True
            if isinstance(split_obj, (list, tuple)):
                if len(split_obj) == 0:
                    return True
                return _is_sample_record(split_obj[0])
            return False
        
        def _build_mask(A_any):
            A_arr = np.asarray(A_any)
            if _cached_mask_policy == "edges":
                m = (A_arr > 0.5)
            elif _cached_mask_policy == "non_edges":
                m = (A_arr <= 0.5)
            elif _cached_mask_policy == "all":
                m = np.ones_like(A_arr, dtype=bool)

            return m.astype(bool)

        def _wrap_with_mask_policy(ds_like: Any):
            if not self.mask_policy:
                return ds_like

            try:
                if hasattr(ds_like, "__len__") and len(ds_like) == 0:
                    return ds_like
                probe = ds_like[0]
                if not isinstance(probe, (list, tuple)) or not _is_square_matrix(probe[0]):
                    return ds_like
            except Exception:
                return ds_like

            class _MaskPolicyDataset(Dataset):
                def __init__(self, base):
                    self.base = base
                    self._pipeline_persistent_workers = bool(getattr(base, "_pipeline_persistent_workers", False))

                def __len__(self):
                    return len(self.base)

                def __getitem__(self, idx):
                    item = self.base[idx]
                    comps = list(item)

                    A0 = comps[0]
                    A_t = A0 if isinstance(A0, torch.Tensor) else torch.as_tensor(np.asarray(A0))
                    if A_t.ndim != 2 or A_t.shape[0] != A_t.shape[1]:
                        raise ValueError("First element must be an NxN adjacency matrix.")
                    N = int(A_t.shape[0])

                    # Extract remaining NxN matrices purely by shape, mirroring collate_fn_pad
                    matrices = []
                    for c in comps[1:]:
                        if isinstance(c, dict):
                            continue
                        arr = c if isinstance(c, torch.Tensor) else torch.as_tensor(np.asarray(c))
                        if arr.ndim != 2 or tuple(arr.shape) != (N, N):
                            raise ValueError(
                                f"[SAMPLE SHAPE] Sample contains a matrix component of shape {tuple(arr.shape)}; label "
                                f"and evaluation mask matrices must match the adjacency at ({N}, {N}). "
                                f"Roles are assigned in order, so a mis-shaped matrix would shift the remaining "
                                f"components into the wrong slots."
                            )
                        matrices.append(arr)

                    # If the user already provided >= 2 matrices (L and Mask), respect their tuple and skip policy
                    if len(matrices) >= 2:
                        return item

                    # Otherwise, they provided 0 or 1 matrices (assumed to be L by position)
                    L = matrices[0] if len(matrices) > 0 else torch.zeros((N, N), dtype=torch.float32)
                    feats = next((c for c in comps if isinstance(c, dict)), {})
                    M = _build_mask(A0)
                    
                    return A0, feats, L, M

            return _MaskPolicyDataset(ds_like)

        def _mk_loader(ds_like, shuffle: bool) -> DataLoader:
            ds_like = _wrap_with_mask_policy(ds_like)
            effective_shuffle = bool(shuffle and len(ds_like) > 0)
            kwargs = dict(
                batch_size=bs,
                shuffle=effective_shuffle,
                num_workers=nw,
                collate_fn=collate,
                pin_memory=pin,
                persistent_workers=bool(nw > 0 and getattr(ds_like, "_pipeline_persistent_workers", False))
            )

            if effective_shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed)
                kwargs["generator"] = g

            return DataLoader(ds_like, **kwargs)

        if hasattr(bench, "splits") and isinstance(bench.splits, dict):
            train = bench.splits.get("train", [])
            val = bench.splits.get("val", [])
            test = bench.splits.get("test", [])

            # Single-graph case: split definitions are masks
            if _is_square_matrix(train) or _is_square_matrix(val) or _is_square_matrix(test):
                safe_splits = dict(bench.splits)
                for k in ("train", "val", "test"):
                    if k not in safe_splits:
                        ref = train if _is_square_matrix(train) else (val if _is_square_matrix(val) else test)
                        safe_splits[k] = (
                            torch.zeros_like(ref, dtype=torch.bool)
                            if isinstance(ref, torch.Tensor)
                            else np.zeros_like(ref, dtype=bool)
                        )
                bench_view = copy.copy(bench)
                bench_view.splits = safe_splits
                return _build_single_graph_loaders_from_bench(
                    bench_view, self.hooks, batch_size=bs, num_workers=nw, task=self
                )

            # Pre-divided split collections: preserve membership and create loaders directly
            if _split_is_predivided(train) and _split_is_predivided(val) and _split_is_predivided(test):
                if len(train) == 0 and len(val) == 0 and len(test) == 0:
                    raise ValueError("[INVALID DATASET] Provided pre-divided splits must contain data in at least one split.")

                train = _augment_with_canonical_features(train, bench, self.hooks, self.directed)
                val = _augment_with_canonical_features(val, bench, self.hooks, self.directed)
                test = _augment_with_canonical_features(test, bench, self.hooks, self.directed)
                tr = _mk_loader(train, shuffle=True)
                va = _mk_loader(val, shuffle=False)
                te = _mk_loader(test, shuffle=False)
                return (tr, va, te)

            raise ValueError(
                "[INVALID DATASET] bench.splits must contain either single-graph split masks "
                "or pre-divided train/val/test sample collections."
            )

        # Generative benchmark path
        if self.num_graphs is None or int(self.num_graphs) < 1:
            raise ValueError("[INVALID CONFIGURATION] Generated datasets require num_graphs >= 1.")

        # Don't resample new graphs; re-batch the existing ones
        ds = getattr(self, "_active_run_dataset", None)
        if ds is None:
            specs = bench.sample_specs(
                num_graphs=self.num_graphs,
                min_nodes=self.min_nodes,
                max_nodes=self.max_nodes
            )

            # Normalise feature_set so the synthetic path matches the single-graph/provided-splits path.
            hooks_for_generation = copy.copy(self.hooks)
            hooks_for_generation.feature_set = _normalise_feature_set(
                getattr(self.hooks, "feature_set", False),
                directed=self.directed
            )

            ds = bench.generate_dataset(
                specs,
                hooks=hooks_for_generation,
                prepackage=True,
                directed=self.directed,
                seed=self.seed
            )

            ds = _augment_with_canonical_features(ds, bench, self.hooks, self.directed)
            ds = _wrap_with_mask_policy(ds)
            self._active_run_dataset = ds

        tr, va, te = bench.make_loaders(
            dataset=ds,
            batch_size=bs,
            ratios=self.ratios,
            collate_fn=collate,
            seed=self.seed,
            pin_memory=pin,
            num_workers=nw
        )

        return (tr, va, te)


class _SingleGraphDataset(Dataset):
    def __init__(self, A: torch.Tensor, feats: dict, L: torch.Tensor, mask: torch.Tensor):
        self.A = A
        self.L = L
        self.mask = mask
        self.feats = feats

    def __len__(self): return 1

    def __getitem__(self, _idx):
        # collate_fn_pad expects items like (A, feats_dict, L, mask)
        return self.A, self.feats, self.L, self.mask


def _coerce_bench(bench_like):
    """
    Accept dict or object; validate the required single-graph bench fields.
    Only truly essential pieces are required:
      - A (adjacency, (N,N) torch.FloatTensor)
      - splits (dict with 'train'/'val'/'test' bool (N,N) masks)
    """
    bench = SimpleNamespace(**bench_like) if isinstance(bench_like, dict) else bench_like

    # Minimal validation: we really need an adjacency and splits
    missing = []
    if not hasattr(bench, "A"): missing.append("A")
    if not hasattr(bench, "splits"): missing.append("splits")
    if missing:
        raise TypeError(
            "Bench is missing required fields: " + ", ".join(missing) +
            ". Supported single-graph tasks must expose those fields directly on task.bench."
        )
    return bench


def _normalise_feature_spec(x, directed: bool) -> Tuple[List[str], Dict[str, str]]:
    """
    Normalise hooks.feature_set to ordered feature keys plus explicit custom types.

    Semantics:
      True                       -> orientation-aware automatic canonical feature set
      list                       -> canonical names/macros and/or `(name, type)` custom entries
      str                        -> [str]
      False/None/[]              -> []

    Custom entries must be written as `(name, "node")` or `(name, "edge")`.
    Canonical names and macros remain plain strings.

    Returns:
        feature_list: flattened, ordered, de-duplicated feature keys.
        custom_types: mapping from custom feature name to "node" or "edge".
    """
    if x is True:
        return list(DIRECTED_AUTO_FEATURES if directed else UNDIRECTED_AUTO_FEATURES), {}
    elif not x:
        requested = []
    elif isinstance(x, str):
        requested = [x]
    else:
        requested = list(x)

    expanded: List[str] = []
    custom_types: Dict[str, str] = {}
    reserved_custom_names = {"adj", "mask", "_N"}

    for item in requested:
        if isinstance(item, (list, tuple)):
            if len(item) != 2:
                raise ValueError(
                    f"[CUSTOM FEATURE] Custom feature declarations must be (name, type) pairs; got {item!r}."
                )
            k, v_type = item
            if not isinstance(k, str) or not k:
                raise ValueError(f"[CUSTOM FEATURE] Custom feature name must be a non-empty string; got {k!r}.")
            if v_type not in ("node", "edge"):
                raise ValueError(
                    f"[CUSTOM FEATURE] Invalid type {v_type!r} for '{k}'. Must be 'node' or 'edge'."
                )
            if k in reserved_custom_names:
                raise ValueError(f"[CUSTOM FEATURE] '{k}' is reserved by the pipeline and cannot be used as a custom feature name.")
            if k in FeatureRegistry.CANONICAL:
                raise ValueError(
                    f"[CUSTOM FEATURE] '{k}' is pipeline-owned. Request canonical features by name without a custom type."
                )
            previous = custom_types.get(k)
            if previous is not None and previous != v_type:
                raise ValueError(
                    f"[CUSTOM FEATURE] Conflicting types for '{k}': {previous!r} and {v_type!r}."
                )
            custom_types[k] = v_type
            expanded.append(k)
            continue

        if not isinstance(item, str):
            raise TypeError(
                f"[FEATURE SET] Entries must be feature-name strings or (name, type) custom declarations; got {item!r}."
            )
        if item == "_N":
            raise ValueError("[FEATURE SET] '_N' is reserved internal metadata and cannot be requested as a feature.")
        if item == "powers":
            expanded += [p for p in ("power_2", "power_3", "power_4", "power_5")
                         if p in FeatureRegistry.CANONICAL]
        elif item == "endpoint_degree":
            expanded += [d for d in ("deg_row", "deg_col")
                         if d in FeatureRegistry.CANONICAL]
        else:
            expanded.append(item)

    seen, out = set(), []
    for k in expanded:
        if k == "adj":
            print("[INFO] ignoring 'adj' in feature_set; it is handled by manifest/gating.")
            continue
        if k not in FeatureRegistry.CANONICAL and k not in custom_types:
            print(f"[WARN] Unknown/non-canonical feature '{k}' dropped. "
                  f"Allowed canonical features: {sorted(FeatureRegistry.CANONICAL)}. "
                  f"Declare custom features as ('{k}', 'node') or ('{k}', 'edge').")
            continue
        if k not in seen:
            seen.add(k)
            out.append(k)

    return out, custom_types


def _normalise_feature_set(x, directed: bool) -> List[str]:
    """Return only the ordered feature-key list from `_normalise_feature_spec`."""
    feature_list, _ = _normalise_feature_spec(x, directed=directed)
    return feature_list


def _resolve_loader_pin_memory(owner: Any = None) -> bool:
    """Resolve DataLoader pin_memory with a backwards-compatible CUDA-aware default."""
    if owner is None:
        return torch.cuda.is_available()
    val = getattr(owner, "pin_memory", None)
    if val is None:
        return torch.cuda.is_available()
    return bool(val) and torch.cuda.is_available()


def _build_single_graph_loaders_from_bench(bench, hooks, batch_size: int = 16, num_workers: int = 0, task=None):
    """
    Build train/val/test DataLoaders for the common single-graph case, using the
    task/benchmark's `extract_features` function and the provided split masks.

    Each split loader wraps exactly one graph sample; train/val/test differ only by
    the supervision mask, so shuffling and DataLoader generator seeding are irrelevant here.
    """
    bench = _coerce_bench(bench)
    task_obj = task

    # A: accept numpy or torch; keep on CPU here, models will .to(DEVICE) later.
    A = bench.A
    if not isinstance(A, torch.Tensor):
        A = torch.as_tensor(A, dtype=torch.float32)
    else:
        A = A.to(dtype=torch.float32, device='cpu')

    # splits: expect dict[str] -> (N,N) bool-like; coerce to torch.bool
    safe_splits = dict(bench.splits)
    for k in ("train", "val", "test"):
        if k not in safe_splits:
            safe_splits[k] = torch.zeros_like(A, dtype=torch.bool)

    splits = {k: (v if isinstance(v, torch.Tensor) else torch.as_tensor(v)).bool()
            for k, v in safe_splits.items()}

    # Normalise requested features and call your extractor on numpy
    A_np = A.detach().cpu().numpy()
    directed = bool(getattr(task_obj, "directed", False))
    feature_list = _normalise_feature_set(getattr(hooks, 'feature_set', False), directed=directed)
    if hasattr(bench, 'extract_features') and callable(bench.extract_features):
        if "directed" in inspect.signature(bench.extract_features).parameters:
            feats_np = bench.extract_features(A_np, feature_set=feature_list, directed=directed)
        else:
            feats_np = bench.extract_features(A_np, feature_set=feature_list)
    else:
        # If the extractor lives on the task, try that as well
        if task_obj and hasattr(task_obj, 'extract_features'):
            if "directed" in inspect.signature(task_obj.extract_features).parameters:
                feats_np = task_obj.extract_features(A_np, feature_set=feature_list, directed=directed)
            else:
                feats_np = task_obj.extract_features(A_np, feature_set=feature_list)
        else:
            # Without bench/task extract_features, canonical auto-derivation is unavailable
            feats_np = {}

    # Ensure numpy dtype/shape sanity and convert to torch
    feats_t = {}
    for k, v in feats_np.items():
        vv = np.asarray(v)
        if vv.ndim == 0:
            feats_t[k] = torch.as_tensor(float(vv), dtype=torch.float32)
        elif vv.ndim == 2:
            # Allow N x N (pairwise) OR N x F / F x N (node features)
            if vv.shape == (A.shape[0], A.shape[0]) or A.shape[0] in vv.shape:
                feats_t[k] = torch.from_numpy(vv.astype(np.float32))
            else:
                print(
                    f"[WARN] feature '{k}' has shape {vv.shape}; expected ({A.shape[0]},{A.shape[0]}) "
                    f"or one dimension equal to {A.shape[0]}. Skipping."
                )
                continue
        elif vv.ndim == 1:
            # node feature: length N
            if vv.shape[0] != A.shape[0]:
                print(f"[WARN] feature '{k}' has length {vv.shape[0]}; expected {A.shape[0]}. Skipping.")
                continue
            feats_t[k] = torch.from_numpy(vv.astype(np.float32))
        else:
            # ignore non 1D/2D features
            continue

    # Labels from hooks.label_fn with the same dispatch rules as GraphBenchmark.generate_dataset()
    A_true_src = getattr(bench, "A_true", None)
    A_true_np = A_np if A_true_src is None else np.asarray(A_true_src, dtype=np.float32)

    G_true = getattr(bench, "G_true", None)
    if G_true is None:
        G_true = getattr(bench, "G", None)
    if G_true is None:
        G_src = A_true_np
        G_true = nx.from_numpy_array(
            (G_src > 0).astype(np.uint8),
            create_using=nx.DiGraph if directed else nx.Graph,
        )

    fast_label_fn = GraphBenchmark._compile_label_fn(getattr(hooks, "label_fn", None))
    L_np = fast_label_fn(A_np, A_true_np, G_true)
    L = torch.as_tensor(L_np, dtype=torch.long if np.issubdtype(L_np.dtype, np.integer) else torch.float32)

    # Datasets/loaders (single item each). Use our collate_fn_pad to keep the runner consistent.
    nw = int(num_workers)
    train_ds = _SingleGraphDataset(A, feats_t, L, splits["train"])
    val_ds   = _SingleGraphDataset(A, feats_t, L, splits["val"])
    test_ds  = _SingleGraphDataset(A, feats_t, L, splits["test"])

    pin = _resolve_loader_pin_memory(task_obj)
    collate = partial(collate_fn_pad, num_classes=int(getattr(task_obj, "num_classes", 1)))
    train_loader = DataLoader(train_ds, batch_size, shuffle=False, num_workers=nw, collate_fn=collate, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size, shuffle=False, num_workers=nw, collate_fn=collate, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size, shuffle=False, num_workers=nw, collate_fn=collate, pin_memory=pin)
    return train_loader, val_loader, test_loader


def _augment_with_canonical_features(ds_like, bench, hooks, directed):
    """
    Wrap a pre-divided split dataset so requested canonical features are made
    available for that split when they are not already present.

    Canonical features are pipeline-owned, split-level outputs. Custom features
    remain user-supplied and are used only where the dataset provides them.

    The first sample is intentionally the authoritative split-schema probe.
    This is not a per-sample repair pass. If a requested canonical key is absent
    from sample 0, the dataset is wrapped and that feature is derived for every
    item on access. If the key is present in sample 0, the split is trusted to
    provide it consistently and later samples are not scanned or repaired.
    """
    requested = _normalise_feature_set(getattr(hooks, "feature_set", False), directed=directed)
    if not requested:
        return ds_like

    # Need extract_features on bench
    if not hasattr(bench, "extract_features") or not callable(bench.extract_features):
        return ds_like

    # Check whether extract_features accepts a 'directed' kwarg
    _ef_accepts_directed = "directed" in inspect.signature(bench.extract_features).parameters

    # Probe one sample to determine which requested features are absent
    try:
        if hasattr(ds_like, "__len__") and len(ds_like) == 0:
            return ds_like
        probe = ds_like[0]
        if not isinstance(probe, (list, tuple)):
            return ds_like
    except Exception:
        return ds_like

    probe_feats = next((c for c in probe if isinstance(c, dict)), None)
    present = set(probe_feats.keys()) if probe_feats is not None else set()
    missing = [k for k in requested if k not in present and k != "shortest_path" and k in FeatureRegistry.CANONICAL]
    if not missing:
        return ds_like

    class _CanonicalAugmentedDataset(Dataset):
        def __init__(self, base):
            self.base = base
            self._cached_missing = [None] * len(base)
            self._pipeline_persistent_workers = True

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            item = self.base[idx]
            comps = list(item)

            # Adjacency is always the first element
            A0 = comps[0]
            A_np = np.asarray(A0, dtype=np.float32)

            # Locate the feature dict in the tuple
            feat_idx = None
            for i, c in enumerate(comps):
                if isinstance(c, dict):
                    feat_idx = i
                    break

            if feat_idx is not None:
                fdict = dict(comps[feat_idx])
            else:
                fdict = {}

            cached = self._cached_missing[idx]
            if cached is None:
                if _ef_accepts_directed:
                    cached = bench.extract_features(A_np, feature_set=missing, directed=directed)
                else:
                    cached = bench.extract_features(A_np, feature_set=missing)

                cached = {k: v for k, v in cached.items() if k in missing}
                self._cached_missing[idx] = cached

            for k, v in cached.items():
                if k not in fdict:
                    fdict[k] = v

            # Rebuild the tuple with the augmented dict
            if feat_idx is not None:
                comps[feat_idx] = fdict
            else:
                comps.insert(1, fdict)

            return tuple(comps)

    return _CanonicalAugmentedDataset(ds_like)


def _pick_probe_loader(train_loader, val_loader, test_loader):
    return train_loader if len(train_loader.dataset) > 0 else (
        val_loader if len(val_loader.dataset) > 0 else test_loader
    )


def _resolve_loaders(task, cfg, batch_size=None):
    """
    Resolves the provided task object into standard train, validation, and test PyTorch DataLoaders.

    This function acts as the central gatekeeper enforcing the pipeline's "Supported task shapes"
    rule. It prevents invalid task structures (like direct dataloader injection) from proceeding
    downstream.

    Supported task entry paths only:
      1. ProvidedSplitsTask-style tasks (or subclasses) explicitly exposing a `_build_loaders(...)` method.
         This handles multi-graph datasets, generated datasets, and pre-divided split collections.
      2. Single-graph tasks exposing exactly `task.bench` + `task.hooks`.
         This handles the standard single-graph benchmarking path where splitting is defined
         via boolean masks.

    Args:
        task: The task object to resolve.
        cfg: Configuration object containing standard model/training parameters.
        batch_size: Optional override for the batch size. If None, it is inferred from `cfg`.

    Returns:
        tuple: (train_loader, val_loader, test_loader)

    Raises:
        RuntimeError: If the task exposes direct dataloader attributes (`train_dataloader`, etc.),
                      or if it does not conform to one of the two supported shapes above.
    """
    bs = batch_size if batch_size is not None else getattr(cfg, "batch_size", None)
    loader_sig = (
        int(bs) if bs is not None else None,
        int(getattr(task, "num_workers", 0)),
        getattr(task, "pin_memory", None),
    )

    # 1) Supported adapter path: ProvidedSplitsTask / subclasses
    if hasattr(task, "_build_loaders") and callable(task._build_loaders):
        cached_sig = getattr(task, "_active_run_loader_sig", None)
        cached_loaders = getattr(task, "_active_run_loaders", None)
        if cached_loaders is not None and cached_sig == loader_sig:
            return cached_loaders

        loaders = task._build_loaders(batch_size=bs)
        task._active_run_loaders = loaders
        task._active_run_loader_sig = loader_sig
        return loaders

    # 2) Reject unsupported direct dataloader tasks explicitly
    if any(hasattr(task, m) for m in ("train_dataloader", "val_dataloader", "test_dataloader")):
        raise RuntimeError(
            "Direct task-provided train_dataloader/val_dataloader/test_dataloader entry points are unsupported. "
            "Use ProvidedSplitsTask (or a subclass implementing _build_loaders), or expose task.bench + task.hooks "
            "for the supported single-graph path."
        )

    # 3) Supported single-graph path: explicit bench + hooks only
    bench_obj = getattr(task, "bench", None)
    hooks = getattr(task, "hooks", None)

    if bench_obj is not None and hooks is not None:
        return _build_single_graph_loaders_from_bench(
            bench_obj, hooks, batch_size=bs, num_workers=int(getattr(task, "num_workers", 0)), task=task
        )

    raise RuntimeError(
        "Unsupported task shape. Use ProvidedSplitsTask (or subclass) or expose task.bench + task.hooks "
        "for the supported single-graph path."
    )


def _task_to_meta_dict(task):
    def _get_safe_hooks(h_obj):
        if h_obj is None:
            return None
        res = {h: getattr(h_obj, h) for h in ("feature_set", "orientation", "ensure_connected", "allow_adj_channel") if hasattr(h_obj, h)}
        return res if res else None

    def _is_safe_sequence(v):
        return isinstance(v, (list, tuple)) and all(
            isinstance(x, (int, float, str, bool, type(None))) for x in v
        )

    if _dc.is_dataclass(task):
        # Shallow field extraction to avoid catastrophic deep-copies of datasets/loaders
        merged = {f.name: getattr(task, f.name) for f in _dc.fields(task)}
        merged.update(getattr(task, "__dict__", {}))

        # Preserve only public primitive metadata, dropping private runtime state and heavy objects
        safe_scalar_types = (int, float, str, bool, type(None))
        d = {
            k: v for k, v in merged.items()
            if not str(k).startswith("_") and (
                isinstance(v, safe_scalar_types) or _is_safe_sequence(v)
            )
        }

        # Sanitise raw hooks object to prevent pickling crashes on callables
        safe_hooks = _get_safe_hooks(merged.get("hooks"))
        if safe_hooks:
            d["hooks"] = safe_hooks

        # Deepcopy after hook sanitisation so the copied payload stays lightweight and checkpoint-safe
        return copy.deepcopy(d)

    if isinstance(task, dict):
        return dict(task)

    # Shallow pick of common fields from SimpleNamespace-like tasks
    keys = (
        "name", "directed", "eval_on_existing_edges_only",
        "num_graphs", "min_nodes", "max_nodes", "ratios"
    )
    meta = {k: getattr(task, k) for k in keys if hasattr(task, k)}

    # Summarise hooks without trying to serialise callables
    safe_hooks = _get_safe_hooks(getattr(task, "hooks", None))
    if safe_hooks:
        meta["hooks"] = safe_hooks
    return meta


# ============================================================
# 4) Model zoo (same architectures, registry-aware)
# ============================================================
class MatrixMLPBase(nn.Module):
    """
    Base class for per-edge Matrix MLPs. Expects self.net to be defined by subclasses.

    Each edge position is scored independently — there is no spatial coupling between (i, j) pairs.
    Supported training/evaluation paths gather only supervised pairs and score them through
    `_forward_flat`, skipping padding and non-supervised positions.
    """
    def _forward_flat(self, x2d: torch.Tensor) -> torch.Tensor:
        """
        Returns logits on the same device as model parameters.
        """
        param = next(self.parameters())
        dev = param.device
        dtype = param.dtype

        if x2d.size(0) == 0:
            return torch.empty((0, self.out_dim), device=dev, dtype=dtype)
        
        xb = x2d.to(device=dev, dtype=dtype, non_blocking=True)
        return self.net(xb)


class MatrixFeatureMLP(MatrixMLPBase):
    """Channels-last MLP over per-edge features."""

    def __init__(self, in_channels: int, hidden_dim: int = 128, num_classes: int = 1, dropout: float = 0.1):
        super().__init__()
        self.out_dim = 1 if num_classes == 1 else num_classes
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, self.out_dim),
        )


class MatrixFeatureDeepMLP(MatrixMLPBase):
    """Deeper MLP with the same device-safe forward as MatrixFeatureMLP."""
    def __init__(self, in_channels: int, hidden_dim: int = 256, depth: int = 4,
                 dropout: float = 0.3, num_classes: int = 1):
        super().__init__()
        layers: List[nn.Module] = []
        h = hidden_dim
        in_dim = in_channels
        for _ in range(max(1, depth)):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            in_dim = h

        self.out_dim = 1 if num_classes == 1 else num_classes
        layers.append(nn.Linear(in_dim, self.out_dim))

        self.net = nn.Sequential(*layers)


class MatrixFeatureCNN(nn.Module):
    """Size-agnostic CNN with kxk head (default 3x3)."""
    def __init__(
        self,
        in_channels: int,
        hidden: int = 64,
        head_kernel: int = 3,
        num_classes: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        assert head_kernel in (1, 3, 5)
        self.num_classes = num_classes

        Drop = nn.Identity if dropout <= 0.0 else (lambda: nn.Dropout2d(dropout))

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            Drop(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            Drop(),
        )
        self.head = nn.Conv2d(hidden, num_classes, kernel_size=head_kernel, padding=head_kernel // 2)

    def forward(self, x_bchw_std: torch.Tensor) -> torch.Tensor:
        h = self.stem(x_bchw_std)
        logits = self.head(h)                     # (B, K, N, N) or (B,1,N,N)
        if self.num_classes == 1:
            return logits.squeeze(1)              # (B, N, N)
        else:
            return logits                         # (B, K, N, N)


class PatchTransformer(nn.Module):
    r"""
    Patchified Transformer over an adjacency "image" (B, C, N, N).

    Highlights
    ----------
    - Supports non-overlap (stride == patch) and overlap (stride < patch).
    - Robust patchify/unpatchify:
        * Fast reshape path when stride == patch
        * Fold + normalisation when stride < patch
    - Positional encoding for the EXACT token grid (Sr x Sc).
    - Tokenisation policy:
        * "keep_all" (default): keep all tiles as tokens.
        * "from_mask": keep tokens only where the provided _task_mask has any True in the tile.
        * "auto": use mask if it's dense enough; otherwise fall back to keep_all (min_keep_ratio threshold).
    - Optional tiny decoder (ConvTranspose2d + 1x1) to restore edge-level detail from tokens.

    Outputs
    -------
    - If num_classes == 1: logits shape (B, N, N)
    - Else:                logits shape (B, K, N, N)
    """
    def __init__(
        self,
        in_channels: int,
        d_model: int = 384,
        n_layers: int = 6,
        n_heads: int = 6,
        patch: int = 64,
        stride: Optional[int] = None,     # None => stride == patch (no overlap)
        dropout: float = 0.10,
        num_classes: int = 1,
        max_tokens: int = 32768,          # warn if Sr*Sc exceeds this
        use_decoder: bool = True,         # tiny upsampling head
        token_policy: str = "keep_all",  # "from_mask" | "keep_all" | "auto"
        min_keep_ratio: float = 0.0,      # used only when token_policy == "auto"
    ):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D sin/cos."
        self.patch = int(patch)
        self.stride = int(patch) if stride is None else int(stride)
        self.num_classes = int(num_classes)
        self.max_tokens = int(max_tokens)
        self.use_decoder = bool(use_decoder)
        self.token_policy = token_policy
        self.min_keep_ratio = float(min_keep_ratio)

        token_dim = in_channels * self.patch * self.patch  # (C * P * P)

        # --- token embedding path (pre-norm -> proj -> post-norm -> dropout) ---
        self.token_ln   = nn.LayerNorm(token_dim)
        self.embed      = nn.Linear(token_dim, d_model)
        self.embed_ln   = nn.LayerNorm(d_model)
        self.embed_drop = nn.Dropout(dropout)

        # --- Transformer encoder (pre-LN is easier to optimise) ---
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True, norm_first=True, dropout=dropout
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.out_ln = nn.LayerNorm(d_model)

        # --- Heads ---
        if self.use_decoder:
            # Path A: tiny decoder (token features -> feature map -> upsample -> logits)
            self.up       = nn.ConvTranspose2d(d_model, d_model, kernel_size=self.patch, stride=self.stride)
            self.out_conv = nn.Conv2d(d_model, self.num_classes, kernel_size=1)
        else:
            # Path B: linear to P*P*K then unpatchify
            self.head     = nn.Linear(d_model, self.patch * self.patch * self.num_classes)

    # -------------------- helpers --------------------
    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, int, int]:
        """
        Pad BCHW to next multiple of 'multiple' on both H and W (pad bottom/right).
        Returns: (x_pad, H_orig, W_orig)
        """
        B, C, H, W = x.shape
        Hm = ((H + multiple - 1) // multiple) * multiple
        Wm = ((W + multiple - 1) // multiple) * multiple
        if Hm == H and Wm == W:
            return x, H, W
        pad = (0, Wm - W, 0, Hm - H)  # (left, right, top, bottom) for 2D pad on BCHW
        return F.pad(x, pad), H, W

    def _pad_features_and_mask(
        self, x: torch.Tensor, m: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """Pad features (B,C,N,N) and mask (B,N,N) to a multiple of 'patch' (kernel size)."""
        x_pad, H_orig, W_orig = self._pad_to_multiple(x, self.patch)
        _, _, Hp, Wp = x_pad.shape
        if m.size(1) != Hp or m.size(2) != Wp:
            dh, dw = Hp - m.size(1), Wp - m.size(2)
            pad = (0, dw, 0, dh)
            m_pad = F.pad(m.float(), pad).bool()
        else:
            m_pad = m
        assert Hp == Wp, "This model expects square inputs after padding."
        return x_pad, m_pad, H_orig, W_orig

    def _patchify(self, x: torch.Tensor, mask: torch.Tensor):
        """
        x:    (B, C, H_pad, W_pad)
        mask: (B, H_pad, W_pad)
        Returns:
          tokens:   (B, L, C*P*P)
          tok_keep: (B, L) boolean
          SS:       (Sr, Sc) token grid size
        """
        B, C, _, _ = x.shape
        P, S = self.patch, self.stride

        # features → (B, C, Sr, Sc, P, P)
        xf = x.unfold(2, P, S).unfold(3, P, S)              # (B, C, Sr, Sc, P, P)
        Sr, Sc = xf.size(2), xf.size(3)

        # (B, C, Sr, Sc, P, P) -> (B, Sr, Sc, C, P, P) -> (B, L, C*P*P)
        xf = xf.permute(0, 2, 3, 1, 4, 5).contiguous()
        tokens = xf.reshape(B, -1, C * P * P)               # L inferred

        # mask → (B, Sr, Sc, P, P) -> reduce over P dims -> (B, Sr, Sc) -> (B, L)
        mf = mask.unfold(1, P, S).unfold(2, P, S)           # (B, Sr, Sc, P, P)
        tok_keep_grid = mf.any(dim=-1).any(dim=-1)          # (B, Sr, Sc)
        tok_keep = tok_keep_grid.reshape(B, -1)             # (B, L)

        return tokens, tok_keep, (Sr, Sc)

    def _unpatchify_linear(self, tlogits: torch.Tensor, SS: Tuple[int, int], N_orig: int,
                           tok_keep: torch.Tensor) -> torch.Tensor:
        """
        Linear head path: tokens -> P*P*K -> image logits.
        Reshape when stride==patch; fold+normalisation when stride<patch.
        """
        B, L, D = tlogits.shape
        P, S, K = self.patch, self.stride, self.num_classes
        Sr, Sc = SS

        if S == P:
            if K == 1:
                assert D == P * P, "Head must output P*P when num_classes==1."
                x = tlogits.view(B, Sr, Sc, P, P).permute(0, 1, 3, 2, 4).contiguous()
                x = x.view(B, Sr * P, Sc * P)  # (B, H_pad, W_pad)
                return x[..., :N_orig, :N_orig]
            else:
                assert D == P * P * K, "Head must output P*P*K when num_classes>1."
                x = tlogits.view(B, Sr, Sc, P, P, K).permute(0, 5, 1, 3, 2, 4).contiguous()
                x = x.view(B, K, Sr * P, Sc * P)  # (B, K, H_pad, W_pad)
                return x[..., :N_orig, :N_orig]

        # general overlap path: fold + normalisation
        H_pad = (Sr - 1) * S + P
        W_pad = (Sc - 1) * S + P

        # Cast the boolean keep_mask to floats to count the valid overlapping patches
        keep_mask = tok_keep.to(tlogits.dtype)  # (B, L)

        p_mask = keep_mask.unsqueeze(1).expand(B, P * P, L)
        norm = F.fold(p_mask, output_size=(H_pad, W_pad), kernel_size=P, stride=S)

        if K == 1:
            assert D == P * P, "Head must output P*P when num_classes==1."
            patches = tlogits.permute(0, 2, 1).contiguous()  # (B, P*P, L) => C=1 inside fold

            # Scrub the linear bias out of dropped tokens before folding
            patches = patches.masked_fill(~tok_keep.unsqueeze(1), 0.0)

            out = F.fold(patches, output_size=(H_pad, W_pad), kernel_size=P, stride=S)  # (B, 1, H, W)
            out = (out / norm.clamp_min(1e-9)).squeeze(1)  # (B, H, W)
            return out[..., :N_orig, :N_orig]
        else:
            assert D == P * P * K, "Head must output P*P*K when num_classes>1."
            patches = tlogits.view(B, L, P * P, K).permute(0, 3, 2, 1).contiguous()  # (B, K, P*P, L)

            # Scrub the linear bias out of dropped tokens before folding using native broadcasting
            patches = patches.masked_fill(~tok_keep.view(B, 1, 1, L), 0.0)

            patches = patches.view(B * K, P * P, L)
            out = F.fold(patches, output_size=(H_pad, W_pad), kernel_size=P, stride=S).view(B, K, H_pad, W_pad)

            out = out / norm.clamp_min(1e-9)
            return out[..., :N_orig, :N_orig]

    @staticmethod
    def _pos_2d_sincos(Sr: int, Sc: int, d_model: int, device) -> torch.Tensor:
        """Return (1, Sr*Sc, d_model) 2D sin/cos positional encoding."""
        assert d_model % 4 == 0
        d = d_model // 2

        def pe_1d(L: int, dim: int) -> torch.Tensor:
            pos = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(1)
            i = torch.arange(dim // 2, device=device, dtype=torch.float32).unsqueeze(0)
            denom = torch.exp(-math.log(10000.0) * (2 * i) / dim)
            pe = torch.zeros(L, dim, device=device)
            pe[:, 0::2] = torch.sin(pos * denom)
            pe[:, 1::2] = torch.cos(pos * denom)
            return pe

        pe_r = pe_1d(Sr, d)
        pe_c = pe_1d(Sc, d)
        pe2d = torch.cat(
            (pe_r.unsqueeze(1).expand(-1, Sc, -1),
             pe_c.unsqueeze(0).expand(Sr, -1, -1)),
            dim=-1
        ).reshape(1, Sr * Sc, d_model)
        return pe2d

    # -------------------- forward --------------------
    def forward(self, x_bchw_std: torch.Tensor, _task_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x_bchw_std : (B, C, N, N) standardised BCHW
        _task_mask : (B, N, N) bool/float — used only to decide which tiles are kept as tokens
        """
        assert x_bchw_std.dim() == 4 and x_bchw_std.size(-2) == x_bchw_std.size(-1), "Input must be (B,C,N,N)."
        B, _, N, _ = x_bchw_std.shape
        device = x_bchw_std.device

        # ---- Build tile-validity image m according to policy (ONLY affects which tiles become tokens) ----
        if self.token_policy == "keep_all":
            m = torch.ones((B, N, N), dtype=torch.bool, device=device)
        elif self.token_policy == "auto":
            base = (_task_mask.to(device=device, dtype=torch.bool)
                    if _task_mask is not None else torch.zeros(B, N, N, dtype=torch.bool, device=device))
            keep_ratio_per_graph = base.float().mean(dim=(1, 2))
            meets_threshold = (keep_ratio_per_graph > self.min_keep_ratio).view(B, 1, 1)
            m = torch.where(meets_threshold, base, torch.ones((B, N, N), dtype=torch.bool, device=device))
        else:  # "from_mask"
            if _task_mask is not None and _task_mask.any():
                m = _task_mask.to(device=device, dtype=torch.bool)
            else:
                # Historical safety fallback: if mask is absent or empty, keep all.
                m = torch.ones((B, N, N), dtype=torch.bool, device=device)

        # Pad features + mask to multiples of self.patch (kernel). (Stride may be <= self.patch; unfold handles it.)
        x_pad, m_pad, H_orig, W_orig = self._pad_features_and_mask(x_bchw_std, m)
        assert H_orig == W_orig, "This model expects square inputs (N x N)."
        N_orig = H_orig

        # Patchify
        tokens, tok_keep, SS = self._patchify(x_pad, m_pad)   # tokens: (B, L, C*P*P)
        Sr, Sc = SS
        L = tokens.size(1)

        if L > self.max_tokens:
            warnings.warn(f"[PatchTransformer] L={L} exceeds max_tokens={self.max_tokens}.", stacklevel=2)

        # Token embed
        tokens = self.token_ln(tokens)
        emb = self.embed(tokens)
        emb = self.embed_drop(self.embed_ln(emb))   # (B, L, d_model)

        # Positional encodings for exact token grid
        pos = self._pos_2d_sincos(Sr, Sc, emb.size(-1), emb.device)  # (1, L, d_model)

        # Key padding mask (True=ignore)
        key_padding_mask = (~tok_keep).to(emb.device, non_blocking=True)

        # Encode
        h = self.enc(emb + pos, src_key_padding_mask=key_padding_mask)  # (B, L, d_model)
        h = self.out_ln(h)

        # Zero out ignored tokens, so they don't smear garbage logits during overlapping folds/upsamples
        h = h.masked_fill(~tok_keep.unsqueeze(-1), 0.0)

        # Decode
        if self.use_decoder:
            # (B, L, D) -> (B, D, Sr, Sc) -> upsample -> logits
            Dm = h.size(-1)
            feat = h.view(B, Sr, Sc, Dm).permute(0, 3, 1, 2).contiguous()  # (B, d_model, Sr, Sc)
            up = self.up(feat)  # (B, d_model, H_pad, W_pad)
            logits_full = self.out_conv(up)  # (B, K, H_pad, W_pad)
            logits = logits_full[..., :N_orig, :N_orig]  # crop to (N,N)
            if self.num_classes == 1:
                logits = logits.squeeze(1)  # (B, N, N)
        else:
            # Linear head + unpatchify
            tlogits = self.head(h)  # (B, L, P*P*K)
            logits = self._unpatchify_linear(tlogits, (Sr, Sc), N_orig, tok_keep)  # Pass tok_keep to scale overlap

        return logits


# ============================================================
# 5) Task-agnostic Trainer/Evaluator
# ============================================================
@dataclass
class TNNTrainConfig:
    """
    Configuration parameters for the edge classification training pipeline.

    Selected attributes:
        use_mask_channel (bool, optional): Whether to append the mask as an input channel.
        lr (float): Learning rate for optimisation.
        rf_neg_pos_ratio (float): Target ratio of negative to positive samples for the
            Random Forest path. This controls hard negative downsampling when collecting
            the capped RF training table. Dense models (MLPs, CNN, Transformer) ignore
            this parameter and process the full highly-skewed dataset, achieving balance
            mathematically via dynamic `pos_weight` in their BCE loss.
    """
    use_mask_channel: Optional[bool] = None  # None -> infer per model (CNN/Tx=True, MLP/RF=False)

    # optimisation
    lr: float = 3e-4
    weight_decay: float = 1e-2
    epochs: int = 10
    batch_size: int = 16
    grad_clip: float = 0.0  # 0.0 disables clipping; set to e.g. 1.0 to enable
    early_stop_patience: int = 0  # 0 => disabled

    # CNN specifics
    cnn_hidden: int = 64
    cnn_head_kernel: int = 3  # Perfectly preserves (N, N) spatial dimensions via padding, allowed: {1,3,5}
    cnn_dropout: float = 0.10

    # What to zero at the supervised (i,j) pixel when building BCHW inputs:
    #   "all"      → zero every channel at (i,j)
    #   "adj_only" → zero only the 'adj' channel at (i,j); non-adj features remain visible
    #   "none"     → no redaction; all channels remain visible at (i,j)
    supervised_redaction_policy: Literal["all", "adj_only", "none"] = "adj_only"

    # MLPs
    mlp_hidden: int = 128
    mlp_dropout: float = 0.10
    deep_mlp_hidden: int = 256
    deep_mlp_layers: int = 4
    deep_mlp_dropout: float = 0.3

    # Transformer
    tx_dmodel: int = 256
    tx_layers: int = 4
    tx_heads: int = 4
    tx_dropout: float = 0.2
    tx_token_budget: int = 1024  # Transformer token cap (≈ ceil(N/P)^2 )
    tx_token_policy: str = "keep_all"
    tx_min_keep_ratio: float = 0.0   # used only when tx_token_policy == "auto"
    tx_use_decoder: bool = True
    tx_force_adj_channel: bool = True  # TX keeps 'adj' unless you turn this off
    tx_patch_overlap: bool = False
    tx_scheduler: str = "none"  # ["none", "cosine", "step"]

    # RandomForest-specific
    rf_neg_pos_ratio: float = 4.0

    # saving
    save_dir: Optional[str] = "saved_checkpoints"  # None to skip saving

    # numerical stability
    display_decimals: int = 4          # number of decimals when printing metrics
    display_truncate: bool = False     # if True, truncate instead of round to avoid printing 1.0000 for 0.99995

    # How to choose the probability threshold for binary tasks:
    #   "f1"   → maximise F1 of the positive class (default)
    #   "bacc" → maximise balanced accuracy (TPR+TNR)/2
    threshold_metric: Literal["f1", "bacc"] = "f1"

    # Which validation metric selects the best epoch/checkpoint (and early stopping):
    #   "f1"    → use F1 (default)
    #   "bacc"  → use balanced accuracy
    #   "auroc" → use AUROC (falls back to F1 if AUROC is undefined)
    select_by: Literal["f1", "bacc", "auroc"] = "f1"


def _format_duration(seconds: float) -> str:
    """Render an elapsed duration as '1d 13h 47m 12s', adding ms under two minutes."""
    total_ms = int(round(float(seconds) * 1000.0))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    d, rem = divmod(total_s, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)

    parts = []
    if d:
        parts.append(f"{d}d")
    if d or h:
        parts.append(f"{h}h")
    if d or h or m:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    if total_s < 120:
        parts.append(f"{ms}ms")
    return " ".join(parts)


def _cuda_gc(tag: str = ""):
    """
    Run Python GC + CUDA cache cleanup.
    If `tag` is provided, print before/after GPU memory stats (GiB).
    Safe on CPU-only machines (no output).
    """
    have_cuda = torch.cuda.is_available()
    if have_cuda:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        if tag:
            alloc_before = torch.cuda.memory_allocated()
            reserv_before = torch.cuda.memory_reserved()

        # cleanup
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

        if tag:
            gib = 1024 ** 3
            alloc_after = torch.cuda.memory_allocated()
            reserv_after = torch.cuda.memory_reserved()
            freed_res = max(0, reserv_before - reserv_after) / gib
            freed_all = max(0, alloc_before - alloc_after) / gib
            print(
                f"[CUDA.GC] {tag}: freed {freed_res:.3f} GiB reserved "
                f"({reserv_before/gib:.3f}→{reserv_after/gib:.3f}), "
                f"{freed_all:.3f} GiB allocated "
                f"({alloc_before/gib:.3f}→{alloc_after/gib:.3f})"
            )


def get_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1", fallback_thr: float = 0.5) -> float:
    """
    Computes the optimal binary classification threshold based on the selected metric.
    
    If the provided data split is degenerate (e.g., contains only positive edges or only 
    negative edges), evaluating PR/ROC curves is mathematically undefined. In these cases, 
    the function catches the invalid state and utilizes state momentum by returning the 
    `fallback_thr` (usually the last known good threshold from a previous epoch).
    """
    if y_prob.size == 0 or len(np.unique(y_true)) < 2: 
        return fallback_thr
    
    metric = (metric or "f1").lower()
    
    if metric == "bacc":
        fpr, tpr, ts = roc_curve(y_true, y_prob)
        if len(ts) < 2:
            return fallback_thr
        
        # Search the real thresholds rather than picking ts[1] when the sentinel wins
        baccs = 0.5 * (tpr + (1.0 - fpr))
        best_i = 1 + int(np.argmax(baccs[1:]))
        return float(ts[best_i])
    
    ps, rs, ts = precision_recall_curve(y_true, y_prob)
    f1s = 2.0 * ps * rs / np.maximum(ps + rs, 1e-12)
    best_i = np.argmax(f1s)
    
    if ts.size == 0: 
        return fallback_thr
    
    return float(ts[min(best_i, len(ts) - 1)])


def forward_logits_common(
    A, feats, mask, *,
    registry, feature_keys, keep_idx,
    model_key, model, device
):
    """
    Build BCHW inputs from (A, feats, mask), standardise with registry stats,
    and run the selected head.

    Redaction is applied during `registry.stack_channels_BCHW`, before
    standardisation, according to `registry.supervised_redaction_policy`
    ("all" | "adj_only" | "none").

    Returns:
        logits: (B, N, N) for binary heads, or (B, K, N, N) for multiclass heads.
    """
    # ------------------------------
    # 1) Move adjacency/mask to device (feats moved inside stacker as needed)
    # ------------------------------
    if not torch.is_tensor(A):
        raise TypeError("A must be a Tensor of shape (B,N,N)")
    A = A.to(device, non_blocking=True)
    m_in = None
    if mask is not None:
        if not torch.is_tensor(mask):
            raise TypeError("mask must be a Tensor or None")
        m_in = mask.to(device, non_blocking=True)

    # ------------------------------
    # 2) Assemble channels
    # ------------------------------
    # x_bchw: (B, C_all, N, N)
    x_bchw = registry.stack_channels_BCHW(
        A, feats, m_in, feature_keys,
        include_adj=("adj" in getattr(registry, "manifest", []))
    )

    if keep_idx is None or len(keep_idx) == 0:
        raise ValueError("keep_idx is empty; no input channels to feed the model.")
        
    # ------------------------------
    # 3) Standardise the manifest channels
    # ------------------------------
    x_bchw_std = registry.standardise_bchw(x_bchw)

    # ------------------------------
    # 4) Forward through the selected head - TX can optionally take a task mask
    # ------------------------------
    if model_key == "transformer":
        logits = model(x_bchw_std, _task_mask=m_in)
    else:
        logits = model(x_bchw_std)

    return logits


def train_and_eval_one_model(task, registry, loaders, model_key: str, cfg, runtime):
    """
    Train a single model (mlp | deep_mlp | cnn | transformer) and evaluate on Val/Test.

    This version:
      - Never includes 'adj' in feature_keys (it's handled by the registry/manifest).
      - Builds keep_idx from registry.manifest and includes exactly one 'adj' for TX (or if allow_adj_channel=True).
      - Uses the same stack -> standardise manifest-channel contract in both train and eval.
      - Leaves your optimiser/scheduler structure unchanged (uses existing 'criterion', 'optimiser', 'scaler' names).
    """
    # ------------------------------------------------------------
    # 0) Unpack loaders & basics
    # ------------------------------------------------------------
    _t_model_start = time.monotonic()
    train_loader, val_loader, test_loader = loaders
    num_classes = int(getattr(task, "num_classes", 1))
    epochs = int(getattr(cfg, "epochs", 80))
    lr = float(getattr(cfg, "lr", 1e-3))
    weight_decay = float(getattr(cfg, "weight_decay", 0.0))
    max_grad_norm = float(getattr(cfg, "grad_clip", 0.0))
    edges_only = bool(getattr(task, "eval_on_existing_edges_only", False))

    disp_dec = int(getattr(cfg, "display_decimals", 4))
    disp_trunc = bool(getattr(cfg, "display_truncate", False))

    def _fmt(v):
        if disp_trunc and not math.isnan(v):
            p = 10 ** disp_dec
            v = int(v * p) / p
        return f"{v:.{disp_dec}f}"

    def _get(name, default):
        return getattr(cfg, name, default)

    # ------------------------------------------------------------
    # 1) Resolve feature_keys for this task and fit registry fresh
    #    (Do not include 'adj' here - registry handles it via manifest)
    # ------------------------------------------------------------
    feature_keys = list(getattr(runtime, "feature_keys", []))

    # Decide mask-channel per model (or honor cfg override)
    if getattr(cfg, "use_mask_channel", None) is None:
        registry.use_mask_channel = (model_key in ("cnn", "transformer"))
        print(f"[{model_key.upper()}] mask channel default → {registry.use_mask_channel}")
    else:
        registry.use_mask_channel = bool(cfg.use_mask_channel)
        print(f"[{model_key.upper()}] mask channel override from cfg → {registry.use_mask_channel}")

    allow_adj = bool(getattr(getattr(task, "hooks", object()), "allow_adj_channel", False))
    include_adj = allow_adj or (model_key == "transformer" and bool(getattr(cfg, "tx_force_adj_channel", True)))
    registry.fit(train_loader, feature_keys, include_adj=include_adj)
    manifest_names = list(getattr(registry, "manifest", feature_keys))
    print("registry.manifest:", manifest_names)

    # ------------------------------------------------------------
    # 2) Keep indices by NAME (gate 'adj' by task/TX override; dedup 'adj')
    # -----------------------------------------------------------------------------
    # There are THREE distinct “masks”/concepts:
    #   (A) Supervision/Evaluation mask  → built by effective_mask(...). Always used.
    #       - Controls which (i,j) pairs contribute to loss/metrics.
    #       - Not an input feature. Independent of everything below.
    #
    #   (B) Transformer token-keep mask  → passed to TX forward(..., _task_mask=...).
    #       - Decides which patches become tokens for token_policy in {"from_mask","auto"}.
    #       - Also not an input feature. Independent of (C).
    #
    #   (C) *Mask input channel*         → a BCHW feature channel named "mask".
    #       - Controlled by cfg.use_mask_channel (None ⇒ infer per model).
    #       - If ON, we append the 0/1 mask matrix as a feature channel to the input tensor.
    #
    # Adjacency input channel (“adj”):
    #   - Task gate:    hooks.allow_adj_channel  (applies to ALL models when True)
    #   - TX override:  cfg.tx_force_adj_channel (applies only to Transformer)
    #   - Inclusion rule (per model):
    #       include_adj = hooks.allow_adj_channel or (is_transformer and cfg.tx_force_adj_channel)
    #
    # Transformer truth table (for adj channel):
    #   allow_adj_channel | tx_force_adj_channel | include 'adj'?
    #   ------------------+----------------------+---------------
    #        False        |        False         |      NO
    #        False        |        True          |      YES  (forced)
    #        True         |        False         |      YES  (task allows)
    #        True         |        True          |      YES
    #
    # Mask input channel (“mask”):
    #   - Controlled by cfg.use_mask_channel
    #   - If cfg.use_mask_channel is None, we infer: CNN/Transformer → True, MLP/RF → False.
    #   - This is independent from the adj channel and from (A)/(B).
    # -----------------------------------------------------------------------------
    # Build the final ordered list of channel names to keep, by name
    names: List[str] = []
    if include_adj and ("adj" in manifest_names):
        names.append("adj")
    if getattr(registry, "use_mask_channel", False) and ("mask" in manifest_names):
        names.append("mask")

    # Add all other channels except duplicates of 'adj'/'mask'
    for n in manifest_names:
        if n in ("adj", "mask"):
            continue
        names.append(n)

    # Map back to manifest indices
    keep_idx = [manifest_names.index(n) for n in names if n in manifest_names]
    eff_in_ch = len(keep_idx)
    if eff_in_ch == 0:
        raise ValueError("Effective in_channels is 0. Enable allow_adj_channel or tx_force_adj_channel.")

    if model_key in ("transformer", "cnn"):
        print(f"[{model_key.upper()}] using channels:", names, " (dedup adj)" if ("adj" in names) else "")

    # expose dense runtime state for eval / checkpoint metadata
    runtime.keep_idx = keep_idx
    runtime.eff_in_ch = eff_in_ch

    # ------------------------------------------------------------
    # 3) Build model with effective in_channels (your existing recipes)
    # ------------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_key == "mlp":
        model = MatrixFeatureMLP(
            in_channels=eff_in_ch,
            hidden_dim=_get("mlp_hidden", 256),
            dropout=_get("mlp_dropout", 0.10),
            num_classes=num_classes,
        ).to(DEVICE)

    elif model_key == "deep_mlp":
        model = MatrixFeatureDeepMLP(
            in_channels=eff_in_ch,
            hidden_dim=_get("deep_mlp_hidden", 256),
            depth=_get("deep_mlp_layers", 4),
            dropout=_get("deep_mlp_dropout", 0.10),
            num_classes=num_classes,
        ).to(DEVICE)

    elif model_key == "cnn":
        head_k = int(cfg.cnn_head_kernel)
        if head_k not in (1, 3, 5):
            head_k = 3
        model = MatrixFeatureCNN(
            in_channels=eff_in_ch,
            hidden=int(cfg.cnn_hidden),
            head_kernel=head_k,
            num_classes=num_classes,
            dropout=float(getattr(cfg, "cnn_dropout", 0.0)),
        ).to(DEVICE)

    elif model_key == "transformer":
        # Size TX from the observed graph size
        probe_loader = _pick_probe_loader(train_loader, val_loader, test_loader)
        A_probe, _, _, _ = next(iter(probe_loader))
        N_ref = int(A_probe.shape[-1])
        if getattr(task, "_active_run_dataset", None) is not None and getattr(task, "max_nodes", None) is not None:
            N_ref = max(N_ref, int(task.max_nodes))

        budget = _get("tx_token_budget", 1024)
        use_dec = _get("tx_use_decoder", True)
        d_model = _get("tx_dmodel", 384)
        n_layers = _get("tx_layers", 6)
        n_heads = _get("tx_heads", 6)
        dropout = _get("tx_dropout", 0.10)
        patch_overlap = bool(_get("tx_patch_overlap", False))
        token_pol = _get("tx_token_policy", "keep_all")
        min_keep = _get("tx_min_keep_ratio", 0.0)
        CANDIDATES = (2, 4, 8, 16, 24, 32, 48, 64)

        def tokens_for(N: int, P: int, S: int) -> int:
            H_pad = math.ceil(N / P) * P
            Sr = (H_pad - P) // S + 1
            return Sr * Sr

        # Choose the smallest patch whose configured stride policy fits the sizing budget
        chosen = None
        for P in CANDIDATES:
            S = max(1, P // 2) if patch_overlap else P
            L = tokens_for(N_ref, P, S)
            if L <= budget:
                chosen = (P, S, L)
                break

        # Fallback: use the largest patch with the configured stride policy
        if chosen is None:
            P_max = max(CANDIDATES)
            S_max = max(1, P_max // 2) if patch_overlap else P_max
            chosen = (P_max, S_max, tokens_for(N_ref, P_max, S_max))

        P_adapt, S_effective, L_est = chosen
        model = PatchTransformer(
            in_channels=eff_in_ch,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            patch=P_adapt,
            stride=None if S_effective == P_adapt else S_effective,
            dropout=dropout,
            num_classes=num_classes,
            max_tokens=budget,
            use_decoder=use_dec,
            token_policy=token_pol,
            min_keep_ratio=min_keep
        ).to(DEVICE)

        print(f"[TX] N≈{N_ref}, patch={P_adapt}, stride={S_effective}, tokens≈{L_est}, "
              f"decoder={use_dec}, policy={token_pol}")

    # ------------------------------------------------------------
    # 4) Criterion / Optimizer / Scaler
    # ------------------------------------------------------------
    criterion = None if num_classes == 1 else torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available()) # type: ignore[attr-defined]

    scheduler = None
    if model_key == "transformer":
        sched_name = str(getattr(cfg, "tx_scheduler", "none")).lower()
        if sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif sched_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)

    # ------------------------------------------------------------
    # 5) Train
    # ------------------------------------------------------------
    best_state = None
    # selection metric (default = F1; optional "auroc", "bacc")
    select_by = str(getattr(cfg, "select_by", "f1")).lower()
    if select_by == "auroc":
        _select_key = "AUROC"
    elif select_by == "bacc":
        _select_key = "BAcc"
    else:
        _select_key = "F1"

    best_val_sel = float("-inf")
    no_improve = 0
    patience = int(getattr(cfg, "early_stop_patience", 0))  # 0 -> no early stop
    last_known_thr = 0.5
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, denom_sum = 0.0, 0
        for A, feats, L, mask in train_loader:
            optimizer.zero_grad(set_to_none=True)

            # Build an effective train mask on the CPU, then select supervised indices
            m_cpu = effective_mask(mask, A, registry.directed)

            # Train only on observed edges if task asks for it
            if edges_only:
                m_cpu = m_cpu & (A > 0.5)

            if not m_cpu.any():
                continue

            idx_cpu = torch.nonzero(m_cpu, as_tuple=False)  # (E, 3): [b, i, j]

            with torch.amp.autocast(
                    "cuda",
                    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    enabled=torch.cuda.is_available()
            ):
                if model_key in ("mlp", "deep_mlp"):
                    A_dev = A.to(DEVICE, non_blocking=True)
                    m_dev = mask.to(DEVICE, non_blocking=True)
                    x_bchw = registry.stack_channels_BCHW(
                        A_dev, feats, m_dev, feature_keys,
                        include_adj=("adj" in getattr(registry, "manifest", []))
                    )
                    x_bchw_std = registry.standardise_bchw(x_bchw)

                    b_dev = idx_cpu[:, 0].to(DEVICE, non_blocking=True)
                    i_dev = idx_cpu[:, 1].to(DEVICE, non_blocking=True)
                    j_dev = idx_cpu[:, 2].to(DEVICE, non_blocking=True)
                    z_gathered = model._forward_flat(x_bchw_std[b_dev, :, i_dev, j_dev])
                    logits = None
                else:
                    z_gathered = None
                    logits = forward_logits_common(
                        A, feats, mask,
                        registry=registry,
                        feature_keys=feature_keys,
                        keep_idx=keep_idx,
                        model_key=model_key,
                        model=model,
                        device=DEVICE
                    )

            dev = DEVICE if logits is None else logits.device
            idx = idx_cpu.to(dev, non_blocking=True)

            # Binary path: BCE-with-logits (fp32), true mean per edge
            if num_classes == 1:
                if z_gathered is not None:
                    z_sel = z_gathered.squeeze(-1).to(torch.float32)
                else:
                    # Normalise logits to (B, H, W)
                    if logits.dim() == 4 and logits.size(1) == 1:
                        z = logits[:, 0, :, :]
                    else:
                        z = logits
                    z_sel = z[idx[:, 0], idx[:, 1], idx[:, 2]].to(torch.float32)

                y_sel = (L[idx_cpu[:, 0], idx_cpu[:, 1], idx_cpu[:, 2]] > 0.5) \
                    .to(dev, non_blocking=True).to(torch.float32)

                # Stable pos_weight computed from the batch
                pw = _pos_weight(y_sel)

                # Keep logits finite and bounded for stable BCE math
                z_sel = torch.nan_to_num(z_sel, nan=0.0, posinf=0.0, neginf=0.0).clamp(-50, 50)

                # Per-edge reduction=none → mean—this is the true per-edge training loss
                loss_elems = torch.nn.functional.binary_cross_entropy_with_logits(
                    z_sel, y_sel, pos_weight=pw, reduction="none"
                )
                loss = loss_elems.mean()
                denom = int(z_sel.numel())

            # Multi-class path: CE (fp32) over supervised positions selected by effective_mask
            else:
                if z_gathered is not None:
                    z_sel = z_gathered.to(torch.float32)
                else:
                    # Normalise logits to (B, H, W, K)
                    if logits.dim() == 4 and logits.size(1) == num_classes:
                        z = logits.permute(0, 2, 3, 1).contiguous()
                    else:
                        z = logits
                    z_sel = z[idx[:, 0], idx[:, 1], idx[:, 2], :].to(torch.float32)

                y_sel = L[idx_cpu[:, 0], idx_cpu[:, 1], idx_cpu[:, 2]] \
                    .to(dev, non_blocking=True).long()

                loss = criterion(z_sel, y_sel)
                denom = int(y_sel.numel())

            if denom == 0:
                continue

            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            loss_sum += float(loss.detach().to("cpu")) * max(1, denom)
            denom_sum += max(1, denom)

        avg = (loss_sum / denom_sum) if denom_sum else float("nan")

        # Per-epoch Val probe (now capture + print macro-F1)
        val_m = _eval_split(model_key, model, val_loader, registry, edges_only, cfg, feature_keys, keep_idx,
                            num_classes, fallback_thr=last_known_thr, criterion=criterion)
        last_known_thr = val_m.get("thr", last_known_thr)
        val_f1 = float(val_m.get("F1_macro", val_m.get("f1", float("nan"))))
        label = "F1_macro" if "F1_macro" in val_m else "f1"
        print(f"[{model_key.upper()}] Epoch {epoch:02d} — Train loss/edge: {_fmt(avg)} | Val {label}: {_fmt(val_f1)}")

        if scheduler is not None:
            scheduler.step()

        # choose selection metric (default F1; optional BAcc and AUROC with fallback)
        if _select_key == "AUROC":
            sel_val = float(val_m.get("auroc", val_m.get("AUC_macro", float("nan"))))
            if not (sel_val == sel_val):  # NaN guard
                sel_val = float(val_f1)
        elif _select_key == "BAcc":
            sel_val = float(val_m.get("bacc", val_m.get("BAcc_macro", float("nan"))))
            if not (sel_val == sel_val):  # NaN guard
                sel_val = float(val_f1)
        else:
            sel_val = float(val_f1)

        if math.isfinite(sel_val) and sel_val > best_val_sel:
            best_val_sel = sel_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        elif not math.isfinite(sel_val):
            # Empty / no-eval validation split: keep NaN explicit and do not count it as "no improvement"
            pass
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(
                    f"[{model_key.upper()}] Early stop @ epoch {epoch} (best Val {_select_key}={best_val_sel:.4f})")
                break

    # ------------------------------------------------------------
    # 6) Final Val/Test, pick a threshold if binary
    # ------------------------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    val_m = _eval_split(model_key, model, val_loader, registry, edges_only, cfg, feature_keys, keep_idx,
                        num_classes, fixed_threshold=None, fallback_thr=last_known_thr, criterion=criterion)

    thr = val_m.get("thr", 0.5) if num_classes == 1 else None
    test_m = _eval_split(
        model_key, model, test_loader, registry,
        edges_only, cfg, feature_keys, keep_idx, num_classes,
        fixed_threshold=thr, criterion=criterion
    )

    # ---------------------
    # Save one checkpoint per (task, timestamp, model) under saved_checkpoints/<task>/<timestamp>/<model_key>.pth
    # prefer the best validation weights if you tracked them; otherwise current
    # ---------------------
    meta = _build_meta(
        model_key=model_key,
        task=task,
        registry=registry,
        cfg=cfg,
        runtime=runtime,
        best_thr=thr
    )
    meta["elapsed_seconds"] = round(time.monotonic() - _t_model_start, 3)

    state_to_save = best_state if best_state is not None else {k: v.cpu() for k, v in model.state_dict().items()}
    ckpt_path = save_pipeline_checkpoint(model_key, state_to_save, task, cfg, meta)

    return val_m, test_m, thr, ckpt_path, meta["elapsed_seconds"]


# ---------------------------------------------------------------------
# Hooks + feature utilities (single source of truth, no globals needed)
# ---------------------------------------------------------------------
def _hooks_get(hooks, key, default=None):
    """Safe access for TaskHooks (object) and dict hooks."""
    if hooks is None:
        return default
    if isinstance(hooks, dict):
        return hooks.get(key, default)
    return getattr(hooks, key, default)


@torch.no_grad()
def _eval_split(model_key, model, loader, registry, edges_only, cfg, feature_keys, keep_idx,
                num_classes, fixed_threshold=None, fallback_thr=0.5, *, criterion=None):
    """
    Eval one split. Returns a dict with keys:
      - Binary:  Acc, P, R, F1, AUROC, AUPRC, loss/edge, thr (optional)
      - Multi:   Acc, P_macro, R_macro, F1_macro, AUC_macro, AUPRC_macro, loss/edge
    Shapes:
      logits: (B,N,N) or (B,1,N,N) for binary; (B,K,N,N) or (B,N,N,K) for multi.
    """
    model.eval()
    DEVICE = next(model.parameters()).device

    # Storage
    all_logits_sel = []
    all_y_sel = []
    denom_sum = 0 if num_classes > 1 else None
    loss_sum = 0.0 if num_classes > 1 else None

    for A, feats, L, mask in loader:
        # MLP fast path: gather supervised pairs directly, skip full-grid forward
        if model_key in ("mlp", "deep_mlp"):
            m_cpu = effective_mask(mask, A, registry.directed)
            if edges_only is True:
                m_cpu = m_cpu & (A > 0.5)
            if not m_cpu.any():
                continue

            idx_cpu = torch.nonzero(m_cpu, as_tuple=False)  # (E, 3): [b, i, j]
            b_idx, i_idx, j_idx = idx_cpu[:, 0], idx_cpu[:, 1], idx_cpu[:, 2]

            # Build and standardize BCHW on device — same ops as forward_logits_common
            # but without running the model on the full (B, N_max, N_max) grid
            A_dev = A.to(DEVICE, non_blocking=True)
            m_dev = mask.to(DEVICE, non_blocking=True)
            x_bchw = registry.stack_channels_BCHW(
                A_dev, feats, m_dev, feature_keys,
                include_adj=("adj" in getattr(registry, "manifest", []))
            )
            x_bchw_std = registry.standardise_bchw(x_bchw)

            # Index (B, C_eff, N, N) at supervised positions → (E, C_eff).
            # Mixed advanced+slice indexing: advanced dims 0/2/3 broadcast to (E,),
            # slice on dim 1 gives C_eff, result is (E, C_eff).
            b_dev = b_idx.to(DEVICE, non_blocking=True)
            i_dev = i_idx.to(DEVICE, non_blocking=True)
            j_dev = j_idx.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, enabled=torch.cuda.is_available()):
                x_gathered = x_bchw_std[b_dev, :, i_dev, j_dev]  # (E, C_eff)
                z_sel = model._forward_flat(x_gathered)           # (E, 1) or (E, K)

            if num_classes == 1:
                z_sel = z_sel.squeeze(-1)  # (E,)
                y_sel = (L[b_idx, i_idx, j_idx] > 0.5).to(torch.float32)
                all_logits_sel.append(z_sel.detach().to(torch.float32).cpu())
                all_y_sel.append(y_sel)
            else:
                y_sel_cpu = L[b_idx, i_idx, j_idx].to(torch.int64)
                y_sel = y_sel_cpu.to(DEVICE, non_blocking=True)
                loss = criterion(z_sel.float(), y_sel)
                loss_sum += float(loss.detach().to("cpu")) * int(y_sel.numel())
                denom_sum += int(y_sel.numel())
                all_logits_sel.append(torch.softmax(z_sel.float(), dim=-1).cpu())
                all_y_sel.append(y_sel_cpu)
            continue

        # --- Full-grid path for spatial models (cnn, transformer) ---
        with torch.amp.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                                enabled=torch.cuda.is_available()):
            logits = forward_logits_common(
                A, feats, mask,
                registry=registry,
                feature_keys=feature_keys,
                keep_idx=keep_idx,
                model_key=model_key,
                model=model,
                device=DEVICE,
            )

        # Build an effective mask on the CPU, then index
        m_cpu = effective_mask(mask, A, registry.directed)
        if edges_only is True:
            m_cpu = m_cpu & (A > 0.5)

        if not m_cpu.any():
            continue

        idx_cpu = torch.nonzero(m_cpu, as_tuple=False)  # (E,3): [b,i,j]
        dev = logits.device
        idx = idx_cpu.to(dev, non_blocking=True)
        if num_classes == 1:
            # Normalise logits to (B,H,W)
            if logits.dim() == 4 and logits.size(1) == 1:
                z = logits[:, 0, :, :]
            else:
                z = logits

            z_sel = z[idx[:, 0], idx[:, 1], idx[:, 2]]  # (E,)

            # Evaluate > 0.5 entirely on the CPU where L and idx_cpu already reside
            y_sel = (L[idx_cpu[:, 0], idx_cpu[:, 1], idx_cpu[:, 2]] > 0.5).to(torch.float32)

            all_logits_sel.append(z_sel.detach().to(torch.float32).cpu())
            all_y_sel.append(y_sel)
        else:
            # Normalise to (B,H,W,K)
            if logits.dim() == 4 and logits.size(1) == num_classes:
                z = logits.permute(0, 2, 3, 1).contiguous()
            else:
                z = logits

            z_sel = z[idx[:, 0], idx[:, 1], idx[:, 2], :]  # (E,K)
            y_sel_cpu = L[idx_cpu[:, 0], idx_cpu[:, 1], idx_cpu[:, 2]].to(torch.int64)
            y_sel = y_sel_cpu.to(dev, non_blocking=True)
            loss = criterion(z_sel.float(), y_sel)
            loss_sum += float(loss.detach().to("cpu")) * int(y_sel.numel())
            denom_sum += int(y_sel.numel())

            all_logits_sel.append(torch.softmax(z_sel.float(), dim=-1).cpu())
            all_y_sel.append(y_sel_cpu)

    # Aggregate
    if len(all_y_sel) == 0:
        # No supervision available on this split
        out = {"loss/edge": float("nan")}
        if num_classes == 1:
            out.update(dict(accuracy=float("nan"), precision=float("nan"), recall=float("nan"),
                            f1=float("nan"), auroc=float("nan"), auprc=float("nan"), bacc=float("nan")))
        else:
            out.update(dict(accuracy=float("nan"), P_macro=float("nan"), R_macro=float("nan"), F1_macro=float("nan"),
                             AUC_macro=float("nan"), AUPRC_macro=float("nan"), BAcc_macro=float("nan"), f1=float("nan")))

        return out

    y_all = torch.cat(all_y_sel, dim=0).numpy()
    if num_classes == 1:
        z_tensor = torch.cat(all_logits_sel, dim=0)
        y_tensor = torch.from_numpy(y_all).to(torch.float32)

        pw = _pos_weight(y_tensor)
        loss_avg = float(
            torch.nn.BCEWithLogitsLoss(
                pos_weight=pw
            )(z_tensor, y_tensor).item()
        )

        # 1) Stable sigmoid (avoid overflow)
        prob = torch.sigmoid(z_tensor).numpy()

        # 2) Auto-threshold:
        #    - On Val (fixed_threshold is None): extract exact probability splits via PR/ROC curve
        #      and break ties by picking the lowest valid threshold (favouring recall).
        #    - On Test: reuse the tuned threshold passed in
        if fixed_threshold is None:
            metric = str(getattr(cfg, "threshold_metric", "f1")).lower()
            best_thr = get_optimal_threshold(y_all, prob, metric, fallback_thr=fallback_thr)
        else:
            best_thr = float(fixed_threshold)

        # 3) Final predictions at chosen threshold
        pred = (prob >= best_thr)

        # 4) Basic metrics (safe with numpy)
        tp = np.logical_and(pred == 1, y_all == 1).sum()
        fp = np.logical_and(pred == 1, y_all == 0).sum()
        tn = np.logical_and(pred == 0, y_all == 0).sum()
        fn = np.logical_and(pred == 0, y_all == 1).sum()
        acc = (tp + tn) / max(1, tp + tn + fp + fn)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec / max(1e-12, (prec + rec))) if (prec + rec) > 0 else 0.0
        tnr = tn / max(1, tn + fp)  # specificity/TNR

        # BAcc averages TPR and TNR; one is undefined when a class is absent
        bacc = 0.5 * (rec + tnr) if ((y_all == 1).any() and (y_all == 0).any()) else float("nan")
        auroc = float("nan")
        auprc = float("nan")

        # Need both positives and negatives
        if (y_all == 1).any() and (y_all == 0).any():
            try:
                auroc = roc_auc_score(y_all, prob)
                auprc = average_precision_score(y_all, prob)
            except Exception:
                pass

        return {
            "loss/edge": float(loss_avg),
            "accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "auroc": auroc, "auprc": auprc,
            "thr": best_thr,
            "bacc": float(bacc),
            "_prob": prob.tolist(),  # per-edge probabilities in dataset order
            "_y": y_all.astype(int).tolist(),
        }

    else:
        # Prob and preds
        loss_avg = loss_sum / max(1, denom_sum)
        prob = torch.cat(all_logits_sel, dim=0).numpy()
        K = prob.shape[1]
        pred = prob.argmax(axis=1)

        # Metrics
        acc = float(accuracy_score(y_all, pred)) if y_all.size else float("nan")
        P_macro = float(precision_score(y_all, pred, average="macro", zero_division=0))
        R_macro = float(recall_score(y_all, pred, average="macro", zero_division=0))
        F1_macro = float(f1_score(y_all, pred, average="macro", zero_division=0))

        # Balanced accuracy = macro-average recall; guard degenerate splits
        if len(np.unique(y_all)) >= 2:
            BAcc_macro = float(balanced_accuracy_score(y_all, pred))
        else:
            BAcc_macro = float("nan")

        AUC_macro = float("nan")
        AUPRC_macro = float("nan")
        aucs, aprs = [], []
        for k in range(K):
            y_bin = (y_all == k).astype(np.int32)
            # Skip degenerate classes (no pos or no neg)
            if y_bin.sum() == 0 or (y_bin.size - y_bin.sum()) == 0:
                continue
            pk = prob[:, k]
            try:
                aucs.append(roc_auc_score(y_bin, pk))
                aprs.append(average_precision_score(y_bin, pk))
            except Exception:
                pass

        if len(aucs):
            AUC_macro = float(np.mean(aucs))

        if len(aprs):
            AUPRC_macro = float(np.mean(aprs))

        return {
            "loss/edge": float(loss_avg),
            "accuracy": float(acc),
            "P_macro": P_macro, "R_macro": R_macro, "F1_macro": F1_macro, "f1": F1_macro,
            "AUC_macro": AUC_macro, "AUPRC_macro": AUPRC_macro,
            "BAcc_macro": BAcc_macro,
            "_prob": prob.astype(np.float32).tolist(),
            "_y": y_all.astype(int).tolist(),
        }


def run_random_forest_for_task(
        task,
        loaders,
        registry,
        cfg,
        feature_keys,
        *,
        n_estimators=400,
        rf_neg_pos_ratio=4.0,
        max_edges=20_000_000,
        n_jobs=-1,
        edges_only=False,
        display_decimals=4,
        display_truncate=False,
        allow_adj_channel: bool = False,
        threshold_metric: str = "f1"
):
    """
    Train & eval a RandomForest on per-edge features assembled from BCHW stacks.
    Standardisation uses the train stats from `registry`. Build (E,C) by
    gathering selected BCHW rows on the effective mask (optionally gating to edges_only).

    Supports Binary and Multiclass (inferred from task.num_classes).
    """
    _t_model_start = time.monotonic()
    train_loader, val_loader, test_loader = loaders
    d = display_decimals

    def _fmt(v):
        if display_truncate and not math.isnan(v):
            p = 10 ** d
            v = int(v * p) / p
        return f"{v:.{d}f}"

    num_classes = int(getattr(task, "num_classes", 1))

    # 1) Inherit safely resolved keys from the parent pipeline
    rf_keys = list(feature_keys)

    # Prevent RF from inheriting a stale mask channel requirement
    registry.use_mask_channel = False

    # Fit/refresh registry stats for RF with the pipeline-resolved list
    registry.fit(train_loader, rf_keys, include_adj=bool(allow_adj_channel))

    def _iter_split_rows(loader: DataLoader):
        for A, feats, L, mask in loader:
            x_bchw = registry.stack_channels_BCHW(
                A, feats, mask, rf_keys, include_adj=("adj" in getattr(registry, "manifest", []))
            )
            x_bchw = registry.standardise_bchw(x_bchw)

            m = effective_mask(mask, A, registry.directed)
            if edges_only:
                m = m & (A > 0.5)
            if not m.any():
                continue

            idx = torch.nonzero(m, as_tuple=False)
            X_c = x_bchw[idx[:, 0], :, idx[:, 1], idx[:, 2]].numpy()
            y_c = L[idx[:, 0], idx[:, 1], idx[:, 2]].numpy()
            yield X_c, y_c

    # Collect a split into (E,C) and labels (with optional reservoir sampling)
    def _collect_split(loader: DataLoader, enforce_cap: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(getattr(task, "seed", None))
        X_res, y_res = None, None
        seen_pos, seen_neg, seen_multi, seen_bin = 0, 0, 0, 0
        
        cap = max_edges if enforce_cap and max_edges else float('inf')
        is_capped_binary = (num_classes == 1 and rf_neg_pos_ratio > 0 and cap != float('inf'))
        is_global_capped_binary = (num_classes == 1 and rf_neg_pos_ratio <= 0 and cap != float('inf'))
        max_pos = int(cap / (1.0 + rf_neg_pos_ratio)) if is_capped_binary else cap
        max_neg = cap - max_pos if is_capped_binary else cap

        with torch.no_grad():
            for X_c, y_c in _iter_split_rows(loader):
                if X_res is None:
                    alloc_size = int(cap) if (enforce_cap and max_edges and (num_classes > 1 or is_global_capped_binary)) \
                        else (int(max_pos + max_neg) if enforce_cap and max_edges else 0)
                    X_res = np.empty((alloc_size, X_c.shape[1]), dtype=np.float32) if alloc_size else []
                    y_res = np.empty((alloc_size,), dtype=np.int64 if num_classes > 1 else np.float32) if alloc_size else []

                if not enforce_cap or not max_edges:
                    X_res.append(X_c)
                    y_res.append((y_c > 0.5).astype(np.float32) if num_classes == 1 else y_c.astype(np.int64))
                    continue

                if num_classes > 1:
                    # Clamp: a full reservoir must start replacement at row 0
                    n_fill = max(0, min(int(cap) - seen_multi, X_c.shape[0]))
                    if n_fill > 0:
                        X_res[seen_multi:seen_multi + n_fill] = X_c[:n_fill]
                        y_res[seen_multi:seen_multi + n_fill] = y_c[:n_fill]
                        seen_multi += n_fill

                    for i in range(n_fill, X_c.shape[0]):
                        j = rng.integers(0, seen_multi + 1)
                        if j < cap:
                            X_res[j], y_res[j] = X_c[i], y_c[i]
                        seen_multi += 1
                elif is_global_capped_binary:
                    y_c = (y_c > 0.5).astype(np.float32)
                    # Clamp: a full reservoir must start replacement at row 0, never a negative index.
                    n_fill = max(0, min(int(cap) - seen_bin, X_c.shape[0]))
                    if n_fill > 0:
                        X_res[seen_bin:seen_bin + n_fill] = X_c[:n_fill]
                        y_res[seen_bin:seen_bin + n_fill] = y_c[:n_fill]
                        seen_bin += n_fill

                    for i in range(n_fill, X_c.shape[0]):
                        j = rng.integers(0, seen_bin + 1)
                        if j < cap:
                            X_res[j], y_res[j] = X_c[i], y_c[i]
                        seen_bin += 1
                else:
                    y_c = (y_c > 0.5).astype(np.float32)

                    # Fast path while both reservoirs are still filling: no random draw is made in that phase, so
                    # bulk copying retains the same rows in the same order and leaves the RNG stream untouched.
                    room_pos = int(max_pos) - seen_pos
                    room_neg = int(max_neg) - seen_neg
                    start = 0
                    if room_pos > 0 and room_neg > 0:
                        pos_i = np.flatnonzero(y_c == 1.0)
                        neg_i = np.flatnonzero(y_c != 1.0)

                        # Bulk-copy only the prefix before either reservoir fills. Rows after
                        # that must go through the replacement loop so seen_pos/seen_neg stay exact.
                        cut = X_c.shape[0]
                        if pos_i.size > room_pos:
                            cut = min(cut, int(pos_i[room_pos]))
                        if neg_i.size > room_neg:
                            cut = min(cut, int(neg_i[room_neg]))

                        n_pos = int(np.searchsorted(pos_i, cut))
                        n_neg = int(np.searchsorted(neg_i, cut))
                        if n_pos > 0:
                            X_res[seen_pos:seen_pos + n_pos] = X_c[pos_i[:n_pos]]
                            y_res[seen_pos:seen_pos + n_pos] = y_c[pos_i[:n_pos]]
                            seen_pos += n_pos
                        if n_neg > 0:
                            base = int(max_pos) + seen_neg
                            X_res[base:base + n_neg] = X_c[neg_i[:n_neg]]
                            y_res[base:base + n_neg] = y_c[neg_i[:n_neg]]
                            seen_neg += n_neg
                        start = cut
                    for i in range(start, X_c.shape[0]):
                        if (y_c[i] == 1.0):
                            if seen_pos < max_pos:
                                X_res[seen_pos], y_res[seen_pos] = X_c[i], y_c[i]
                            else:
                                j = rng.integers(0, seen_pos + 1)
                                if j < max_pos:
                                    X_res[j], y_res[j] = X_c[i], y_c[i]
                            seen_pos += 1
                        else:
                            if seen_neg < max_neg:
                                X_res[int(max_pos) + seen_neg], y_res[int(max_pos) + seen_neg] = X_c[i], y_c[i]
                            else:
                                j = rng.integers(0, seen_neg + 1)
                                if j < max_neg:
                                    X_res[int(max_pos) + j], y_res[int(max_pos) + j] = X_c[i], y_c[i]
                            seen_neg += 1

        if X_res is None:
            return np.empty((0, 0), np.float32), np.empty((0,), np.float32)
        
        if not enforce_cap or not max_edges:
            return np.concatenate(X_res, axis=0), np.concatenate(y_res, axis=0)

        if num_classes > 1:
            valid = min(seen_multi, int(cap))
            return X_res[:valid], y_res[:valid]
        
        if is_global_capped_binary:
            valid = min(seen_bin, int(cap))
            return X_res[:valid], y_res[:valid]
        
        valid_pos = min(seen_pos, int(max_pos))
        if valid_pos > 0:
            target_neg = max(1, int(valid_pos * rf_neg_pos_ratio))
        else:
            # No positives to balance against: keep the available negatives up to the cap
            target_neg = int(max_neg)

        neg_lo = int(max_pos)
        held_neg = min(seen_neg, int(max_neg))
        valid_neg = min(held_neg, target_neg)
        if valid_neg < held_neg:
            # max_neg exceeds any real split, so the reservoir holds negatives in stream order
            pick = rng.choice(held_neg, size=valid_neg, replace=False) + neg_lo
        else:
            pick = np.arange(neg_lo, neg_lo + valid_neg)

        X_final = np.concatenate([X_res[:valid_pos], X_res[pick]], axis=0)
        y_final = np.concatenate([y_res[:valid_pos], y_res[pick]], axis=0)
        perm = rng.permutation(len(y_final))
        return X_final[perm], y_final[perm]

    # Train
    Xtr, ytr = _collect_split(train_loader, enforce_cap=True)

    def _empty_metrics() -> Dict[str, float]:
        if num_classes == 1:
            return {
                "loss/edge": None,
                "accuracy": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "bacc": float("nan"),
                "auroc": float("nan"),
                "auprc": float("nan"),
                "_prob": [],
                "_y": [],
            }
        return {
            "loss/edge": None,
            "accuracy": float("nan"),
            "P_macro": float("nan"),
            "R_macro": float("nan"),
            "F1_macro": float("nan"),
            "f1": float("nan"),
            "AUC_macro": float("nan"),
            "AUPRC_macro": float("nan"),
            "BAcc_macro": float("nan"),
            "_prob": [],
            "_y": [],
        }

    if Xtr.shape[0] == 0:
        print("[RF] No training edges after masking/subsampling; skipping.")
        return {"val": _empty_metrics(), "test": _empty_metrics(), "thr": None, "ckpt": None}

    if num_classes == 1:
        n_total = len(ytr)
        pos_frac = float(np.count_nonzero(ytr == 1)) / n_total if n_total > 0 else 0.0
        print(f"[RF] Train set: X={Xtr.shape}, pos_frac={_fmt(pos_frac)}")
    else:
        print(f"[RF] Train set: X={Xtr.shape}, Multiclass K={num_classes}")

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=getattr(task, "seed", None),
        class_weight=None,
        max_features="sqrt",
    )
    rf.fit(Xtr, ytr)

    def _predict_probas(model, X, num_classes) -> np.ndarray:
        if num_classes == 1:
            # Binary: return probs for class 1
            if hasattr(model, "classes_") and len(model.classes_) == 2:
                return model.predict_proba(X)[:, 1]
            cls = int(model.classes_[0]) if hasattr(model, "classes_") else 1
            return np.full(X.shape[0], 1.0 if cls == 1 else 0.0, dtype=np.float32)
        else:
            # Multiclass: return (N, K)
            probs = model.predict_proba(X)
            # Handle edge case where some classes might be missing in training
            if probs.shape[1] != num_classes:
                full_probs = np.zeros((X.shape[0], num_classes), dtype=np.float32)
                for i, c in enumerate(model.classes_):
                    if c < num_classes:
                        full_probs[:, c] = probs[:, i]
                return full_probs
            return probs

    def _predict_split(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        prob_parts, y_parts = [], []

        with torch.no_grad():
            for X_c, y_c in _iter_split_rows(loader):
                prob_parts.append(_predict_probas(rf, X_c, num_classes))
                y_parts.append(
                    (y_c > 0.5).astype(np.float32)
                    if num_classes == 1 else y_c.astype(np.int64)
                )

        if not y_parts:
            prob_shape = (0,) if num_classes == 1 else (0, num_classes)
            y_dtype = np.float32 if num_classes == 1 else np.int64
            return np.empty(prob_shape, dtype=np.float32), np.empty((0,), dtype=y_dtype)

        return np.concatenate(prob_parts, axis=0), np.concatenate(y_parts, axis=0)

    def _metrics_from_probs(
            probs: np.ndarray,
            labels: np.ndarray,
            fixed_thr: Optional[float] = None,
            threshold_metric: str = "f1",
    ) -> Tuple[Optional[float], Dict[str, float]]:

        y = np.asarray(labels).astype(np.int32)

        # --- Multiclass Path ---
        if num_classes > 1:
            preds = probs.argmax(axis=1)
            acc = float(np.mean(preds == y))
            f1_mac = float(f1_score(y, preds, average="macro", zero_division=0))
            p_mac = float(precision_score(y, preds, average="macro", zero_division=0))
            r_mac = float(recall_score(y, preds, average="macro", zero_division=0))
            bacc = float(balanced_accuracy_score(y, preds)) if len(np.unique(y)) >= 2 else float("nan")

            # AUC/AP Macro (OvR); skip unrepresented classes
            aucs, aprs = [], []
            for k in range(num_classes):
                y_bin = (y == k).astype(np.int32)
                if y_bin.sum() == 0 or (y_bin.size - y_bin.sum()) == 0:
                    continue
                pk = probs[:, k]
                try:
                    aucs.append(roc_auc_score(y_bin, pk))
                except Exception:
                    pass
                try:
                    aprs.append(average_precision_score(y_bin, pk))
                except Exception:
                    pass
            
            auc_mac = float(np.mean(aucs)) if aucs else float("nan")
            auprc_mac = float(np.mean(aprs)) if aprs else float("nan")

            return None, {
                "loss/edge": None,
                "accuracy": acc, "BAcc_macro": bacc,
                "F1_macro": f1_mac, "f1": f1_mac,
                "P_macro": p_mac, "R_macro": r_mac,
                "AUC_macro": auc_mac, "AUPRC_macro": auprc_mac
            }

        # --- Binary Path ---
        pr = np.asarray(probs, dtype=np.float64)
        if pr.size == 0:
            return None, {
                "loss/edge": None,
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "accuracy": float("nan"),
                "bacc": float("nan"),
                "auroc": float("nan"),
                "auprc": float("nan"),
                "TP": 0, "TN": 0, "FP": 0, "FN": 0
            }

        # pick threshold
        if fixed_thr is None:
            thr = get_optimal_threshold(y, pr, threshold_metric)
        else:
            thr = float(fixed_thr)

        p = (pr >= thr).astype(np.int32)
        TP = int(np.count_nonzero((y == 1) & (p == 1)))
        TN = int(np.count_nonzero((y == 0) & (p == 0)))
        FP = int(np.count_nonzero((y == 0) & (p == 1)))
        FN = int(np.count_nonzero((y == 1) & (p == 0)))

        has_neg, has_pos = bool(np.any(y == 0)), bool(np.any(y == 1))
        if has_neg and has_pos:
            auc = float(roc_auc_score(y, pr))
            auprc = float(average_precision_score(y, pr))
        else:
            auc, auprc = float("nan"), float("nan")

        # Compute metrics manually in O(1) to avoid redundant O(N) array traversals by sklearn
        prec = TP / max(1, TP + FP)
        rec = TP / max(1, TP + FN)
        f1 = (2.0 * prec * rec / max(1e-12, prec + rec)) if (prec + rec) > 0 else 0.0
        acc = (TP + TN) / max(1, TP + TN + FP + FN)

        # BAcc averages TPR and TNR; one is undefined when a class is absent
        bacc = 0.5 * (rec + (TN / max(1, TN + FP))) if (has_pos and has_neg) else float("nan")

        return (thr if fixed_thr is None else None), {
            "loss/edge": None,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "accuracy": float(acc),
            "bacc": float(bacc),
            "auroc": float(auc),
            "auprc": float(auprc),
            "TP": TP, "TN": TN, "FP": FP, "FN": FN
        }

    # --- Val/Test ---
    pv, yv = _predict_split(val_loader)
    if yv.size == 0:
        thr = 0.5 if num_classes == 1 else None
        val_m = _empty_metrics()
    else:
        thr, val_m = _metrics_from_probs(pv, yv, fixed_thr=None, threshold_metric=threshold_metric)
        val_m["_prob"] = pv.astype(np.float32).tolist()
        val_m["_y"] = yv.astype(int).tolist()

    pt, yt = _predict_split(test_loader)
    if yt.size == 0:
        test_m = _empty_metrics()
    else:
        _, test_m = _metrics_from_probs(pt, yt, fixed_thr=thr, threshold_metric=threshold_metric)
        test_m["_prob"] = pt.astype(np.float32).tolist()
        test_m["_y"] = yt.astype(int).tolist()

    if num_classes == 1:
        print(
            f"[RF] thr={_fmt(float(thr))} | "
            f"Test: F1={_fmt(float(test_m.get('f1', 0.0)))}  "
            f"P={_fmt(float(test_m.get('precision', 0.0)))}  "
            f"R={_fmt(float(test_m.get('recall', 0.0)))}"
        )
    else:
        print(
            f"[RF] Test: F1_macro={_fmt(float(test_m.get('F1_macro', 0.0)))}  "
            f"Acc={_fmt(float(test_m.get('accuracy', 0.0)))}  "
            f"AUC_macro={_fmt(float(test_m.get('AUC_macro', 0.0)))}"
        )

    meta = _build_meta(
        model_key="rf",
        task=task,
        registry=registry,
        cfg=cfg,
        runtime=SimpleNamespace(
            feature_keys=list(rf_keys),
            keep_idx=None,
            eff_in_ch=None,
        ),
        best_thr=thr
    )
    meta["rf"] = {
        "n_estimators": int(n_estimators),
        "n_jobs": int(n_jobs),
        "max_edges": int(max_edges),
        "rf_neg_pos_ratio": float(rf_neg_pos_ratio),
        "threshold_metric": str(threshold_metric),
    }

    meta["elapsed_seconds"] = round(time.monotonic() - _t_model_start, 3)
    ckpt_path = save_pipeline_checkpoint("rf", {"sklearn_model": rf}, task, cfg, meta)

    return {"val": val_m, "test": test_m, "thr": thr, "ckpt": ckpt_path, "elapsed_seconds": meta["elapsed_seconds"]}


def print_final_summary_table(
    results: Dict[str, dict],
    task,
    header: Optional[str] = None,
    order: Optional[Sequence[str]] = None,
    display_decimals: int = 4,
    display_truncate: bool = False
) -> None:
    """
    Pretty-printer for final test results that tolerates missing keys
    and auto-switches between binary and macro metrics. If `order` is provided,
    models are shown in that order with any extras appended.
    """
    num_classes = int(getattr(task, "num_classes", 1))
    disp_dec = int(display_decimals)
    disp_trunc = bool(display_truncate)

    def _get(d: dict | None, keys):
        if not d:
            return None
        if isinstance(keys, str):
            return d.get(keys, None)
        for k in keys:
            if k in d:
                return d[k]
        return None

    def _fmt(v, d: int = disp_dec):
        if v is None:
            return "-"
        try:
            x = float(v)
        except Exception:
            return "-"

        if disp_trunc and not math.isnan(x):
            p = 10 ** d
            x = int(x * p) / p
        return f"{x:.{d}f}"

    BIN_KEYS = {"precision", "recall", "f1", "auroc", "auprc", "bacc", "accuracy"}
    MAC_KEYS = {
        "P_macro", "precision_macro", "R_macro", "recall_macro", "F1_macro", "f1_macro",
        "AUC_macro", "auc_macro", "AUPRC_macro", "auprc_macro", "BAcc_macro", "bacc_macro"
    }

    def acc_with_optional_bacc(acc_val, bacc_val):
        s = _fmt(acc_val)
        if bacc_val is not None:
            s += f" [BAcc={_fmt(bacc_val)}]"
        return s

    keys = list(results.keys())
    if order:
        key_lut = {str(k).lower(): k for k in keys}
        seen = set()
        ordered = []
        for ok in order:
            rk = key_lut.get(str(ok).lower())
            if rk is not None and rk not in seen:
                ordered.append(rk)
                seen.add(rk)
        for k in keys:
            if k not in seen:
                ordered.append(k)
                seen.add(k)
        keys = ordered

    def _has_data(d) -> bool:
        if not d:
            return False
        try:
            return not math.isnan(float(d.get("accuracy", float("nan"))))
        except Exception:
            return False

    use_test = any(_has_data(results.get(mk, {}).get("test")) for mk in keys)
    use_val = (not use_test) and any(_has_data(results.get(mk, {}).get("val")) for mk in keys)

    print("\n" + "#" * 80)
    if not use_test and not use_val:
        print("FINAL SUMMARY — no evaluation data in any split; nothing to report.")
        print("#" * 80)
        return

    split_name = "Test" if use_test else "Val"
    print(header if header is not None else f"FINAL SUMMARY ({split_name})")
    print("#" * 80)

    for mk in keys:
        dct = results.get(mk, {})
        thr = dct.get("thr", None)
        t = dct.get("test", {}) if use_test else dct.get("val", {})

        has_macro = any(k in t for k in MAC_KEYS)
        has_binary = any(k in t for k in BIN_KEYS)
        is_multiclass = has_macro or (not has_binary and num_classes > 1)

        acc = _get(t, ["accuracy"])
        loss = _get(t, ["loss/edge", "loss_per_edge"])

        if not is_multiclass:
            prec = t.get("precision")
            rec = t.get("recall")
            f1 = t.get("f1")
            auroc = t.get("auroc")
            auprc = t.get("auprc")
            bacc = t.get("bacc")
            thr_s = _fmt(thr, d=disp_dec)

            print(
                f"[{mk.upper()}] thr={thr_s} | "
                f"{split_name}: F1={_fmt(f1)}  "
                f"AUPRC={_fmt(auprc)} [AUROC={_fmt(auroc)}]  "
                f"Acc={acc_with_optional_bacc(acc, bacc)}  "
                f"P={_fmt(prec)}  R={_fmt(rec)}  "
                f"loss/edge={_fmt(loss)}"
            )
        else:
            p_mac = _get(t, ["P_macro", "precision_macro"])
            r_mac = _get(t, ["R_macro", "recall_macro"])
            f1_mac = _get(t, ["F1_macro", "f1_macro"])
            auc_mac = _get(t, ["AUC_macro", "auc_macro"])
            auprc_m = _get(t, ["AUPRC_macro", "auprc_macro"])
            bacc_m = _get(t, ["BAcc_macro", "bacc_macro"])

            print(
                f"[{mk.upper()}] {split_name}: F1_macro={_fmt(f1_mac)}  "
                f"AUPRC_macro={_fmt(auprc_m)} [AUC_macro={_fmt(auc_mac)}]  "
                f"Acc={acc_with_optional_bacc(acc, bacc_m)}  "
                f"P_macro={_fmt(p_mac)}  R_macro={_fmt(r_mac)}  "
                f"loss/edge={_fmt(loss)}"
            )


def _preferred_model_order() -> Sequence[str]:
    """Return the canonical display order for mixed NN/GNN runs."""
    return (
        "mlp",
        "deep_mlp",
        "sage",
        "rf",
        "gin",
        "cnn",
        "gcn",
        "transformer",
        "edge_tx",
        "gps"
    )


def finalise_summary(results: Dict[str, Any], task, header: Optional[str] = None,
                     display_decimals: int = 4, display_truncate: bool = False) -> None:
    """
    Print a single consolidated summary for a (possibly merged) results mapping.
    Accepts either the raw results dict or the full pipeline bundle with a "results" key.
    Safe to call multiple times on the same object; prints once.
    """
    bundle = results
    res_map = results
    if isinstance(results, dict) and "results" in results and isinstance(results["results"], dict):
        res_map = results["results"]
    if isinstance(bundle, dict) and bundle.get("_summary_printed"):
        return

    # Use the explicit parameters instead of reading from cfg
    print_final_summary_table(
        res_map,
        task,
        display_decimals=display_decimals,
        display_truncate=display_truncate,
        header=header,
        order=_preferred_model_order()
    )

    if isinstance(bundle, dict):
        bundle["_summary_printed"] = True


def _reset_pipeline_runtime_state(task=None) -> None:
    FeatureRegistry._stats_cache.clear()
    FeatureRegistry._warned_pad_features.clear()

    if task is not None:
        for attr in ("_active_run_loaders", "_active_run_loader_sig", "_active_run_dataset", "_latest_results"):
            if hasattr(task, attr):
                delattr(task, attr)

        if bool(getattr(task, "_owns_bench_instance", False)):
            task._bench_instance = None
            task._owns_bench_instance = False


# ============================================================
# 6) Main runner
# ============================================================
def run_pipeline_for_task(task, models, cfg, *, quiet: bool = False):
    """
    End-to-end runner for the dense edge-classification pipeline.

    Run lifecycle:
        - This call starts a new run unless it is later followed in the same script/cell
          by `run_gnn_suite(...)` or `run_gnn_edges_suite(...)` on the same task.
        - In that dense -> GNN case, both stages are treated as one combined run and
          share a single merged results bundle / final summary table.
        - A second top-level pipeline entrypoint in the same script/cell starts a fresh
          run and finalizes the previous one first.
        - Final summary printing and runtime-state reset are owned by the shared run
          lifecycle manager rather than by this function directly.

    Args:
        task: TaskSpec or ProvidedSplitsTask implementing train/val/test loaders, or exposing (bench + hooks).
              Must define task.directed. hooks.allow_adj_channel gates 'adj' as input.
        models: Iterable of accepted model keys
                {"mlp","deep_mlp","cnn","transformer","rf"}.
                GNN keys are handled by the gnn_bridge suite helpers.
        cfg: A TNNTrainConfig object containing training knobs. Common attributes:
             - batch_size, epochs, lr, weight_decay, grad_clip, display_decimals
             - use_mask_channel, supervised_redaction_policy
             - tx_force_adj_channel (whether TX forces 'adj'), tx_* hyperparameters
             - threshold_metric/select_by for threshold tuning, save_dir for checkpoint root.

    Behaviour:
        - Infers eval mask policy from task.directed.
        - Resolves canonical feature_keys once (no 'adj').
        - For each model, configures the effective mask-channel setting, re-fits FeatureRegistry on the train split,
          builds keep_idx (handles adj gating), and runs the shared trainer/evaluator.
        - Saves checkpoints under saved_checkpoints/<task.name>/<timestamp>/<model>.pth with meta bundle.
        - Returns the active run bundle in the same {"results": ..., "metadata": ...} shape
          used by the rest of the pipeline.
    """
    print(f"=== Task: {task.name} (directed={task.directed}) ===")

    # Lifecycle decision must happen before anything that reads or writes run-scoped state
    bundle = begin_or_attach_run(
        task_key=(id(task), str(getattr(task, "name", "task"))),
        stage="tnn",
        bundle_factory=lambda: {"results": {}, "metadata": {}},
        can_attach=lambda active_stage, next_stage, active_task_key, next_task_key: (
            active_stage == ["tnn"] and next_stage in {"gnn_full", "gnn_edges"} and active_task_key == next_task_key
        ),
        reset_cb=lambda: _reset_pipeline_runtime_state(task),
        summary_cb=lambda b, quiet, dd, dt: (
            None if quiet else finalise_summary(b, task, display_decimals=dd, display_truncate=dt)
        ),
        quiet=quiet,
        display_decimals=int(getattr(cfg, "display_decimals", 4)),
        display_truncate=bool(getattr(cfg, "display_truncate", False))
    )
    results = bundle["results"]

    # Task-controlled adjacency channel only (Transformer override is applied later in the TX branch)
    allow_adj = bool(getattr(getattr(task, "hooks", object()), "allow_adj_channel", False))

    # Feature keys — no 'adj' here; it is managed by the registry manifest and gated later
    hooks = getattr(task, "hooks", None)
    requested, custom_feature_types = _normalise_feature_spec(
        _hooks_get(hooks, "feature_set", []), directed=task.directed
    )
    feature_keys = list(requested)

    registry = FeatureRegistry(
        use_mask_channel=getattr(cfg, "use_mask_channel", None),
        directed=getattr(task, "directed", True),
        supervised_redaction_policy=str(getattr(cfg, "supervised_redaction_policy", "adj_only")),
        custom_feature_types=custom_feature_types
    )

    # Supported dense policy: when adjacency input is disabled and no dense features were requested,
    # use the default structural feature set for dense loader resolution without retaining it on the task
    _restore_feature_set = False
    _original_feature_set = None
    if not feature_keys and not allow_adj:
        feature_keys = ["degree", "deg_row", "deg_col", "clustering_coeff", "cn", "jaccard", "adamic_adar"]
        if hooks is not None:
            _original_feature_set = hooks.feature_set
            hooks.feature_set = list(feature_keys)
            _restore_feature_set = True
        print(
            "[WARN] Dense no-feature mode is unavailable when allow_adj_channel=False. "
            "Using the default dense structural feature set for loader resolution: "
            f"{feature_keys}"
        )

    # Resolve loaders
    try:
        train_loader, val_loader, test_loader = _resolve_loaders(task, cfg)
    finally:
        if _restore_feature_set:
            hooks.feature_set = _original_feature_set

    # Expose dense runtime state to subroutines without mutating cfg
    print("feature_keys:", feature_keys)
    runtime = SimpleNamespace(
        feature_keys=list(feature_keys),
        keep_idx=None,
        eff_in_ch=None,
    )

    seen_models = set()
    deduped_models = []
    for mk in models:
        if mk in seen_models:
            print(f"[TNN PIPELINE MODEL SELECTION] Duplicate model key {mk!r} ignored.", flush=True)
            continue
        seen_models.add(mk)
        deduped_models.append(mk)

    for mk in deduped_models:
        print("\n" + "=" * 80)
        print(f"Running model: {mk.upper()}")
        print("=" * 80)
        _model_elapsed = None

        if mk in ("mlp", "deep_mlp", "cnn", "transformer"):
            val_m, test_m, thr, ckpt, elapsed = train_and_eval_one_model(
                task=task,
                registry=registry,
                loaders=(train_loader, val_loader, test_loader),
                model_key=mk,
                cfg=cfg,
                runtime=runtime
            )
            entry = dict(val=val_m, test=test_m, thr=thr, ckpt=ckpt)
            results[mk] = entry
            _model_elapsed = elapsed

        elif mk == "rf":
            edges_only = bool(getattr(task, "eval_on_existing_edges_only", False))
            rf_neg_ratio = getattr(cfg, "rf_neg_pos_ratio", 4.0)
            rf_out = run_random_forest_for_task(
                task, (train_loader, val_loader, test_loader), registry, cfg, runtime.feature_keys,
                n_estimators=400, rf_neg_pos_ratio=float(rf_neg_ratio), max_edges=20_000_000,
                n_jobs=-1,
                edges_only=edges_only,
                display_decimals=getattr(cfg, "display_decimals", 4),
                display_truncate=getattr(cfg, "display_truncate", False),
                allow_adj_channel=allow_adj,
                threshold_metric=str(getattr(cfg, "threshold_metric", "f1")).lower()
            )
            entry = dict(val=rf_out["val"], test=rf_out["test"], thr=rf_out.get("thr"), ckpt=rf_out.get("ckpt"))
            results[mk] = entry
            _model_elapsed = rf_out.get("elapsed_seconds")

        elif mk in ("sage", "gcn", "gin", "edge_tx", "gps"):
            print(
                f"[TNN PIPELINE MODEL SELECTION] GNN model key {mk!r} is not supported by run_pipeline_for_task "
                f"and will be skipped. Use run_gnn_suite(..., encoders=[...]) or "
                f"run_gnn_edges_suite(..., encoders=[...]) instead.",
                flush=True
            )
            continue
        else:
            raise ValueError(
                f"Unknown model key: {mk}. "
                f"Allowed dense keys are: mlp, deep_mlp, cnn, transformer, rf."
            )

        _cuda_gc(f"after-{mk.lower()}")
        if _model_elapsed is not None:
            print(f"[TIME] Training total: {_format_duration(_model_elapsed)}")

    # Construct the metadata bundle
    meta = {
        "task": _task_to_meta_dict(task),
        "registry": {
            "base_feature_keys": list(feature_keys),
            "directed": registry.directed
        }
    }

    bundle.setdefault("metadata", {}).update(meta)
    setattr(task, "_latest_results", bundle)
    return bundle

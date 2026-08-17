# An Architecture Evaluation Framework for edge-level Deep Graph Transformation Tasks

Architecture evaluation framework for binary and multiclass edge prediction tasks on graphs, supporting both dense (image-style) models and GNN encoders with a shared evaluation contract.

---

## Overview

This pipeline frames **edge classification** as a supervised learning problem on graphs: given an observed adjacency matrix and optional node/edge features, predict a label for every edge (or candidate edge) in the graph. It supports both binary prediction (edge present/absent, edge has a property or not) and multiclass prediction (edge type), on both directed and undirected graphs.

The pipeline provides two architecturally separate model families under a shared evaluation and reporting contract:

- **Dense pipeline** — Treats the adjacency matrix and features as multi-channel images in BCHW format and processes them with standard architectures (MLP, CNN, Patch Transformer) and a Random Forest baseline.
- **GNN pipeline** — Uses graph neural network encoders (GCN, GraphSAGE, GIN, Edge-Masked Transformer, GPS) to produce node embeddings, then scores edges through a shared edge decoder.

Both pipelines share evaluation masking (`effective_mask`), metric computation, the summary printer, checkpoint saving, and a run lifecycle manager that supports standalone or combined dense→GNN execution in a single script or notebook cell.

A built-in synthetic graph generator (`GraphBenchmark`) produces datasets from 10 graph families with native support for directed and undirected graphs, enabling controlled benchmarking across diverse topologies. Users define a task by writing a short task file — a label function, a feature request, and configuration — then call the provided runner entry points.

---

## Installation

```bash
pip install -r requirements.txt
```

Developed and tested in a Conda environment with Python 3.12.

Key dependencies: PyTorch, scikit-learn, NetworkX, NumPy.

The pipeline auto-detects CUDA and uses GPU acceleration with mixed precision when available; CPU-only execution is fully supported. For GPU use, install PyTorch and PyTorch Geometric builds compatible with your CUDA setup.

Task-owned datasets, downloaded data, and reusable task caches should live under `data/`. The exact subdirectory layout is defined by each task file, not by the pipeline. Model checkpoints are written under `saved_checkpoints/`.

---

## Citation requirement

If you use this framework, or any benchmark task distributed with it, in a publication, you must cite the associated ICGT 2026 paper:

> Machowczyk, A., Heckel, R. (2026). Benchmark First: Defining Tasks for Graph Transformation Learning. In: Archibald, B., Semeráth, O. (eds) Graph Transformation. ICGT 2026. Lecture Notes in Computer Science, vol 16624. Springer, Cham. [https://doi.org/10.1007/978-3-032-29730-3_13](https://doi.org/10.1007/978-3-032-29730-3_13).

The full citation is also provided in the [Citation](#citation) section below.

---

## Getting Started

Users interact with the pipeline by writing a **task file**: define a label function, configure features and settings, create a task object, then call the runner entry points. The repository includes ready-to-run example task files.

### Running task files

Task files should be run as modules from the repository root:

```bash
python -m tasks.symmetric-closure
```

For task files that take command-line arguments:

```bash
python -m tasks.graph-denoising --model cnn
```

### Example: symmetric closure on directed graphs

```python
import numpy as np
from pipeline.EdgeClassification import TaskHooks, ProvidedSplitsTask, TNNTrainConfig, run_pipeline_for_task
from pipeline.GNNBridge import GNNTrainConfig, run_gnn_suite


def label_symmetric_closure(A_obs: np.ndarray) -> np.ndarray:
    """L[i, j] = 1 if A[i,j] or A[j,i] exists. No self-loops."""
    L = ((A_obs + A_obs.T) > 0).astype(np.float32)
    np.fill_diagonal(L, 0.0)
    return L


hooks = TaskHooks(
    label_fn=label_symmetric_closure,
    feature_set=["powers", "deg_row", "deg_col", "triangles",
                 "clustering_coeff", "cn", "jaccard", "adamic_adar", "transpose"],
    allow_adj_channel=True
)

task = ProvidedSplitsTask(
    name="symmetric_closure",
    directed=True,
    hooks=hooks,
    num_graphs=400, min_nodes=6, max_nodes=140,
    ratios=(0.7, 0.2, 0.1),
    seed=42
)

# Dense pipeline — all five models
cfg = TNNTrainConfig(epochs=30, lr=3e-4)
results = run_pipeline_for_task(task, ["mlp", "deep_mlp", "cnn", "transformer", "rf"], cfg)

# GNN pipeline — attaches to the same run
gnn_cfg = GNNTrainConfig(epochs=30, lr=3e-4)
run_gnn_suite(task=task, encoders=("gcn", "sage", "gin", "edge_tx", "gps"), cfg=gnn_cfg)
```

### Task file template

Replace the placeholders with your task-specific label logic, feature requests, graph-generation settings, and training configuration.

```python
import numpy as np
from pipeline.EdgeClassification import TaskHooks, ProvidedSplitsTask, TNNTrainConfig, run_pipeline_for_task
from pipeline.GNNBridge import GNNTrainConfig, run_gnn_suite


def label_fn(A_obs: np.ndarray) -> np.ndarray:
    """
    Return the supervision matrix for your task.

    Requirements on supported paths:
    - return a matrix aligned with the graph size
    - use valid class labels at supervised positions
    - do not rely on ignore-index padding conventions
    """
    raise NotImplementedError("Replace with your task-specific label function")


hooks = TaskHooks(
    label_fn=label_fn,
    feature_set=[
        # Replace with the canonical/custom features your task needs
        # Set to True to request all features available
    ],
    allow_adj_channel=True
)

task = ProvidedSplitsTask(
    name="my_task",
    directed=False,          # or True
    hooks=hooks,
    num_graphs=100,          # replace as needed
    min_nodes=8,
    max_nodes=32,
    ratios=(0.7, 0.2, 0.1),
    seed=42                  # optional but recommended for reproducibility
                             # no seed will generate and print one to the console
)

dense_cfg = TNNTrainConfig(
    epochs=10,               # leave out a config option to use the default!
    lr=3e-4,
    batch_size=16
)

MODELS = ["mlp", "deep_mlp", "cnn", "transformer", "rf"]
results = run_pipeline_for_task(task, MODELS, dense_cfg)

gnn_cfg = GNNTrainConfig(
    epochs=40,
    lr=3e-4,
    batch_size=16
)

run_gnn_suite(
    task=task,
    encoders=("gcn", "sage", "gin", "edge_tx", "gps"),
    cfg=gnn_cfg
)
```

### TaskHooks reference

| Parameter | Type | Description |
|-----------|------|-------------|
| `label_fn` | callable or `None` | Computes the label matrix L. Receives `A_obs`, `A_true`, and/or `G_true` based on parameter name matching. `None` yields all-zero labels. |
| `feature_set` | `bool` or `list[str | tuple[str, Literal["node", "edge"]]]` | `True` for the orientation-aware automatic canonical feature set, `False` for none, or a list containing canonical names/macros and typed custom declarations such as `("my_feature", "node")` or `("my_feature", "edge")`. |
| `allow_adj_channel` | `bool` | If `True`, include the observed adjacency as an explicit input channel for dense models. |
| `orientation` | `str` or `None` | Optional post-generation orientation mode (e.g. `"dag"`). |
| `ensure_connected` | `bool` | If `True`, enforce connectivity via proportional multi-stitching after generation. |

### Supported runner entry points

There are exactly **three** supported entry points:

**1. Dense / Random Forest models**

```python
from pipeline.EdgeClassification import run_pipeline_for_task
bundle = run_pipeline_for_task(task, models=["mlp", "deep_mlp", "cnn", "transformer", "rf"], cfg=cfg)
```

Accepted model keys: `mlp`, `deep_mlp`, `cnn`, `transformer`, `rf`.
Passing GNN keys (`sage`, `gcn`, `gin`, `edge_tx`, `gps`) prints a warning and skips them — use the GNN entry points below.

**2. Full-batch GNN models**

```python
from pipeline.GNNBridge import run_gnn_suite
bundle = run_gnn_suite(task, encoders=["gcn", "sage", "gin", "edge_tx", "gps"], cfg=cfg)
```

**3. Scalable Mode**

```python
from pipeline.GNNBridge import run_gnn_edges_suite
bundle = run_gnn_edges_suite(task, encoders=["sage"], cfg=cfg)
```

`run_gnn_edges_suite(...)` is the dedicated Scalable Mode runner for memory-efficient GNN execution on larger graphs. It processes one graph at a time, encodes that graph once, selects the supervised pairs through the normal effective mask, and scores only those pairs without materialising a dense `(N, N, Fe)` decoder-side edge feature tensor. It forces `batch_size=1`. This does not restrict the task to a single graph: multi-graph datasets are processed sequentially, one graph at a time.

### Supported task shapes

All three runners accept only these task shapes:

1. **`ProvidedSplitsTask` (or a subclass)** — for generated multi-graph datasets with ratio-based splitting, or pre-split datasets exposed through `bench.splits`. Ratio-based splitting uses integer cut points; zero-sized splits are allowed.
2. **Single-graph task exposing `task.bench` + `task.hooks`** — for the single-graph benchmark path.

### Run semantics

A **run** is the unit of summary printing and runtime-state cleanup.

Supported run shapes:

1. A standalone dense call.
2. A standalone GNN call.
3. A combined dense → GNN run on the same task in the same script/notebook cell.

Rules:

- A notebook cell is treated like a script.
- `run_pipeline_for_task(...)` followed by `run_gnn_suite(...)` or `run_gnn_edges_suite(...)` on the same task in the same script/cell is one combined run.
- A second top-level runner call after that starts a new run and finalises the previous one first.
- Final summary printing and runtime-state reset happen automatically at run finalisation.
- Runtime-state reset includes pipeline-owned temporary state such as summary bookkeeping and run-scoped loader / generated-dataset caches.
- `quiet=True` suppresses the final summary table only. It does **not** affect cleanup or state reset.
- In a combined run, the latest attached stage controls final summary display options (`quiet`, `display_decimals`, `display_truncate`).
- Within a run, the pipeline reuses the same resolved `(train_loader, val_loader, test_loader)` objects for later attached stages whenever the requested loader configuration is compatible.
- Checkpoints saved during the same run share one run timestamp, even when dense and GNN stages use separate config objects.

---

## Architecture Overview

```text
┌───────────────────────────────────────────────────────────────┐
│                    Supported task shapes                      │
│  1) ProvidedSplitsTask / subclass                             │
│  2) Single-graph task exposing task.bench + task.hooks        │
└────────────┬──────────────────────────┬───────────────────────┘
             │                          │
    ┌────────▼─────────────┐    ┌───────▼──────────────┐
    │   Dense Pipeline     │    │    GNN Pipeline      │
    │ EdgeClassification.py│    │    GNNBridge.py      │
    │                      │    │                      │
    │ BCHW stacking        │    │ Per-graph (X, E)     │
    │ Dataset-level stats  │    │ Per-graph z-score    │
    │ Image-style heads    │    │ Encode → Decode      │
    └────────┬─────────────┘    └────────┬─────────────┘
             │                           │
             └─────────────┬─────────────┘
                           │
               ┌───────────▼───────────────┐
               │      Shared contracts     │
               │      effective_mask()     │
               │      FeatureRegistry      │
               │      finalise_summary()   │
               │    _utils.run_lifecycle   │
               │ save_pipeline_checkpoint()│
               └───────────────────────────┘
```

The dense and GNN pipelines are **architecturally separate** and intentionally differ in preprocessing, normalisation, and several configurable behaviours. They share evaluation masking logic (`effective_mask`), feature math (`_utils.features`), metric computation, and the summary printer.

---

## Supported Graph Families

The built-in `GraphBenchmark` generates synthetic graphs from the following families.

| Family | Description | Directed support |
|--------|-------------|------------------|
| `erdos_renyi` | Uniform random edges (p = 3.5/N) | Native (NetworkX) |
| `barabasi_albert` | Preferential attachment (m=2); scale-free degree distribution | Bollobás scale-free model |
| `watts_strogatz` | Small-world with local structure and random rewiring | Custom directed analogue |
| `random_regular` | Strict 4-regular graph | Eulerian circuit orientation (in=out=2) |
| `stochastic_block` | 2-block community structure (p\_in > p\_out) | Native (NetworkX) |
| `powerlaw_cluster` | Holme-Kim: scale-free hubs with high clustering | Directed triad formation |
| `random_geometric` | Spatial proximity with radius r = √(3.5/Nπ) | K-nearest neighbours (K=3) |
| `balanced_tree` | Full tree with branching factor r ∈ \[2, 4\] | Native (root-outward edges) |
| `tree_plus_chords` | Random labelled tree + up to 0.75N chord edges | BFS-directed tree + non-reciprocal chords |
| `shape_cycle` | Incomplete ring topologies (~50% of nodes) + background forest | Forward-directed rings, randomly oriented forest |

Most families hold average degree constant as node count varies: `erdos_renyi`, `random_geometric` and `tree_plus_chords` target ~3.5 explicitly; `barabasi_albert`, `powerlaw_cluster`, `random_regular` and `watts_strogatz` sit around 4; `balanced_tree` and `shape_cycle` around 2. `stochastic_block` is the exception — its `p_in`/`p_out` are drawn from fixed ranges independent of node count, so its average degree grows with graph size (~0.18 x N). They also use an organic mutation step targeting 10–15% edge dropout to break algorithmic perfection. Because an integer number of edges must be removed, that range is not guaranteed when the graph does not contain enough edges to realise it. Small or sparse graphs may therefore have a lower realised dropout fraction or zero removals. Optional post-generation connectivity enforcement via proportional multi-stitching is available through `hooks.ensure_connected`.

---

## Model Zoo

### Dense models

All dense models consume BCHW tensors (adjacency + features stacked as channels, padded to N\_max across the batch).

| Key | Architecture | Notes |
|-----|-------------|-------|
| `mlp` | Multi-layer perceptron | Per-pixel MLP over channel features |
| `deep_mlp` | Deeper MLP variant | Additional hidden layers |
| `cnn` | Convolutional network | Spatial convolutions on the N×N feature grid |
| `transformer` | Patch Transformer | Divides the N×N grid into patches processed as tokens; configurable token masking policy (`keep_all`, `from_mask`, `auto`) and optional 50% patch overlap |
| `rf` | Random Forest | scikit-learn; extracts per-edge feature vectors from the channel stack |

For the Patch Transformer, `cfg.tx_patch_overlap=False` is the default and preserves the current non-overlapping tokenisation (`stride == patch`). Setting `cfg.tx_patch_overlap=True` enables 50% overlap (`stride = max(1, patch // 2)`). Adaptive patch size selection accounts for the selected stride when estimating the token sizing budget.

### GNN models

All GNN encoders produce per-node embeddings that feed into a shared edge decoder. The encoders use LayerNorm (configurable affine / non-affine via `cfg.learnable_layer_norm`) and support training-time DropEdge regularisation.

| Key | Encoder | Propagation | Notes |
|-----|---------|-------------|-------|
| `gcn` | GCN with residual connections and Jumping Knowledge | Dense: D⁻¹⁄²AD⁻¹⁄² (undirected) or row-stochastic (directed) | JK concatenation projected back to hidden dim |
| `sage` | Mean-aggregator GraphSAGE | Sparse row-stochastic | Self + neighbour linear transforms |
| `gin` | GIN with learnable ε | Sparse binary adjacency | (1+ε)·H + Σ neighbours → 2-layer MLP |
| `edge_tx` | Edge-Masked Transformer | Dense attention restricted to the A+I neighbourhood | Standard `nn.TransformerEncoder` with a structural attention mask |
| `gps` | GPS / GraphGPS-style encoder | Sparse local GINE-style message passing plus global full self-attention | Uses GPS-owned LapPE/RWSE structural encodings and a shared edge decoder |

Aliases for `edge_tx`: `transformer`, `tx`, `edge-transformer`, `edge_transformer`, `edge-tx`.
Aliases for `gps`: `graph_gps`, `graph-gps`.

**Edge decoder** (shared across all GNN encoders): Scores each candidate edge (i, j) by concatenating \[z\_i, z\_j, |z\_i − z\_j|, z\_i ⊙ z\_j, E\_ij\] and passing through a 3-layer MLP. For undirected graphs, logits are symmetrised as (s\_ij + s\_ji) / 2.

---

## Evaluation Metrics

### Binary tasks

| Metric | Description |
|--------|-------------|
| `accuracy` | Overall classification accuracy at the tuned threshold |
| `precision` | Positive predictive value |
| `recall` | True positive rate (sensitivity) |
| `f1` | Harmonic mean of precision and recall |
| `bacc` | Balanced accuracy: mean of TPR and TNR |
| `auroc` | Area under the ROC curve (threshold-independent) |
| `auprc` | Area under the precision-recall curve |
| `loss/edge` | BCE with pos\_weight, computed globally over the split for neural models. Unavailable for Random Forest (`-` in the final summary) |

Threshold tuning is performed on the validation split. The dense pipeline supports configurable leading metrics (`cfg.threshold_metric`, `cfg.select_by`); the GNN pipeline uses fixed validation-F1 for both threshold tuning and checkpoint selection.

Tuning always returns a threshold drawn from the observed score distribution. When no threshold on that split beats chance, which is normal in early epochs and on small validation splits, the tuner returns the best available real threshold over the degenerate predict-all-negative point, so a reported `bacc` below 0.5 reflects the split rather than a tuning failure. Degenerate splits (all-positive or all-negative) fall back to the previous epoch's threshold.

### Multiclass tasks

| Metric | Description |
|--------|-------------|
| `accuracy` | Overall classification accuracy |
| `F1_macro` | Macro-averaged F1 across classes |
| `P_macro` / `R_macro` | Macro-averaged precision / recall |
| `BAcc_macro` | Balanced accuracy |
| `AUC_macro` / `AUPRC_macro` | Macro-averaged one-vs-rest AUROC / AUPRC |
| `loss/edge` | Cross-entropy loss for neural models; unavailable for Random Forest (`-` in the final summary) |

---

## Configuration Reference

### TNNTrainConfig (dense pipeline)

Representative fields:

```python
@dataclass
class TNNTrainConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-2
    epochs: int = 10
    batch_size: int = 16
    grad_clip: float = 0.0            # 0 disables
    early_stop_patience: int = 0      # 0 disables
    use_mask_channel: Optional[bool] = None   # None → infer per model
    supervised_redaction_policy: str = "adj_only"
    threshold_metric: str = "f1"      # "f1" | "bacc" (tuned on the validation split)
    select_by: str = "f1"             # "f1" | "bacc" | "auroc"
    tx_force_adj_channel: bool = True
    tx_patch_overlap: bool = False     # False: stride=patch; True: 50% overlap
    save_dir: Optional[str] = "saved_checkpoints"
    # ... plus model-specific hyperparameters (mlp_hidden, cnn_hidden, tx_*, etc.)
```

### GNNTrainConfig (GNN pipeline)

Representative fields:

```python
@dataclass
class GNNTrainConfig:
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    batch_size: int = 16
    hidden: int = 128
    layers: int = 3
    heads: int = 4
    dropout: float = 0.10
    dropedge_p: float = 0.10
    lap_pe_k: int = 0
    gps_lap_pe_k: int = 16
    gps_lap_pe_sign_flip: bool = True
    gps_rwse_steps: int = 16
    gnn_zero_supervised: bool = False
    learnable_layer_norm: bool = True
    scheduler: str = "cosine"         # "none" | "cosine"
    neg_pos_ratio: Optional[float] = None
    use_tree_aux_loss: bool = False
    save_dir: Optional[str] = "saved_checkpoints"
```

Notes:

- GNN threshold tuning and checkpoint selection are always F1-based. Dense-only knobs such as `threshold_metric` and `select_by` do not apply.
- `grad_clip=1.0` keeps gradient clipping enabled by default in the GNN pipeline; set `grad_clip=0.0` to disable it.
- `learnable_layer_norm=False` switches the GNN encoder LayerNorm modules to the non-affine variant.
- `lap_pe_k` is the general LapPE request in both GNN runners, and GPS additionally enforces at least `gps_lap_pe_k` LapPE columns. GPS also uses `gps_lap_pe_sign_flip` and `gps_rwse_steps`.
- `gps_rwse_steps` controls GPS-owned random-walk structural encodings. These are not part of `hooks.feature_set` and are not requested by `feature_set=True`.
- `gnn_zero_supervised` controls adjacency redaction uniformly across GNN encoders; GPS derives its model-owned structural processing from that same model-visible adjacency.
- `neg_pos_ratio` controls negative-to-positive sampling ratio for binary GNN training. `None` (default) disables ratio-based sampling and uses `pos_weight` instead.
- `use_tree_aux_loss=True` adds an auxiliary penalty pulling the expected number of predicted edges toward `N-1`. It does not detect cycles or enforce connectivity. Only meaningful for spanning-tree tasks. Automatically disabled in Scalable Mode.
- Summary display formatting is set on the runner call itself, e.g. `run_gnn_suite(..., display_decimals=6, display_truncate=True)`.

---

## Feature System

### Feature flow

```text
hooks.feature_set (user request)
  → _normalise_feature_set()       [expand macros, deduplicate, validate]
  → feature_keys                   [never includes 'adj']

feature_keys
  → dense BCHW assembly            [task features + runner-managed structural inputs]
  → model input tensor             [channels selected by the runner]

feature_keys
  → GNN feature_schema             [requested keys, typed into node_keys / edge_keys]
  → full-matrix GNN assembly       [may append heavy pairwise structural keys]
  → GPS-owned PE/SE                [LapPE minimum + optional RWSE, controlled by GNNTrainConfig]
  → Scalable Mode                  [encode graph, compute structural features only for selected pairs]
```

Notes:

- In the dense pipeline, adjacency-derived pairwise channels are computed during BCHW assembly from the supervised-redacted observed adjacency.
- The dense runner prevents empty model inputs by adding the default dense structural feature set when needed: `degree`, `deg_row`, `deg_col`, `clustering_coeff`, `cn`, `jaccard`, `adamic_adar`.
- In the full-matrix GNN path, edge matrices may include appended heavy structural pairwise features even if they were not explicitly requested as standalone edge tensors.
- `pairwise_batch_from_adj(...)` returns keys in its own fixed helper insertion order, not the caller's requested key order.
- In Scalable Mode, node features discovered by untargeted schema inference continue to flow through the encoder. Untargeted discovery scans at most 64 batches total across the resolved loaders, so features that first appear after that scan window are not added to the schema. The decoder does not materialise a dense `(N, N, Fe)` edge-feature tensor. Instead, it computes its fixed structural pair-feature vector only for the supervised pairs selected for scoring. User-supplied decoder-side edge features outside that fixed structural set do not reach the Scalable Mode decoder.

### Adjacency as input channel

| Setting | Scope | Effect |
|---------|-------|--------|
| `hooks.allow_adj_channel=True` | All dense models | Include `adj` for all models |
| `cfg.tx_force_adj_channel=True` | Transformer only | Force `adj` for TX even if hooks say no |

When `hooks.allow_adj_channel=False`, dense no-feature mode is unavailable. In that case, `run_pipeline_for_task(...)` temporarily uses the default dense structural feature set during loader resolution, then restores the caller's original `task.hooks.feature_set`.

GNN encoders never consume `adj` as a feature channel — they receive it as the propagation matrix.

### Supervised redaction policy

Controls what gets zeroed at supervised `(i, j)` positions in BCHW inputs:

| Policy | Behaviour |
|--------|----------|
| `"adj_only"` (default) | Zero the `adj` channel; derived channels reflect that redaction |
| `"all"` | Zero every applicable input channel at task-mask positions |
| `"none"` | No redaction |

#### GNN-side redaction

The GNN pipeline uses a separate flag, `cfg.gnn_zero_supervised`, which controls whether supervised edges are zeroed in the adjacency matrix before message passing.

| Value | Behaviour |
|-------|----------|
| `False` (default) | No redaction; the GNN encoder sees the full observed adjacency |
| `True` | Zeros the adjacency at supervised `(i, j)` positions before encoding, functionally equivalent to `supervised_redaction_policy="adj_only"` on the dense side |

`gnn_zero_supervised` controls the shared adjacency path for every GNN encoder. GPS computes its LapPE/RWSE and local message-passing topology from the same adjacency visible to the model. This does not globally blank user-supplied edge feature tensors.

### Existing-edge-only evaluation

`task.eval_on_existing_edges_only` controls whether non-edges are scored and included in the training loss.

| Value | Behaviour |
|-------|----------|
| `False` (default) | All `(i, j)` pairs selected by `effective_mask` contribute to the loss, including non-edges |
| `True` | Only pairs where the observed adjacency is non-zero are scored; non-edges are excluded from the loss |

Use `True` for tasks where the objective operates exclusively on existing edges (e.g. edge deletion or edge property classification) and scoring non-edges would be meaningless. During inference, non-edges can still be scored, but the model will not have been trained on them.

### Custom features

Canonical and custom features follow different ownership rules:

- **Canonical features** are pipeline-owned and remain plain strings in `hooks.feature_set`. If requested, the pipeline makes them available.
- **Custom features** are user-owned. Declare each custom name and its type directly in `hooks.feature_set` as `(name, "node")` or `(name, "edge")`, and provide its tensor in the sample feature dictionaries wherever it is used.
- Custom feature types are part of the task declaration and therefore do not depend on run-scoped registration state.
- `adj`, `mask`, and `_N` are reserved pipeline names and cannot be declared as custom features. Canonical feature names also cannot be redeclared as custom.
- In the dense pipeline, custom feature names must not collide with channel names generated by another requested feature. For example, `"degree"` generates `degree_row` and `degree_col`, so combining it with a custom feature named `degree_row` is invalid.

Example:

```python
hooks = TaskHooks(
    feature_set=[
        "degree",
        "cn",
        ("my_node_feat", "node"),
        ("my_edge_feat", "edge"),
    ]
)
```

The custom type is mandatory; custom node/edge semantics are not inferred from a square tensor. Canonical strings, macros, `True`, and `False` retain their existing meaning.

For supported pre-divided datasets, requested canonical features are treated as split-level pipeline outputs. Custom features remain user-supplied and may be present only where the user provides them.

Non-square 2D inputs are treated as node features when no explicit custom type is available. `(F, N)` node inputs are transposed to `(N, F)` when the second dimension matches the graph dimension.

### Feature macros

| Macro | Expands to |
|-------|-----------|
| `"powers"` | `power_2`, `power_3`, `power_4`, `power_5` |
| `"endpoint_degree"` | `deg_row`, `deg_col` |

`twohop` is a 1D node feature counting unique nodes at exactly graph distance 2 (excluding the source and its direct neighbours; outward reachability when directed). `power_2` is the distinct pairwise length-2 walk count.

`"degree"` is the actual 1D node-degree feature (row/out-degree under the directed row-wise convention). Use `"endpoint_degree"` (or explicit `deg_row`, `deg_col`) for the pairwise endpoint-degree channels.

---

## Masking System

Three distinct mask concepts exist. Confusing them is a common source of errors.

### (A) Supervision / Evaluation mask

Built by `effective_mask(mask, A, directed)`. Controls which `(i, j)` pairs contribute to loss and metrics.

- **Undirected**: strict upper-triangle off-diagonal only (each pair evaluated once). Asymmetric masks are rejected.
- **Directed**: full matrix; diagonal `(i, i)` included only where `A[i, i] > 0`.
- Always used. Independent of input features.

Label validity contract on supported paths:

- This pipeline does **not** use ignore-index labels for padding or masking.
- Padding is represented by `mask == False`, not by a sentinel target such as `-1`.
- `collate_fn_pad(...)` pads `L` with `0` and pads `mask` with `False`.
- For multiclass tasks, label matrices must contain finite integer class IDs in `[0, num_classes - 1]`. Invalid values are rejected during collation.

### (B) Transformer token-keep mask

Passed as `_task_mask` to `PatchTransformer.forward()`. Decides which spatial patches become tokens based on `token_policy`:

- `"keep_all"`: all patches are tokens (default).
- `"from_mask"`: only patches overlapping `True` mask entries.
- `"auto"`: adaptive based on `min_keep_ratio`.

Not an input feature. The full patch grid is always embedded and passed to the encoder. The policy controls which patches are attended to, via a key–padding mask. Token count depends only on graph size and patch size, so these policies do not reduce the transformer's sequence length.

### (C) Mask input channel

A BCHW feature channel named `"mask"`, controlled by `cfg.use_mask_channel`:

- `None` → inferred per model: CNN/Transformer = `True`, MLP/RF = `False`
- `True` / `False` → explicit override for the neural models; RF never takes the `"mask"` channel and ignores this setting

This is a model input, not a supervision signal.

---

## Data Flow

### Task → Loaders

```text
ProvidedSplitsTask._build_loaders()
  │
  ├─ bench.splits is dict of (N, N) masks?
  │    → _build_single_graph_loaders_from_bench (single-graph path)
  │
  ├─ bench.splits is dict of sample collections?
  │    → Preserve split membership
  │    → Auto-derive any requested canonical features for the split
  │    → Wrap with mask_policy
  │
  └─ No splits?
       → first loader resolution in a run:
            bench.sample_specs → bench.generate_dataset → build loaders → cache on task
       → later compatible loader resolutions in the same run:
            reuse cached loaders directly
       → next top-level run:
            clear run cache and regenerate for auto-generated tasks

single-graph task exposing task.bench + task.hooks
  │
  └─ _resolve_loaders(...)
       → _coerce_bench(...)
       → _build_single_graph_loaders_from_bench(...)
```

### Loader → Model (Dense)

```text
collate_fn_pad(batch)
  → (A_batch, F_list, L_batch, M_batch)  [padded to N_max]

train_and_eval_one_model(...)
  → registry.fit(train_loader, feature_keys)              [compute channel stats]
  → forward_logits_common(A, feats, mask, ...)
       → registry.stack_channels_BCHW(...)                [assemble + redact]
       → x_bchw = x_bchw[:, keep_idx, :, :]               [slice to effective channels]
       → x_bchw_std = registry.standardise_bchw(...)      [apply train stats to kept channels]
       → model(x_bchw_std)                                [or model(x_bchw_std, _task_mask=m) for TX]
```

### Loader → Model (GNN, full-matrix)

```text
collate_fn_pad(batch)
  → (A_batch, F_list, L_batch, M_batch)

train_one_gnn(...)
  → _infer_feature_schema([train, val, test], requested_keys=feature_keys)
  → _redact_supervised_edges(A, mask, ...)        [zero supervised in A]
  → _prepare_features_batch(A, feats, mask, ...)  [per-graph X, E, node_mask]
       → _assemble_features_for_graph(...)        [build X(N, F), E(N, N, Fe)]
            → zscore_nodes_per_graph(X)
            → zscore_edges_per_graph(E)
  → GPS only: _prepare_gps_local_edge_features(...) [local message-passing edge features]
  → model.enc.forward(A, X_batch, node_mask)        [encode → Z_batch]
  → model.dec.forward(Z_batch, node_mask, E_batch)  [decode → logits]
```

### Loader → Model (GNN, Scalable Mode)

```text
run_gnn_edges_suite(...)
  → _infer_feature_schema(...)
  → _redact_supervised_edges(A, mask, ...)
  → _prepare_features_batch(..., build_edge_mats=False, append_pairwise=False)
  → model.encode_only(...)                 [encode the graph once]
  → effective_mask(...)                    [select supervised pairs]
  → optional neg_pos_ratio sampling        [binary tasks only]
  → model.score_pairs_on_demand(...)       [compute pair features and score selected pairs]
```

`run_gnn_edges_suite(...)` is the dedicated Scalable Mode runner and forces `batch_size=1`, so one graph is processed at a time. A task may still contain multiple graphs, which are processed sequentially. The graph encoder operates on the graph as a whole. Memory savings come from avoiding the dense decoder-side edge feature representation and computing structural decoder features only for the selected supervised pairs.

---

## Output Format

### Bundle structure

All three runner entry points return a `bundle` dictionary:

```python
{
    "results": {
        "<model_key>": {
            "val":  { "f1": ..., "precision": ..., "recall": ..., ..., "_prob": [...], "_y": [...] },
            "test": { "f1": ..., "precision": ..., "recall": ..., ..., "_prob": [...], "_y": [...] },
            "thr":  0.42,           # tuned threshold (binary) or None (multiclass)
            "ckpt": "saved_checkpoints/task_name/YYYYMMDD_HHMMSS/model_key.pth",
        },
        # ...
    },
    "metadata": {
        "task": { "name": "...", "directed": "...", },
        # ...
    }
}
```

For non-empty evaluated splits, metrics dicts also carry `_prob` (per-pair predicted probabilities) and `_y` (per-pair integer labels) for downstream analysis. Empty splits or splits with no supervised pairs may instead return only the applicable `NaN` summary metrics and may omit `_prob` / `_y`. GNN validation with no finite selectable result may use the minimal `{"f1": NaN}` fallback. These forms are intentional and do not represent valid zero-valued metrics. The most recent bundle is also attached to the task as `task._latest_results` for the duration of the run. It is cleared when the run is finalised. In a combined dense → GNN run, both stages write into the same bundle, so `results` contains keys for all models across both pipelines.

### Checkpoint format

```python
{
    "state_dict": { },    # model weights
    "meta": {
        "model_key": "str",
        "best_val_threshold": "float | None",
        "task": { },
        "cfg": { },
        # pipeline-specific fields may also be present
    }
}
```

Saved to `saved_checkpoints/<task.name>/<timestamp>/<model_key>.pth` via `save_pipeline_checkpoint(...)`.

Common dense-pipeline metadata fields include: `manifest`, `use_mask_channel`, `directed`, `supervised_redaction_policy`, `edges_only`, `seed`, `feature_keys`, `keep_idx`, `eff_in_ch`.

Common GNN-pipeline metadata fields include: `directed`, `feature_schema`, `pairwise_on_demand`.

The exact `meta` payload is pipeline-specific; only `state_dict`, `model_key`, `task`, and the nested `cfg` subset should be treated as universally stable.

---

## Reproducibility

```python
task = ProvidedSplitsTask(..., seed=42)
# Internally calls seed_everything(42), which locks:
#   random, np.random, torch.manual_seed, CUDA seeds, cuDNN deterministic mode
```

If no seed is provided, a random seed is generated and printed. The seed controls graph generation order, dataset splitting, dataloader shuffling, and model initialisation.

Reproducibility follows the documented run sequence. Constructing a task seeds the relevant random-number generators; starting another top-level run with the same existing task does not rewind those generators to their initial state. Repeating the same execution sequence with the same seed reproduces the same sequence of runs.

---

## GPU and Mixed Precision

The pipeline auto-detects CUDA availability:

- **GPU present**: training and evaluation run on CUDA with automatic mixed precision (bfloat16 where supported, float16 otherwise) via `torch.amp`. DataLoader `pin_memory` is enabled by default.
- **CPU only**: fully supported; mixed precision is disabled, all computation runs in float32.

No configuration is needed — hardware detection is automatic.

---

## Key Design Decisions

The following are **intentional** and should not be flagged as bugs, inconsistencies, or dead code.

### Two pipelines, two normalisation strategies

| Aspect | Dense pipeline | GNN pipeline |
|--------|----------------|--------------|
| Normalisation | Dataset-level channel stats (fit on train) | Per-graph z-score |
| Input format | BCHW tensor (all graphs padded to `N_max`) | Per-graph lists of `(N, F)` and `(N, N, Fe)` |
| Adjacency redaction | Inside `stack_channels_BCHW` during channel assembly | Via `_redact_supervised_edges` before encoding |

The redaction logic is duplicated across pipelines because it integrates at different architectural points. The dense path redacts during BCHW assembly (where it can also redact derived channels), while the GNN path redacts the raw adjacency before message passing.

### Threshold tuning and checkpoint selection

The dense pipeline exposes `cfg.threshold_metric` and `cfg.select_by` for configurable threshold tuning and checkpoint selection. The GNN pipeline uses a fixed validation-F1 policy for both. This is intentional and should not be flagged as an inconsistency.

### Encoder normalisation in the GNN suite

All GNN encoders use LayerNorm. The single knob `cfg.learnable_layer_norm` controls whether those LayerNorm modules use learnable affine parameters (`True` = standard, `False` = non-affine). This setting is shared across the GNN suite for comparability and does not apply to the dense pipeline. GPS also owns additional structural encoding knobs for LapPE and RWSE.

### Scalable Mode uses a reduced decoder feature path

Scalable Mode is the memory-saving GNN execution strategy used by `run_gnn_edges_suite(...)`. In the standard full-matrix GNN path, decoder-side edge features may be represented as a dense `(N, N, Fe)` tensor. Scalable Mode avoids materialising that tensor.

Technically:
- The graph is first encoded to produce node embeddings. The normal effective mask then selects the supervised `(i, j)` pairs. For binary tasks, `neg_pos_ratio` may optionally reduce the selected negative pairs before scoring.
- The decoder computes its structural inputs only for those selected pairs. Its fixed structural feature vector contains endpoint degree statistics together with common-neighbour, Jaccard, and Adamic–Adar statistics derived from the adjacency visible to the model.
- Node features, including typed custom node features, flow through the encoder. User-supplied edge features outside the fixed Scalable Mode structural set do not reach the decoder.

Scalable Mode forces `batch_size=1`, so it processes one graph at a time. This does not restrict the dataset to a single graph. `use_tree_aux_loss=True` is disabled in Scalable Mode. GPS uses `gps_lap_pe_k`, `gps_lap_pe_sign_flip`, and `gps_rwse_steps` as encoder-side structural encodings in both runners.

Full-matrix GPS also uses the assembled edge-feature channels for its local GINE-style message passing, normalised over the observed message-passing edges. Those local attributes are stored only for observed edges; the dense `(N, N, Fe)` tensor remains the decoder representation. Scalable Mode skips edge-matrix assembly before `encode_only()`, so its GPS local branch is constructed with `edge_dim=0`.

These differences from full-matrix decoding are intentional and should not be reported as inconsistencies unless observed behaviour contradicts this contract.

### Eval loss vs training loss

For neural models, training computes `pos_weight` per-batch for numerical stability. Evaluation recomputes a single global `pos_weight` across the full split for a comparable, stable metric. The reported `loss/edge` is therefore not identical to the average of per-batch training losses. Random Forest has no corresponding `loss/edge` and displays `-` in the final summary.

### `run_pipeline_for_task` rejects GNN keys

Passing `sage`, `gcn`, `gin`, `edge_tx`, or `gps` to `run_pipeline_for_task` prints a warning and skips that key; the remaining dense models still train. GNN models require different preprocessing and must be run through `run_gnn_suite` or `run_gnn_edges_suite`.

### Loader caching within a run

For `ProvidedSplitsTask`, the supported entry points resolve loaders via `_resolve_loaders`, which caches the `(train_loader, val_loader, test_loader)` tuple for the active run. Later attached stages reuse the same DataLoader objects when the requested configuration is compatible. A new top-level run clears that cache.

### Task wrapping and `base_task` traversal

The pipeline supports task wrapping. Some internal utilities inspect `base_task` when the top-level task object does not explicitly define a required property. Metadata serialisation is taken from the top-level task object used by the runner.

---

## Code Review Scope

When reviewing this codebase, only report findings that match the requested review categories. The standard categories and what qualifies for each:

### 1. Bugs

Code that will crash or produce silently wrong results **on a supported code path** (see "Getting Started" above). Verify by tracing the full call chain from an entry point — if the code is only reachable through unsupported direct instantiation of internal classes or unsupported task shapes, it is not a bug in the pipeline.

Before reporting: read the target code twice. Check for guards (`torch.where`, `.clamp_min`, conditional branches) that may already handle the case you are about to flag.

### 2. Redundant code

Code that performs identical work twice or can be removed or simplified without changing behaviour. This includes unnecessary type casts, duplicate computations, and re-derivations of values already available in scope. Suboptimal code where a logically equivalent, more efficient implementation exists also qualifies as redundancy. For hot-path performance claims that would change encoder adjacency routing, feature assembly, or training-loop behaviour, report them as findings only when supported by profiling or concrete measurements on a supported code path; otherwise treat them as non-blocking hypotheses, not recommended changes.

### 3. Inconsistent behaviour

Two supported code paths that should produce identical results for identical inputs but do not. Before reporting: check whether the difference is documented in "Key Design Decisions" above — the dense and GNN pipelines intentionally differ in normalisation, threshold tuning, redaction integration, and the full-matrix versus Scalable Mode decoder path.

### 4. Dead code

Code that is unreachable from any supported entry point, or code that only functions under unsupported usage patterns. This includes conditional branches that can never evaluate to true on supported paths. Before reporting: check whether the code is forward-looking scaffolding documented in "Key Design Decisions."

### 5. Internal cleanup and other unclassified issues

Attribute initialisation, naming, or structural issues that affect maintainability. Style preferences (staticmethod vs instance method, import organisation, defensive no-op calls) do not qualify.

### Out of scope

The following do not qualify under any category and should not be reported:

- **Style preferences**: staticmethod vs instance method, import grouping, variable naming conventions.
- **Used imports**: every import in this codebase is used. An import used in only one function is not "narrowly scoped" or "dead."
- **Defensive calls**: `gc.collect()`, `torch.cuda.ipc_collect()`, `torch.cuda.synchronize()` exist as safety nets and are not dead code.
- **Numerical safety nets in forward paths**: `torch.nan_to_num`, `torch.clamp`, and input bounding in encoder or decoder forward methods guard against silent NaN or Inf propagation.
- **Code organisation choices**: whether two runners with shared setup should be merged into one parameterised function are structural preferences.

### Common false-positive patterns (do not report)

- **Flagging documented runtime lifecycle behaviour as a bug.** State that is explicitly owned by a run (for example, summary bookkeeping or other pipeline runtime state reset at run finalisation) should not be reported as a defect unless the behaviour contradicts the documented run semantics.
- **Flagging the behaviour a parameter controls as a bug.** If a configurable flag changes pipeline behaviour, that behaviour is the flag's purpose, not a defect.
- **"Pipeline A does X but pipeline B does Y."** Check "Key Design Decisions" first. The dense and GNN pipelines intentionally differ in normalisation, redaction integration, threshold policy, and feature assembly.
- **Intermediate state that gets overwritten.** Trace the full call chain from the supported entry point to the point of use. If a value is set early and unconditionally overwritten before it is consumed, the early value is not a bug.
- **Redundancy that serves a distinct purpose at each site.** Two operations that look identical may operate on different data (pre- vs post-standardisation, raw vs z-scored). Verify the numeric values are actually identical before reporting redundancy.
- **"Runner A does X but Runner B does Y" within the same pipeline.** The three supported entry points serve architecturally different strategies. Each runner's policy choices are specific to its strategy.
- **`getattr` fallback values on attributes that `__init__` always sets.** If a constructor unconditionally assigns `self.x = value`, a downstream `getattr(self, "x", fallback)` is defensive.
- **Issues behind active guards.** If a guard disables a feature for a particular mode, hypothetical issues within the guarded-off code are not bugs.
- **Assuming a called function cannot handle an observed input.** If a utility is called with a given shape on multiple exercised supported paths without issues, it handles that shape. Do not flag it as a bug without evidence of failure.

---

## Limitations and Future Work

### Unsupported usage

Only the documented runner entry points and task shapes are supported. Direct instantiation of internal classes, calling underscore-prefixed helpers, invoking individual dataloader methods, or relying on direct task-level `train_dataloader` / `val_dataloader` / `test_dataloader` entry points is unsupported and may crash, rebuild expensive state, or produce incorrect results. Internal components are optimised for pipeline-owned, already-sanitised inputs and do not contain defensive guards for unexpected shapes, sparse/dense mismatches, or missing masks.

### Internal components

| Component | Why it is internal |
|-----------|--------------------|
| `_GCNEncoder`, `_SAGEEncoder`, `_GINEncoder`, `_EdgeMaskedTransformer`, `_GPSEncoder` | Expect pre-normalised adjacency and feature tensors |
| `_EdgeFeatureDecoder` | Expects node embeddings from a compatible encoder |
| `GraphEdgeClassifier` | Runtime state (`feature_schema`, `dropedge_p`, `lap_pe_k`, `learnable_layer_norm`, pairwise-mode flags) is injected by the GNN runners after construction; direct instantiation is unsupported |
| `_eval_split` (`EdgeClassification`) | Coupled to registry state and `keep_idx` from training setup |
| `_gnn_eval_split` | Coupled to model attributes set during training |
| `_build_single_graph_loaders_from_bench` | Assumes `_coerce_bench` has validated inputs |
| `GraphBenchmark._compile_label_fn` | Assumes parameter names follow alias conventions |

### Feature requests and possible extensions

The following are **not currently supported**. If you need one, open an issue and include a task file showing why the feature is required and which supported runner path it affects.

**Training and evaluation**

- **Stronger GNN supervision redaction.** `gnn_zero_supervised` redacts the shared adjacency path but does not automatically blank all supervised edge-feature tensors.
- **Early stopping for the GNN pipeline.** `TNNTrainConfig` exposes `early_stop_patience`; `GNNTrainConfig` does not currently provide an equivalent.
- **Configurable leading metric for the GNN pipeline.** The GNN pipeline is currently fixed to validation F1 for threshold tuning and checkpoint selection.
- **Configurable self-loop supervision policy.** Undirected evaluation excludes diagonal pairs; directed evaluation includes them only where `A[i, i] > 0`.

**Data support and graph generation**

- **Weighted / non-binary adjacency support.** The pipeline currently assumes binary adjacency matrices.
- **Generated self-loops.** User-provided datasets may contain self-loops, but the built-in generator does not produce them.
- **User-registered graph families.** A future extension could let users register new graph families from the task side.
- **Spawn-based DataLoader multiprocessing portability.** Pipeline-owned dataset wrappers are currently designed for the normal fork-based multiprocessing path used by typical Linux training environments. Full support for spawn-based worker environments may require module-level dataset wrappers and explicitly serialisable worker state.

**Losses and extensibility**

- **User-defined auxiliary loss components.** Task-specific loss logic currently lives in the shared pipeline.
- **Custom decoder-side edge features in Scalable Mode.** The Scalable Mode decoder uses a fixed structural pair-feature representation. Arbitrary user-supplied edge-feature tensors outside that structural set do not reach the decoder.
- **Batched graph processing in Scalable Mode.** Scalable Mode currently processes one graph at a time and forces `batch_size=1`.
- **Dense support for multi-column custom node features.** 1D node features are currently expanded to paired row/col channels; multi-column `(N, F)` custom node features are not yet supported.

**Validation strictness and limits**

- **Pipeline-level strict mode.** Some malformed inputs currently warn and continue where feasible.
- **Per-feature strictness controls for dynamic padding.**
- **Configurable RF edge cap.** The RF path uses a fixed maximum of 20,000,000 edges. Its capped training collector preallocates the reservoir at that maximum so replacement sampling remains one-pass and in-place. Small training splits reserve the corresponding array address space.

---

## Contributing Tasks and Feature Requests

This benchmark is intended to grow as a community resource. We welcome contributions that add new graph transformation tasks or request task-enabling features that the current pipeline cannot yet express.

For task contributions, please include the task motivation, graph type, input-output relation, structural change type, expected output shape, label semantics, evaluation mask semantics, and whether the task runs through a supported task shape today.

For feature requests, please tie the request to a concrete task or task family. Explain why the current interface cannot express the task, which runner path it affects, and whether existing tasks should continue to behave unchanged.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full task checklist and feature-request guidance.

---

## File Structure

```text
pipeline/EdgeClassification.py      Dense pipeline: collation, FeatureRegistry, models, training, RF, summary
pipeline/GNNBridge.py               GNN pipeline: encoders, decoder, feature assembly, training
pipeline/GraphBenchmark.py          Graph generation, dataset building, splitting
pipeline/_utils/features.py         Shared pairwise feature math (cn, jaccard, adamic_adar, etc.)
pipeline/_utils/run_lifecycle.py    Run ownership, run boundaries, final summary triggering, runtime-state cleanup
tasks/                              Runnable task modules; run from the repo root with python -m tasks.<name>
task_utils/                         Dataset builders and task-level utility scripts
data/                               Task-owned datasets and reusable caches
saved_checkpoints/                  Generated model checkpoints grouped by task and run timestamp
requirements.txt                    Python dependency list
LICENSE.md                          Creative Commons Attribution-ShareAlike 4.0 International license
CONTRIBUTING.md                     Task contribution checklist and task-driven feature-request guidance
```

---

## Review Checklist for Future Changes

When modifying the codebase, verify these before treating a change as complete:

1. **Supported entry points still work**
   - `run_pipeline_for_task`
   - `run_gnn_suite`
   - `run_gnn_edges_suite`

2. **Empty-split behaviour stays explicit**
   - No-evaluation splits should surface as `NaN` metrics, not as valid zero-valued metrics that can accidentally win checkpoint selection.

3. **Feature-shape inference matches runtime assembly**
   - Dense `registry.manifest` and runtime channel expansion must agree.
   - GNN `feature_schema` and runtime node/edge assembly must agree.
   - GNN schema inference scans all three loaders for requested keys. Changes to loader construction or feature dict population must not silently remove keys from only one split.

4. **Scalable Mode behaviour stays documented**
   - The decoder uses the fixed Scalable Mode structural pair-feature representation
   - User-supplied edge features outside that structural set do not reach the Scalable Mode decoder
   - Node features, including custom node features, continue to flow through the encoder via schema inference
   - Scalable Mode continues to enforce `batch_size=1`
   - `run_gnn_edges_suite(...)` continues to follow the documented Scalable Mode behaviour
   - GPS `gps_lap_pe_k`, `gps_lap_pe_sign_flip`, and `gps_rwse_steps` remain active in `run_gnn_edges_suite(...)`
   - `use_tree_aux_loss` remains disabled in Scalable Mode

5. **Run finalisation and state reset remain correct across sequential tasks**
   - Standalone dense runs finalise cleanly without manual reset.
   - Standalone GNN runs finalise cleanly without manual reset.
   - Dense → GNN combined runs produce one merged final summary table.
   - A second top-level runner call finalises the previous run before starting a new one.
   - `quiet=True` suppresses printing only and does not affect cleanup.

6. **Checkpoint metadata stays intentional**
   - Shared keys remain stable.
   - Pipeline-specific keys are updated in the README if their semantics change.

---

## Citation

If you use this framework, or any benchmark task distributed with it, in a publication, you must cite:

> Machowczyk, A., Heckel, R. (2026). Benchmark First: Defining Tasks for Graph Transformation Learning. In: Archibald, B., Semeráth, O. (eds) Graph Transformation. ICGT 2026. Lecture Notes in Computer Science, vol 16624. Springer, Cham. [https://doi.org/10.1007/978-3-032-29730-3_13](https://doi.org/10.1007/978-3-032-29730-3_13).

## Licence

This project is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).

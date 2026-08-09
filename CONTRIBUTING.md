# Contributing Tasks and Feature Requests

Thank you for your interest in contributing.

This project is intended to grow as a community benchmark for graph-to-graph learning. New task ideas are very welcome, including ideas that the current pipeline cannot yet express.

You do not need a finished implementation or a complete task classification to start a discussion. A short proposal describing the graph transformation you want to study is enough. The taxonomy and checklists below are there to help develop and compare ideas.

We are especially interested in:

1. New benchmark tasks
2. Feature requests that would make new benchmark tasks possible

---

## 1. Contributing a New Task

A task can be proposed at any stage, from an early idea to a complete runnable implementation.

As the proposal develops, the following information helps make the task easier to understand, reproduce, evaluate, and compare with other tasks in the benchmark.

### Task identity

Helpful details include:

- Task name
- Short description
- Motivation
- Relevant paper, dataset, or algorithmic reference, if applicable
- Whether the task is synthetic, dataset-backed, or both

Example:

```text
Task name: Symmetric Closure Completion

Description:
Given a directed graph, predict which reverse edges must be added so that every existing edge has a corresponding inverse edge.

Motivation:
Tests whether a model can learn a structural relation from adjacency patterns.
```

---

## 2. Task Classification Guide

The benchmark taxonomy is provided to make tasks easier to describe and compare. Use the parts that are relevant or known when you submit the idea; classifications can be refined during discussion.

### Complexity

If known, note the expected complexity of the task.

Examples:

- Linear
- Polynomial
- NP-hard
- NP-complete
- Unknown
- Depends on variant

If the complexity is unknown or unclear, that is fine; `Unknown` is a valid starting point.

```text
Complexity:
Polynomial
```

### Input-output relation

If known, describe the relation between the input graph and valid output graph.

Useful categories include:

- Deterministic
- Nondeterministic
- Optimisation
- Heuristic optimisation
- Probabilistic
- Unknown or proposed extension

```text
Input-output relation:
Deterministic
```

### Graph type

Describe the graph type required by the task as far as you know.

Relevant details may include:

- Directed or undirected
- Simple graph, multigraph, hypergraph, or other
- Homogeneous, bipartite, heterogeneous, typed, labelled, or attributed
- Whether node features are required
- Whether edge features are required
- Whether self-loops are required
- Whether weighted adjacency is required

```text
Graph type:
Directed simple graph with optional edge attributes
```

### Structural change type

Describe what the task requires the model to predict.

The existing categories include:

- Edge structure completion: add only
- Edge structure completion: delete only
- Edge structure completion: add and delete
- Edge attribute computation: classification
- Edge attribute computation: regression
- Node attribute computation: classification
- Node attribute computation: regression
- Node creation
- Node deletion
- Node-edge co-transformation
- Other

```text
Structural change:
Edge structure completion: add only
```

---

## 3. Current Pipeline Compatibility

If you know whether the task is supported by the current pipeline, note that here. It is also completely valid to propose a task first and work out pipeline compatibility during discussion.

The current implementation is strongest for edge classification tasks. A task is more likely to run without framework changes if it can be represented as:

- A graph adjacency matrix
- A label matrix aligned with the graph
- An optional evaluation mask
- Optional node or edge features
- A supported task object using the documented runner paths

Supported task shapes are:

1. `ProvidedSplitsTask` or a subclass
2. A single-graph task exposing `task.bench` and `task.hooks`

Direct task-provided `train_dataloader`, `val_dataloader`, or `test_dataloader` methods are not supported.

If useful, you can describe the current status with one of the following labels:

```text
Pipeline status:
Runs on the current pipeline
```

```text
Pipeline status:
Requires a feature request
```

```text
Pipeline status:
Conceptual task proposal only
```

---

## 4. If You Have an Implementation

If you already have a task running on the current pipeline, a minimal runnable task file is very helpful.

A typical task file may include:

- A `label_fn`
- A `TaskHooks` configuration
- A `ProvidedSplitsTask` or supported single-graph task setup
- A clear `directed=True` or `directed=False` setting
- Feature requests through `feature_set`
- Split settings
- A fixed seed, where possible
- At least one supported runner call

A minimal task can follow this shape:

```python
import numpy as np

from EdgeClassification import (
    TaskHooks,
    ProvidedSplitsTask,
    TNNTrainConfig,
    run_pipeline_for_task,
)

from gnn_bridge import (
    GNNTrainConfig,
    run_gnn_suite,
)


def label_fn(A_obs: np.ndarray) -> np.ndarray:
    """
    Return a label matrix aligned with A_obs.

    The returned matrix must have shape (N, N).
    """
    raise NotImplementedError("Replace with task-specific label logic")


hooks = TaskHooks(
    label_fn=label_fn,
    feature_set=[
        # Add the canonical or custom features required by the task.
    ],
    allow_adj_channel=True,
)

task = ProvidedSplitsTask(
    name="my_task",
    directed=False,
    hooks=hooks,
    num_graphs=100,
    min_nodes=8,
    max_nodes=32,
    ratios=(0.7, 0.2, 0.1),
    seed=42,
)

dense_cfg = TNNTrainConfig(
    epochs=10,
    lr=3e-4,
    batch_size=16,
)

dense_results = run_pipeline_for_task(
    task,
    models=["mlp", "deep_mlp", "cnn", "transformer", "rf"],
    cfg=dense_cfg,
)

gnn_cfg = GNNTrainConfig(
    epochs=40,
    lr=3e-4,
    batch_size=16,
)

gnn_results = run_gnn_suite(
    task=task,
    encoders=("gcn", "sage", "gin", "edge_tx"),
    cfg=gnn_cfg,
)
```

---

## 5. Optional Task Submission Guide

Use this checklist as a guide when developing a task proposal. Include whatever is known or relevant; you do not need every item before opening a discussion:

- [ ] Task name
- [ ] Short task description
- [ ] Motivation for including the task
- [ ] Complexity classification, if known
- [ ] Input-output relation classification
- [ ] Graph type classification
- [ ] Structural change classification
- [ ] Whether the task is deterministic, nondeterministic, optimisation-based, heuristic, or probabilistic
- [ ] Whether the task is synthetic, dataset-backed, or both
- [ ] Whether the task runs on the current pipeline
- [ ] A runnable task file, if currently supported
- [ ] A feature request, if not currently supported
- [ ] Dataset source and licence, if using external data
- [ ] Reproduction notes, including seed and split policy
- [ ] Expected output shape and label semantics
- [ ] Evaluation mask semantics, if using a custom mask

---

## 6. Dataset-Backed Tasks

If your task uses an external dataset, the following details are especially helpful where available:

- Dataset name
- Dataset source
- Dataset licence or access terms
- Download instructions
- Preprocessing steps
- Expected file structure
- Whether the dataset can be redistributed
- Whether the dataset should be downloaded manually by users
- Any citation requirements

Do not commit large datasets directly.

If the dataset cannot be redistributed, provide a loader or clear instructions instead.

---

## 7. Feature Requests

Feature requests are welcome when they are tied to a concrete task or class of tasks, including tasks that are only at the proposal stage.

The most useful requests explain what new benchmark capability they would enable.

Helpful details include:

- Feature name
- Task or task family that needs it
- Why the current pipeline cannot express the task
- Expected input shape
- Expected output shape
- Whether the feature affects dense models, GNN models, or both
- Whether it affects data generation, feature extraction, masking, training, evaluation, or reporting
- A minimal example or pseudocode, where possible

Example:

```text
Feature request:
Support weighted adjacency matrices.

Needed for:
Tasks where edge weights are part of the input graph and cannot be represented as separate edge features without changing the task semantics.

Current limitation:
The current pipeline expects binary adjacency matrices.

Affected areas:
Collation, feature extraction, masking, dense channel stacking, GNN message passing, and checkpoint metadata.
```

---

## 8. Optional Feature Request Guide

Use this checklist to add detail where it is useful. An early feature request does not need to answer every item:

- [ ] The task that motivates the request
- [ ] Why the task cannot be represented with the current supported interface
- [ ] Whether the request affects dense models, GNN models, or both
- [ ] Whether it changes task semantics or only implementation convenience
- [ ] A minimal example
- [ ] Expected behaviour
- [ ] Any known edge cases
- [ ] Whether existing tasks should continue to behave unchanged

---

## 9. Suggested Issue Titles

Clear issue titles make proposals easier to find and discuss.

Examples:

```text
Task proposal: Minimum Dominating Set as node classification
```

```text
Task implementation: Directed transitive reduction
```

```text
Feature request: Support weighted adjacency matrices
```

```text
Feature request: Add hypergraph task interface
```

Less descriptive examples:

```text
New idea
```

```text
Improve benchmark
```

```text
Support more graphs
```

---

## 10. Helpful Information for Pull Requests

If you are submitting a task implementation, the following information is helpful:

- The task implementation
- A short task description
- The task classification
- Any required feature notes
- A minimal runnable example
- Reproducibility settings
- Documentation updates, if needed

For pull requests implementing a task-enabling feature, helpful information includes:

- The motivating task
- The implementation change
- Any changes to supported task semantics
- Any changes to masking, features, metrics, or runner behaviour
- Tests or examples showing the new capability
- Notes on backwards compatibility

If a contribution changes existing task semantics, please document that change clearly so existing users can understand the effect.

---

## 11. Scope

The contribution scope is intentionally broad in terms of benchmark tasks, but narrow in terms of project direction.

In scope:

- New graph transformation tasks
- New task generators
- Dataset-backed task definitions
- Task-specific feature requirements
- Feature requests needed to support new tasks
- Improvements that make task definitions clearer or more reproducible

Out of scope unless directly tied to a benchmark task:

- General model architecture experimentation
- Unrelated training-loop refactors
- General-purpose graph library features
- Large framework rewrites
- Style-only changes
- New dependencies without a task-driven reason

---

## 12. What Makes a Good Task?

A good benchmark task is not just a piece of code. It should help clarify what kinds of graph transformations a model can or cannot learn.

Strong tasks usually have at least one of the following properties:

- They test a clear structural operation
- They cover an underrepresented part of the taxonomy
- They require a graph type not yet well covered
- They introduce a different input-output relation
- They expose a limitation of current model families
- They are reproducible
- They can be evaluated consistently
- They are simple enough to understand but hard enough to be informative

We especially welcome tasks that expand coverage beyond the current edge-classification-heavy setting, provided the required pipeline extensions are clearly described.

---

## 13. Questions and Early Proposals

If you are unsure whether a task fits, you are welcome to open an issue at any stage.

A short task proposal is enough to start the discussion. Share whatever you already know about the task, such as its name, intended graph transformation, expected labels or outputs, or possible pipeline requirements. Missing details can be worked out during discussion.

The goal is to make it easy for the community to propose, discuss, implement, and compare graph transformation tasks in a shared framework.

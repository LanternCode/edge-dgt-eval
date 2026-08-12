# SPDX-License-Identifier: CC-BY-SA-4.0

import numpy as np
import torch
import networkx as nx
from torch import Tensor
from typing import Dict, Optional, Sequence

CANONICAL_FEATURES = {
    "transpose", "power_2", "power_3", "power_4", "power_5",
    "degree", "deg_row", "deg_col", "deg_diff",
    "triangles", "clustering_coeff",
    "cn", "jaccard", "adamic_adar",
    "shortest_path", "twohop"
}

DIRECTED_AUTO_FEATURES = (
    "transpose", "power_3", "power_4", "power_5",
    "degree", "deg_row", "deg_col", "deg_diff",
    "triangles", "clustering_coeff",
    "cn", "jaccard", "adamic_adar",
    "shortest_path",
)

UNDIRECTED_AUTO_FEATURES = (
    "power_3", "power_4", "power_5",
    "degree", "deg_row", "deg_col", "deg_diff",
    "triangles", "clustering_coeff",
    "cn", "jaccard", "adamic_adar",
    "shortest_path",
)

def shortest_path_from_adj(A: Tensor, *, is_directed: bool = False) -> Tensor:
    """
    Compute all-pairs shortest-path distances from the provided adjacency.

    The caller is responsible for passing the graph view the model is allowed to
    see. In the training pipeline this should usually be the supervised-redacted
    adjacency, so direct held-out edges do not leak through distance-1 entries.

    Returns an (N, N) float tensor on A.device, with unreachable pairs encoded as -1.
    """
    A01 = (A.detach().cpu().numpy() > 0).astype(np.uint8)
    graph_type = nx.DiGraph if is_directed else nx.Graph
    G = nx.from_numpy_array(A01, create_using=graph_type)
    D = np.full(A01.shape, -1, dtype=np.float32)
    for src, lengths in nx.all_pairs_shortest_path_length(G):
        for dst, dist in lengths.items():
            D[src, dst] = float(dist)
    return torch.as_tensor(D, device=A.device, dtype=torch.float32)


def pairwise_batch_from_adj(A_batch: Tensor, keys: Sequence[str], *, is_directed: bool = False) -> Dict[str, Tensor]:
    """
    Computes dense pairwise features for a batch of (or a single) adjacency matrices.
    Dynamically supports both 2D (N, N) and 3D (B, N, N) inputs natively.
    Returned keys follow this helper's fixed internal insertion order, not the caller's requested key order.
    """
    orig_dim = A_batch.dim()
    A = A_batch.unsqueeze(0) if orig_dim == 2 else A_batch
        
    A01 = torch.gt(A, 0).to(dtype=torch.float32)
    B, N, _ = A01.shape
    row_deg = A01.sum(dim=-1)
    col_deg = A01.sum(dim=-2) if is_directed else row_deg
    
    results: Dict[str, Tensor] = {}
    
    if "deg_row" in keys:
        results["deg_row"] = row_deg.view(B, N, 1).expand(B, N, N)
    if "deg_col" in keys:
        results["deg_col"] = col_deg.view(B, 1, N).expand(B, N, N)
    if "deg_diff" in keys:
        results["deg_diff"] = (row_deg.view(B, N, 1) - col_deg.view(B, 1, N)).abs()
        
    if "twohop" in keys:
        # A^2 yields the exact number of 2-hop paths from i to j
        results["twohop"] = A01 @ A01

    if any(k in keys for k in ("cn", "jaccard")):
        # Directed cn and twohop are the same product
        if is_directed and "twohop" in results:
            cn = results["twohop"].clone()
        else:
            cn = (A01 @ A01) if is_directed else (A01 @ A01.transpose(-1, -2))
        cn.diagonal(dim1=-2, dim2=-1).fill_(0.0)
        
        if "cn" in keys:
            results["cn"] = cn
            
        if "jaccard" in keys:
            union = row_deg.view(B, N, 1) + col_deg.view(B, 1, N) - cn
            results["jaccard"] = torch.where(union > 0, cn / union, torch.zeros_like(union))
            
    if "adamic_adar" in keys:
        safe_inv_log = torch.where(row_deg > 1, 1.0 / torch.log(row_deg), torch.zeros_like(row_deg))
        rhs = A01 if is_directed else A01.transpose(-1, -2)
        W = (A01 * safe_inv_log.view(B, 1, N)) @ rhs
        W.diagonal(dim1=-2, dim2=-1).fill_(0.0)
        results["adamic_adar"] = W

    # Squeeze the batch dimension back out if the user passed a 2D matrix
    if orig_dim == 2:
        return {k: v.squeeze(0) for k, v in results.items()}
        
    return results


def pairwise_for_pairs(
        A: Tensor,
        src: Tensor,
        dst: Tensor,
        keys: Sequence[str],
        *,
        is_directed: bool = True,
        row_deg: Optional[Tensor] = None,
        col_deg: Optional[Tensor] = None
) -> Dict[str, Tensor]:
    """
    Compute pairwise features only for specific (src, dst) pairs.
    Materialises a dense binary adjacency view internally and computes outputs only for the requested pairs.
    """
    if A.is_sparse:
        A01 = (A.to_dense() > 0).to(dtype=torch.float32)
    else:
        A01 = torch.gt(A, 0).to(dtype=torch.float32)

    results: Dict[str, Tensor] = {}
    if row_deg is None:
        row_deg = A01.sum(dim=1)
    if col_deg is None:
        col_deg = A01.sum(dim=0) if is_directed else row_deg

    # score_pairs_on_demand requests exactly the heavy pairwise keys. Endpoint degrees are computed by the caller itself.
    Au = A01[src]
    Av = A01.t()[dst] if is_directed else A01[dst]
    cn = (Au * Av).sum(dim=1)  # (M,)

    for key in keys:
        if key == "cn":
            results[key] = cn
        elif key == "jaccard":
            union = row_deg[src] + col_deg[dst] - cn
            results[key] = torch.where(union > 0, cn / union, torch.zeros_like(union))
        elif key == "adamic_adar":
            invlog = torch.where(row_deg > 1, 1.0 / torch.log(row_deg), torch.zeros_like(row_deg))
            results[key] = ((Au * Av) * invlog.view(1, -1)).sum(dim=1)
        else:
            raise KeyError(f"Unsupported key for pairwise_for_pairs: {key}")

    return results

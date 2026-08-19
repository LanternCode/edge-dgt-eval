# SPDX-License-Identifier: CC-BY-SA-4.0

import numpy as np
import torch
import networkx as nx
from torch import Tensor
from typing import Dict, Optional, Sequence
from functools import lru_cache

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


@lru_cache(maxsize=512)
def _shortest_path_cached(packed_adj: bytes, n: int, is_directed: bool) -> np.ndarray:
    A01 = np.unpackbits(
        np.frombuffer(packed_adj, dtype=np.uint8),
        count=n * n
    ).reshape(n, n)
    graph_type = nx.DiGraph if is_directed else nx.Graph
    G = nx.from_numpy_array(A01, create_using=graph_type)
    dist_dtype = np.int16 if n <= np.iinfo(np.int16).max else np.int32
    D = np.full((n, n), -1, dtype=dist_dtype)
    for src, lengths in nx.all_pairs_shortest_path_length(G):
        for dst, dist in lengths.items():
            D[src, dst] = dist
    return D


def shortest_path_from_adj(
        A: Tensor, *, is_directed: bool = False, valid_n: Optional[int] = None
) -> Tensor:
    """
    Compute all pair shortest-path distances from the provided adjacency.

    The caller is responsible for passing the graph view the model is allowed to
    see. In the training pipeline this should usually be the supervised, redacted
    adjacency, so direct held-out edges do not leak through distance-1 entries.

    Returns an (N, N) float tensor on A.device, with unreachable pairs encoded as -1.
    """
    full_n = int(A.shape[0])
    n = full_n if valid_n is None else int(valid_n)
    A01 = A[:n, :n].detach().cpu().numpy() > 0
    packed_adj = np.packbits(A01.reshape(-1)).tobytes()
    D_core = _shortest_path_cached(packed_adj, n, bool(is_directed))

    if n == full_n:
        D = D_core
    else:
        D = np.full((full_n, full_n), -1, dtype=D_core.dtype)
        D[:n, :n] = D_core
        np.fill_diagonal(D[n:, n:], 0)

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
        
    if any(k in keys for k in ("cn", "jaccard")):
        cn = (A01 @ A01) if is_directed else (A01 @ A01.transpose(-1, -2))
        
        if "cn" in keys:
            results["cn"] = cn
            
        if "jaccard" in keys:
            union = row_deg.view(B, N, 1) + col_deg.view(B, 1, N) - cn
            results["jaccard"] = torch.where(union > 0, cn / union, torch.zeros_like(union))
            
    if "adamic_adar" in keys:
        safe_inv_log = torch.where(row_deg > 1, 1.0 / torch.log(row_deg), torch.zeros_like(row_deg))
        rhs = A01 if is_directed else A01.transpose(-1, -2)
        W = (A01 * safe_inv_log.view(B, 1, N)) @ rhs
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
        col_deg: Optional[Tensor] = None,
        chunk_size: int = 4096,
        prebinarized: bool = False
) -> Dict[str, Tensor]:
    """
    Compute pairwise features only for specific (src, dst) pairs.
    Pair rows are processed in chunks so temporary memory is O(chunk_size * N).
    """
    if prebinarized:
        A01 = A.to_dense() if A.is_sparse else A
        A01 = A01.to(dtype=torch.float32)
    elif A.is_sparse:
        A01 = (A.to_dense() > 0).to(dtype=torch.float32)
    else:
        A01 = torch.gt(A, 0).to(dtype=torch.float32)

    if row_deg is None:
        row_deg = A01.sum(dim=1)
    else:
        row_deg = row_deg.to(device=A01.device, dtype=torch.float32)
    if col_deg is None:
        col_deg = A01.sum(dim=0) if is_directed else row_deg
    else:
        col_deg = col_deg.to(device=A01.device, dtype=torch.float32)

    allowed = {"cn", "jaccard", "adamic_adar"}
    for key in keys:
        if key not in allowed:
            raise KeyError(f"Unsupported key for pairwise_for_pairs: {key}")

    M = int(src.numel())
    step = max(1, int(chunk_size))
    results: Dict[str, Tensor] = {
        key: torch.empty((M,), device=A01.device, dtype=torch.float32)
        for key in keys
    }
    invlog = None
    if "adamic_adar" in results:
        invlog = torch.where(
            row_deg > 1,
            1.0 / torch.log(row_deg),
            torch.zeros_like(row_deg)
        )

    A01_t = A01.t() if is_directed else A01
    for start in range(0, M, step):
        end = min(start + step, M)
        src_c = src[start:end]
        dst_c = dst[start:end]

        Au = A01[src_c]
        Av = A01_t[dst_c]
        shared = Au * Av
        cn = shared.sum(dim=1)

        if "cn" in results:
            results["cn"][start:end] = cn
        if "jaccard" in results:
            union = row_deg[src_c] + col_deg[dst_c] - cn
            results["jaccard"][start:end] = torch.where(
                union > 0,
                cn / union,
                torch.zeros_like(union)
            )
        if "adamic_adar" in results:
            shared.mul_(invlog.view(1, -1))
            results["adamic_adar"][start:end] = shared.sum(dim=1)

    return results

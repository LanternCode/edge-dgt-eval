import argparse
import gc
import hashlib
import os
import re
import numpy as np
import pandas as pd
import torch
from pipeline.EdgeClassification import ProvidedSplitsTask, TNNTrainConfig, TaskHooks, run_pipeline_for_task
from pipeline.GNNBridge import GNNTrainConfig, run_gnn_edges_suite
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


_BERT_FEATURE_CACHE_VERSION = "v2_masked_mean_camelcase"

_LOCAL_CONTEXT_FEATURES = (
    "ctx_cand_head_cos",
    "ctx_cand_tail_cos",
    "ctx_head_tail_cos",
    "ctx_cand_head_l2",
    "ctx_cand_tail_l2",
    "ctx_head_tail_l2",
    "ctx_head_log_count",
    "ctx_tail_log_count",
)


def _ensure_conceptnet_dataset(data_dir: str, noise_level: str) -> str:
    data_dir_abs = os.path.abspath(data_dir)
    required = [
        os.path.join(data_dir_abs, "train.txt"),
        os.path.join(data_dir_abs, "valid.txt"),
        os.path.join(data_dir_abs, "test.txt"),
        os.path.join(data_dir_abs, "errors", f"{noise_level}-error.txt"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing ConceptNet denoising dataset files:\n" + "\n".join(missing))
    return data_dir_abs


def _load_gold_conceptnet_poisoned(
        dataset_base: str,
        noise_level: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the released GOLD ConceptNet files into two distinct collections.

    The clean files use (relation, head, tail), while the error file uses
    (head, relation, tail). Clean triples and released errors are combined into
    the poisoned candidate graph; the clean collection is retained only for
    count reporting.
    """
    clean_cols = ["r", "h", "t"]
    df_train = pd.read_csv(os.path.join(dataset_base, "train.txt"), sep="\t", header=None, names=clean_cols, dtype=str)
    df_valid = pd.read_csv(os.path.join(dataset_base, "valid.txt"), sep="\t", header=None, names=clean_cols, dtype=str)
    df_test = pd.read_csv(os.path.join(dataset_base, "test.txt"), sep="\t", header=None, names=clean_cols, dtype=str)

    clean_all = pd.concat([df_train, df_valid, df_test], ignore_index=True)
    clean_all.drop_duplicates(subset=["h", "r", "t"], keep="first", inplace=True)
    clean_all.reset_index(drop=True, inplace=True)
    clean_all["label"] = np.int8(1)

    del df_train, df_valid, df_test
    gc.collect()

    err_path = os.path.join(dataset_base, "errors", f"{noise_level}-error.txt")
    noise_all = pd.read_csv(err_path, sep="\t", header=None, names=["h", "r", "t"], dtype=str)
    noise_all["label"] = np.int8(0)

    candidates = pd.concat([clean_all, noise_all], ignore_index=True)
    candidates.reset_index(drop=True, inplace=True)

    del noise_all
    gc.collect()
    return clean_all, candidates


def _clean_conceptnet_text(s: str) -> str:
    s = str(s)
    for pref in ["/c/en/", "/c/", "/r/"]:
        if s.startswith(pref):
            s = s[len(pref):]
    s = s.replace("_", " ").replace("-", " ").replace("/", " ")
    return " ".join(s.split())


def _clean_conceptnet_relation_text(s: str) -> str:
    s = _clean_conceptnet_text(s)
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    return " ".join(s.split())


def _load_bert_local(model_dir: str, device: torch.device) -> Tuple[BertTokenizer, BertModel]:
    # use_fast=False ensures we bypass the Rust tokenizer memory leak
    if not model_dir or not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Missing local BERT directory: {model_dir!r}. "
            "Download google-bert/bert-base-uncased and place config.json, vocab.txt, "
            "tokenizer_config.json, and pytorch_model.bin there."
        )
    tok = BertTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=False)
    mdl = BertModel.from_pretrained(model_dir, local_files_only=True).to(device)
    mdl.eval()
    return tok, mdl


def _stable_hash_texts(texts: Sequence[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _bert_model_cache_identity(model_dir: str) -> str:
    """Build a stable local-model identity without hashing the full weight file."""
    model_dir_abs = os.path.abspath(model_dir)
    h = hashlib.sha256(model_dir_abs.encode("utf-8", errors="ignore"))

    for filename in ("config.json", "tokenizer_config.json", "vocab.txt", "pytorch_model.bin", "model.safetensors"):
        path = os.path.join(model_dir_abs, filename)
        if not os.path.exists(path):
            continue
        stat = os.stat(path)
        h.update(filename.encode("utf-8"))
        h.update(str(int(stat.st_size)).encode("ascii"))
        if stat.st_size <= 2_000_000:
            with open(path, "rb") as f:
                h.update(f.read())

    return h.hexdigest()[:16]


def _load_cached_projection(path: str, expected_rows: int, expected_dim: int) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None

    arr = np.load(path, mmap_mode="r")
    if tuple(arr.shape) == (int(expected_rows), int(expected_dim)):
        return arr

    print(
        f"[ConceptNet denoising task] Ignoring incompatible cache {path!r}: "
        f"shape={tuple(arr.shape)}, expected={(int(expected_rows), int(expected_dim))}.",
        flush=True,
    )
    del arr
    return None


def _make_random_projection_torch(D: int, k: int, seed: int, device: torch.device) -> torch.Tensor:
    rng = np.random.default_rng(int(seed))
    W = rng.standard_normal(size=(D, k)).astype(np.float32)
    W /= np.sqrt(float(D))
    return torch.from_numpy(W).to(device=device)


def _encode_and_project_streaming(
        texts: Sequence[str],
        tok: BertTokenizer,
        mdl: BertModel,
        W: torch.Tensor,
        device: torch.device,
        batch_size: int,
        max_length: int,
) -> np.ndarray:
    n = len(texts)
    k = int(W.shape[1])
    out_np = np.empty((n, k), dtype=np.float32)

    pos = 0
    for i in tqdm(range(0, n, batch_size), desc="BERT encode+proj", unit="batch"):
        batch = list(texts[i:i + batch_size])
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {kk: vv.to(device) for kk, vv in inputs.items()}
        with torch.no_grad():
            hidden = mdl(**inputs).last_hidden_state
            token_mask = inputs["attention_mask"].unsqueeze(-1).to(dtype=hidden.dtype)
            emb = (hidden * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1.0)
            proj = emb @ W
        bsz = int(proj.shape[0])
        out_np[pos:pos + bsz] = proj.detach().cpu().numpy().astype(np.float32)
        pos += bsz

        # Explicit garbage collection per batch prevents Python fragmentation
        del inputs, hidden, token_mask, emb, proj
        if i % (batch_size * 100) == 0:
            torch.cuda.empty_cache()

    if pos != n:
        raise RuntimeError(f"Projection write mismatch: wrote {pos}, expected {n}.")
    return out_np


def _load_or_build_entity_relation_features(
        cache_dir: str,
        entity_texts: Sequence[str],
        relation_texts: Sequence[str],
        bert_dir: str,
        proj_dim: int,
        proj_seed: int,
        bert_batch_size: int,
        bert_max_length: int,
        device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    os.makedirs(cache_dir, exist_ok=True)

    ent_hash = _stable_hash_texts(entity_texts)
    rel_hash = _stable_hash_texts(relation_texts)
    bert_identity = _bert_model_cache_identity(bert_dir)
    cache_tag = (
        f"{_BERT_FEATURE_CACHE_VERSION}_bert{bert_identity}"
        f"_len{int(bert_max_length)}"
    )

    ent_path = os.path.join(
        cache_dir,
        f"entity_proj_{cache_tag}_{ent_hash}_k{proj_dim}_seed{proj_seed}.npy",
    )
    rel_path = os.path.join(
        cache_dir,
        f"rel_proj_{cache_tag}_{rel_hash}_k{proj_dim}_seed{proj_seed}.npy",
    )

    ent_proj = _load_cached_projection(ent_path, len(entity_texts), int(proj_dim))
    rel_proj = _load_cached_projection(rel_path, len(relation_texts), int(proj_dim))

    if ent_proj is not None and rel_proj is not None:
        return ent_proj, rel_proj

    tok, mdl = _load_bert_local(bert_dir, device)
    W = _make_random_projection_torch(768, int(proj_dim), int(proj_seed), device)

    if ent_proj is None:
        ent_proj_raw = _encode_and_project_streaming(entity_texts, tok, mdl, W, device, bert_batch_size,
                                                     bert_max_length)
        np.save(ent_path, ent_proj_raw)
        del ent_proj_raw
        ent_proj = np.load(ent_path, mmap_mode="r")

    if rel_proj is None:
        rel_proj_raw = _encode_and_project_streaming(relation_texts, tok, mdl, W, device, bert_batch_size,
                                                     bert_max_length)
        np.save(rel_path, rel_proj_raw)
        del rel_proj_raw
        rel_proj = np.load(rel_path, mmap_mode="r")

    del mdl
    del tok
    del W
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ent_proj, rel_proj


def _compute_gold_metrics(prob_clean: np.ndarray, y_clean: np.ndarray) -> Tuple[float, float, int]:
    y_clean = y_clean.astype(np.int64)
    noise_score = 1.0 - prob_clean.astype(np.float64)
    y_noisy = (y_clean == 0).astype(np.int64)

    k = int(y_noisy.sum())
    auc = float("nan")
    if len(np.unique(y_noisy)) == 2:
        auc = float(roc_auc_score(y_noisy, noise_score))

    if k <= 0:
        return auc, float("nan"), k

    order = np.argsort(-noise_score)
    topk = order[:k]
    recall_at_k = float((y_noisy[topk] == 1).sum() / k)
    return auc, recall_at_k, k


def _build_vocab_and_indices(triples: pd.DataFrame) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, np.ndarray]:
    # 1. Extract uniques first to avoid massive string duplication
    h_unq = triples["h"].unique()
    t_unq = triples["t"].unique()
    entities_arr = pd.unique(np.concatenate([h_unq, t_unq]))
    relations_arr = pd.unique(triples["r"])

    triple_h, triple_t, triple_r = _index_triples_with_vocab(
        triples,
        entities_arr,
        relations_arr,
    )
    return entities_arr.tolist(), relations_arr.tolist(), triple_h, triple_t, triple_r


def _index_triples_with_vocab(
        triples: pd.DataFrame,
        entities: Sequence[str],
        relations: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index triples against an existing shared entity/relation vocabulary."""
    entity_type = pd.CategoricalDtype(categories=entities)
    relation_type = pd.CategoricalDtype(categories=relations)

    triple_h = triples["h"].astype(entity_type).cat.codes.to_numpy(dtype=np.int32).copy()
    triple_t = triples["t"].astype(entity_type).cat.codes.to_numpy(dtype=np.int32).copy()
    triple_r = triples["r"].astype(relation_type).cat.codes.to_numpy(dtype=np.int32).copy()

    if np.any(triple_h < 0) or np.any(triple_t < 0) or np.any(triple_r < 0):
        raise ValueError("Reference triples contain entities or relations missing from the shared candidate vocabulary.")

    return triple_h, triple_t, triple_r


def _build_incidence_csr(n_entities: int, triple_h: np.ndarray, triple_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h_counts = np.bincount(triple_h, minlength=n_entities)
    t_counts = np.bincount(triple_t, minlength=n_entities)
    degrees = (h_counts + t_counts).astype(np.int32)

    offsets = np.empty(n_entities + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(degrees, out=offsets[1:])

    indices = np.empty(offsets[-1], dtype=np.int32)
    pos = offsets[:-1].copy()

    for i in range(len(triple_h)):
        h = triple_h[i]
        t = triple_t[i]
        indices[pos[h]] = i
        pos[h] += 1
        indices[pos[t]] = i
        pos[t] += 1

    return offsets, indices


def _encode_triple_key(h: int, r: int, t: int, n_entities: int, n_relations: int) -> int:
    return (int(h) * int(n_relations) + int(r)) * int(n_entities) + int(t)


def _make_observed_set(triple_h: np.ndarray, triple_r: np.ndarray, triple_t: np.ndarray, n_entities: int,
                       n_relations: int) -> np.ndarray:
    keys = triple_h.astype(np.int64)
    keys *= int(n_relations)
    keys += triple_r
    keys *= int(n_entities)
    keys += triple_t
    keys.sort()
    return keys


def _build_tile_nodes_for_candidate(
        h_id: int,
        r_id: int,
        t_id: int,
        incidence_offsets: torch.Tensor,
        incidence_indices: torch.Tensor,
        context_h: torch.Tensor,
        context_r: torch.Tensor,
        context_t: torch.Tensor,
        n_entities: int,
        tile_size: int,
        nei_k: int,
        rng: np.random.Generator,
) -> np.ndarray:
    P = int(tile_size)
    K = int(nei_k)

    if P < 3:
        raise ValueError("tile_size must be at least 3 for head, candidate triple, and tail nodes.")

    def _is_exact_candidate(triple_idx: int) -> bool:
        return (
            int(context_h[triple_idx].item()) == int(h_id)
            and int(context_r[triple_idx].item()) == int(r_id)
            and int(context_t[triple_idx].item()) == int(t_id)
        )

    def _sample_neighbors(eid: int) -> List[int]:
        start = int(incidence_offsets[int(eid)].item())
        end = int(incidence_offsets[int(eid) + 1].item())
        if start == end or K <= 0:
            return []

        neighbors = incidence_indices[start:end].numpy()
        neighbors = np.asarray(
            [int(j) for j in neighbors.tolist() if not _is_exact_candidate(int(j))],
            dtype=np.int32,
        )
        if neighbors.size <= K:
            return neighbors.tolist()

        chosen = rng.choice(neighbors.size, size=K, replace=False)
        return neighbors[chosen].tolist()

    head_neighbors = _sample_neighbors(h_id)
    tail_neighbors = _sample_neighbors(t_id)

    # Interleave endpoint neighbourhoods so one high-degree endpoint cannot consume
    # the tile. Context comes from the full poisoned graph, but every occurrence of
    # the exact candidate (head, relation, tail) is excluded before sampling.
    ordered_triples: List[int] = []
    seen_triples = set()
    for offset in range(max(len(head_neighbors), len(tail_neighbors))):
        for neighbors in (head_neighbors, tail_neighbors):
            if offset >= len(neighbors):
                continue
            triple_idx = int(neighbors[offset])
            if triple_idx in seen_triples:
                continue
            seen_triples.add(triple_idx)
            ordered_triples.append(triple_idx)

    fixed = [int(h_id), -2, int(t_id)]
    ordered: List[int] = list(fixed)

    for triple_idx in ordered_triples:
        if len(ordered) >= P:
            break
        ordered.append(int(n_entities + triple_idx))

    seen_entities = {int(h_id), int(t_id)}
    for triple_idx in ordered_triples:
        if len(ordered) >= P:
            break
        for entity_id in (
            int(context_h[triple_idx].item()),
            int(context_t[triple_idx].item()),
        ):
            if entity_id in seen_entities:
                continue
            seen_entities.add(entity_id)
            ordered.append(entity_id)
            if len(ordered) >= P:
                break

    if len(ordered) < P:
        ordered.extend([-1] * (P - len(ordered)))

    return np.asarray(ordered, dtype=np.int32)

def _candidate_triple_proj(h_id: int, r_id: int, t_id: int, ent_proj: np.ndarray, rel_proj: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [ent_proj[int(h_id)], rel_proj[int(r_id)], ent_proj[int(t_id)]],
        axis=0,
    ).astype(np.float32, copy=False)


def _build_precomputed_context_stats(
        triple_h: np.ndarray,
        triple_r: np.ndarray,
        triple_t: np.ndarray,
        ent_proj: np.ndarray,
        rel_proj: np.ndarray,
        n_entities: int,
        n_relations: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute poisoned-graph incident-triple sums, counts, and exact-triple multiplicities."""
    proj_dim = int(ent_proj.shape[1])
    semantic_dim = 3 * proj_dim

    triple_proj = np.concatenate(
        [
            np.asarray(ent_proj[triple_h.astype(np.int64)], dtype=np.float32),
            np.asarray(rel_proj[triple_r.astype(np.int64)], dtype=np.float32),
            np.asarray(ent_proj[triple_t.astype(np.int64)], dtype=np.float32),
        ],
        axis=1,
    )

    entity_context_sum = np.zeros((int(n_entities), semantic_dim), dtype=np.float32)
    np.add.at(entity_context_sum, triple_h.astype(np.int64), triple_proj)
    np.add.at(entity_context_sum, triple_t.astype(np.int64), triple_proj)

    entity_context_count = (
        np.bincount(triple_h.astype(np.int64), minlength=int(n_entities))
        + np.bincount(triple_t.astype(np.int64), minlength=int(n_entities))
    ).astype(np.int32)

    keys = triple_h.astype(np.int64)
    keys *= int(n_relations)
    keys += triple_r.astype(np.int64)
    keys *= int(n_entities)
    keys += triple_t.astype(np.int64)
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    candidate_multiplicity = counts[inverse].astype(np.int32, copy=False)

    del triple_proj, keys, inverse, counts
    gc.collect()
    return entity_context_sum, entity_context_count, candidate_multiplicity


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def _normalised_l2(a: np.ndarray, b: np.ndarray) -> float:
    dim = max(1, int(np.asarray(a).shape[0]))
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)) / np.sqrt(float(dim)))


def _build_A_and_feats_for_tile(
        nodes: np.ndarray,
        n_entities: int,
        triple_h: torch.Tensor,
        triple_r: torch.Tensor,
        triple_t: torch.Tensor,
        ent_proj: np.ndarray,
        rel_proj: np.ndarray,
        entity_context_sum: torch.Tensor,
        entity_context_count: torch.Tensor,
        exact_candidate_count: int,
        cand_h: int,
        cand_r: int,
        cand_t: int,
        include_x: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    P = int(nodes.shape[0])
    proj_dim = int(ent_proj.shape[1])
    semantic_dim = 3 * proj_dim

    A = np.zeros((P, P), dtype=np.float32)

    local: Dict[int, int] = {}
    for i, gid in enumerate(nodes.tolist()):
        if gid >= 0 and gid not in local:
            local[int(gid)] = int(i)

    # Candidate triple node (pos=1): connect to h (pos=0) and t (pos=2)
    if P >= 3 and nodes[1] == -2:
        A[0, 1] = 1.0
        A[1, 0] = 1.0
        A[2, 1] = 1.0
        A[1, 2] = 1.0

    def _connect_entity_triple(ent_gid: int, tri_pos: int) -> None:
        pos_ent = local.get(int(ent_gid))
        if pos_ent is None:
            return
        A[pos_ent, tri_pos] = 1.0
        A[tri_pos, pos_ent] = 1.0

    for j, gid in enumerate(nodes.tolist()):
        if gid < 0 or gid == -2:
            continue
        if gid < n_entities:
            continue
        tr_idx = int(gid - n_entities)
        if tr_idx < 0 or tr_idx >= int(triple_h.shape[0]):
            continue
        _connect_entity_triple(int(triple_h[tr_idx].item()), j)
        _connect_entity_triple(int(triple_t[tr_idx].item()), j)

    node_type = np.zeros((P,), dtype=np.float32)
    proj = np.zeros((P, semantic_dim), dtype=np.float32)

    cand_proj = _candidate_triple_proj(cand_h, cand_r, cand_t, ent_proj, rel_proj)

    def _candidate_excluded_context_mean(entity_id: int) -> Tuple[np.ndarray, int]:
        total = entity_context_sum[int(entity_id)].numpy().astype(np.float32, copy=True)
        count = int(entity_context_count[int(entity_id)].item())
        incident_copies = int(exact_candidate_count) * (
            int(int(cand_h) == int(entity_id)) + int(int(cand_t) == int(entity_id))
        )
        if incident_copies > 0:
            total -= float(incident_copies) * cand_proj
            count -= int(incident_copies)
        if count <= 0:
            return np.zeros_like(cand_proj), 0
        return total / float(count), count

    head_context, head_context_count = _candidate_excluded_context_mean(int(cand_h))
    tail_context, tail_context_count = _candidate_excluded_context_mean(int(cand_t))
    context_values = {
        "ctx_cand_head_cos": _safe_cosine(cand_proj, head_context),
        "ctx_cand_tail_cos": _safe_cosine(cand_proj, tail_context),
        "ctx_head_tail_cos": _safe_cosine(head_context, tail_context),
        "ctx_cand_head_l2": _normalised_l2(cand_proj, head_context),
        "ctx_cand_tail_l2": _normalised_l2(cand_proj, tail_context),
        "ctx_head_tail_l2": _normalised_l2(head_context, tail_context),
        "ctx_head_log_count": float(np.log1p(head_context_count)),
        "ctx_tail_log_count": float(np.log1p(tail_context_count)),
    }

    for i, gid in enumerate(nodes.tolist()):
        if gid == -2:
            node_type[i] = 1.0
            proj[i] = cand_proj
            continue
        if gid < 0:
            continue

        if gid >= n_entities:
            node_type[i] = 1.0
            tr_idx = int(gid - n_entities)
            h_id = int(triple_h[tr_idx].item())
            r_id = int(triple_r[tr_idx].item())
            t_id = int(triple_t[tr_idx].item())
            proj[i] = _candidate_triple_proj(h_id, r_id, t_id, ent_proj, rel_proj)
        else:
            proj[i, :proj_dim] = ent_proj[int(gid)]

    proj_t = torch.from_numpy(proj)
    node_type_t = torch.from_numpy(node_type)

    if include_x:
        # GNN path: pack node type plus the role-aware [head | relation | tail]
        # semantic representation into a single (P, 1+3*proj_dim) matrix.
        # torch.cat allocates a fresh tensor so proj_t / proj are not referenced
        # by any slice and can be freed by the GC immediately after this line.
        # Emitting individual bert_p* views alongside feats["x"] would double the
        # bert data in RAM (proj kept alive by views + a full copy inside x) and
        # triple the node-feature dimension seen by the GNN encoder.
        feats: Dict[str, torch.Tensor] = {
            "x": torch.cat([node_type_t.unsqueeze(1), proj_t], dim=1)
        }
    else:
        # EC / FeatureRegistry path: individual 1-D keys are needed so that the
        # registry can broadcast them into row/col channels.  x is not emitted here
        # because it is not in the canonical feature_set for the dense pipeline.
        feats = {"node_type": node_type_t}
        for d in range(semantic_dim):
            feats[f"bert_p{d}"] = proj_t[:, d]

        for key in _LOCAL_CONTEXT_FEATURES:
            edge_feature = torch.zeros((P, P), dtype=torch.float32)
            edge_feature[0, 1] = float(context_values[key])
            feats[key] = edge_feature

    return torch.from_numpy(A), feats


class ConceptNetDenoisingTrainDataset(Dataset):
    """
    Train on candidate observed-vs-corrupted triples. Clean/noisy labels are not used.

    Candidate triples contain clean triples plus released errors. Neighbourhoods and
    corruption rejection are derived from the full poisoned graph.
    """

    def __init__(
            self,
            candidate_h: np.ndarray,
            candidate_r: np.ndarray,
            candidate_t: np.ndarray,
            context_h: np.ndarray,
            context_r: np.ndarray,
            context_t: np.ndarray,
            incidence_offsets: np.ndarray,
            incidence_indices: np.ndarray,
            observed_set: np.ndarray,
            entity_context_sum: np.ndarray,
            entity_context_count: np.ndarray,
            candidate_multiplicity: np.ndarray,
            ent_proj: np.ndarray,
            rel_proj: np.ndarray,
            n_entities: int,
            n_relations: int,
            tile_size: int,
            nei_k: int,
            neg_per_pos: int,
            seed: int,
            include_x: bool,
    ) -> None:
        # Wrap datasets in torch Tensors so PyTorch multiprocess handles them in Shared Memory.
        self.candidate_h = torch.from_numpy(candidate_h)
        self.candidate_r = torch.from_numpy(candidate_r)
        self.candidate_t = torch.from_numpy(candidate_t)
        self.context_h = torch.from_numpy(context_h)
        self.context_r = torch.from_numpy(context_r)
        self.context_t = torch.from_numpy(context_t)
        self.incidence_offsets = torch.from_numpy(incidence_offsets)
        self.incidence_indices = torch.from_numpy(incidence_indices)
        self.observed_set = torch.from_numpy(observed_set)
        self.entity_context_sum = torch.from_numpy(entity_context_sum)
        self.entity_context_count = torch.from_numpy(entity_context_count)
        self.candidate_multiplicity = torch.from_numpy(candidate_multiplicity)

        self.ent_proj = ent_proj
        self.rel_proj = rel_proj
        self.n_entities = int(n_entities)
        self.n_relations = int(n_relations)
        self.tile_size = int(tile_size)
        self.nei_k = int(nei_k)
        self.neg_per_pos = int(neg_per_pos)
        self.seed = int(seed)
        self.include_x = bool(include_x)

        self.n_candidates = int(self.candidate_h.shape[0])

    def __len__(self) -> int:
        return self.n_candidates * (1 + self.neg_per_pos)

    def __getitem__(self, idx: int):
        idx = int(idx)
        rng = np.random.default_rng(self.seed + idx)

        if idx < self.n_candidates:
            base = idx
            h = int(self.candidate_h[base].item())
            r = int(self.candidate_r[base].item())
            t = int(self.candidate_t[base].item())
            exact_candidate_count = int(self.candidate_multiplicity[base].item())
            y = 1.0
        else:
            exact_candidate_count = 0
            y = 0.0
            negative_offset = idx - self.n_candidates
            base = int(negative_offset // self.neg_per_pos)
            r = int(self.candidate_r[base].item())

            corrupt_head = bool(rng.integers(0, 2) == 0)
            h0 = int(self.candidate_h[base].item())
            t0 = int(self.candidate_t[base].item())
            h = h0
            t = t0

            found = False
            for _ in range(500):
                ent_new = int(rng.integers(0, self.n_entities))
                if corrupt_head:
                    if ent_new == h0:
                        continue
                    cand_key = _encode_triple_key(ent_new, r, t0, self.n_entities, self.n_relations)
                    idx_s = int(torch.searchsorted(self.observed_set, cand_key).item())
                    if idx_s < len(self.observed_set) and int(self.observed_set[idx_s].item()) == cand_key:
                        continue
                    h = ent_new
                    found = True
                    break

                if ent_new == t0:
                    continue
                cand_key = _encode_triple_key(h0, r, ent_new, self.n_entities, self.n_relations)
                idx_s = int(torch.searchsorted(self.observed_set, cand_key).item())
                if idx_s < len(self.observed_set) and int(self.observed_set[idx_s].item()) == cand_key:
                    continue
                t = ent_new
                found = True
                break

            if not found:
                raise RuntimeError("Negative sampling failed to find a non-observed corruption.")

        nodes = _build_tile_nodes_for_candidate(
            h,
            r,
            t,
            self.incidence_offsets,
            self.incidence_indices,
            self.context_h,
            self.context_r,
            self.context_t,
            self.n_entities,
            self.tile_size,
            self.nei_k,
            rng,
        )
        A, feats = _build_A_and_feats_for_tile(
            nodes,
            self.n_entities,
            self.context_h,
            self.context_r,
            self.context_t,
            self.ent_proj,
            self.rel_proj,
            self.entity_context_sum,
            self.entity_context_count,
            exact_candidate_count,
            h,
            r,
            t,
            self.include_x,
        )

        P = int(nodes.shape[0])
        L = torch.zeros((P, P), dtype=torch.float32)
        M = torch.zeros((P, P), dtype=torch.bool)
        L[0, 1] = float(y)
        M[0, 1] = True
        return A, feats, L, M


class ConceptNetDenoisingEvalDataset(Dataset):
    """Test candidate triples with clean/noisy labels used only for metrics."""

    def __init__(
            self,
            candidate_h: np.ndarray,
            candidate_r: np.ndarray,
            candidate_t: np.ndarray,
            y_clean: np.ndarray,
            context_h: np.ndarray,
            context_r: np.ndarray,
            context_t: np.ndarray,
            incidence_offsets: np.ndarray,
            incidence_indices: np.ndarray,
            entity_context_sum: np.ndarray,
            entity_context_count: np.ndarray,
            candidate_multiplicity: np.ndarray,
            ent_proj: np.ndarray,
            rel_proj: np.ndarray,
            n_entities: int,
            n_relations: int,
            tile_size: int,
            nei_k: int,
            seed: int,
            include_x: bool,
    ) -> None:
        self.candidate_h = torch.from_numpy(candidate_h)
        self.candidate_r = torch.from_numpy(candidate_r)
        self.candidate_t = torch.from_numpy(candidate_t)
        self.y_clean = torch.from_numpy(y_clean)
        self.context_h = torch.from_numpy(context_h)
        self.context_r = torch.from_numpy(context_r)
        self.context_t = torch.from_numpy(context_t)
        self.incidence_offsets = torch.from_numpy(incidence_offsets)
        self.incidence_indices = torch.from_numpy(incidence_indices)
        self.entity_context_sum = torch.from_numpy(entity_context_sum)
        self.entity_context_count = torch.from_numpy(entity_context_count)
        self.candidate_multiplicity = torch.from_numpy(candidate_multiplicity)

        self.ent_proj = ent_proj
        self.rel_proj = rel_proj
        self.n_entities = int(n_entities)
        self.n_relations = int(n_relations)
        self.tile_size = int(tile_size)
        self.nei_k = int(nei_k)
        self.seed = int(seed)
        self.include_x = bool(include_x)

        self.n_candidates = int(self.candidate_h.shape[0])

    def __len__(self) -> int:
        return self.n_candidates

    def __getitem__(self, idx: int):
        idx = int(idx)
        rng = np.random.default_rng(self.seed + idx)

        h = int(self.candidate_h[idx].item())
        r = int(self.candidate_r[idx].item())
        t = int(self.candidate_t[idx].item())
        exact_candidate_count = int(self.candidate_multiplicity[idx].item())

        nodes = _build_tile_nodes_for_candidate(
            h,
            r,
            t,
            self.incidence_offsets,
            self.incidence_indices,
            self.context_h,
            self.context_r,
            self.context_t,
            self.n_entities,
            self.tile_size,
            self.nei_k,
            rng,
        )
        A, feats = _build_A_and_feats_for_tile(
            nodes,
            self.n_entities,
            self.context_h,
            self.context_r,
            self.context_t,
            self.ent_proj,
            self.rel_proj,
            self.entity_context_sum,
            self.entity_context_count,
            exact_candidate_count,
            h,
            r,
            t,
            self.include_x,
        )

        P = int(nodes.shape[0])
        L = torch.zeros((P, P), dtype=torch.float32)
        M = torch.zeros((P, P), dtype=torch.bool)
        L[0, 1] = float(self.y_clean[idx].item())
        M[0, 1] = True
        return A, feats, L, M


@dataclass
class _Bench:
    splits: Dict[str, Dataset]


def _extract_prob_and_y(bundle: Dict[str, dict], model_key: str) -> Tuple[np.ndarray, np.ndarray]:
    entry = bundle["results"][model_key]["test"]
    prob = np.asarray(entry.get("_prob", []), dtype=np.float64)
    y = np.asarray(entry.get("_y", []), dtype=np.int64)
    return prob, y


def _print_gold_style_metrics(bundle: Dict[str, dict], model_key: str) -> None:
    prob, y = _extract_prob_and_y(bundle, model_key)
    if prob.size == 0 or y.size == 0:
        print("[GOLD metrics] Missing _prob/_y; cannot compute metrics.")
        return
    auc, r_at_k, k = _compute_gold_metrics(prob, y)
    print(f"[GOLD protocol] AUC(noisy>clean)={auc:.4f}  Recall@k={r_at_k:.4f} (k=#noisy={k})")


def _make_task(
        data_dir: str,
        noise_level: str,
        cache_dir: str,
        tile_size: int,
        nei_k: int,
        proj_dim: int,
        proj_seed: Optional[int],
        bert_dir: str,
        bert_batch_size: int,
        bert_max_length: int,
        batch_size: int,
        num_workers: int,
        neg_per_pos: int,
        seed: Optional[int],
        include_x: bool,
        device: torch.device,
) -> ProvidedSplitsTask:
    dataset_base = _ensure_conceptnet_dataset(data_dir, noise_level)
    clean_reference, candidates = _load_gold_conceptnet_poisoned(dataset_base, noise_level)

    y_clean = candidates["label"].to_numpy(dtype=np.int8, copy=True)

    entities, relations, candidate_h, candidate_t, candidate_r = _build_vocab_and_indices(candidates)
    n_entities = len(entities)
    n_relations = len(relations)

    print(
        f"[ConceptNet denoising task] clean_reference={len(clean_reference)} "
        f"candidates={len(candidate_h)} noisy={int((y_clean == 0).sum())}",
        flush=True,
    )

    del clean_reference, candidates
    gc.collect()

    entity_texts = [_clean_conceptnet_text(e) for e in entities]
    relation_texts = [_clean_conceptnet_relation_text(r) for r in relations]

    del entities
    del relations
    gc.collect()

    # Triple nodes use the role-aware [head | relation | tail] concatenation,
    # while entity nodes occupy the head segment and leave the other segments zero.
    semantic_dim = 3 * int(proj_dim)
    proj_keys = tuple(f"bert_p{d}" for d in range(semantic_dim))

    context_keys = tuple(_LOCAL_CONTEXT_FEATURES)
    hooks = TaskHooks(
        label_fn=None,
        feature_set=[
            "deg_row", "deg_col", "deg_diff", "cn", "jaccard", "adamic_adar", "power_3",
            ("node_type", "node"),
            *[(k, "node") for k in proj_keys],
            *[(k, "edge") for k in context_keys]
        ]
    )

    # Let ProvidedSplitsTask resolve the authoritative run seed first.
    bench = _Bench(splits={})
    task = ProvidedSplitsTask(
        name=f"ConceptNet_Denoise_{noise_level}",
        directed=True,
        hooks=hooks,
        bench=bench,
        eval_on_existing_edges_only=True,
        pin_memory=False,
        num_workers=num_workers,
        seed=seed
    )

    resolved_seed = int(task.seed)
    resolved_proj_seed = resolved_seed if proj_seed is None else int(proj_seed)
    print(f"[ConceptNet denoising task] proj_seed={resolved_proj_seed}", flush=True)

    ent_proj, rel_proj = _load_or_build_entity_relation_features(
        cache_dir,
        entity_texts,
        relation_texts,
        bert_dir,
        proj_dim,
        resolved_proj_seed,
        bert_batch_size,
        bert_max_length,
        device
    )

    del entity_texts
    del relation_texts
    gc.collect()

    entity_context_sum, entity_context_count, candidate_multiplicity = _build_precomputed_context_stats(
        candidate_h,
        candidate_r,
        candidate_t,
        ent_proj,
        rel_proj,
        n_entities,
        n_relations,
    )
    print(
        f"[ConceptNet denoising task] precomputed local semantic means: "
        f"entities={n_entities} semantic_dim={3 * int(proj_dim)}",
        flush=True,
    )

    incidence_offsets, incidence_indices = _build_incidence_csr(
        n_entities,
        candidate_h,
        candidate_t,
    )

    observed_set = _make_observed_set(
        candidate_h,
        candidate_r,
        candidate_t,
        n_entities,
        n_relations,
    )

    train_ds = ConceptNetDenoisingTrainDataset(
        candidate_h,
        candidate_r,
        candidate_t,
        candidate_h,
        candidate_r,
        candidate_t,
        incidence_offsets,
        incidence_indices,
        observed_set,
        entity_context_sum,
        entity_context_count,
        candidate_multiplicity,
        ent_proj,
        rel_proj,
        n_entities,
        n_relations,
        tile_size,
        nei_k,
        neg_per_pos,
        resolved_seed,
        include_x,
    )
    val_ds: List = []
    test_ds = ConceptNetDenoisingEvalDataset(
        candidate_h,
        candidate_r,
        candidate_t,
        y_clean,
        candidate_h,
        candidate_r,
        candidate_t,
        incidence_offsets,
        incidence_indices,
        entity_context_sum,
        entity_context_count,
        candidate_multiplicity,
        ent_proj,
        rel_proj,
        n_entities,
        n_relations,
        tile_size,
        nei_k,
        resolved_seed,
        include_x,
    )

    bench.splits = {"train": train_ds, "val": val_ds, "test": test_ds}

    return task


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ConceptNet denoising: GOLD-style training (observed vs corrupted), no clean/noisy labels in training.")
    parser.add_argument("--model", required=True,
                        choices=["mlp", "deep_mlp", "cnn", "transformer", "rf", "gcn", "gin", "sage", "edge_tx", "gps"])
    parser.add_argument("--data_dir", default="data/conceptnet_denoising/raw")
    parser.add_argument("--noise_level", default="C-20")
    parser.add_argument("--cache_dir", default="data/conceptnet_denoising/bert_projection_cache")
    parser.add_argument("--tile_size", type=int, default=128)
    parser.add_argument("--tile_size_gnn", type=int, default=128)
    parser.add_argument("--nei_k", type=int, default=64)
    parser.add_argument("--proj_dim", type=int, default=32)
    parser.add_argument("--proj_seed", type=int, default=None)
    parser.add_argument("--bert_dir", default=os.environ.get("CONCEPTNET_BERT_DIR", "data/conceptnet_denoising/bert_base_uncased"))
    parser.add_argument("--bert_batch_size", type=int, default=64)
    parser.add_argument("--bert_max_length", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--neg_per_pos", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gnn_lr", type=float, default=1e-3)
    parser.add_argument("--gnn_weight_decay", type=float, default=0.0)
    parser.add_argument("--gnn_grad_clip", type=float, default=1.0)
    parser.add_argument("--gnn_scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--dense_lr", type=float, default=1e-4)
    parser.add_argument("--dense_weight_decay", type=float, default=1e-3)
    parser.add_argument("--dense_grad_clip", type=float, default=1.0)
    parser.add_argument("--cnn_hidden", type=int, default=64)
    parser.add_argument("--cnn_head_kernel", type=int, choices=[1, 3, 5], default=3)
    parser.add_argument("--cnn_dropout", type=float, default=0.10)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    include_x = args.model in ["gcn", "gin", "sage", "edge_tx", "gps"]

    is_gnn = args.model in ["gcn", "gin", "sage", "edge_tx", "gps"]
    tile_size = int(args.tile_size_gnn) if is_gnn else int(args.tile_size)
    batch_size = 1 if is_gnn else int(args.batch_size)

    task = _make_task(
        args.data_dir,
        args.noise_level,
        args.cache_dir,
        tile_size,
        args.nei_k,
        args.proj_dim,
        args.proj_seed,
        args.bert_dir,
        args.bert_batch_size,
        args.bert_max_length,
        batch_size,
        args.num_workers,
        args.neg_per_pos,
        args.seed,
        include_x,
        device,
    )

    if is_gnn:
        print(f"--- Running GNN Pipeline for model: {args.model} ---", flush=True)
        gnn_cfg = GNNTrainConfig(
            epochs=int(args.epochs),
            lr=float(args.gnn_lr),
            weight_decay=float(args.gnn_weight_decay),
            hidden=int(args.hidden),
            layers=int(args.layers),
            dropedge_p=0.0,
            scheduler=str(args.gnn_scheduler),
            neg_pos_ratio=1.0,
            gnn_zero_supervised=True,
            batch_size=1,
            grad_clip=float(args.gnn_grad_clip)
        )
        bundle = run_gnn_edges_suite(task=task, encoders=[args.model], cfg=gnn_cfg)
        _print_gold_style_metrics(bundle, args.model)
    else:
        print(f"--- Running EC Pipeline for model: {args.model} ---", flush=True)
        ec_cfg = TNNTrainConfig(
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            early_stop_patience=4,
            threshold_metric="bacc",
            rf_neg_pos_ratio=4.0,
            lr=float(args.dense_lr),
            weight_decay=float(args.dense_weight_decay),
            grad_clip=float(args.dense_grad_clip),
            cnn_hidden=int(args.cnn_hidden),
            cnn_head_kernel=int(args.cnn_head_kernel),
            cnn_dropout=float(args.cnn_dropout)
        )
        bundle = run_pipeline_for_task(task=task, models=[args.model], cfg=ec_cfg)
        _print_gold_style_metrics(bundle, args.model)

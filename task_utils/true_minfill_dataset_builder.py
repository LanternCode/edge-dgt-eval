# SPDX-License-Identifier: CC-BY-SA-4.0

import argparse
import hashlib
import itertools
import json
import os
import random
import shutil
import networkx as nx
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from pyomo.environ import Binary, ConcreteModel, ConstraintList, Objective, SolverFactory, Var, minimize, value
from pyomo.opt import TerminationCondition
from tqdm import tqdm
from pipeline.EdgeClassification import FeatureRegistry
from pipeline.GraphBenchmark import GraphBenchmark


def _default_out_dir() -> str:
    return str(Path(__file__).resolve().parents[1] / "data" / "minfill_true_1000")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_jsonl(path: str, records: Iterable[dict]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _read_jsonl(path: str) -> List[dict]:
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(json.loads(s))
    return out


def _edges_hash(num_nodes: int, edges: Sequence[Tuple[int, int]]) -> str:
    pairs = sorted((u, v) if u < v else (v, u) for u, v in edges if u != v)
    s = f"n={num_nodes};" + ";".join(f"{u},{v}" for u, v in pairs)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _graph_to_edges(g: nx.Graph) -> List[Tuple[int, int]]:
    edges = [(int(u), int(v)) if int(u) < int(v) else (int(v), int(u)) for u, v in g.edges() if u != v]
    return sorted(set(edges))


def _generate_non_chordal_graph(n: int, p: float) -> nx.Graph:
    while True:
        g = nx.erdos_renyi_graph(n, p, seed=None)
        g.remove_edges_from(nx.selfloop_edges(g))
        if nx.is_chordal(g):
            continue

        if not nx.is_connected(g):
            comps = list(nx.connected_components(g))
            largest = max(comps, key=len)
            g = g.subgraph(largest).copy()
            mapping = {old: i for i, old in enumerate(sorted(g.nodes()))}
            g = nx.relabel_nodes(g, mapping, copy=True)

        if g.number_of_nodes() >= 6 and not nx.is_chordal(g):
            return g


def _cycle_has_chord(h: nx.Graph, cycle: List[int]) -> bool:
    m = len(cycle)
    if m < 4:
        return False
    for i in range(m):
        u = cycle[i]
        for j in range(i + 2, m):
            if i == 0 and j == m - 1:
                continue
            v = cycle[j]
            if h.has_edge(u, v):
                return True
    return False


def _reduce_to_chordless_cycle(h: nx.Graph, cycle: List[int]) -> List[int]:
    c = list(cycle)
    while True:
        m = len(c)
        if m < 4:
            raise RuntimeError("Cycle collapsed below length 4 during chordless reduction.")
        chord_found = False
        for i in range(m):
            u = c[i]
            for j in range(i + 2, m):
                if i == 0 and j == m - 1:
                    continue
                v = c[j]
                if h.has_edge(u, v):
                    chord_found = True
                    path1 = c[i: j + 1]
                    path2 = c[j:] + c[: i + 1]
                    if len(path1) >= 4:
                        c = path1
                    else:
                        c = path2
                    break
            if chord_found:
                break
        if not chord_found:
            return c


def _find_chordless_cycle_certificate(h: nx.Graph) -> List[int]:
    if nx.is_chordal(h):
        raise RuntimeError("Graph is chordal; no certificate exists.")

    try:
        breaker = nx.algorithms.chordal._find_chordality_breaker(h)
    except Exception:
        breaker = None

    if breaker:
        u, v, w = breaker
        h_minus_v = h.copy()
        if v in h_minus_v:
            h_minus_v.remove_node(v)

        try:
            path_w_to_u = nx.shortest_path(h_minus_v, source=w, target=u)
        except nx.NetworkXNoPath:
            path_w_to_u = nx.shortest_path(h, source=w, target=u)
            path_w_to_u = [node for node in path_w_to_u if node != v]

        cycle = [u, v] + path_w_to_u
        if cycle[-1] == u:
            cycle = cycle[:-1]

        cycle = _reduce_to_chordless_cycle(h, cycle)
        if not _cycle_has_chord(h, cycle):
            return cycle

    cycles = nx.cycle_basis(h)
    for cyc in cycles:
        if len(cyc) < 4:
            continue
        cycle = _reduce_to_chordless_cycle(h, cyc)
        if len(cycle) >= 4 and not _cycle_has_chord(h, cycle):
            return cycle

    raise RuntimeError("Failed to construct a chordless cycle certificate from a non-chordal graph.")


def _solver_set_time_limit(solver, solver_name: str, time_limit_s: Optional[float]) -> None:
    if time_limit_s is None:
        return
    name = solver_name.lower().strip()
    if name == "cbc":
        solver.options["seconds"] = float(time_limit_s)
    elif name == "gurobi":
        solver.options["TimeLimit"] = float(time_limit_s)
    elif name == "cplex":
        solver.options["timelimit"] = float(time_limit_s)
    elif name == "glpk":
        solver.options["tmlim"] = int(time_limit_s)
    else:
        solver.options["timelimit"] = float(time_limit_s)


def _candidate_pairs_biconnected(g: nx.Graph) -> Set[Tuple[int, int]]:
    edge_set = {tuple(sorted((int(u), int(v)))) for u, v in g.edges() if u != v}

    candidates: Set[Tuple[int, int]] = set()
    for comp in nx.biconnected_components(g):
        nodes = sorted(int(x) for x in comp)
        if len(nodes) < 3:
            continue
        for u, v in itertools.combinations(nodes, 2):
            e = (u, v) if u < v else (v, u)
            if e not in edge_set:
                candidates.add(e)

    if not candidates:
        n = g.number_of_nodes()
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in edge_set:
                    candidates.add((i, j))

    return candidates


def _heuristic_upper_bound(g: nx.Graph) -> Tuple[int, List[Tuple[int, int]]]:
    h, _ = nx.algorithms.chordal.complete_to_chordal_graph(g)
    g_edges = set(_graph_to_edges(g))
    h_edges = set(_graph_to_edges(h))
    added = sorted(h_edges - g_edges)
    return len(added), added


def solve_true_min_fill_chordal_completion(
        g: nx.Graph,
        solver_name: str,
        *,
        time_limit_s: Optional[float],
        max_cuts: int,
        tee: bool,
) -> List[Tuple[int, int]]:
    candidates = sorted(_candidate_pairs_biconnected(g))
    candidate_set = set(candidates)

    ub, warm_added = _heuristic_upper_bound(g)

    model = ConcreteModel()
    model.x = Var(candidates, domain=Binary)
    model.constraints = ConstraintList()
    model.constraints.add(sum(model.x[i, j] for (i, j) in candidates) <= ub)
    model.obj = Objective(expr=sum(model.x[i, j] for (i, j) in candidates), sense=minimize)

    warm_set = set(warm_added)
    for (i, j) in candidates:
        model.x[i, j].value = 1 if (i, j) in warm_set else 0

    solver = SolverFactory(solver_name)
    _solver_set_time_limit(solver, solver_name=solver_name, time_limit_s=time_limit_s)

    def _current_fill() -> List[Tuple[int, int]]:
        fill: List[Tuple[int, int]] = []
        for (i, j) in candidates:
            if value(model.x[i, j]) >= 0.5:
                fill.append((int(i), int(j)))
        return sorted(set(fill))

    for _ in range(max_cuts + 1):
        results = solver.solve(model, tee=tee)
        term = results.solver.termination_condition
        if term != TerminationCondition.optimal:
            raise RuntimeError(f"MILP not optimal: termination_condition={term}")

        fill_edges = _current_fill()
        h = nx.Graph(g)
        h.add_edges_from(fill_edges)

        if nx.is_chordal(h):
            return fill_edges

        cycle = _find_chordless_cycle_certificate(h)
        m = len(cycle)

        chord_pairs: List[Tuple[int, int]] = []
        for a in range(m):
            for b in range(a + 2, m):
                if a == 0 and b == m - 1:
                    continue
                u, v = int(cycle[a]), int(cycle[b])
                e = (u, v) if u < v else (v, u)
                if e in candidate_set:
                    chord_pairs.append(e)

        chord_pairs = sorted(set(chord_pairs))
        if not chord_pairs:
            raise RuntimeError("Chordless cycle found but no available chord variables to cut it.")

        model.constraints.add(sum(model.x[u, v] for (u, v) in chord_pairs) >= 1)

    raise RuntimeError(f"Exceeded max_cuts={max_cuts} without reaching chordality.")


@dataclass
class GenConfig:
    out_dir: str
    shard_id: int
    num_shards: int
    graphs_total: int
    min_nodes: int
    max_nodes: int
    er_p: float
    solver_name: str
    solver_time_limit_s: Optional[float]
    max_cuts: int
    max_attempts_per_graph: int
    workers: int
    tee: bool


def _graphs_for_shard(graphs_total: int, shard_id: int, num_shards: int) -> int:
    base = graphs_total // num_shards
    rem = graphs_total % num_shards
    return base + (1 if shard_id < rem else 0)


def _worker_generate_records(args: Tuple[int, GenConfig]) -> List[dict]:
    worker_id, cfg = args

    graphs_target = _graphs_for_shard(cfg.graphs_total, cfg.shard_id, cfg.num_shards)
    per_worker = graphs_target // cfg.workers
    extra = graphs_target % cfg.workers
    my_target = per_worker + (1 if worker_id < extra else 0)

    records: List[dict] = []
    local_idx = 0

    while len(records) < my_target:
        n = random.randint(cfg.min_nodes, cfg.max_nodes)
        accepted = False

        for _ in range(cfg.max_attempts_per_graph):
            g = _generate_non_chordal_graph(n=n, p=cfg.er_p)
            if g.number_of_nodes() < cfg.min_nodes:
                continue

            try:
                fill_edges = solve_true_min_fill_chordal_completion(
                    g,
                    solver_name=cfg.solver_name,
                    time_limit_s=cfg.solver_time_limit_s,
                    max_cuts=cfg.max_cuts,
                    tee=cfg.tee,
                )
            except Exception:
                continue

            edges = _graph_to_edges(g)
            fill_edges = [(u, v) if u < v else (v, u) for u, v in fill_edges]
            fill_edges = sorted(set(fill_edges))

            h = nx.Graph(g)
            h.add_edges_from(fill_edges)
            if not nx.is_chordal(h):
                continue

            gid = int(cfg.shard_id) * 10_000_000 + int(worker_id) * 100_000 + int(local_idx)
            local_idx += 1

            rec = {
                "graph_id": gid,
                "num_nodes": int(g.number_of_nodes()),
                "edges": [[int(u), int(v)] for (u, v) in edges],
                "fill_edges": [[int(u), int(v)] for (u, v) in fill_edges],
                "graph_hash": _edges_hash(int(g.number_of_nodes()), edges),
                "split": None,
            }
            records.append(rec)
            accepted = True
            break

        if not accepted:
            continue

    return records


def infer_shard_from_slurm(shard_id: Optional[int], num_shards: Optional[int]) -> Tuple[int, int]:
    if shard_id is not None and num_shards is not None:
        return int(shard_id), int(num_shards)

    slurm_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    slurm_count = os.environ.get("SLURM_ARRAY_TASK_COUNT")

    if slurm_task is not None and slurm_count is not None:
        return int(slurm_task), int(slurm_count)

    if shard_id is None or num_shards is None:
        raise ValueError("Provide --shard_id and --num_shards, or run under a Slurm array.")
    return int(shard_id), int(num_shards)


def cmd_generate(args: argparse.Namespace) -> None:
    shard_id, num_shards = infer_shard_from_slurm(args.shard_id, args.num_shards)

    cfg = GenConfig(
        out_dir=args.out_dir,
        shard_id=shard_id,
        num_shards=num_shards,
        graphs_total=args.graphs,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        er_p=args.er_p,
        solver_name=args.solver,
        solver_time_limit_s=args.solver_time_limit_s,
        max_cuts=args.max_cuts,
        max_attempts_per_graph=args.max_attempts_per_graph,
        workers=args.workers,
        tee=args.tee,
    )

    shard_dir = os.path.join(cfg.out_dir, "_work", "shards")
    _ensure_dir(shard_dir)

    shard_path = os.path.join(shard_dir, f"graphs_shard_{cfg.shard_id:04d}.jsonl")
    if os.path.exists(shard_path) and not args.overwrite:
        raise FileExistsError(f"Shard already exists: {shard_path} (use --overwrite)")

    ctx = get_context("spawn")
    work = [(wid, cfg) for wid in range(cfg.workers)]
    all_records: List[dict] = []

    if cfg.workers == 1:
        all_records = _worker_generate_records((0, cfg))
    else:
        with ctx.Pool(processes=cfg.workers) as pool:
            for recs in tqdm(pool.imap_unordered(_worker_generate_records, work), total=len(work), desc="Workers"):
                all_records.extend(recs)

    random.shuffle(all_records)

    _write_jsonl(shard_path, all_records)
    print(f"Wrote {len(all_records)} records to {shard_path}")


def _record_to_adjacency_np(rec: dict) -> np.ndarray:
    n = int(rec["num_nodes"])
    a = np.zeros((n, n), dtype=np.float32)
    for u, v in rec["edges"]:
        uu = int(u)
        vv = int(v)
        a[uu, vv] = 1.0
        a[vv, uu] = 1.0
    np.fill_diagonal(a, 0.0)
    return a


def _compute_features_for_records(
        records: Sequence[dict],
        feature_keys: Optional[List[str]],
) -> Tuple[Dict[int, Dict[str, "torch.Tensor"]], Dict[int, str]]:
    bench = GraphBenchmark(show_progress=False)

    if feature_keys is None:
        keys = sorted(FeatureRegistry.CANONICAL)
    else:
        keys = feature_keys

    features_by_id: Dict[int, Dict[str, "torch.Tensor"]] = {}
    graph_hashes: Dict[int, str] = {}

    for rec in tqdm(records, desc="Computing features", leave=False):
        gid = int(rec["graph_id"])
        a = _record_to_adjacency_np(rec)
        a_bin = (a > 0.5).astype(np.float32)
        feats_np = bench.extract_features(a_bin, feature_set=keys)

        feats: Dict[str, "torch.Tensor"] = {}
        for k, v in feats_np.items():
            feats[str(k)] = v if isinstance(v, torch.Tensor) else torch.as_tensor(v).float()

        features_by_id[gid] = feats
        graph_hashes[gid] = str(rec.get("graph_hash", ""))

    return features_by_id, graph_hashes


def cmd_features(args: argparse.Namespace) -> None:
    shard_id, num_shards = infer_shard_from_slurm(args.shard_id, args.num_shards)

    shard_dir = os.path.join(args.out_dir, "_work", "shards")
    graphs_path = os.path.join(shard_dir, f"graphs_shard_{shard_id:04d}.jsonl")
    feats_path = os.path.join(shard_dir, f"features_shard_{shard_id:04d}.pt")

    if not os.path.exists(graphs_path):
        raise FileNotFoundError(f"Missing shard graphs file: {graphs_path}")

    if os.path.exists(feats_path) and not args.overwrite:
        raise FileExistsError(f"Shard features already exist: {feats_path} (use --overwrite)")

    records = _read_jsonl(graphs_path)

    feature_keys = None
    if args.feature_set and args.feature_set != "canonical":
        feature_keys = [s.strip() for s in args.feature_set.split(",") if s.strip()]

    features_by_id, graph_hashes = _compute_features_for_records(records, feature_keys=feature_keys)

    payload = {
        "graph_ids": sorted(features_by_id.keys()),
        "feature_set": "canonical" if feature_keys is None else args.feature_set,
        "features": features_by_id,
        "graph_hashes": graph_hashes,
        "shard_id": int(shard_id),
        "num_shards": int(num_shards),
    }

    torch.save(payload, feats_path)
    print(f"Wrote features for {len(features_by_id)} graphs to {feats_path}")


def _assign_splits(records: List[dict], ratios: Tuple[float, float, float]) -> None:
    if len(ratios) != 3 or not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios must sum to 1, got {ratios!r}")

    random.shuffle(records)

    n_total = len(records)
    n_train = max(1, int(ratios[0] * n_total))
    n_val = max(1, int(ratios[1] * n_total))

    for i, rec in enumerate(records):
        if i < n_train:
            rec["split"] = "train"
        elif i < n_train + n_val:
            rec["split"] = "val"
        else:
            rec["split"] = "test"


def cmd_merge(args: argparse.Namespace) -> None:
    work_dir = os.path.join(args.out_dir, "_work")
    shard_dir = os.path.join(work_dir, "shards")
    if not os.path.isdir(shard_dir):
        raise FileNotFoundError(f"Missing shards directory: {shard_dir}")

    graph_shards = sorted(
        f for f in os.listdir(shard_dir) if f.startswith("graphs_shard_") and f.endswith(".jsonl")
    )
    feat_shards = sorted(
        f for f in os.listdir(shard_dir) if f.startswith("features_shard_") and f.endswith(".pt")
    )

    if not graph_shards:
        raise FileNotFoundError("No graph shards found.")
    if not feat_shards:
        raise FileNotFoundError("No feature shards found. Run `features` first.")

    all_records: List[dict] = []
    for fn in graph_shards:
        all_records.extend(_read_jsonl(os.path.join(shard_dir, fn)))

    all_records = sorted(all_records, key=lambda r: int(r["graph_id"]))
    old_to_new: Dict[int, int] = {}
    for new_id, rec in enumerate(all_records):
        old_id = int(rec["graph_id"])
        old_to_new[old_id] = int(new_id)
        rec["graph_id"] = int(new_id)

    _assign_splits(all_records, ratios=(args.train_ratio, args.val_ratio, args.test_ratio))

    out_jsonl = os.path.join(args.out_dir, args.out_jsonl)
    _write_jsonl(out_jsonl, all_records)
    print(f"Wrote merged dataset JSONL: {out_jsonl} (graphs={len(all_records)})")

    merged_features: Dict[int, Dict[str, "torch.Tensor"]] = {}
    merged_hashes: Dict[int, str] = {}

    for fn in feat_shards:
        payload = torch.load(os.path.join(shard_dir, fn), map_location="cpu")
        feats = payload["features"]
        hashes = payload.get("graph_hashes", {})

        for old_id_str, fdict in feats.items():
            old_id = int(old_id_str)
            if old_id not in old_to_new:
                continue
            new_id = old_to_new[old_id]
            merged_features[new_id] = fdict
            if old_id in hashes:
                merged_hashes[new_id] = str(hashes[old_id])

    out_pt = out_jsonl + ".features.pt"
    final_payload = {
        "graph_ids": sorted(merged_features.keys()),
        "feature_set": "canonical",
        "features": merged_features,
        "graph_hashes": merged_hashes,
    }
    torch.save(final_payload, out_pt)
    print(f"Wrote merged features: {out_pt} (graphs={len(merged_features)})")

    shutil.rmtree(work_dir, ignore_errors=True)
    print(f"Removed temporary work directory: {work_dir}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("true_minfill_dataset_builder")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate JSONL shard(s) of true min-fill chordal completions (CPU-bound)")
    g.add_argument("--out_dir", type=str, default=_default_out_dir())
    g.add_argument("--graphs", type=int, default=1000)
    g.add_argument("--min_nodes", type=int, default=6)
    g.add_argument("--max_nodes", type=int, default=140)
    g.add_argument("--er_p", type=float, default=0.3)
    g.add_argument("--solver", type=str, default="cbc")
    g.add_argument("--solver_time_limit_s", type=float, default=120.0)
    g.add_argument("--max_cuts", type=int, default=2000)
    g.add_argument("--max_attempts_per_graph", type=int, default=80)
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--shard_id", type=int, default=None)
    g.add_argument("--num_shards", type=int, default=None)
    g.add_argument("--tee", action="store_true")
    g.add_argument("--overwrite", action="store_true")

    f = sub.add_parser("features", help="Compute feature shard for a JSONL shard (CPU-bound in most setups)")
    f.add_argument("--out_dir", type=str, default=_default_out_dir())
    f.add_argument("--feature_set", type=str, default="canonical", help="'canonical' or comma-separated keys")
    f.add_argument("--shard_id", type=int, default=None)
    f.add_argument("--num_shards", type=int, default=None)
    f.add_argument("--overwrite", action="store_true")

    m = sub.add_parser("merge", help="Merge shards into final JSONL and .features.pt")
    m.add_argument("--out_dir", type=str, default=_default_out_dir())
    m.add_argument("--out_jsonl", type=str, default="min_fill_dataset.jsonl")
    m.add_argument("--train_ratio", type=float, default=0.8)
    m.add_argument("--val_ratio", type=float, default=0.1)
    m.add_argument("--test_ratio", type=float, default=0.1)

    return p


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "features":
        cmd_features(args)
    elif args.cmd == "merge":
        cmd_merge(args)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()

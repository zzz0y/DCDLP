from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import networkx as nx
import pandas as pd

from .cn_intervention import intervene_cn
from .degree_intervention import intervene_degree
from .validator import InterventionError, common_neighbor_set


def build_intervention_cache(
    graph: nx.Graph, pairs, types: list[str], output: Path, seed: int,
    forbidden_edges: set[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    rows = []
    for index, (raw_u, raw_v) in enumerate(pairs):
        u, v = int(raw_u), int(raw_v)
        for offset, kind in enumerate(types):
            local_seed = seed * 1_000_003 + index * 101 + offset
            pre_cn = len(common_neighbor_set(graph, u, v))
            try:
                if kind.startswith("cn_"):
                    delta = 1 if "plus" in kind else -1
                    cf, log = intervene_cn(graph, u, v, delta, local_seed, forbidden_edges, str(index))
                else:
                    endpoint = "v" if "degree_v" in kind else "u"
                    delta = 1 if "plus" in kind else -1
                    cf, log = intervene_degree(graph, u, v, endpoint, delta, local_seed, forbidden_edges, str(index))
                row = asdict(log)
                row.update({
                    "pre_degree_u": graph.degree(u), "pre_degree_v": graph.degree(v),
                    "post_degree_u": cf.degree(u), "post_degree_v": cf.degree(v),
                    "pre_cn": pre_cn, "post_cn": len(common_neighbor_set(cf, u, v)),
                })
            except InterventionError as exc:
                row = {"pair_id": str(index), "u": u, "v": v, "intervention_type": kind,
                       "valid": False, "failure_code": exc.code, "seed": local_seed}
            rows.append(row)
    frame = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(output, index=False)
    except ImportError:
        frame.to_json(output.with_suffix(".jsonl"), orient="records", lines=True)
    return frame

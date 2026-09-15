"""Generic paired analysis for conditional-CN selective-routing runs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from dcdlp.evaluation.statistics import paired_bootstrap_ci
from dcdlp.utils import array_hash, write_json


PREDICTION_METRICS = (
    "mrr", "hits10", "hits20", "hits50", "hits100", "auc", "ap",
    "macro_mrr", "worst_group_mrr", "group_gap",
)
MECHANISM_METRICS = (
    "target_response", "cross_response", "routing_selectivity",
    "cross_sensitivity", "routing_ratio",
    "abs_delta_score_degree", "abs_delta_score_cn",
    "abs_delta_score_residual", "abs_delta_score_interaction",
    "interaction_share_before", "interaction_share_after",
    "interaction_share_cap_violation_before",
)


@dataclass(frozen=True)
class RunArtifact:
    result_path: Path
    record: dict
    mechanism_path: Path | None
    probe_path: Path | None

    @property
    def key(self) -> tuple[str, int]:
        return str(self.record.get("dataset", "")), int(self.record["seed"])


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_result_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    candidates = sorted(path.glob("raw/*.json"))
    if not candidates:
        candidates = sorted(path.glob("*.json"))
    if not candidates:
        candidates = sorted(path.rglob("raw/*.json"))
    return candidates


def _resolve_related(
    result_path: Path,
    record: dict,
    explicit_key: str,
    category: str,
    filename: str,
) -> Path | None:
    explicit = record.get(explicit_key)
    if explicit:
        candidate = Path(explicit)
        if not candidate.is_absolute():
            candidate = result_path.parent / candidate
        if candidate.exists():
            return candidate.resolve()
    run_id = result_path.stem
    roots = [result_path.parent, result_path.parent.parent]
    for root in roots:
        direct = root / category / run_id / filename
        if direct.exists():
            return direct.resolve()
        matches = list(root.glob(f"{category}/**/{run_id}/{filename}"))
        if matches:
            return matches[0].resolve()
    return None


def discover_runs(paths: Iterable[Path]) -> list[RunArtifact]:
    artifacts: list[RunArtifact] = []
    seen: set[Path] = set()
    for supplied in paths:
        for path in _candidate_result_files(supplied.resolve()):
            path = path.resolve()
            if path in seen:
                continue
            try:
                record = _read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            if not {"dataset", "seed", "metrics", "config_hash"} <= set(record):
                continue
            seen.add(path)
            artifacts.append(RunArtifact(
                path,
                record,
                _resolve_related(
                    path, record, "mechanism_predictions",
                    "mechanism_audit", "intervention_predictions.csv",
                ),
                _resolve_related(
                    path, record, "probe_results",
                    "probe", "posthoc_probe.csv",
                ),
            ))
    return artifacts


def _index_unique(
    artifacts: list[RunArtifact], label: str
) -> dict[tuple[str, int], RunArtifact]:
    indexed: dict[tuple[str, int], RunArtifact] = {}
    for artifact in artifacts:
        if artifact.key in indexed:
            raise ValueError(f"Duplicate {label} run for {artifact.key}")
        indexed[artifact.key] = artifact
    return indexed


def _prediction_pair_hash(artifact: RunArtifact) -> str | None:
    prediction = artifact.record.get("predictions")
    if not prediction:
        return None
    path = Path(prediction)
    if not path.is_absolute():
        path = artifact.result_path.parent / path
    if not path.exists():
        return None
    frame = pd.read_csv(path, usecols=["u", "v"])
    return array_hash(frame[["u", "v"]].to_numpy(dtype=np.int64))


def validate_pair(
    baseline: RunArtifact, candidate: RunArtifact
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for field in ("dataset", "seed", "protocol_train", "protocol_eval"):
        if baseline.record.get(field) != candidate.record.get(field):
            reasons.append(f"{field} mismatch")
    baseline_integrity = baseline.record.get("data_integrity", {})
    candidate_integrity = candidate.record.get("data_integrity", {})
    for field in ("split_hash", "test_positive_hash", "candidate_hash"):
        left, right = baseline_integrity.get(field), candidate_integrity.get(field)
        if not left or not right:
            reasons.append(f"{field} missing")
        elif left != right:
            reasons.append(f"{field} mismatch")
    left_prediction = _prediction_pair_hash(baseline)
    right_prediction = _prediction_pair_hash(candidate)
    if left_prediction and right_prediction and left_prediction != right_prediction:
        reasons.append("prediction positive-pair order mismatch")
    return not reasons, reasons


def pair_runs(
    baseline_runs: list[RunArtifact], candidate_runs: list[RunArtifact]
) -> tuple[list[tuple[RunArtifact, RunArtifact]], list[dict]]:
    baseline = _index_unique(baseline_runs, "baseline")
    candidate = _index_unique(candidate_runs, "candidate")
    pairs: list[tuple[RunArtifact, RunArtifact]] = []
    audit_rows: list[dict] = []
    for key in sorted(set(baseline) | set(candidate)):
        left, right = baseline.get(key), candidate.get(key)
        if left is None or right is None:
            audit_rows.append({
                "dataset": key[0], "seed": key[1], "status": "MISSING",
                "reason": "baseline missing" if left is None else "candidate missing",
            })
            continue
        valid, reasons = validate_pair(left, right)
        audit_rows.append({
            "dataset": key[0], "seed": key[1],
            "status": "AVAILABLE" if valid else "INVALID_PAIR",
            "reason": "; ".join(reasons),
            "baseline_result": str(left.result_path),
            "candidate_result": str(right.result_path),
        })
        if valid:
            pairs.append((left, right))
    return pairs, audit_rows


def prediction_deltas(
    pairs: list[tuple[RunArtifact, RunArtifact]]
) -> pd.DataFrame:
    rows: list[dict] = []
    for baseline, candidate in pairs:
        left, right = baseline.record["metrics"], candidate.record["metrics"]
        for metric in PREDICTION_METRICS:
            if metric not in left or metric not in right:
                continue
            rows.append({
                "dataset": baseline.key[0], "seed": baseline.key[1],
                "metric": metric, "baseline": float(left[metric]),
                "candidate": float(right[metric]),
                "delta": float(right[metric]) - float(left[metric]),
            })
        for group in ("HH", "HL", "LH", "LL"):
            left_group = left.get("groups", {}).get(group, {})
            right_group = right.get("groups", {}).get(group, {})
            for name in ("mrr", "count"):
                if name in left_group and name in right_group:
                    rows.append({
                        "dataset": baseline.key[0], "seed": baseline.key[1],
                        "metric": f"group_{group}_{name}",
                        "baseline": float(left_group[name]),
                        "candidate": float(right_group[name]),
                        "delta": float(right_group[name]) - float(left_group[name]),
                    })
    return pd.DataFrame(rows)


def _safe_statistic(function, *args) -> float:
    try:
        value = function(*args)
        value = value.statistic if hasattr(value, "statistic") else value
        return float(value) if np.isfinite(value) else np.nan
    except (ValueError, ZeroDivisionError):
        return np.nan


def _safe_pvalue(function, *args) -> float:
    try:
        value = float(function(*args).pvalue)
        return value if np.isfinite(value) else np.nan
    except (ValueError, ZeroDivisionError):
        return np.nan


def paired_statistics(deltas: pd.DataFrame, bootstrap_seed: int) -> pd.DataFrame:
    rows: list[dict] = []
    if deltas.empty:
        return pd.DataFrame(rows)
    for (dataset, metric), frame in deltas.groupby(["dataset", "metric"]):
        values = frame["delta"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        ci_low, ci_high = paired_bootstrap_ci(
            values, seed=bootstrap_seed
        )
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append({
            "dataset": dataset, "metric": metric, "count": len(values),
            "mean_delta": float(values.mean()),
            "median_delta": float(np.median(values)), "std_delta": std,
            "bootstrap_ci_low": ci_low, "bootstrap_ci_high": ci_high,
            "paired_t_statistic": _safe_statistic(stats.ttest_1samp, values, 0.0),
            "paired_t_pvalue": _safe_pvalue(stats.ttest_1samp, values, 0.0),
            "wilcoxon_statistic": _safe_statistic(stats.wilcoxon, values),
            "wilcoxon_pvalue": _safe_pvalue(stats.wilcoxon, values),
            "cohen_dz": float(values.mean() / std) if std > 0 else np.nan,
            "paired_sign_effect": float(
                (np.sum(values > 1e-12) - np.sum(values < -1e-12))
                / len(values)
            ),
            "interpretation": "exploratory_5_seed_screening",
        })
    return pd.DataFrame(rows)


def _failure_codes(frame: pd.DataFrame) -> str:
    if frame.empty or "failure_code" not in frame:
        return "{}"
    counts = frame.loc[
        frame.get("status", "") != "AVAILABLE", "failure_code"
    ].fillna("MISSING").value_counts().to_dict()
    return json.dumps({str(key): int(value) for key, value in counts.items()}, sort_keys=True)


def mechanism_comparisons(
    pairs: list[tuple[RunArtifact, RunArtifact]]
) -> pd.DataFrame:
    rows: list[dict] = []
    for baseline, candidate in pairs:
        if baseline.mechanism_path is None or candidate.mechanism_path is None:
            rows.append({
                "dataset": baseline.key[0], "seed": baseline.key[1],
                "intervention_kind": "ALL", "metric": "MISSING",
                "status": "MISSING", "reason": "intervention artifact missing",
            })
            continue
        left = pd.read_csv(baseline.mechanism_path)
        right = pd.read_csv(candidate.mechanism_path)
        for kind in ("cn", "degree"):
            left_kind = left[left["intervention_kind"] == kind]
            right_kind = right[right["intervention_kind"] == kind]
            left_valid = left_kind[left_kind["status"] == "AVAILABLE"]
            right_valid = right_kind[right_kind["status"] == "AVAILABLE"]
            keys = ["pair_id", "u", "v", "intervention_kind"]
            merged = left_valid.merge(
                right_valid, on=keys, suffixes=("_baseline", "_candidate")
            )
            union_count = len(pd.concat([
                left_valid[keys], right_valid[keys]
            ]).drop_duplicates())
            coverage = len(merged) / union_count if union_count else 0.0
            common = {
                "dataset": baseline.key[0], "seed": baseline.key[1],
                "intervention_kind": kind,
                "status": "AVAILABLE" if len(merged) else "MISSING",
                "paired_valid": len(merged),
                "baseline_attempted": len(left_kind),
                "baseline_valid": len(left_valid),
                "candidate_attempted": len(right_kind),
                "candidate_valid": len(right_valid),
                "intersection_coverage": coverage,
                "baseline_failure_codes": _failure_codes(left_kind),
                "candidate_failure_codes": _failure_codes(right_kind),
            }
            if merged.empty:
                rows.append({**common, "metric": "MISSING"})
                continue
            for metric in MECHANISM_METRICS:
                left_column, right_column = (
                    f"{metric}_baseline", f"{metric}_candidate"
                )
                if left_column not in merged or right_column not in merged:
                    continue
                left_values = merged[left_column].to_numpy(dtype=float)
                right_values = merged[right_column].to_numpy(dtype=float)
                rows.append({
                    **common, "metric": metric,
                    "baseline": float(np.nanmean(left_values)),
                    "candidate": float(np.nanmean(right_values)),
                    "delta": float(np.nanmean(right_values - left_values)),
                })
    frame = pd.DataFrame(rows)
    if frame.empty or "delta" not in frame:
        return frame
    available = frame[
        (frame["status"] == "AVAILABLE") & frame["delta"].notna()
    ]
    aggregate_rows: list[dict] = []
    for (dataset, kind, metric), local in available.groupby(
        ["dataset", "intervention_kind", "metric"]
    ):
        aggregate_rows.append({
            "dataset": dataset, "seed": "aggregate",
            "intervention_kind": kind, "metric": metric,
            "status": "AVAILABLE", "paired_valid": int(local["paired_valid"].sum()),
            "baseline": float(local["baseline"].mean()),
            "candidate": float(local["candidate"].mean()),
            "delta": float(local["delta"].mean()),
            "intersection_coverage": float(local["intersection_coverage"].mean()),
        })
    return pd.concat([frame, pd.DataFrame(aggregate_rows)], ignore_index=True)


def probe_comparisons(
    pairs: list[tuple[RunArtifact, RunArtifact]]
) -> pd.DataFrame:
    rows: list[dict] = []
    for baseline, candidate in pairs:
        if baseline.probe_path is None or candidate.probe_path is None:
            rows.append({
                "dataset": baseline.key[0], "model_seed": baseline.key[1],
                "probe": "MISSING", "metric": "MISSING", "status": "MISSING",
                "reason": "probe artifact missing",
            })
            continue
        left = pd.read_csv(baseline.probe_path)
        right = pd.read_csv(candidate.probe_path)
        keys = ["probe", "role", "probe_seed"]
        merged = left.merge(right, on=keys, suffixes=("_baseline", "_candidate"))
        for _, item in merged.iterrows():
            for metric in ("r2", "mae", "spearman"):
                left_value = float(item[f"{metric}_baseline"])
                right_value = float(item[f"{metric}_candidate"])
                rows.append({
                    "dataset": baseline.key[0], "model_seed": baseline.key[1],
                    "probe": item["probe"], "role": item["role"],
                    "probe_seed": int(item["probe_seed"]), "metric": metric,
                    "status": "AVAILABLE", "baseline": left_value,
                    "candidate": right_value, "delta": right_value - left_value,
                })
    frame = pd.DataFrame(rows)
    if frame.empty or "delta" not in frame:
        return frame
    aggregate_rows: list[dict] = []
    available = frame[
        (frame["status"] == "AVAILABLE") & frame["delta"].notna()
    ]
    for (dataset, probe, role, metric), local in available.groupby(
        ["dataset", "probe", "role", "metric"]
    ):
        aggregate_rows.append({
            "dataset": dataset, "model_seed": "aggregate", "probe": probe,
            "role": role, "probe_seed": "aggregate", "metric": metric,
            "status": "AVAILABLE", "baseline": float(local["baseline"].mean()),
            "candidate": float(local["candidate"].mean()),
            "delta": float(local["delta"].mean()), "count": len(local),
        })
    return pd.concat([frame, pd.DataFrame(aggregate_rows)], ignore_index=True)


def _metric_deltas(frame: pd.DataFrame, metric: str) -> np.ndarray:
    if frame.empty:
        return np.asarray([], dtype=float)
    local = frame[frame["metric"] == metric]
    return local["delta"].dropna().to_numpy(dtype=float)


def make_decision(
    pairs: list[tuple[RunArtifact, RunArtifact]],
    prediction: pd.DataFrame,
    mechanism: pd.DataFrame,
    probes: pd.DataFrame,
    pair_audit: list[dict],
) -> dict:
    mrr = _metric_deltas(prediction, "mrr")
    baseline_mrr = prediction.loc[
        prediction["metric"] == "mrr", "baseline"
    ].to_numpy(dtype=float) if not prediction.empty else np.asarray([])
    relative_drop = (
        max(0.0, -float(mrr.mean())) / float(baseline_mrr.mean())
        if len(mrr) and len(baseline_mrr) and baseline_mrr.mean() > 0
        else np.inf
    )

    def mechanism_values(kind: str, metric: str) -> np.ndarray:
        if mechanism.empty or "delta" not in mechanism:
            return np.asarray([])
        local = mechanism[
            (mechanism["seed"] != "aggregate")
            & (mechanism["intervention_kind"] == kind)
            & (mechanism["metric"] == metric)
            & (mechanism["status"] == "AVAILABLE")
        ]
        return local["delta"].dropna().to_numpy(dtype=float)

    route = {
        kind: mechanism_values(kind, "routing_selectivity")
        for kind in ("cn", "degree")
    }
    target = {
        kind: mechanism_values(kind, "target_response")
        for kind in ("cn", "degree")
    }
    cross = {
        kind: mechanism_values(kind, "cross_sensitivity")
        for kind in ("cn", "degree")
    }
    routing_ok = all(
        len(values)
        and values.mean() > 0
        and np.sum(values > 0) > len(values) / 2
        for values in route.values()
    )
    target_ok = all(
        len(values) and values.mean() >= -1e-12
        for values in target.values()
    )
    cross_improved = any(
        len(values) and values.mean() < 0 for values in cross.values()
    )
    cross_probe = probes[
        (probes.get("model_seed") == "aggregate")
        & (probes.get("role") == "cross_leakage")
        & (probes.get("metric") == "r2")
    ] if not probes.empty and "role" in probes else pd.DataFrame()
    probe_improved = bool(
        len(cross_probe) and (cross_probe["delta"] < 0).any()
    )

    interaction_rows = mechanism[
        (mechanism.get("seed") == "aggregate")
        & (mechanism.get("metric") == "interaction_share_before")
    ] if not mechanism.empty and "metric" in mechanism else pd.DataFrame()
    cap = min([
        float(pair[1].record.get("config", {}).get(
            "interaction_share_cap",
            pair[1].record.get("loss_weights", {}).get(
                "interaction_share_cap", 0.2
            ),
        ))
        for pair in pairs
    ], default=0.2)
    interaction_mean = (
        float(interaction_rows["candidate"].mean())
        if len(interaction_rows) else np.nan
    )
    modes_audited = bool(pairs) and all(
        pair[1].record.get("interaction_mode") == "audited" for pair in pairs
    )
    interaction_ok = bool(
        modes_audited and np.isfinite(interaction_mean)
        and 1e-12 < interaction_mean <= cap
    )
    mrr_ok = bool(len(mrr) and relative_drop <= 0.01)
    support_ok = cross_improved or probe_improved
    complete = bool(pairs) and all(
        row["status"] == "AVAILABLE" for row in pair_audit
    ) and len(mrr) == len(pairs)

    worst = _metric_deltas(prediction, "worst_group_mrr")
    gap = _metric_deltas(prediction, "group_gap")
    group_improved = bool(
        len(worst) and len(gap) and worst.mean() >= 0 and gap.mean() <= 0
    )
    mrr_improved = bool(len(mrr) and mrr.mean() > 0)

    if complete and mrr_ok and routing_ok and target_ok and support_ok and interaction_ok:
        status = (
            "GO_TO_MULTI_DATASET_CONFIRMATION"
            if group_improved or not mrr_improved
            else "CONDITIONAL_GO"
        )
    elif complete and mrr_improved and not (
        routing_ok and target_ok and probe_improved
    ):
        status = "PIVOT_TO_CALIBRATION"
    else:
        status = "STOP_DCDLP_RESCUE"
    return {
        "decision": status,
        "confirmatory_claim_allowed": False,
        "screening_scope": "Cora 5-seed exploratory route screening",
        "criteria": {
            "complete_pairs": {"passed": complete, "count": len(pairs)},
            "relative_mrr_drop_le_1pct": {
                "passed": mrr_ok, "relative_drop": relative_drop,
                "mean_delta": float(mrr.mean()) if len(mrr) else None,
            },
            "routing_selectivity": {
                kind: {
                    "passed": bool(
                        len(values) and values.mean() > 0
                        and np.sum(values > 0) > len(values) / 2
                    ),
                    "mean_delta": float(values.mean()) if len(values) else None,
                    "improved_seeds": int(np.sum(values > 0)),
                    "seed_count": len(values),
                }
                for kind, values in route.items()
            },
            "target_response_not_reduced": {
                kind: {
                    "passed": bool(len(values) and values.mean() >= -1e-12),
                    "mean_delta": float(values.mean()) if len(values) else None,
                }
                for kind, values in target.items()
            },
            "cross_or_probe_leakage_improved": {
                "passed": support_ok,
                "cross_sensitivity": cross_improved,
                "probe_r2": probe_improved,
            },
            "audited_interaction_controlled_nonzero": {
                "passed": interaction_ok,
                "mean_share": interaction_mean if np.isfinite(interaction_mean) else None,
                "cap": cap,
                "audited_mode": modes_audited,
            },
            "group_robustness_improved": {
                "passed": group_improved,
                "worst_group_mean_delta": float(worst.mean()) if len(worst) else None,
                "group_gap_mean_delta": float(gap.mean()) if len(gap) else None,
            },
        },
    }


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        pd.DataFrame([{"status": "MISSING"}]).to_csv(path, index=False)
    else:
        frame.to_csv(path, index=False)


def _report(decision: dict, pair_audit: list[dict]) -> str:
    criteria = decision["criteria"]
    return "\n".join([
        "# Phase-2 Selective Routing 决策报告",
        "",
        f"决策：`{decision['decision']}`",
        "",
        "本报告仅用于 Cora 5-seed 路线筛选，不构成确认性显著性结论。",
        "",
        "## 配对完整性",
        "",
        f"- 合法配对：{criteria['complete_pairs']['count']}",
        f"- 完整性通过：{criteria['complete_pairs']['passed']}",
        f"- 配对审计：`{json.dumps(pair_audit, ensure_ascii=False)}`",
        "",
        "## 冻结门槛",
        "",
        f"- MRR 相对下降不超过 1%：{criteria['relative_mrr_drop_le_1pct']}",
        f"- Routing selectivity：{criteria['routing_selectivity']}",
        f"- Target response：{criteria['target_response_not_reduced']}",
        f"- Cross/probe leakage：{criteria['cross_or_probe_leakage_improved']}",
        f"- Audited interaction：{criteria['audited_interaction_controlled_nonzero']}",
        f"- Group robustness（支持证据）：{criteria['group_robustness_improved']}",
        "",
        "所有统计检验均为探索性辅助；不会仅凭 5-seed t-test 宣称显著。",
        "",
    ])


def analyze(
    baseline_paths: list[Path],
    candidate_paths: list[Path],
    output_dir: Path,
    *,
    bootstrap_seed: int = 0,
) -> dict:
    baseline = discover_runs(baseline_paths)
    candidate = discover_runs(candidate_paths)
    pairs, pair_audit = pair_runs(baseline, candidate)
    prediction = prediction_deltas(pairs)
    statistics = paired_statistics(prediction, bootstrap_seed)
    mechanism = mechanism_comparisons(pairs)
    probes = probe_comparisons(pairs)
    decision = make_decision(
        pairs, prediction, mechanism, probes, pair_audit
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(statistics, output_dir / "phase2_selective_paired_stats.csv")
    _write_csv(prediction, output_dir / "phase2_selective_paired_deltas.csv")
    _write_csv(mechanism, output_dir / "phase2_selective_mechanism.csv")
    _write_csv(probes, output_dir / "phase2_selective_probe.csv")
    write_json(output_dir / "phase2_selective_decision.json", {
        **decision, "pair_audit": pair_audit,
        "baseline_runs": [str(item.result_path) for item in baseline],
        "candidate_runs": [str(item.result_path) for item in candidate],
    })
    (output_dir / "PHASE2_SELECTIVE_DECISION_REPORT.md").write_text(
        _report(decision, pair_audit), encoding="utf-8"
    )
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generic baseline/candidate selective-routing audit"
    )
    parser.add_argument("--baseline-run", action="append", required=True)
    parser.add_argument("--candidate-run", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    decision = analyze(
        [Path(value) for value in args.baseline_run],
        [Path(value) for value in args.candidate_run],
        Path(args.output_dir),
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


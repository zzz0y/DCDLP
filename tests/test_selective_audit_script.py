import json
from pathlib import Path

import pandas as pd

from scripts.analyze_phase2_selective_routing import analyze


def _write_run(root: Path, label: str, mrr: float, routing: float) -> Path:
    run_id = f"{label}_seed0"
    raw = root / label / "raw"
    mechanism_dir = root / label / "mechanism_audit" / run_id
    probe_dir = root / label / "probe" / run_id
    raw.mkdir(parents=True)
    mechanism_dir.mkdir(parents=True)
    probe_dir.mkdir(parents=True)
    prediction = raw / f"{run_id}_predictions.csv"
    pd.DataFrame([{"u": 0, "v": 1}]).to_csv(prediction, index=False)
    mechanism = mechanism_dir / "intervention_predictions.csv"
    rows = []
    for kind in ("cn", "degree"):
        rows.append({
            "pair_id": "0", "u": 0, "v": 1,
            "intervention_kind": kind, "status": "AVAILABLE",
            "failure_code": "", "target_response": 1.0 + routing,
            "cross_response": 1.0, "routing_selectivity": routing,
            "cross_sensitivity": 1.0 - routing,
            "routing_ratio": 1.0 + routing,
            "abs_delta_score_degree": 0.5,
            "abs_delta_score_cn": 0.5,
            "abs_delta_score_residual": 0.1,
            "abs_delta_score_interaction": 0.05,
            "interaction_share_before": 0.1,
            "interaction_share_after": 0.1,
            "interaction_share_cap_violation_before": 0,
        })
    pd.DataFrame(rows).to_csv(mechanism, index=False)
    probe = probe_dir / "posthoc_probe.csv"
    probe_rows = []
    for name, role in (
        ("z_cn_to_degree", "cross_leakage"),
        ("z_degree_to_degree", "target_retention"),
    ):
        probe_rows.append({
            "probe": name, "role": role, "probe_seed": 0,
            "r2": 0.2 if label == "candidate" else 0.3,
            "mae": 1.0, "spearman": 0.2,
        })
    pd.DataFrame(probe_rows).to_csv(probe, index=False)
    result = raw / f"{run_id}.json"
    record = {
        "dataset": "cora", "seed": 0, "protocol_train": "uniform",
        "protocol_eval": "heart", "config_hash": label,
        "interaction_mode": "audited",
        "config": {"interaction_share_cap": 0.2},
        "metrics": {
            "mrr": mrr, "hits10": 0.5, "hits20": 0.6,
            "hits50": 0.7, "hits100": 0.8, "auc": 0.7, "ap": 0.7,
            "macro_mrr": mrr, "worst_group_mrr": mrr,
            "group_gap": 0.1,
            "groups": {group: {"mrr": mrr, "count": 1}
                       for group in ("HH", "HL", "LH", "LL")},
        },
        "data_integrity": {
            "split_hash": "same", "test_positive_hash": "same",
            "candidate_hash": "same",
        },
        "predictions": str(prediction),
        "mechanism_predictions": str(mechanism),
        "probe_results": str(probe),
    }
    result.write_text(json.dumps(record), encoding="utf-8")
    return result


def test_generic_selective_audit_uses_cli_artifacts_without_model_hardcode(tmp_path):
    baseline = _write_run(tmp_path, "baseline", 0.30, 0.4)
    candidate = _write_run(tmp_path, "candidate", 0.31, 0.6)
    output = tmp_path / "analysis"
    decision = analyze([baseline], [candidate], output, bootstrap_seed=7)
    assert decision["decision"] in {
        "GO_TO_MULTI_DATASET_CONFIRMATION", "CONDITIONAL_GO"
    }
    for name in (
        "phase2_selective_paired_stats.csv",
        "phase2_selective_paired_deltas.csv",
        "phase2_selective_mechanism.csv",
        "phase2_selective_probe.csv",
        "phase2_selective_decision.json",
        "PHASE2_SELECTIVE_DECISION_REPORT.md",
    ):
        assert (output / name).exists()


def test_pairing_rejects_candidate_hash_mismatch(tmp_path):
    baseline = _write_run(tmp_path, "baseline", 0.30, 0.4)
    candidate = _write_run(tmp_path, "candidate", 0.31, 0.6)
    record = json.loads(candidate.read_text(encoding="utf-8"))
    record["data_integrity"]["candidate_hash"] = "different"
    candidate.write_text(json.dumps(record), encoding="utf-8")
    decision = analyze([baseline], [candidate], tmp_path / "analysis")
    assert decision["decision"] == "STOP_DCDLP_RESCUE"
    assert decision["criteria"]["complete_pairs"]["passed"] is False


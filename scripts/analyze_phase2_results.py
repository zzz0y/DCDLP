"""Paired seed-level analysis for the official phase-2 result bundle."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp, wilcoxon


MODES = {
    "raw_audited": ("raw", "audited"),
    "residual_audited": ("raw_plus_residual", "audited"),
    "residual_disabled": ("raw_plus_residual", "disabled"),
}
METRICS = ("mrr", "worst_group_mrr", "group_gap", "train_seconds")


def load_records(results_dir: Path) -> dict[tuple[str, int], dict]:
    records: dict[tuple[str, int], dict] = {}
    for path in sorted((results_dir / "raw").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        key = (str(record["cn_feature_mode"]), int(record["seed"]))
        interaction = str(record["interaction_mode"])
        records[(f"{key[0]}::{interaction}", key[1])] = record
    return records


def metric_value(record: dict, metric: str) -> float:
    if metric == "train_seconds":
        return float(record["runtime"]["train_seconds"])
    return float(record["metrics"][metric])


def paired_rows(records: dict[tuple[str, int], dict]) -> list[dict]:
    mode_keys = {
        name: (f"{feature}::{interaction}",)
        for name, (feature, interaction) in MODES.items()
    }
    seeds = sorted({seed for _, seed in records})
    rows: list[dict] = []
    comparisons = (
        ("residual_vs_raw", "residual_audited", "raw_audited"),
        ("interaction_vs_disabled", "residual_audited", "residual_disabled"),
    )
    for comparison, left_name, right_name in comparisons:
        left_key = mode_keys[left_name][0]
        right_key = mode_keys[right_name][0]
        for seed in seeds:
            left = records.get((left_key, seed))
            right = records.get((right_key, seed))
            if left is None or right is None:
                continue
            row = {"comparison": comparison, "seed": seed}
            for metric in METRICS:
                row[f"{metric}_left"] = metric_value(left, metric)
                row[f"{metric}_right"] = metric_value(right, metric)
                row[f"delta_{metric}"] = row[f"{metric}_left"] - row[f"{metric}_right"]
            rows.append(row)
    return rows


def p_values(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    ttest = float(ttest_1samp(values, 0.0).pvalue)
    try:
        wilcoxon_p = float(wilcoxon(values, alternative="two-sided").pvalue)
    except ValueError:
        wilcoxon_p = None
    return ttest, wilcoxon_p


def analyze(results_dir: Path, output_dir: Path) -> dict:
    records = load_records(results_dir)
    rows = paired_rows(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    delta_path = output_dir / "phase2_paired_deltas.csv"
    if rows:
        with delta_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        delta_path.write_text("comparison,seed\n", encoding="utf-8")

    summary: list[dict] = []
    for comparison in ("residual_vs_raw", "interaction_vs_disabled"):
        local = [row for row in rows if row["comparison"] == comparison]
        for metric in METRICS:
            values = [float(row[f"delta_{metric}"]) for row in local]
            ttest_p, wilcoxon_p = p_values(values)
            summary.append({
                "comparison": comparison,
                "metric": metric,
                "n": len(values),
                "mean_delta": float(np.mean(values)) if values else None,
                "std_delta": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "ttest_p": ttest_p,
                "wilcoxon_p": wilcoxon_p,
            })
    summary_path = output_dir / "phase2_paired_stats.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    def mean_delta(comparison: str, metric: str) -> float:
        values = [
            float(row[f"delta_{metric}"])
            for row in rows if row["comparison"] == comparison
        ]
        return float(np.mean(values)) if values else float("nan")

    decision = {
        "paired_seed_count": len({row["seed"] for row in rows}),
        "comparisons": {
            "residual_vs_raw": {
                "mean_delta_mrr": mean_delta("residual_vs_raw", "mrr"),
                "mean_delta_worst_group_mrr": mean_delta("residual_vs_raw", "worst_group_mrr"),
                "mean_delta_group_gap": mean_delta("residual_vs_raw", "group_gap"),
                "mrr_improved": mean_delta("residual_vs_raw", "mrr") > 0,
                "worst_group_non_degraded": mean_delta("residual_vs_raw", "worst_group_mrr") >= 0,
                "gap_non_increased": mean_delta("residual_vs_raw", "group_gap") <= 0,
            },
            "interaction_vs_disabled": {
                "mean_delta_mrr": mean_delta("interaction_vs_disabled", "mrr"),
                "mean_delta_worst_group_mrr": mean_delta("interaction_vs_disabled", "worst_group_mrr"),
                "mean_delta_group_gap": mean_delta("interaction_vs_disabled", "group_gap"),
            },
        },
    }
    gates = decision["comparisons"]["residual_vs_raw"]
    decision["verdict"] = (
        "PASS_ALL_PRIMARY_GATES"
        if gates["mrr_improved"]
        and gates["worst_group_non_degraded"]
        and gates["gap_non_increased"]
        else "PARTIAL_MRR_ONLY"
    )
    (output_dir / "phase2_paired_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    decision = analyze(args.results_dir, args.output_dir)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

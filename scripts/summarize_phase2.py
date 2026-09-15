"""Summarize the phase-2 suites without mixing legacy result files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev


SUITES = {"phase2_a5_raw", "phase2_a5_residual", "phase2_a5_no_interaction"}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "results" / "manifest.csv"
    result_to_suite: dict[str, str] = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("suite") in SUITES and row.get("status") == "COMPLETED":
                    result_file = row.get("result_file", "")
                    if result_file:
                        result_to_suite[Path(result_file).name] = row["suite"]

    records: list[dict] = []
    for path in (root / "results" / "raw").glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # A mechanism mode alone is not an experiment identity.  Require the
        # completed phase-2 manifest row so a future full-suite run cannot be
        # silently mixed into this comparison.
        suite = result_to_suite.get(path.name)
        if suite in SUITES:
            record["_suite"] = suite
            records.append(record)

    groups: dict[tuple[str, str, str], list[dict]] = {}
    for record in records:
        key = (
            str(record.get("dataset", "")),
            str(record.get("cn_feature_mode", "legacy_unknown")),
            str(record.get("interaction_mode", "legacy_unknown")),
        )
        groups.setdefault(key, []).append(record)

    rows: list[dict] = []
    for (dataset, feature_mode, interaction_mode), values in sorted(groups.items()):
        mrr = [float(item["metrics"]["mrr"]) for item in values]
        worst = [float(item["metrics"]["worst_group_mrr"]) for item in values]
        gap = [float(item["metrics"]["group_gap"]) for item in values]
        runtime = [float(item["runtime"]["train_seconds"]) for item in values]
        candidates = sorted({
            str(item.get("evaluation_candidates", {}).get("source", "missing"))
            for item in values
        })
        rows.append({
            "dataset": dataset,
            "cn_feature_mode": feature_mode,
            "interaction_mode": interaction_mode,
            "runs": len(values),
            "mrr_mean": mean(mrr),
            "mrr_std": stdev(mrr) if len(mrr) > 1 else 0.0,
            "worst_group_mrr_mean": mean(worst),
            "group_gap_mean": mean(gap),
            "train_seconds_mean": mean(runtime),
            "evaluation_candidate_sources": ";".join(candidates),
        })

    output = root / "results" / "aggregate" / "phase2_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "cn_feature_mode", "interaction_mode", "runs",
        "mrr_mean", "mrr_std", "worst_group_mrr_mean", "group_gap_mean",
        "train_seconds_mean", "evaluation_candidate_sources",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Phase-2 result groups: {len(rows)}")
    for row in rows:
        print(
            f"  {row['dataset']:8s} {row['cn_feature_mode']:17s} "
            f"interaction={row['interaction_mode']:10s} n={row['runs']} "
            f"MRR={row['mrr_mean']:.4f} candidates={row['evaluation_candidate_sources']}"
        )
    print("Phase-2 verdict: READY_FOR_REVIEW" if rows else "Phase-2 verdict: INCOMPLETE_NO_RESULTS")
    print(output)


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev


DATASETS = ("cora", "citeseer")
ABLATIONS = ("A0", "A5", "A14")
SEEDS = {0, 1, 2}
PROTOCOL = "uniform"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    values: dict[tuple[str, str], list[float]] = {
        (dataset, ablation): [] for dataset in DATASETS for ablation in ABLATIONS
    }
    for path in (root / "results" / "raw").glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        key = (record.get("dataset"), record.get("ablation"))
        if (key in values and record.get("seed") in SEEDS and
                record.get("protocol_train") == PROTOCOL):
            metric = record.get("metrics", {}).get("mrr")
            if metric is not None:
                values[key].append(float(metric))

    rows = []
    promising = True
    for dataset in DATASETS:
        means = {
            ablation: mean(values[(dataset, ablation)])
            for ablation in ABLATIONS if values[(dataset, ablation)]
        }
        if set(means) != set(ABLATIONS):
            promising = False
        for ablation in ABLATIONS:
            scores = values[(dataset, ablation)]
            score_mean = mean(scores) if scores else None
            rows.append({
                "dataset": dataset,
                "ablation": ablation,
                "runs": len(scores),
                "mrr_mean": score_mean,
                "mrr_std": stdev(scores) if len(scores) > 1 else 0.0 if scores else None,
                "delta_vs_A0": (score_mean - means["A0"]
                                if score_mean is not None and "A0" in means else None),
                "delta_vs_A5": (score_mean - means["A5"]
                                if score_mean is not None and "A5" in means else None),
            })
        if set(means) == set(ABLATIONS):
            promising &= means["A14"] > means["A0"] and means["A14"] > means["A5"]

    output = root / "results" / "aggregate" / "pilot_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("Pilot MRR summary (A0=GAE, A5=branches only, A14=full DCDLP)")
    for row in rows:
        value = "missing" if row["mrr_mean"] is None else f"{row['mrr_mean']:.4f}"
        print(f"  {row['dataset']:8s} {row['ablation']:3s} n={row['runs']} MRR={value}")
    verdict = "PROMISING" if promising else "MIXED_OR_INCOMPLETE"
    print(f"Pilot verdict: {verdict}")
    print(output)


if __name__ == "__main__":
    main()

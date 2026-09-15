from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def flatten(record: dict) -> dict:
    row = {key: value for key, value in record.items() if key not in {"metrics", "runtime"}}
    # Legacy JSON files predate the explicit mechanism fields.  Keep them
    # visible as legacy rather than silently merging them with phase-2 runs.
    row.setdefault("cn_feature_mode", "legacy_unknown")
    row.setdefault("interaction_mode", "legacy_unknown")
    row.update({f"metric_{key}": value for key, value in record.get("metrics", {}).items() if not isinstance(value, dict)})
    row.update({f"runtime_{key}": value for key, value in record.get("runtime", {}).items()})
    return row


def aggregate_result_root(result_root: Path) -> Path:
    records = []
    for path in (result_root / "raw").glob("*.json"):
        try:
            records.append(flatten(json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[invalid] {path}: {exc}")
    if not records:
        raise SystemExit("No completed JSON results were found; aggregation did not invent missing values.")
    frame = pd.DataFrame(records)
    output = result_root / "aggregate" / "all_runs.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    metric_columns = [column for column in frame if column.startswith("metric_")]
    group_columns = [
        "dataset", "model", "ablation", "protocol_train", "protocol_eval",
        "cn_feature_mode", "interaction_mode",
    ]
    summary = frame.groupby(group_columns, dropna=False)[metric_columns].agg(["mean", "std", "count"])
    summary.to_csv(result_root / "aggregate" / "summary.csv")
    latex = summary.to_latex(float_format=lambda value: f"{value:.4f}")
    tables = result_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "paper_main.tex").write_text(latex, encoding="utf-8")
    print(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result_root = Path(args.results_dir)
    if not result_root.is_absolute():
        result_root = root / result_root
    aggregate_result_root(result_root)


if __name__ == "__main__":
    main()

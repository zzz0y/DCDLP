from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_figure(fig, base: Path) -> None:
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "results" / "aggregate" / "all_runs.csv"
    if not source.exists():
        raise SystemExit("Run scripts/aggregate_results.py first; figures never use fabricated values.")
    frame = pd.read_csv(source)
    output = root / "results" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.prop_cycle": plt.cycler(color=["#0072B2", "#D55E00", "#009E73", "#CC79A7"])})
    metrics = [item for item in ["metric_mrr", "metric_ap", "metric_worst_group_mrr", "metric_group_gap"] if item in frame]
    if metrics:
        grouped = frame.groupby("dataset")[metrics].mean()
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        grouped.plot.bar(ax=ax)
        ax.set_ylabel("Metric")
        ax.set_title("Completed DCDLP runs")
        ax.legend(frameon=False)
        save_figure(fig, output / "figure_main_and_group_metrics")
    if "runtime_train_seconds" in frame and "metric_mrr" in frame:
        fig, ax = plt.subplots(figsize=(4.5, 3.4))
        ax.scatter(frame["runtime_train_seconds"], frame["metric_mrr"])
        ax.set_xlabel("Training time (s)")
        ax.set_ylabel("MRR")
        ax.set_title("Performance--efficiency")
        save_figure(fig, output / "figure_performance_efficiency")
    print(output)


if __name__ == "__main__":
    main()


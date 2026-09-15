from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPOSITORIES = {
    "HeaRT": "https://github.com/juanhui28/HeaRT.git",
    "NeuralCommonNeighbor": "https://github.com/GraphPKU/NeuralCommonNeighbor.git",
    "DegreeCorrectedLP": "https://github.com/skojaku/degree-corrected-link-prediction-benchmark.git",
    "subgraph-sketching": "https://github.com/melifluos/subgraph-sketching.git",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--third-party", action="store_true")
    parser.add_argument("--datasets", nargs="*", default=[])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.third_party:
        for name, url in REPOSITORIES.items():
            destination = root / "third_party" / name
            if destination.exists():
                print(f"[skip] {destination}")
                continue
            subprocess.run(["git", "clone", "--depth", "1", url, str(destination)], check=True)
    for dataset in args.datasets:
        protocol = "ogb" if dataset.startswith("ogbl-") else "standard"
        subprocess.run([
            sys.executable, "-m", "dcdlp.cli", "prepare-data", "--dataset", dataset,
            "--protocol", protocol,
        ], cwd=root, check=True)


if __name__ == "__main__":
    main()

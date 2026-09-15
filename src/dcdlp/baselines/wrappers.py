from __future__ import annotations

from pathlib import Path


OFFICIAL_REPOSITORIES = {
    "heart": "https://github.com/juanhui28/HeaRT.git",
    "ncn": "https://github.com/GraphPKU/NeuralCommonNeighbor.git",
    "degree_corrected": "https://github.com/skojaku/degree-corrected-link-prediction-benchmark.git",
    "buddy": "https://github.com/melifluos/subgraph-sketching.git",
}


def require_official_checkout(root: Path, name: str) -> Path:
    path = root / name
    if not path.exists():
        raise FileNotFoundError(
            f"Official {name} checkout is missing at {path}. Run scripts/download_data.py --third-party. "
            "Original third_party sources are intentionally never modified."
        )
    return path

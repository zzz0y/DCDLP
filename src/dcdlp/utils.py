from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def stable_hash(value: Any) -> str:
    if is_dataclass(value):
        value = asdict(value)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def array_hash(value: Any) -> str:
    """Hash an array by dtype, shape, and complete contiguous byte content."""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "untracked"


def json_safe(item: Any) -> Any:
    if isinstance(item, dict):
        return {key: json_safe(child) for key, child in item.items()}
    if isinstance(item, (list, tuple)):
        return [json_safe(child) for child in item]
    if isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
        return None
    if isinstance(item, np.integer):
        return int(item)
    return item


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, default=str, allow_nan=False), encoding="utf-8")


def read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

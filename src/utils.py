"""
src/utils.py - shared utilities for the Hindi sentiment project.
"""

from __future__ import annotations

import csv
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    os.environ["PYTHONHASHSEED"] = str(seed)


def log_run(
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    out_dir: str | Path = "results",
    filename: str = "runs.csv",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / filename

    row = {
        "timestamp": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **config,
        **metrics,
    }

    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return csv_path

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except IndexError:
        return Path.cwd()


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip())
    return cleaned.strip("-") or "experiment"


def create_run_dir(experiment_name: str) -> Path:
    """
    Create a timestamped run directory with a plots subfolder.

    Returns:
        Path to runs/YYYYMMDD_HHMMSS_<experiment_name>/
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = _safe_name(experiment_name)

    runs_root = _repo_root() / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    run_dir = runs_root / f"{timestamp}_{safe_name}"
    counter = 1
    while run_dir.exists():
        run_dir = runs_root / f"{timestamp}_{safe_name}_{counter}"
        counter += 1

    run_dir.mkdir(parents=True)
    (run_dir / "plots").mkdir()
    return run_dir

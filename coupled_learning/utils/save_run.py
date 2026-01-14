from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except IndexError:
        return Path.cwd()


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def save_run(run_dir: Path | str, meta: Dict[str, Any], histories: Dict[str, Any]) -> None:
    """
    Save run metadata and histories to disk.

    Args:
        run_dir: Run directory path (created by create_run_dir).
        meta: Metadata dict to write to meta.json.
        histories: Mapping of name -> array or dict-of-arrays.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    meta_out = dict(meta) if meta else {}
    meta_out.setdefault("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
    meta_out.setdefault("git_commit", _git_commit_hash())

    meta_path = run_dir / "meta.json"
    meta_path.write_text(json.dumps(meta_out, indent=2, default=_json_default) + "\n")

    history_summaries = []

    for name, data in histories.items():
        if isinstance(data, dict):
            arrays = {}
            shapes = []
            for key, value in data.items():
                arr = np.asarray(value)
                arrays[key] = arr
                shapes.append(f"{key} {arr.shape}")
            np.savez(run_dir / f"{name}.npz", **arrays)
            history_summaries.append(f"{name}.npz: " + ", ".join(shapes))
        else:
            arr = np.asarray(data)
            if arr.ndim <= 1:
                csv_path = run_dir / f"{name}.csv"
                np.savetxt(csv_path, arr, delimiter=",")
                history_summaries.append(f"{name}.csv: {arr.shape}")
            else:
                np.savez(run_dir / f"{name}.npz", data=arr)
                history_summaries.append(f"{name}.npz: data {arr.shape}")

    readme_lines = [
        "Run contents",
        "",
        "- meta.json: experiment metadata (seed, hyperparameters, git hash)",
        "- plots/: output figures saved by scripts/plot_run.py",
        "- history files:",
    ]
    for summary in history_summaries:
        readme_lines.append(f"  - {summary}")
    readme_lines.append("")
    readme_lines.append("Shapes are shown as (rows, columns) where applicable.")

    (run_dir / "README.txt").write_text("\n".join(readme_lines) + "\n")

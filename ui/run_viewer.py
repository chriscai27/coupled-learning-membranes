from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np


def repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except IndexError:
        return Path.cwd()


def _ensure_repo_on_path() -> None:
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


import streamlit as st

_ensure_repo_on_path()

from scripts.plot_run import (  # noqa: E402
    load_histories,
    plot_evolution,
    plot_parameter_updates,
    plot_error,
    plot_network_configuration,
)


def list_runs(runs_root: Path) -> List[Path]:
    if not runs_root.exists():
        return []
    runs = [p for p in runs_root.iterdir() if p.is_dir()]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def read_meta(run_dir: Path) -> Dict[str, str]:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return {}


def available_nodes(histories: Dict[str, Dict[str, np.ndarray]]) -> int:
    max_nodes = 0
    for data in histories.values():
        if "node_positions_free" in data:
            positions = np.asarray(data["node_positions_free"])
            if positions.ndim == 3:
                max_nodes = max(max_nodes, positions.shape[1])
    return max_nodes


def available_edges(histories: Dict[str, Dict[str, np.ndarray]]) -> List[int]:
    edge_ids = set()
    for data in histories.values():
        if "rest_lengths" in data:
            rest = np.asarray(data["rest_lengths"])
            if rest.ndim >= 2:
                edge_ids.update(range(rest.shape[1]))
        for key in data.keys():
            if key.startswith("L") and key[1:].isdigit():
                edge_ids.add(int(key[1:]) - 1)
    return sorted(edge_ids)


def main() -> None:
    st.set_page_config(page_title="Run Viewer", layout="wide")
    st.title("Run Viewer")

    runs_root = repo_root() / "runs"
    runs = list_runs(runs_root)
    if not runs:
        st.info(f"No runs found in {runs_root}")
        return

    run_labels = [run.name for run in runs]
    selected_label = st.selectbox("Select a run folder", run_labels, index=0)
    run_dir = runs_root / selected_label

    meta = read_meta(run_dir)
    if meta:
        st.subheader("Meta")
        col1, col2, col3 = st.columns(3)
        col1.write(f"experiment_name: {meta.get('experiment_name', 'n/a')}")
        col2.write(f"seed: {meta.get('random_seed', 'n/a')}")
        col3.write(f"iters: {meta.get('iters', 'n/a')}")
        with st.expander("Full meta.json"):
            st.json(meta)

    histories = load_histories(run_dir)
    history_names = list(histories.keys())

    st.subheader("Plot Options")
    mode = st.radio("Mode", ["combined", "separate"], horizontal=True)

    plot_evo = st.checkbox("Evolution plot", value=False)
    plot_params = st.checkbox("Parameter update plot", value=False)
    plot_err = st.checkbox("Error convergence plot", value=False)
    plot_network = st.checkbox("Network configuration plot", value=False)

    selected_histories = st.multiselect(
        "Histories to include",
        history_names,
        default=history_names,
    )

    node_selection = None
    max_nodes = available_nodes(histories)
    if plot_evo and max_nodes > 0:
        options = ["all"] + [str(i) for i in range(max_nodes)]
        selection = st.selectbox("Target node index (0-based)", options, index=0)
        if selection != "all":
            node_selection = int(selection)

    edge_selection = None
    edges = available_edges(histories)
    if plot_params and edges:
        edge_labels = [f"L{idx + 1}" for idx in edges]
        chosen = st.multiselect("Edges to include", edge_labels, default=edge_labels)
        edge_selection = [int(label[1:]) - 1 for label in chosen]

    if st.button("Generate selected plots"):
        output_paths: List[Path] = []
        if plot_evo:
            output_paths.extend(
                plot_evolution(
                    run_dir,
                    mode=mode,
                    histories=selected_histories,
                    node_index=node_selection,
                )
            )
        if plot_params:
            output_paths.extend(
                plot_parameter_updates(
                    run_dir,
                    mode=mode,
                    histories=selected_histories,
                    edge_indices=edge_selection,
                )
            )
        if plot_err:
            output_paths.extend(
                plot_error(
                    run_dir,
                    mode=mode,
                    histories=selected_histories,
                )
            )
        if plot_network:
            output_paths.extend(
                plot_network_configuration(
                    run_dir,
                    histories=selected_histories,
                )
            )

        st.session_state["last_plots"] = [str(p) for p in output_paths]

    last_plots = st.session_state.get("last_plots", [])
    if last_plots:
        st.subheader("Generated Plots")
        for img_path in last_plots:
            if Path(img_path).exists():
                st.image(img_path, caption=Path(img_path).name)
    else:
        st.caption("Select plots and click 'Generate selected plots' to render images.")


if __name__ == "__main__":
    main()

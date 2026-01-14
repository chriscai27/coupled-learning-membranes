from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def _load_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    data = {}
    with np.load(npz_path) as npz:
        for key in npz.files:
            data[key] = np.asarray(npz[key])
    return data


def _iteration_axis(data: Dict[str, np.ndarray]) -> np.ndarray:
    if "cumulative_iter" in data and data["cumulative_iter"].ndim == 1:
        return data["cumulative_iter"]
    if "iteration" in data and data["iteration"].ndim == 1:
        return data["iteration"]
    for value in data.values():
        if value.ndim == 1 and value.size > 1:
            return np.arange(value.size)
        if value.ndim > 1 and value.shape[0] > 1:
            return np.arange(value.shape[0])
    return np.arange(1)


def _save_plot(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"{name}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def _plot_matrix(
    x: np.ndarray,
    mat: np.ndarray,
    title: str,
    ylabel: str,
    out_dir: Path,
    name: str,
    label_prefix: str = "edge",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if mat.ndim == 1:
        ax.plot(x, mat, label=label_prefix)
    else:
        for idx in range(mat.shape[1]):
            ax.plot(x, mat[:, idx], label=f"{label_prefix}{idx + 1}")
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _save_plot(fig, out_dir, name)


def _plot_series(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    ylabel: str,
    out_dir: Path,
    name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    _save_plot(fig, out_dir, name)


def plot_histories(data: Dict[str, np.ndarray], out_dir: Path, prefix: str) -> None:
    x = _iteration_axis(data)

    rest_length_keys = [
        k for k in data.keys()
        if "rest_length" in k or "rest_lengths" in k
    ]
    stiffness_keys = [
        k for k in data.keys()
        if "stiffness" in k or k.startswith("k_")
    ]
    edge_length_keys = [
        k for k in data.keys()
        if "edge_length" in k or "edge_lengths" in k
    ]
    node_position_keys = [k for k in data.keys() if "node_positions" in k]
    error_keys = [
        k for k in data.keys()
        if any(token in k for token in ["error", "loss", "cost"])
    ]

    l_keys = [
        k for k in data.keys()
        if re.fullmatch(r"L\d+", k) is not None
    ]

    for key in rest_length_keys:
        _plot_matrix(
            x,
            data[key],
            title=f"{prefix}: {key}",
            ylabel=key,
            out_dir=out_dir,
            name=f"{prefix}_{key}",
            label_prefix="L",
        )

    for key in l_keys:
        _plot_series(
            x,
            data[key],
            title=f"{prefix}: {key}",
            ylabel=key,
            out_dir=out_dir,
            name=f"{prefix}_{key}",
        )

    for key in stiffness_keys:
        _plot_series(
            x,
            data[key],
            title=f"{prefix}: {key}",
            ylabel=key,
            out_dir=out_dir,
            name=f"{prefix}_{key}",
        )

    for key in edge_length_keys:
        _plot_matrix(
            x,
            data[key],
            title=f"{prefix}: {key}",
            ylabel=key,
            out_dir=out_dir,
            name=f"{prefix}_{key}",
            label_prefix="e",
        )

    for key in error_keys:
        _plot_series(
            x,
            data[key],
            title=f"{prefix}: {key}",
            ylabel=key,
            out_dir=out_dir,
            name=f"{prefix}_{key}",
        )

    for key in ["y_middle", "y1_free", "y1_clamped", "a_est"]:
        if key in data:
            _plot_series(
                x,
                data[key],
                title=f"{prefix}: {key}",
                ylabel=key,
                out_dir=out_dir,
                name=f"{prefix}_{key}",
            )

    for key in node_position_keys:
        positions = data[key]
        if positions.ndim != 3 or positions.shape[0] != x.size:
            continue
        for dim, axis_name in enumerate(["x", "y"]):
            fig, ax = plt.subplots(figsize=(7, 4))
            for idx in range(positions.shape[1]):
                ax.plot(x, positions[:, idx, dim], label=f"node{idx}")
            ax.set_title(f"{prefix}: {key} ({axis_name})")
            ax.set_xlabel("Iteration")
            ax.set_ylabel(axis_name)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            _save_plot(fig, out_dir, f"{prefix}_{key}_{axis_name}")


def load_histories(run_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    histories: Dict[str, Dict[str, np.ndarray]] = {}
    for npz_path in sorted(run_dir.glob("*.npz")):
        histories[npz_path.stem] = _load_npz(npz_path)
    return histories


def _filter_histories(
    histories: Dict[str, Dict[str, np.ndarray]],
    selected: list[str] | None,
) -> Dict[str, Dict[str, np.ndarray]]:
    if selected is None:
        return histories
    return {name: histories[name] for name in selected if name in histories}


def plot_evolution(
    run_dir: Path | str,
    mode: str = "combined",
    histories: list[str] | None = None,
    node_index: int | None = None,
    axis: str = "y",
) -> list[Path]:
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    all_histories = _filter_histories(load_histories(run_dir), histories)
    output_paths: list[Path] = []
    axis_idx = 1 if axis == "y" else 0

    def series_for_history(data: Dict[str, np.ndarray], name: str) -> list[tuple[np.ndarray, str]]:
        if "y_middle" in data:
            return [(np.asarray(data["y_middle"]), f"{name}: y_middle")]
        if "y1_free" in data:
            return [(np.asarray(data["y1_free"]), f"{name}: y1_free")]
        if "node_positions_free" in data:
            positions = np.asarray(data["node_positions_free"])
            if positions.ndim != 3:
                return []
            indices = [node_index] if node_index is not None else list(range(positions.shape[1]))
            series = []
            for idx in indices:
                if idx < 0 or idx >= positions.shape[1]:
                    continue
                series.append((positions[:, idx, axis_idx], f"{name}: node{idx}_{axis}"))
            return series
        return []

    if mode == "separate":
        for name, data in all_histories.items():
            series = series_for_history(data, name)
            if not series:
                continue
            x = _iteration_axis(data)
            fig, ax = plt.subplots(figsize=(7, 4))
            for y, label in series:
                if y.shape[0] != x.shape[0]:
                    x = np.arange(y.shape[0])
                ax.plot(x, y, label=label)
            ax.set_title(f"Evolution: {name}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            output_paths.append(_save_plot(fig, plots_dir, f"evolution_{name}"))
        return output_paths

    fig, ax = plt.subplots(figsize=(7, 4))
    has_data = False
    for name, data in all_histories.items():
        series = series_for_history(data, name)
        if not series:
            continue
        x = _iteration_axis(data)
        for y, label in series:
            x_local = x
            if y.shape[0] != x_local.shape[0]:
                x_local = np.arange(y.shape[0])
            ax.plot(x_local, y, label=label)
            has_data = True
    if has_data:
        ax.set_title("Evolution")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        output_paths.append(_save_plot(fig, plots_dir, "evolution_combined"))
    else:
        plt.close(fig)
    return output_paths


def plot_parameter_updates(
    run_dir: Path | str,
    mode: str = "combined",
    histories: list[str] | None = None,
    edge_indices: list[int] | None = None,
) -> list[Path]:
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    all_histories = _filter_histories(load_histories(run_dir), histories)
    output_paths: list[Path] = []

    def rest_series(data: Dict[str, np.ndarray], name: str) -> list[tuple[np.ndarray, str]]:
        if "rest_lengths" in data:
            rest = np.asarray(data["rest_lengths"])
            if rest.ndim == 1:
                return [(rest, f"{name}: L")]
            indices = edge_indices if edge_indices else list(range(rest.shape[1]))
            series = []
            for idx in indices:
                if idx < 0 or idx >= rest.shape[1]:
                    continue
                series.append((rest[:, idx], f"{name}: L{idx + 1}"))
            return series
        l_keys = [k for k in data.keys() if re.fullmatch(r"L\\d+", k)]
        series = []
        if l_keys:
            l_keys_sorted = sorted(l_keys, key=lambda k: int(k[1:]))
            for key in l_keys_sorted:
                edge_id = int(key[1:]) - 1
                if edge_indices and edge_id not in edge_indices:
                    continue
                series.append((np.asarray(data[key]), f"{name}: {key}"))
        return series

    if mode == "separate":
        for name, data in all_histories.items():
            series = rest_series(data, name)
            if not series:
                continue
            x = _iteration_axis(data)
            fig, ax = plt.subplots(figsize=(7, 4))
            for y, label in series:
                if y.shape[0] != x.shape[0]:
                    x = np.arange(y.shape[0])
                ax.plot(x, y, label=label)
            ax.set_title(f"Parameter Updates: {name}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Rest length")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            output_paths.append(_save_plot(fig, plots_dir, f"params_{name}"))
        return output_paths

    fig, ax = plt.subplots(figsize=(7, 4))
    has_data = False
    for name, data in all_histories.items():
        series = rest_series(data, name)
        if not series:
            continue
        x = _iteration_axis(data)
        for y, label in series:
            x_local = x
            if y.shape[0] != x_local.shape[0]:
                x_local = np.arange(y.shape[0])
            ax.plot(x_local, y, label=label)
            has_data = True
    if has_data:
        ax.set_title("Parameter Updates")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Rest length")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        output_paths.append(_save_plot(fig, plots_dir, "params_combined"))
    else:
        plt.close(fig)
    return output_paths


def plot_error(
    run_dir: Path | str,
    mode: str = "combined",
    histories: list[str] | None = None,
) -> list[Path]:
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    all_histories = _filter_histories(load_histories(run_dir), histories)
    output_paths: list[Path] = []

    def error_series(data: Dict[str, np.ndarray], name: str) -> list[tuple[np.ndarray, str]]:
        series = []
        error_keys = [
            k for k in data.keys()
            if any(token in k for token in ["error", "loss", "cost"])
        ]
        for key in error_keys:
            arr = np.asarray(data[key])
            if arr.ndim == 1:
                series.append((arr, f"{name}: {key}"))
        if series:
            return series
        if "symmetry_error" in data:
            err = np.asarray(data["symmetry_error"])
            if err.ndim == 2:
                series.append((err[:, 0], f"{name}: symmetry_error_1"))
                if err.shape[1] > 1:
                    series.append((err[:, 1], f"{name}: symmetry_error_2"))
            return series
        if "edge_lengths_free" in data:
            lengths = np.asarray(data["edge_lengths_free"])
            if lengths.ndim == 2 and lengths.shape[1] >= 4:
                err1 = np.abs(lengths[:, 0] - lengths[:, 3])
                err2 = np.abs(lengths[:, 1] - lengths[:, 2])
                series.append((err1, f"{name}: |L1-L4|"))
                series.append((err2, f"{name}: |L2-L3|"))
        return series

    if mode == "separate":
        for name, data in all_histories.items():
            series = error_series(data, name)
            if not series:
                continue
            x = _iteration_axis(data)
            fig, ax = plt.subplots(figsize=(7, 4))
            for y, label in series:
                if y.shape[0] != x.shape[0]:
                    x = np.arange(y.shape[0])
                ax.plot(x, y, label=label)
            ax.set_title(f"Error: {name}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Error")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            output_paths.append(_save_plot(fig, plots_dir, f"error_{name}"))
        return output_paths

    fig, ax = plt.subplots(figsize=(7, 4))
    has_data = False
    for name, data in all_histories.items():
        series = error_series(data, name)
        if not series:
            continue
        x = _iteration_axis(data)
        for y, label in series:
            x_local = x
            if y.shape[0] != x_local.shape[0]:
                x_local = np.arange(y.shape[0])
            ax.plot(x_local, y, label=label)
            has_data = True
    if has_data:
        ax.set_title("Error Convergence")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Error")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        output_paths.append(_save_plot(fig, plots_dir, "error_combined"))
    else:
        plt.close(fig)
    return output_paths


def plot_network_configuration(
    run_dir: Path | str,
    histories: list[str] | None = None,
) -> list[Path]:
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    all_histories = _filter_histories(load_histories(run_dir), histories)
    output_paths: list[Path] = []

    for name, data in all_histories.items():
        rest = None
        labels = []
        if "rest_lengths" in data:
            rest = np.asarray(data["rest_lengths"])
            labels = [f"L{idx + 1}" for idx in range(rest.shape[1])] if rest.ndim > 1 else ["L"]
        else:
            l_keys = [k for k in data.keys() if re.fullmatch(r"L\\d+", k)]
            if l_keys:
                l_keys_sorted = sorted(l_keys, key=lambda k: int(k[1:]))
                rest = np.vstack([np.asarray(data[k]) for k in l_keys_sorted]).T
                labels = l_keys_sorted
        if rest is None:
            continue
        if rest.ndim == 1:
            rest = rest[:, None]
            labels = ["L"]
        initial = rest[0]
        final = rest[-1]
        delta = final - initial

        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x - width / 2, initial, width, label="initial")
        ax.bar(x + width / 2, final, width, label="final")
        ax.plot(x, delta, "o--", color="black", label="delta")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"Network Configuration: {name}")
        ax.set_xlabel("Edge")
        ax.set_ylabel("Rest length")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="best")
        output_paths.append(_save_plot(fig, plots_dir, f"network_{name}"))

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot saved run histories")
    parser.add_argument("run_dir", type=str, help="Path to run folder")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(p for p in run_dir.glob("*.npz"))
    for npz_path in npz_files:
        data = _load_npz(npz_path)
        plot_histories(data, plots_dir, npz_path.stem)

    csv_files = sorted(p for p in run_dir.glob("*.csv"))
    for csv_path in csv_files:
        try:
            series = np.loadtxt(csv_path, delimiter=",")
        except Exception:
            continue
        series = np.atleast_1d(series)
        x = np.arange(series.size)
        _plot_series(
            x,
            series,
            title=f"{csv_path.stem}",
            ylabel=csv_path.stem,
            out_dir=plots_dir,
            name=f"{csv_path.stem}",
        )

    print(f"Saved plots to: {plots_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

from coupled_learning.utils.run_manager import create_run_dir
from coupled_learning.utils.save_run import save_run


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run_dir_timestamp(run_dir: Path) -> str:
    parts = run_dir.name.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return _timestamp()


def run_motion_divider(
    iters: int,
    seed: int,
    eta: float,
    threshold: float,
    alpha_scale: float,
    quantize_step: float | None,
    run_dir: Path,
) -> None:
    from examples.test_spring_motion_divider import (
        MotionDividerBackend,
        train_single_target,
    )

    L_BASE_CM = 13.75
    node_positions = np.array(
        [[0.0, 0.0], [0.0, L_BASE_CM], [0.0, 2 * L_BASE_CM]], dtype=float
    )
    edges = [(0, 1), (1, 2)]
    fixed_nodes = [0, 2]

    backend = MotionDividerBackend(node_positions, edges, fixed_nodes)

    trials = [
        {"label": "a", "a": 2.00, "alpha": 0.16},
        {"label": "b", "a": -1.00, "alpha": 0.32},
        {"label": "c", "a": 0.70, "alpha": 0.48},
        {"label": "d", "a": -1.20, "alpha": 0.40},
    ]

    input_node = 2
    input_y = backend.node_positions[input_node, 1]
    targets = [0.5 * input_y + trial["a"] for trial in trials]

    histories: Dict[str, Dict[str, List[float]]] = {}
    cumulative_iterations = 0

    for run_id, target in enumerate(targets):
        trial = trials[run_id]
        history = train_single_target(
            backend,
            a_target=trial["a"],
            y2_input=input_y,
            eta=eta,
            alpha=trial["alpha"] * alpha_scale,
            max_iterations=iters,
            threshold=threshold,
            quantize_step=quantize_step,
        )

        history["cumulative_iter"] = [
            cumulative_iterations + i for i in history["iteration"]
        ]
        cumulative_iterations += max(len(history["iteration"]) - 1, 0)

        histories[f"motion_divider_run_{trial['label']}"] = history

    meta = {
        "experiment_name": "motion_divider",
        "timestamp": _run_dir_timestamp(run_dir),
        "random_seed": seed,
        "iters": iters,
        "eta": eta,
        "threshold": threshold,
        "alpha_scale": alpha_scale,
        "quantize_step": quantize_step,
        "trials": trials,
        "targets": targets,
        "L_base_cm": L_BASE_CM,
        "fixed_nodes": fixed_nodes,
        "edges": edges,
    }

    save_run(run_dir, meta, histories)


class SpringNetwork2D:
    def __init__(
        self,
        node_positions: np.ndarray,
        edges: List[Tuple[int, int]],
        fixed_nodes: List[int],
        stiffness: float,
        rest_lengths: np.ndarray,
    ) -> None:
        self.node_positions = node_positions.copy()
        self.edges = edges
        self.fixed_nodes = set(fixed_nodes)
        self.free_nodes = sorted(set(range(len(node_positions))) - self.fixed_nodes)
        self.stiffness = stiffness
        self.rest_lengths = rest_lengths.copy()

    def _edge_lengths(self, positions: np.ndarray) -> np.ndarray:
        lengths = np.zeros(len(self.edges))
        for edge_id, (i, j) in enumerate(self.edges):
            lengths[edge_id] = np.linalg.norm(positions[i] - positions[j])
        return lengths

    def edge_lengths(self) -> np.ndarray:
        return self._edge_lengths(self.node_positions)

    def _energy(
        self,
        free_xy: np.ndarray,
        penalty_targets: Dict[int, float] | None = None,
        penalty_weight: float = 0.0,
    ) -> float:
        positions = self.node_positions.copy()
        if self.free_nodes:
            positions[self.free_nodes] = free_xy.reshape(-1, 2)

        lengths = self._edge_lengths(positions)
        energy = 0.0
        for edge_id in range(len(self.edges)):
            diff = lengths[edge_id] - self.rest_lengths[edge_id]
            energy += 0.5 * self.stiffness * diff**2

        if penalty_targets:
            for edge_id, target in penalty_targets.items():
                diff = lengths[edge_id] - target
                energy += 0.5 * penalty_weight * diff**2

        return energy

    def solve(
        self,
        penalty_targets: Dict[int, float] | None = None,
        penalty_weight: float = 0.0,
    ) -> None:
        if not self.free_nodes:
            return
        x0 = self.node_positions[self.free_nodes].flatten()
        result = minimize(
            lambda x: self._energy(x, penalty_targets, penalty_weight),
            x0,
            method="BFGS",
        )
        self.node_positions[self.free_nodes] = result.x.reshape(-1, 2)


def run_fig4e_symmetry(
    iters: int,
    seed: int,
    eta: float,
    update_mode: str,
    alpha: float,
    delta: float,
    rest_min: float,
    rest_max: float,
    run_dir: Path,
) -> None:
    set_random_seed(seed)

    node_positions = np.array(
        [
            [-1.0, 0.5],  # A
            [1.0, 0.5],   # B
            [1.0, -0.5],  # C
            [-1.0, -0.5], # D
            [0.0, 0.2],   # E
            [0.0, -0.2],  # F
        ],
        dtype=float,
    )
    fixed_nodes = [0, 1, 2, 3]
    edges = [
        (0, 4),  # L1: A-E
        (1, 4),  # L2: B-E
        (2, 5),  # L3: C-F
        (3, 5),  # L4: D-F
        (4, 5),  # L5: E-F
    ]

    stiffness = 1.0
    rest_lengths = np.random.uniform(rest_min, rest_max, size=len(edges))

    backend = SpringNetwork2D(
        node_positions=node_positions,
        edges=edges,
        fixed_nodes=fixed_nodes,
        stiffness=stiffness,
        rest_lengths=rest_lengths,
    )

    n_steps = iters
    n_edges = len(edges)
    n_free = len(backend.free_nodes)

    rest_length_history = np.zeros((n_steps + 1, n_edges))
    edge_lengths_free = np.zeros((n_steps + 1, n_edges))
    edge_lengths_clamped = np.zeros((n_steps + 1, n_edges))
    node_positions_free = np.zeros((n_steps + 1, n_free, 2))
    node_positions_clamped = np.zeros((n_steps + 1, n_free, 2))
    symmetry_error = np.zeros((n_steps + 1, 2))

    clamp_weight = eta * stiffness

    for t in range(n_steps):
        backend.solve()
        free_lengths = backend.edge_lengths()

        rest_length_history[t] = backend.rest_lengths
        edge_lengths_free[t] = free_lengths
        node_positions_free[t] = backend.node_positions[backend.free_nodes]
        symmetry_error[t] = [
            abs(free_lengths[0] - free_lengths[3]),
            abs(free_lengths[1] - free_lengths[2]),
        ]

        target_14 = 0.5 * (free_lengths[0] + free_lengths[3])
        target_23 = 0.5 * (free_lengths[1] + free_lengths[2])
        penalty_targets = {
            0: target_14,
            3: target_14,
            1: target_23,
            2: target_23,
        }

        backend.solve(penalty_targets=penalty_targets, penalty_weight=clamp_weight)
        clamped_lengths = backend.edge_lengths()
        edge_lengths_clamped[t] = clamped_lengths
        node_positions_clamped[t] = backend.node_positions[backend.free_nodes]

        if update_mode == "quantized":
            diff = clamped_lengths - free_lengths
            delta_step = np.sign(diff)
            delta_step[np.abs(diff) < 1e-12] = 0.0
            backend.rest_lengths = np.clip(
                backend.rest_lengths + delta * delta_step, rest_min, rest_max
            )
        else:
            backend.rest_lengths = np.clip(
                backend.rest_lengths + alpha * (clamped_lengths - free_lengths),
                rest_min,
                rest_max,
            )

    backend.solve()
    free_lengths = backend.edge_lengths()
    rest_length_history[n_steps] = backend.rest_lengths
    edge_lengths_free[n_steps] = free_lengths
    node_positions_free[n_steps] = backend.node_positions[backend.free_nodes]
    symmetry_error[n_steps] = [
        abs(free_lengths[0] - free_lengths[3]),
        abs(free_lengths[1] - free_lengths[2]),
    ]

    target_14 = 0.5 * (free_lengths[0] + free_lengths[3])
    target_23 = 0.5 * (free_lengths[1] + free_lengths[2])
    penalty_targets = {
        0: target_14,
        3: target_14,
        1: target_23,
        2: target_23,
    }
    backend.solve(penalty_targets=penalty_targets, penalty_weight=clamp_weight)
    edge_lengths_clamped[n_steps] = backend.edge_lengths()
    node_positions_clamped[n_steps] = backend.node_positions[backend.free_nodes]

    histories = {
        "fig4e_history": {
            "iteration": np.arange(n_steps + 1),
            "rest_lengths": rest_length_history,
            "edge_lengths_free": edge_lengths_free,
            "edge_lengths_clamped": edge_lengths_clamped,
            "node_positions_free": node_positions_free,
            "node_positions_clamped": node_positions_clamped,
            "symmetry_error": symmetry_error,
        }
    }

    meta = {
        "experiment_name": "fig4e_symmetry",
        "timestamp": _run_dir_timestamp(run_dir),
        "random_seed": seed,
        "iters": iters,
        "eta": eta,
        "update_mode": update_mode,
        "alpha": alpha,
        "delta": delta,
        "rest_length_min": rest_min,
        "rest_length_max": rest_max,
        "stiffness": stiffness,
        "edges": edges,
        "fixed_nodes": fixed_nodes,
        "node_positions_init": node_positions.tolist(),
    }

    save_run(run_dir, meta, histories)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mechanical network replications")
    parser.add_argument(
        "--exp",
        required=True,
        choices=["motion_divider", "fig4e_symmetry"],
        help="Experiment to run",
    )
    parser.add_argument("--iters", type=int, default=200, help="Number of iterations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--eta", type=float, default=None, help="Nudge factor")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Motion divider convergence threshold",
    )
    parser.add_argument(
        "--alpha-scale",
        type=float,
        default=1.0,
        help="Scale factor for motion divider trial alphas",
    )
    parser.add_argument(
        "--quantize-step",
        type=float,
        default=None,
        help="Quantization step for motion divider rest-length updates",
    )
    parser.add_argument(
        "--update",
        choices=["continuous", "quantized"],
        default="continuous",
        help="Update mode for fig4e symmetry network",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Continuous update step for fig4e symmetry network",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.01,
        help="Quantized update step for fig4e symmetry network",
    )
    parser.add_argument(
        "--rest-min",
        type=float,
        default=0.5,
        help="Minimum rest length for fig4e symmetry network",
    )
    parser.add_argument(
        "--rest-max",
        type=float,
        default=2.5,
        help="Maximum rest length for fig4e symmetry network",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    exp_name = args.exp
    run_dir = create_run_dir(exp_name)

    if exp_name == "motion_divider":
        eta = args.eta if args.eta is not None else 1.0
        run_motion_divider(
            iters=args.iters,
            seed=args.seed,
            eta=eta,
            threshold=args.threshold,
            alpha_scale=args.alpha_scale,
            quantize_step=args.quantize_step,
            run_dir=run_dir,
        )
    else:
        eta = args.eta if args.eta is not None else 0.1
        run_fig4e_symmetry(
            iters=args.iters,
            seed=args.seed,
            eta=eta,
            update_mode=args.update,
            alpha=args.alpha,
            delta=args.delta,
            rest_min=args.rest_min,
            rest_max=args.rest_max,
            run_dir=run_dir,
        )

    print(f"Saved run to: {run_dir}")


if __name__ == "__main__":
    main()

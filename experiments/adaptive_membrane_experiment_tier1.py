"""Adaptive Membrane Experiment Runner (Tier 1 PDE backend)."""

import numpy as np
import argparse
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from coupled_learning.backends.membrane_pde_backend import MembranePDEBackend


def _center_heights(z: np.ndarray, backend: MembranePDEBackend) -> np.ndarray:
    centers = backend.get_patch_center_nodes()
    H = np.zeros((backend.n_patches_y, backend.n_patches_x), dtype=float)
    for i in range(backend.n_patches_y):
        for j in range(backend.n_patches_x):
            H[i, j] = z[int(centers[i, j])]
    return H


def run_multi_target_experiment(
    targets: Dict[Tuple[int, int], float],
    n_patches: int = 2,
    pressure_psi: float = 5.0,
    patch_size_mm: float = 2.0,
    thickness_mm: float = 1.0,
    mesh_refinement: int = 6,
    E_min_mpa: float = 0.6,
    E_max_mpa: float = 381.7,
    signal_mode: str = "center",
    eta: float = 0.3,
    alpha: float = 0.05,
    n_iters: int = 100,
    seed: int = 0,
    threshold: float = 0.01,
):
    """
    Multi-target learning on a patch grid.

    targets: dict[(i,j) -> target height mm] at patch centers.
    """
    backend = MembranePDEBackend(
        n_patches_x=n_patches,
        n_patches_y=n_patches,
        patch_size_mm=patch_size_mm,
        thickness_mm=thickness_mm,
        pressure_psi=pressure_psi,
        mesh_refinement=mesh_refinement,
        E_min_mpa=E_min_mpa,
        E_max_mpa=E_max_mpa,
        signal_mode=signal_mode,
        seed=seed,
    )

    # init E in the middle (already), but keep deterministic
    history = {"E_patches": [], "signal": [], "error": [], "error_targets": []}

    target_items = list(targets.items())

    def _compute_errors(z_centers: np.ndarray):
        errs = []
        for (i, j), th in target_items:
            errs.append(abs(z_centers[i, j] - th))
        errs = np.array(errs, dtype=float)
        return errs, float(np.max(errs))

    # iteration 0
    z0 = backend.solve_free()
    z0c = _center_heights(z0, backend)
    errs0, err0 = _compute_errors(z0c)
    history["E_patches"].append(backend.E_patches.copy())
    history["signal"].append(np.zeros_like(backend.E_patches))
    history["error"].append(err0)
    history["error_targets"].append(errs0.tolist())

    if err0 < threshold:
        return history

    for it in range(1, n_iters + 1):
        z_free = backend.solve_free()
        z_free_centers = _center_heights(z_free, backend)

        errs, err = _compute_errors(z_free_centers)
        if err < threshold:
            break

        # clamp targets nudged toward desired
        clamp_targets = {}
        for (i, j), target_h in target_items:
            zf = z_free_centers[i, j]
            clamp_targets[(i, j)] = zf + eta * (target_h - zf)

        z_clamped = backend.solve_clamped(clamp_targets)

        signal = backend.eval_signal(z_free, z_clamped)
        delta_E = alpha * signal

        # Update E globally (Tier1). You can later add masking or normalization here.
        backend.E_patches = np.clip(
            backend.E_patches + alpha * signal,
            backend.E_min,
            backend.E_max,
        )

        # record (note: error is evaluated from the free state at start of loop)
        history["E_patches"].append(backend.E_patches.copy())
        history["signal"].append(signal.copy())
        history["error"].append(err)
        history["error_targets"].append(errs.tolist())

        err_str = ", ".join([f"{e:.4f}" for e in errs])
        updated = int(np.count_nonzero(np.abs(delta_E) > 0.0))
        max_delta = float(np.max(np.abs(delta_E))) if delta_E.size else 0.0
        print(
            f"Iter {it:3d}: error_max={err:.4f} mm (targets: {err_str}), "
            f"max|deltaE|={max_delta:.6f}, patches_updated={updated}/{delta_E.size}"
        )

    return history


def run_single_target_experiment(
    target_patch: Tuple[int, int],
    target_height_mm: float,
    **kwargs
):
    return run_multi_target_experiment(targets={target_patch: target_height_mm}, **kwargs)


def _parse_targets(spec: str) -> Dict[Tuple[int, int], float]:
    # "i,j:height; i,j:height"
    out = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        ij, h = part.split(":")
        i, j = [int(x.strip()) for x in ij.split(",")]
        out[(i, j)] = float(h.strip())
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_patch', type=str, default='1,0', help='Target patch as i,j')
    parser.add_argument('--target_height', type=float, default=0.25, help='Target z-displacement (mm)')
    parser.add_argument('--targets', type=str, default=None,
                        help='Multi-target spec: "i,j:height; i,j:height"')
    parser.add_argument('--n_patches', type=int, default=2, help='Grid size (NxN)')
    parser.add_argument('--pressure', type=float, default=5.0, help='Pressure (psi)')
    parser.add_argument('--patch_size', type=float, default=2.0, help='Patch size (mm)')
    parser.add_argument('--thickness', type=float, default=1.0, help='Membrane thickness (mm)')
    parser.add_argument('--mesh_ref', type=int, default=6, help='Mesh refinement per patch')
    parser.add_argument('--E_min', type=float, default=0.6, help='Min Young modulus (MPa)')
    parser.add_argument('--E_max', type=float, default=381.7, help='Max Young modulus (MPa)')
    parser.add_argument('--signal_mode', type=str, default='center',
                        choices=['center', 'patch_mean'], help='Signal definition')
    parser.add_argument('--eta', type=float, default=0.3, help='Target nudge factor')
    parser.add_argument('--alpha', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--iters', type=int, default=100, help='Max iterations')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed')
    parser.add_argument('--threshold', type=float, default=0.01, help='Convergence threshold (mm)')
    parser.add_argument('--out', type=str, default='runs_tier1', help='Output directory')

    args = parser.parse_args()

    if args.targets:
        targets = _parse_targets(args.targets)
    else:
        i, j = [int(x) for x in args.target_patch.split(",")]
        targets = {(i, j): float(args.target_height)}

    history = run_multi_target_experiment(
        targets=targets,
        n_patches=args.n_patches,
        pressure_psi=args.pressure,
        patch_size_mm=args.patch_size,
        thickness_mm=args.thickness,
        mesh_refinement=args.mesh_ref,
        E_min_mpa=args.E_min,
        E_max_mpa=args.E_max,
        signal_mode=args.signal_mode,
        eta=args.eta,
        alpha=args.alpha,
        n_iters=args.iters,
        seed=args.seed,
        threshold=args.threshold,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"membrane_tier1_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(run_dir / "history.npz",
                        E_patches=np.array(history["E_patches"]),
                        signal=np.array(history["signal"]),
                        error=np.array(history["error"]),
                        error_targets=np.array(history["error_targets"], dtype=object))

    target_items = list(targets.items())
    target_patches = [[int(i), int(j)] for (i, j), _ in target_items]
    target_heights = [float(h) for _, h in target_items]
    meta = {
        "backend": "MembranePDEBackend (Tier1)",
        "targets": {f"{k[0]},{k[1]}": v for k, v in targets.items()},
        "target_patches": target_patches,
        "target_heights_mm": target_heights,
        "n_patches": args.n_patches,
        "pressure_psi": args.pressure,
        "patch_size_mm": args.patch_size,
        "thickness_mm": args.thickness,
        "mesh_refinement": args.mesh_ref,
        "E_min_mpa": args.E_min,
        "E_max_mpa": args.E_max,
        "eta": args.eta,
        "alpha": args.alpha,
        "n_iters": args.iters,
        "seed": args.seed,
        "threshold": args.threshold,
        "final_error_mm": float(history["error"][-1]),
    }

    if len(target_items) == 1:
        meta["target_patch"] = target_patches[0]
        meta["target_height_mm"] = target_heights[0]

    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {run_dir}")

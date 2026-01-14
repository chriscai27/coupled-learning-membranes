"""
Adaptive Membrane Experiment Runner

Sequential single-target learning experiments for pixelated membrane.
"""

import numpy as np
import argparse
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from coupled_learning.backends.pixelated_membrane_backend import PixelatedMembraneBackend


def run_multi_target_experiment(
    targets: Dict[Tuple[int, int], float],
    n_patches: int = 2,
    pressure_psi: float = 5.0,
    patch_size_mm: float = 2.0,
    thickness_mm: float = 1.0,
    mesh_refinement: int = 4,
    E_min_mpa: float = 0.6,
    E_max_mpa: float = 381.7,
    eta: float = 0.3,
    alpha: float = 0.01,
    n_iters: int = 100,
    seed: int = 0,
    threshold: float = 0.01,
):
    """
    Run multi-target learning experiment.

    Args:
        targets: Dict of {(i, j): desired_height_mm}
        n_patches: Grid size (n_patches × n_patches)
        pressure_psi: Bottom pressure (psi)
        patch_size_mm: Patch size (mm)
        thickness_mm: Membrane thickness (mm)
        mesh_refinement: Nodes per patch edge
        E_min_mpa: Minimum Young's modulus (MPa)
        E_max_mpa: Maximum Young's modulus (MPa)
        eta: Nudge factor (0 < eta <= 1)
        alpha: Learning rate
        n_iters: Number of training iterations
        seed: Random seed for initialization
        threshold: Early stop threshold on max target error (mm)
    """
    np.random.seed(seed)

    # Create backend
    backend = PixelatedMembraneBackend(
        n_patches_x=n_patches,
        n_patches_y=n_patches,
        patch_size_mm=patch_size_mm,
        thickness_mm=thickness_mm,
        pressure_psi=pressure_psi,
        mesh_refinement=mesh_refinement,
        E_min_mpa=E_min_mpa,
        E_max_mpa=E_max_mpa,
    )

    # Random initialization of E
    E_init = np.random.uniform(
        backend.E_min,
        backend.E_max,
        size=(n_patches, n_patches)
    )
    backend.set_stiffness_params(E_init)

    # Storage
    history = {
        'E_patches': [E_init.copy()],
        'z_free': [],
        'z_clamped': [],
        'signal': [],
        'error': [],
        'error_targets': [],
    }

    target_items = list(targets.items())
    target_str = ", ".join([f"{patch}→{height:.2f} mm" for patch, height in target_items])

    print(f"Starting experiment: targets {target_str}")
    print(f"Initial E range: [{backend.E_min:.2f}, {backend.E_max:.2f}] MPa")
    print(f"Initialized E: mean={np.mean(E_init):.2f}, std={np.std(E_init):.2f} MPa")

    def _center_heights(z_vals: np.ndarray) -> np.ndarray:
        center_nodes = backend.get_patch_center_nodes()
        return np.array([
            [z_vals[center_nodes[i, j]] for j in range(n_patches)]
            for i in range(n_patches)
        ])

    def _compute_errors(z_centers: np.ndarray) -> Tuple[np.ndarray, float]:
        errors = np.array([
            abs(z_centers[i, j] - target_h)
            for (i, j), target_h in target_items
        ], dtype=float)
        return errors, float(np.max(errors))

    # Iteration 0 logging (free state only)
    z_free = backend.solve_free()
    z_free_centers = _center_heights(z_free)
    errors, error = _compute_errors(z_free_centers)
    history['z_free'].append(z_free_centers.copy())
    history['z_clamped'].append(z_free_centers.copy())
    history['signal'].append(np.zeros_like(z_free_centers))
    history['error'].append(error)
    history['error_targets'].append(errors)

    if error < threshold:
        print(f"Already at target (max error={error:.4f} mm)")
        return history

    for iter_num in range(1, n_iters + 1):
        # Free state
        z_free = backend.solve_free()
        z_free_centers = _center_heights(z_free)

        errors, error = _compute_errors(z_free_centers)
        if error < threshold:
            print(f"Converged at iteration {iter_num} (max error={error:.4f} mm)")
            break

        # Nudge targets toward desired
        clamp_targets = {}
        for (i, j), target_h in target_items:
            z_free_target = z_free_centers[i, j]
            clamp_targets[(i, j)] = z_free_target + eta * (target_h - z_free_target)

        # Clamped state
        z_clamped = backend.solve_clamped(clamp_targets)
        z_clamped_centers = _center_heights(z_clamped)

        # Signal contrast: (z_free)² - (z_clamped)²
        signal = backend.eval_signal(z_free, z_clamped)

        # Update E
        E_new = np.clip(
            backend.E_patches + alpha * signal,
            backend.E_min,
            backend.E_max
        )
        backend.set_stiffness_params(E_new)

        # Log
        history['E_patches'].append(E_new.copy())
        history['z_free'].append(z_free_centers.copy())
        history['z_clamped'].append(z_clamped_centers.copy())
        history['signal'].append(signal.copy())
        history['error'].append(error)
        history['error_targets'].append(errors)

        if iter_num % 10 == 0:
            error_str = ", ".join([f"{e:.4f}" for e in errors])
            print(f"Iter {iter_num:3d}: error_max={error:.4f} mm "
                  f"(targets: {error_str})")

    # Final summary
    print(f"\nFinal results:")
    print(f"  Max error: {history['error'][-1]:.4f} mm")
    print(f"  E range: [{np.min(history['E_patches'][-1]):.2f}, {np.max(history['E_patches'][-1]):.2f}] MPa")

    return history


def run_single_target_experiment(
    target_patch: Tuple[int, int],
    target_height_mm: float,
    n_patches: int = 2,
    pressure_psi: float = 5.0,
    patch_size_mm: float = 2.0,
    thickness_mm: float = 1.0,
    mesh_refinement: int = 4,
    E_min_mpa: float = 0.6,
    E_max_mpa: float = 381.7,
    eta: float = 0.3,
    alpha: float = 0.01,
    n_iters: int = 100,
    seed: int = 0,
    threshold: float = 0.01,
):
    """Run single-target learning experiment."""
    return run_multi_target_experiment(
        targets={target_patch: target_height_mm},
        n_patches=n_patches,
        pressure_psi=pressure_psi,
        patch_size_mm=patch_size_mm,
        thickness_mm=thickness_mm,
        mesh_refinement=mesh_refinement,
        E_min_mpa=E_min_mpa,
        E_max_mpa=E_max_mpa,
        eta=eta,
        alpha=alpha,
        n_iters=n_iters,
        seed=seed,
        threshold=threshold,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_patch', type=str, default='1,0', help='Target patch as i,j')
    parser.add_argument('--target_height', type=float, default=4.0, help='Target z-displacement (mm)')
    parser.add_argument('--n_patches', type=int, default=2, help='Grid size (NxN)')
    parser.add_argument('--pressure', type=float, default=5.0, help='Pressure (psi)')
    parser.add_argument('--patch_size', type=float, default=2.0, help='Patch size (mm)')
    parser.add_argument('--thickness', type=float, default=1.0, help='Membrane thickness (mm)')
    parser.add_argument('--mesh_refinement', type=int, default=4, help='Nodes per patch edge')
    parser.add_argument('--E_min', type=float, default=0.6, help='Minimum Young\'s modulus (MPa)')
    parser.add_argument('--E_max', type=float, default=381.7, help='Maximum Young\'s modulus (MPa)')
    parser.add_argument('--eta', type=float, default=0.3, help='Nudge factor')
    parser.add_argument('--alpha', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--iters', type=int, default=100, help='Training iterations')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    args = parser.parse_args()
    
    # Parse target patch
    target_i, target_j = map(int, args.target_patch.split(','))
    target_patch = (target_i, target_j)
    
    # Run experiment
    history = run_single_target_experiment(
        target_patch=target_patch,
        target_height_mm=args.target_height,
        n_patches=args.n_patches,
        pressure_psi=args.pressure,
        patch_size_mm=args.patch_size,
        thickness_mm=args.thickness,
        mesh_refinement=args.mesh_refinement,
        E_min_mpa=args.E_min,
        E_max_mpa=args.E_max,
        eta=args.eta,
        alpha=args.alpha,
        n_iters=args.iters,
        seed=args.seed,
    )
    
    # Save results
    output_dir = Path('runs') / datetime.now().strftime('%Y%m%d_%H%M%S_adaptive_membrane')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy arrays
    np.savez(
        output_dir / 'history.npz',
        E_patches=np.array(history['E_patches']),
        z_free=np.array(history['z_free']),
        z_clamped=np.array(history['z_clamped']),
        signal=np.array(history['signal']),
        error=np.array(history['error']),
    )
    
    # Save metadata
    meta = {
        'target_patch': target_patch,
        'target_height_mm': args.target_height,
        'n_patches': args.n_patches,
        'pressure_psi': args.pressure,
        'patch_size_mm': args.patch_size,
        'thickness_mm': args.thickness,
        'mesh_refinement': args.mesh_refinement,
        'E_min_mpa': args.E_min,
        'E_max_mpa': args.E_max,
        'eta': args.eta,
        'alpha': args.alpha,
        'n_iters': args.iters,
        'seed': args.seed,
        'final_error_mm': float(history['error'][-1]),
    }
    
    with open(output_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")

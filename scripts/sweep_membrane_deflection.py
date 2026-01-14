#!/usr/bin/env python
"""
Sweep membrane deflection without learning.

Reports how max patch-center displacement changes with:
- Uniform E at E_min vs E_max
- Pressure sweep
- Thickness sweep
"""

import argparse
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Optional, Tuple
import itertools
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from coupled_learning.backends.pixelated_membrane_backend import PixelatedMembraneBackend


def _center_z(backend: PixelatedMembraneBackend) -> np.ndarray:
    z = backend.solve_free()
    center_nodes = backend.get_patch_center_nodes()
    z_centers = np.array([
        [z[center_nodes[i, j]] for j in range(backend.n_patches_x)]
        for i in range(backend.n_patches_y)
    ])
    return z_centers


def _max_center_z(backend: PixelatedMembraneBackend) -> float:
    return float(np.max(_center_z(backend)))


def _uniform_E_backend(
    n_patches: int,
    patch_size_mm: float,
    thickness_mm: float,
    pressure_psi: float,
    mesh_refinement: int,
    use_nonlinear: bool,
    E_value: float,
    E_min_mpa: Optional[float],
    E_max_mpa: Optional[float],
) -> PixelatedMembraneBackend:
    backend_kwargs = {}
    if E_min_mpa is not None:
        backend_kwargs["E_min_mpa"] = E_min_mpa
    if E_max_mpa is not None:
        backend_kwargs["E_max_mpa"] = E_max_mpa

    backend = PixelatedMembraneBackend(
        n_patches_x=n_patches,
        n_patches_y=n_patches,
        patch_size_mm=patch_size_mm,
        thickness_mm=thickness_mm,
        pressure_psi=pressure_psi,
        mesh_refinement=mesh_refinement,
        use_nonlinear=use_nonlinear,
        **backend_kwargs,
    )
    E = np.full((n_patches, n_patches), E_value)
    backend.set_stiffness_params(E)
    return backend


def run_sweeps(
    n_patches: int,
    patch_size_mm: float,
    thickness_mm: float,
    pressure_psi: float,
    mesh_refinement: int,
    use_nonlinear: bool,
    pressure_values: List[float],
    thickness_values: List[float],
    E_min_mpa: Optional[float] = None,
    E_max_mpa: Optional[float] = None,
) -> dict:
    backend_kwargs = {}
    if E_min_mpa is not None:
        backend_kwargs["E_min_mpa"] = E_min_mpa
    if E_max_mpa is not None:
        backend_kwargs["E_max_mpa"] = E_max_mpa

    base_backend = PixelatedMembraneBackend(
        n_patches_x=n_patches,
        n_patches_y=n_patches,
        patch_size_mm=patch_size_mm,
        thickness_mm=thickness_mm,
        pressure_psi=pressure_psi,
        mesh_refinement=mesh_refinement,
        use_nonlinear=use_nonlinear,
        **backend_kwargs,
    )

    E_min = base_backend.E_min
    E_max = base_backend.E_max
    E_mid = 0.5 * (E_min + E_max)

    # E_min / E_max bounds
    backend_min = _uniform_E_backend(
        n_patches,
        patch_size_mm,
        thickness_mm,
        pressure_psi,
        mesh_refinement,
        use_nonlinear,
        E_min,
        E_min_mpa,
        E_max_mpa,
    )
    backend_max = _uniform_E_backend(
        n_patches,
        patch_size_mm,
        thickness_mm,
        pressure_psi,
        mesh_refinement,
        use_nonlinear,
        E_max,
        E_min_mpa,
        E_max_mpa,
    )

    z_min = _max_center_z(backend_min)
    z_max = _max_center_z(backend_max)

    # Pressure sweep at fixed E_mid
    pressure_results = []
    for p in pressure_values:
        backend = _uniform_E_backend(
            n_patches,
            patch_size_mm,
            thickness_mm,
            p,
            mesh_refinement,
            use_nonlinear,
            E_mid,
            E_min_mpa,
            E_max_mpa,
        )
        pressure_results.append((p, _max_center_z(backend)))

    # Thickness sweep at fixed E_mid
    thickness_results = []
    for t in thickness_values:
        backend = _uniform_E_backend(
            n_patches,
            patch_size_mm,
            t,
            pressure_psi,
            mesh_refinement,
            use_nonlinear,
            E_mid,
            E_min_mpa,
            E_max_mpa,
        )
        thickness_results.append((t, _max_center_z(backend)))

    return {
        'settings': {
            'n_patches': n_patches,
            'patch_size_mm': patch_size_mm,
            'thickness_mm': thickness_mm,
            'pressure_psi': pressure_psi,
            'mesh_refinement': mesh_refinement,
            'use_nonlinear': use_nonlinear,
        'E_min': E_min,
        'E_max': E_max,
        'E_mid': E_mid,
    },
        'E_bounds': {
            'E_min': E_min,
            'E_max': E_max,
            'z_max_at_E_min': z_min,
            'z_max_at_E_max': z_max,
        },
        'pressure_sweep': pressure_results,
        'thickness_sweep': thickness_results,
    }


def run_pressure_sweep_per_patch(
    n_patches: int,
    patch_size_mm: float,
    thickness_values: List[float],
    pressure_values: List[float],
    mesh_refinement: int,
    use_nonlinear: bool,
    E_value_mode: str = "min",
    E_min_mpa: Optional[float] = None,
    E_max_mpa: Optional[float] = None,
) -> dict:
    backend_kwargs = {}
    if E_min_mpa is not None:
        backend_kwargs["E_min_mpa"] = E_min_mpa
    if E_max_mpa is not None:
        backend_kwargs["E_max_mpa"] = E_max_mpa

    base_backend = PixelatedMembraneBackend(
        n_patches_x=n_patches,
        n_patches_y=n_patches,
        patch_size_mm=patch_size_mm,
        thickness_mm=thickness_values[0],
        pressure_psi=pressure_values[0],
        mesh_refinement=mesh_refinement,
        use_nonlinear=use_nonlinear,
        **backend_kwargs,
    )
    E_min = base_backend.E_min
    E_max = base_backend.E_max
    if E_value_mode == "max":
        E_value = E_max
    else:
        E_value = E_min

    results: Dict[str, object] = {
        'settings': {
            'n_patches': n_patches,
            'patch_size_mm': patch_size_mm,
            'mesh_refinement': mesh_refinement,
            'use_nonlinear': use_nonlinear,
            'E_min': E_min,
            'E_max': E_max,
            'E_value_mode': E_value_mode,
            'E_value': E_value,
        },
        'pressure_values': pressure_values,
        'thickness_values': thickness_values,
        'z_centers': {},
    }

    for thickness in thickness_values:
        z_centers_per_pressure = []
        for pressure in pressure_values:
            backend = _uniform_E_backend(
                n_patches=n_patches,
                patch_size_mm=patch_size_mm,
                thickness_mm=thickness,
                pressure_psi=pressure,
                mesh_refinement=mesh_refinement,
                use_nonlinear=use_nonlinear,
                E_value=E_value,
                E_min_mpa=E_min_mpa,
                E_max_mpa=E_max_mpa,
            )
            z_centers_per_pressure.append(_center_z(backend))
        results['z_centers'][str(thickness)] = np.array(z_centers_per_pressure)

    return results


def plot_pressure_sweep_per_patch(results: dict, out_dir: Path) -> List[Path]:
    pressure_values = np.array(results['pressure_values'], dtype=float)
    thickness_values = [float(t) for t in results['thickness_values']]
    z_centers = results['z_centers']
    e_label = results['settings'].get('E_value_mode', 'min').upper()
    output_paths: List[Path] = []
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'v', 'D', 'x', '+', '*', 'P']

    for thickness in thickness_values:
        z_stack = np.array(z_centers[str(thickness)])
        n_patches = z_stack.shape[1]

        fig, ax = plt.subplots(figsize=(8, 5))
        style_cycle = itertools.cycle(
            itertools.product(linestyles, markers)
        )
        colors = plt.cm.tab10(np.linspace(0, 1, n_patches * n_patches))
        color_idx = 0
        for i in range(n_patches):
            for j in range(n_patches):
                label = f"({i},{j})"
                linestyle, marker = next(style_cycle)
                ax.plot(
                    pressure_values,
                    z_stack[:, i, j],
                    label=label,
                    linestyle=linestyle,
                    marker=marker,
                    markevery=max(1, len(pressure_values) // 8),
                    linewidth=1.2,
                    markersize=4,
                    alpha=0.8,
                    color=colors[color_idx % len(colors)],
                )
                color_idx += 1

        ax.set_title(f'Pressure vs z_deflection @ thickness={thickness} mm (E={e_label})')
        ax.set_xlabel('Pressure (psi)')
        ax.set_ylabel('z_deflection (mm)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', ncol=2, fontsize=8)

        plt.tight_layout()
        out_path = out_dir / f'pressure_sweep_per_patch_t{thickness:.2f}_E{e_label}.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        output_paths.append(out_path)

    return output_paths


def plot_solution_space_3d(results: dict, out_dir: Path) -> Path:
    pressure_values = np.array(results['pressure_values'], dtype=float)
    thickness_values = np.array(results['thickness_values'], dtype=float)
    z_centers = results['z_centers']
    e_label = results['settings'].get('E_value_mode', 'min').upper()

    sample = np.array(next(iter(z_centers.values())))
    n_patches = sample.shape[1]

    p_min = pressure_values[0]
    p_max = pressure_values[-1]
    x = np.arange(n_patches)
    y = np.arange(n_patches)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(5 * len(thickness_values), 4.5))
    for idx, thickness in enumerate(thickness_values, start=1):
        z_stack = np.array(z_centers[str(thickness)])
        Z_min = z_stack[0]
        Z_max = z_stack[-1]

        ax = fig.add_subplot(1, len(thickness_values), idx, projection='3d')
        ax.plot_surface(X, Y, Z_min, cmap='Blues', alpha=0.75, linewidth=0)
        ax.plot_surface(X, Y, Z_max, cmap='Reds', alpha=0.75, linewidth=0)

        min_idx = np.unravel_index(np.argmin(Z_min), Z_min.shape)
        max_idx = np.unravel_index(np.argmax(Z_max), Z_max.shape)
        ax.scatter(min_idx[1], min_idx[0], Z_min[min_idx], color='blue', s=30)
        ax.scatter(max_idx[1], max_idx[0], Z_max[max_idx], color='red', s=30)
        ax.set_title(f'Thickness={thickness} mm')
        ax.set_xlabel('Patch j')
        ax.set_ylabel('Patch i')
        ax.set_zlabel('z (mm)')

    fig.suptitle(
        f'Membrane surfaces at Pmin={p_min:.1f} psi and Pmax={p_max:.1f} psi (E={e_label})',
        fontsize=12
    )
    plt.tight_layout()
    out_path = out_dir / f'membrane_min_max_3d_E{e_label}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path




def plot_sweeps(results: dict, out_dir: Path) -> Path:
    pressure = np.array(results['pressure_sweep'])
    thickness = np.array(results['thickness_sweep'])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(pressure[:, 0], pressure[:, 1], marker='o')
    axes[0].set_title('Pressure Sweep (E=E_mid)')
    axes[0].set_xlabel('Pressure (psi)')
    axes[0].set_ylabel('Max z_center (mm)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(thickness[:, 0], thickness[:, 1], marker='o', color='tab:orange')
    axes[1].set_title('Thickness Sweep (E=E_mid)')
    axes[1].set_xlabel('Thickness (mm)')
    axes[1].set_ylabel('Max z_center (mm)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / 'deflection_sweeps.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Sweep membrane deflection (no learning)')
    parser.add_argument('--n_patches', type=int, default=2)
    parser.add_argument('--patch_size', type=float, default=2.0)
    parser.add_argument('--thickness', type=float, default=1.0)
    parser.add_argument('--pressure', type=float, default=5.0)
    parser.add_argument('--mesh_refinement', type=int, default=4)
    parser.add_argument('--linear', action='store_true', help='Use linear solver (default: nonlinear)')
    parser.add_argument('--pressure_sweep', type=str, default='1,3,5,7,10')
    parser.add_argument('--thickness_sweep', type=str, default='0.5,1.0,2.0')
    parser.add_argument('--per_patch_sweep', action='store_true',
                        help='Run pressure sweep per patch at E_min')
    parser.add_argument('--pressure_start', type=float, default=1.0)
    parser.add_argument('--pressure_end', type=float, default=10.0)
    parser.add_argument('--pressure_step', type=float, default=0.5)
    parser.add_argument('--per_patch_E', choices=['min', 'max'], default='min',
                        help='Use E_min or E_max for per-patch sweep')
    parser.add_argument('--plot_3d', action='store_true',
                        help='Create 3D pressure/thickness/z surfaces per patch')
    parser.add_argument('--E_min', type=float, default=None, help='Override E_min (MPa)')
    parser.add_argument('--E_max', type=float, default=None, help='Override E_max (MPa)')
    parser.add_argument('--output_dir', type=str, default=None)

    args = parser.parse_args()

    thickness_values = [float(x) for x in args.thickness_sweep.split(',')]

    if args.per_patch_sweep:
        pressure_values = np.arange(
            args.pressure_start,
            args.pressure_end + 1e-9,
            args.pressure_step,
        ).tolist()
        results = run_pressure_sweep_per_patch(
            n_patches=args.n_patches,
            patch_size_mm=args.patch_size,
            thickness_values=thickness_values,
            pressure_values=pressure_values,
            mesh_refinement=args.mesh_refinement,
            use_nonlinear=not args.linear,
            E_value_mode=args.per_patch_E,
            E_min_mpa=args.E_min,
            E_max_mpa=args.E_max,
        )
    else:
        pressure_values = [float(x) for x in args.pressure_sweep.split(',')]
        results = run_sweeps(
            n_patches=args.n_patches,
            patch_size_mm=args.patch_size,
            thickness_mm=args.thickness,
            pressure_psi=args.pressure,
            mesh_refinement=args.mesh_refinement,
            use_nonlinear=not args.linear,
            pressure_values=pressure_values,
            thickness_values=thickness_values,
            E_min_mpa=args.E_min,
            E_max_mpa=args.E_max,
        )

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path('outputs') / f"deflection_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _jsonify(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonify(v) for v in obj]
        return obj

    with open(out_dir / 'sweep_results.json', 'w') as f:
        json.dump(_jsonify(results), f, indent=2)

    if args.per_patch_sweep:
        plot_paths = plot_pressure_sweep_per_patch(results, out_dir)
        if args.plot_3d:
            plot_paths.append(plot_solution_space_3d(results, out_dir))
    else:
        plot_paths = [plot_sweeps(results, out_dir)]

    print('Sweep results saved to:', out_dir)
    for plot_path in plot_paths:
        print('Plot:', plot_path)
    if 'E_bounds' in results:
        print('E bounds:', results['E_bounds'])
    if 'pressure_sweep' in results:
        print('Pressure sweep:', results['pressure_sweep'])
    if 'thickness_sweep' in results:
        print('Thickness sweep:', results['thickness_sweep'])


if __name__ == '__main__':
    main()

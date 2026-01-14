"""
Sweep membrane deflection for Tier 1 PDE backend.

This mirrors sweep_membrane_deflection.py but uses MembranePDEBackend.
Produces:
- z_center vs pressure curves for different thicknesses and E settings
- optional surface plots if matplotlib is available
"""

import numpy as np
import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from coupled_learning.backends.membrane_pde_backend import MembranePDEBackend


def center_height(backend: MembranePDEBackend, z: np.ndarray) -> float:
    # pick geometric center node
    cx, cy = backend.Lx / 2.0, backend.Ly / 2.0
    dist2 = (backend.nodes_x - cx) ** 2 + (backend.nodes_y - cy) ** 2
    idx = int(np.argmin(dist2))
    return float(z[idx])


def run_sweep(
    n_patches: int,
    patch_size_mm: float,
    mesh_refinement: int,
    pressures_psi,
    thicknesses_mm,
    E_values_mpa,
):
    rows = []
    for thickness in thicknesses_mm:
        for E_val in E_values_mpa:
            for p in pressures_psi:
                backend = MembranePDEBackend(
                    n_patches_x=n_patches,
                    n_patches_y=n_patches,
                    patch_size_mm=patch_size_mm,
                    thickness_mm=thickness,
                    pressure_psi=p,
                    mesh_refinement=mesh_refinement,
                    E_min_mpa=min(E_values_mpa),
                    E_max_mpa=max(E_values_mpa),
                    seed=0,
                )
                backend.E_patches[:] = E_val
                z = backend.solve_free()
                rows.append({
                    "pressure_psi": float(p),
                    "thickness_mm": float(thickness),
                    "E_mpa": float(E_val),
                    "z_center_mm": center_height(backend, z),
                    "z_max_mm": float(np.max(z)),
                })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_patches", type=int, default=2)
    parser.add_argument("--patch_size", type=float, default=2.0)
    parser.add_argument("--mesh_ref", type=int, default=6)
    parser.add_argument("--pressures", type=str, default="1,3,5,7,10")
    parser.add_argument("--thicknesses", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--E_values", type=str, default="0.6,381.7")
    parser.add_argument("--out", type=str, default="sweep_tier1.csv")
    args = parser.parse_args()

    pressures = [float(x) for x in args.pressures.split(",")]
    thicknesses = [float(x) for x in args.thicknesses.split(",")]
    E_values = [float(x) for x in args.E_values.split(",")]

    rows = run_sweep(
        n_patches=args.n_patches,
        patch_size_mm=args.patch_size,
        mesh_refinement=args.mesh_ref,
        pressures_psi=pressures,
        thicknesses_mm=thicknesses,
        E_values_mpa=E_values,
    )

    import csv
    out_path = Path(args.out)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {out_path} with {len(rows)} rows.")

#!/usr/bin/env python
"""
Test script for MembranePDEBackend signal/update behavior.

Verifies that signals and updates affect more than the target patch.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from coupled_learning.backends.membrane_pde_backend import MembranePDEBackend


def test_signal_spread(signal_mode: str = "center") -> bool:
    backend = MembranePDEBackend(
        n_patches_x=3,
        n_patches_y=3,
        patch_size_mm=2.0,
        thickness_mm=1.0,
        pressure_psi=5.0,
        mesh_refinement=4,
        signal_mode=signal_mode,
        seed=0,
    )

    target_patch = (1, 1)
    centers = backend.get_patch_center_nodes()
    E0 = backend.E_patches.copy()

    active = 0
    max_abs = 0.0
    alpha = 1.0
    for it in range(3):
        z_free = backend.solve_free()
        z_free_target = z_free[int(centers[target_patch])]
        desired = z_free_target + 0.5
        eta = 1.0
        z_target = z_free_target + eta * (desired - z_free_target)
        z_clamped = backend.solve_clamped({target_patch: z_target})

        signal = backend.eval_signal(z_free, z_clamped)
        if it == 0:
            max_abs = float(np.max(np.abs(signal)))
            if max_abs == 0.0:
                print("Signal is zero everywhere.")
                return False
            active = np.count_nonzero(np.abs(signal) > max_abs * 1e-3)
            print(f"Signal mode: {signal_mode}")
            print(f"Max |signal|: {max_abs:.6f}")
            print(f"Patches with |signal| > 1e-3 * max: {active}/{signal.size}")

        backend.E_patches = np.clip(
            backend.E_patches + alpha * signal,
            backend.E_min,
            backend.E_max,
        )

    delta_E = backend.E_patches - E0
    updated = np.count_nonzero(np.abs(delta_E) > max_abs * 1e-3)
    print(f"Patches with |deltaE| > 1e-3 * max: {updated}/{signal.size}")

    return active > 1 and updated > 1


if __name__ == "__main__":
    print("=" * 60)
    print("TEST: MembranePDEBackend signal spread")
    print("=" * 60)
    ok_center = test_signal_spread("center")
    ok_mean = test_signal_spread("patch_mean")
    passed = ok_center and ok_mean
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")

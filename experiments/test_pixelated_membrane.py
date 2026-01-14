#!/usr/bin/env python
"""
Test script for PixelatedMembraneBackend validation.

Tests:
1. Uniform membrane (all E same) → check symmetry
2. Single patch with low E → should deflect more than neighbors
3. Free vs clamped state comparison
4. Signal computation and learning direction verification

Usage:
    python experiments/test_pixelated_membrane.py
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from coupled_learning import PixelatedMembraneBackend


def test_uniform_membrane():
    """Test 1: Uniform membrane should have symmetric displacement."""
    print("=" * 60)
    print("TEST 1: Uniform Membrane Symmetry")
    print("=" * 60)

    backend = PixelatedMembraneBackend(
        n_patches_x=2,
        n_patches_y=2,
        patch_size_mm=2.0,
        thickness_mm=1.0,
        pressure_psi=5.0,
        mesh_refinement=4,
        use_nonlinear=True
    )

    # Set uniform E
    E_uniform = 100.0  # MPa
    E_patches = np.full((2, 2), E_uniform)
    backend.set_stiffness_params(E_patches)

    # Solve
    z_free = backend.solve_free()
    center_nodes = backend.get_patch_center_nodes()

    print(f"E uniform = {E_uniform} MPa")
    print(f"Pressure = {backend.pressure_psi} psi")
    print(f"\nPatch center z-displacements (mm):")

    z_centers = []
    for i in range(2):
        for j in range(2):
            z = z_free[center_nodes[i, j]]
            z_centers.append(z)
            print(f"  Patch ({i},{j}): z = {z:.6f}")

    # Check symmetry
    # Due to mesh structure, (0,0)≈(1,1) and (0,1)≈(1,0)
    sym_diff_1 = abs(z_centers[0] - z_centers[3])  # (0,0) vs (1,1)
    sym_diff_2 = abs(z_centers[1] - z_centers[2])  # (0,1) vs (1,0)
    max_sym_diff = max(sym_diff_1, sym_diff_2)

    print(f"\nSymmetry check:")
    print(f"  |(0,0) - (1,1)| = {sym_diff_1:.6f}")
    print(f"  |(0,1) - (1,0)| = {sym_diff_2:.6f}")

    passed = max_sym_diff < 1e-6
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_soft_patch_deflection():
    """Test 2: Soft patch should deflect more than stiff neighbors."""
    print("\n" + "=" * 60)
    print("TEST 2: Soft Patch Deflects More")
    print("=" * 60)

    backend = PixelatedMembraneBackend(
        n_patches_x=2,
        n_patches_y=2,
        patch_size_mm=2.0,
        thickness_mm=1.0,
        pressure_psi=5.0,
        mesh_refinement=4,
        use_nonlinear=True
    )

    # One soft patch, others stiff
    E_patches = np.array([
        [381.7, 381.7],  # top row - stiff
        [0.6, 381.7]     # bottom row - patch (1,0) is soft
    ])
    backend.set_stiffness_params(E_patches)

    # Solve
    z_free = backend.solve_free()
    center_nodes = backend.get_patch_center_nodes()

    print("E distribution (MPa):")
    for i in range(2):
        row = "  "
        for j in range(2):
            row += f"{E_patches[i, j]:8.1f}  "
        print(row)

    print(f"\nPatch center z-displacements (mm):")
    z_dict = {}
    for i in range(2):
        for j in range(2):
            z = z_free[center_nodes[i, j]]
            z_dict[(i, j)] = z
            print(f"  Patch ({i},{j}): E = {E_patches[i, j]:6.1f} MPa, z = {z:.4f} mm")

    # Soft patch should have largest displacement
    z_soft = z_dict[(1, 0)]
    z_stiff_max = max(z_dict[(0, 0)], z_dict[(0, 1)], z_dict[(1, 1)])

    print(f"\nSoft patch (1,0): z = {z_soft:.4f} mm")
    print(f"Max stiff patch:  z = {z_stiff_max:.4f} mm")
    print(f"Ratio: {z_soft / z_stiff_max:.2f}x")

    passed = z_soft > z_stiff_max * 3  # Should be at least 3x more
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_free_vs_clamped():
    """Test 3: Free vs clamped state comparison and signal computation."""
    print("\n" + "=" * 60)
    print("TEST 3: Free vs Clamped State and Signal")
    print("=" * 60)

    backend = PixelatedMembraneBackend(
        n_patches_x=2,
        n_patches_y=2,
        patch_size_mm=2.0,
        thickness_mm=1.0,
        pressure_psi=5.0,
        mesh_refinement=4,
        use_nonlinear=True
    )

    # Uniform E
    E_patches = np.full((2, 2), 100.0)
    backend.set_stiffness_params(E_patches)

    # Solve free state
    z_free = backend.solve_free()
    center_nodes = backend.get_patch_center_nodes()

    print("Free state center displacements (mm):")
    for i in range(2):
        for j in range(2):
            z = z_free[center_nodes[i, j]]
            print(f"  Patch ({i},{j}): z_free = {z:.4f}")

    # Target: push patch (0,0) to higher displacement
    target_patch = (0, 0)
    z_free_target = z_free[center_nodes[0, 0]]
    target_height = 0.3  # mm
    eta = 0.3  # nudge factor
    z_target = z_free_target + eta * (target_height - z_free_target)

    print(f"\nTarget: patch (0,0) -> {z_target:.4f} mm (nudged toward {target_height} mm)")

    # Solve clamped state
    z_clamped = backend.solve_clamped({target_patch: z_target})

    print("\nClamped state center displacements (mm):")
    for i in range(2):
        for j in range(2):
            z = z_clamped[center_nodes[i, j]]
            print(f"  Patch ({i},{j}): z_clamped = {z:.4f}")

    # Verify constraint is satisfied
    z_clamped_target = z_clamped[center_nodes[0, 0]]
    constraint_error = abs(z_clamped_target - z_target)
    print(f"\nConstraint check: |z_clamped - z_target| = {constraint_error:.6f}")

    # Compute signal
    signal = backend.eval_signal(z_free, z_clamped)
    print(f"\nSignal (z_free² - z_clamped²):")
    for i in range(2):
        for j in range(2):
            s = signal[i, j]
            print(f"  Patch ({i},{j}): signal = {s:.6f}")

    # Check learning direction
    # Target patch (0,0) was nudged UP (z_clamped > z_free)
    # So signal should be NEGATIVE
    # Negative signal -> decrease E -> softer -> z_free increases
    print(f"\nLearning direction analysis:")
    print(f"  z_clamped[0,0] > z_free[0,0]: {z_clamped_target > z_free_target}")
    print(f"  signal[0,0] < 0: {signal[0, 0] < 0}")
    print(f"  -> Correct: negative signal means decrease E to increase z_free")

    passed = constraint_error < 1e-4 and signal[0, 0] < 0
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_3x3_grid():
    """Test 4: Larger 3x3 grid with multiple targets."""
    print("\n" + "=" * 60)
    print("TEST 4: 3x3 Grid with Center Target")
    print("=" * 60)

    backend = PixelatedMembraneBackend(
        n_patches_x=3,
        n_patches_y=3,
        patch_size_mm=2.0,
        thickness_mm=1.0,
        pressure_psi=5.0,
        mesh_refinement=3,
        use_nonlinear=True
    )

    # Uniform E
    E_patches = np.full((3, 3), 100.0)
    backend.set_stiffness_params(E_patches)

    # Solve free state
    z_free = backend.solve_free()
    center_nodes = backend.get_patch_center_nodes()

    print(f"Grid size: {backend.n_patches_x}x{backend.n_patches_y}")
    print(f"Nodes: {backend.n_nodes}, Edges: {len(backend.edges)}")

    print("\nFree state center displacements (mm):")
    for i in range(3):
        row = "  "
        for j in range(3):
            z = z_free[center_nodes[i, j]]
            row += f"{z:.4f}  "
        print(row)

    # Check that center patch (1,1) has highest displacement
    z_center = z_free[center_nodes[1, 1]]
    z_corners = [
        z_free[center_nodes[0, 0]],
        z_free[center_nodes[0, 2]],
        z_free[center_nodes[2, 0]],
        z_free[center_nodes[2, 2]],
    ]

    print(f"\nCenter patch (1,1): z = {z_center:.4f} mm")
    print(f"Corner patches avg: z = {np.mean(z_corners):.4f} mm")

    passed = z_center > max(z_corners)
    print(f"\nCenter higher than corners: {passed}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_linear_vs_nonlinear():
    """Test 5: Compare linear vs nonlinear solver."""
    print("\n" + "=" * 60)
    print("TEST 5: Linear vs Nonlinear Solver")
    print("=" * 60)

    # Low pressure - should be similar
    for use_nonlinear, solver_name in [(False, "Linear"), (True, "Nonlinear")]:
        backend = PixelatedMembraneBackend(
            n_patches_x=2,
            n_patches_y=2,
            patch_size_mm=2.0,
            thickness_mm=1.0,
            pressure_psi=1.0,  # Low pressure
            mesh_refinement=4,
            use_nonlinear=use_nonlinear
        )

        E_patches = np.full((2, 2), 100.0)
        backend.set_stiffness_params(E_patches)

        z_free = backend.solve_free()
        z_max = np.max(z_free)
        print(f"{solver_name}: max z = {z_max:.4f} mm")

    print("\nNote: For low pressure, linear and nonlinear should be similar.")
    print("For high pressure, nonlinear accounts for geometric stiffening.")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "#" * 60)
    print("PIXELATED MEMBRANE BACKEND VALIDATION TESTS")
    print("#" * 60 + "\n")

    tests = [
        ("Uniform Membrane Symmetry", test_uniform_membrane),
        ("Soft Patch Deflection", test_soft_patch_deflection),
        ("Free vs Clamped State", test_free_vs_clamped),
        ("3x3 Grid", test_3x3_grid),
        ("Linear vs Nonlinear", test_linear_vs_nonlinear),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed, None))
        except Exception as e:
            results.append((name, False, str(e)))

    # Summary
    print("\n" + "#" * 60)
    print("TEST SUMMARY")
    print("#" * 60)

    n_passed = 0
    for name, passed, error in results:
        status = "PASS" if passed else "FAIL"
        if error:
            status += f" (Error: {error})"
        print(f"  {name}: {status}")
        if passed:
            n_passed += 1

    print(f"\n{n_passed}/{len(tests)} tests passed")

    return n_passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

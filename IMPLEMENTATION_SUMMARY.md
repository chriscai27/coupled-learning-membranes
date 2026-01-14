# Adaptive Membrane FEA Implementation Summary

## Completed Implementation

### PixelatedMembraneBackend (`coupled_learning/backends/pixelated_membrane_backend.py`)

A complete FEA solver for pixelated membranes with learnable Young's modulus per patch.

#### Physics Model
- **Spring network representation**: Membrane discretized as 3D spring network
- **Spring types**: Horizontal, vertical, and diagonal springs for mesh stability
- **Energy formulation**:
  - Elastic: `E = (1/2) * k * (L - L0)²` where L is 3D length, L0 is xy rest length
  - Potential: Work done against pressure on free nodes
- **Spring stiffness**: `k = E * h * dx / L0` where E is patch modulus, h is thickness

#### Solver Options
1. **Nonlinear (recommended)**: Energy minimization using L-BFGS-B
   - Handles geometric nonlinearity (large deformations)
   - Captures membrane stretching behavior

2. **Linear**: Direct sparse solve
   - For small deformations only
   - Much faster but less accurate for membranes

#### Key Methods
```python
backend = PixelatedMembraneBackend(
    n_patches_x=2,
    n_patches_y=2,
    patch_size_mm=2.0,
    thickness_mm=1.0,
    pressure_psi=5.0,
    mesh_refinement=4,
    use_nonlinear=True
)

# Set patch stiffness
backend.set_stiffness_params(E_patches)  # (n_patches_y, n_patches_x) array in MPa

# Solve free state (pressure BC only)
z_free = backend.solve_free()

# Solve clamped state (pressure + target constraints)
z_clamped = backend.solve_clamped({(i, j): z_target})

# Compute signal for learning
signal = backend.eval_signal(z_free, z_clamped)  # (z_free² - z_clamped²)
```

#### Validated Behavior
1. **Symmetry**: Uniform E produces symmetric displacement field
2. **Stiffness effect**: Softer patches (lower E) deflect more (~7x for E_min vs E_max)
3. **Constraint enforcement**: Clamped state satisfies target displacement constraints
4. **Signal direction**: Correct learning gradient (negative signal → decrease E → increase z)

### Test Script (`experiments/test_pixelated_membrane.py`)

Comprehensive validation tests:
1. Uniform membrane symmetry check
2. Soft patch deflection test
3. Free vs clamped state comparison
4. 3x3 grid validation
5. Linear vs nonlinear comparison

Run with: `python experiments/test_pixelated_membrane.py`

### Updated Experiment Runner (`experiments/adaptive_membrane_experiment.py`)

Single-target learning experiment with:
- Random E initialization
- Configurable target patch and height
- Full history tracking
- Automatic result saving

Example:
```bash
python experiments/adaptive_membrane_experiment.py \
    --target_patch 0,0 \
    --target_height 0.3 \
    --alpha 100 \
    --iters 100
```

## Project Structure

```
coupled-learning-membranes/
├── coupled_learning/
│   ├── base.py                    # PhysicsBackend abstract class
│   ├── backends/
│   │   ├── __init__.py           # Package exports
│   │   ├── simple_backend.py     # Spring/flow networks
│   │   └── pixelated_membrane_backend.py  # Membrane FEA
│   ├── engine.py
│   ├── signals.py
│   └── update_rules.py
├── experiments/
│   ├── adaptive_membrane_experiment.py
│   ├── test_pixelated_membrane.py
│   └── mechanical_network_replications.py
└── archive/
    └── IMPLEMENTATION_SUMMARY_*.md
```

## Key Parameters

### Material Bounds
- **E_min**: 0.6 MPa (Shore A50, softest)
- **E_max**: 381.7 MPa (solid material, stiffest)

### Pressure Conversion
- 1 psi = 6894.76 Pa = 0.00689476 N/mm²

### Achievable Deflection Range
For 2×2 grid, 2mm patches, 1mm thickness, 5 psi:
- All E_min: ~0.36-0.39 mm at patch centers
- All E_max: ~0.08-0.09 mm at patch centers

## Learning Example

With parameters:
- Target: patch (0,0) → 0.3 mm
- Learning rate α = 100
- Iterations: 100

Results:
- Error reduced from 0.24 mm to 0.014 mm
- E at target decreased from ~142 MPa to 0.69 MPa
- Learning converged successfully

## Known Limitations

1. **Linear solver**: Produces very small displacements for membranes (use nonlinear)
2. **Mesh asymmetry**: Diagonal springs only in one direction may cause slight asymmetry
3. **Large deflections**: For δ/h > 5, geometric nonlinearity is essential

## Next Steps

1. Add symmetric diagonal edges (both \ and /)
2. Implement multi-target learning experiments
3. Add visualization tools for displacement fields
4. Test with 3×3 and larger grids
5. Optimize solver performance for larger meshes

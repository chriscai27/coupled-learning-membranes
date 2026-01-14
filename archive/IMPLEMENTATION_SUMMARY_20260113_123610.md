# Adaptive Membrane Python Prototype - Implementation Summary

## What's Been Created

### 1. PixelatedMembraneBackend (`coupled_learning/backends/pixelated_membrane_backend.py`)

**Key Features:**
- Grid-based membrane: NxN patches, each with learnable E_i
- Mesh generation with configurable refinement
- Patch center node identification
- Signal computation: (z_free)² - (z_clamped)²
- Material bounds: E ∈ [0.6, 381.7] MPa

**Current Status:**
- Structure complete ✓
- Mesh generation working ✓
- Signal computation correct ✓
- **FEA solver: PLACEHOLDER** (needs implementation)

**FEA Solver Options:**
1. **solidspy**: Simple, Pythonic, good for learning
   ```bash
   pip install solidspy
   ```
2. **PyFEA**: Lightweight membrane solver
3. **Custom**: Minimal plate/membrane solver from scratch

### 2. Experiment Runner (`experiments/adaptive_membrane_experiment.py`)

**Capabilities:**
- Single-target learning (sequential experiments)
- Random E initialization
- Full history tracking (E, z_free, z_clamped, signal, error)
- Automatic result saving (npz + json metadata)

**Usage:**
```bash
# Experiment 1: Target at (1,0)
python experiments/adaptive_membrane_experiment.py \
    --target_patch 1,0 \
    --target_height 4.0 \
    --n_patches 2 \
    --pressure 5.0 \
    --iters 100 \
    --seed 0

# Experiment 2: Retrain for (0,1)  
python experiments/adaptive_membrane_experiment.py \
    --target_patch 0,1 \
    --target_height 4.0 \
    --n_patches 2 \
    --pressure 5.0 \
    --iters 100 \
    --seed 1
```

### 3. Updated Documentation
- CLAUDE.md: Comprehensive adaptive membrane section
- README.md: Experiment commands and material parameters

## Signal Design Implementation

**Correct implementation** following Dillavou/Altman/Stern:

```python
def eval_signal(self, z_free, z_clamped):
    """Signal: (z_free)² - (z_clamped)²"""
    signal = np.zeros((n_patches_y, n_patches_x))
    
    for i in range(n_patches_y):
        for j in range(n_patches_x):
            node_idx = center_nodes[i, j]
            signal[i, j] = z_free[node_idx]**2 - z_clamped[node_idx]**2
            
    return signal
```

**Why squared?**
- Energy-like quantity (strain energy ∝ displacement²)
- Consistent with Dillavou's [ΔV]² signal
- Provides correct gradient direction

## Problem Definition (Corrected)

**Key correction:** Only target patches get clamped, not all patches.

```python
def solve_clamped(self, target_patches: Dict[Tuple[int, int], float]):
    """
    Args:
        target_patches: {(i,j): z_target} for ONLY the target patches
    
    Constraints applied:
        - Pressure BC on ALL patches
        - z-displacement constraint ONLY at target patch centers
        - Edge constraints (z=0) on all edge nodes
    """
```

This mirrors Altman's spring network where only target nodes are clamped.

## Pressure & Nonlinearity

**Rule of thumb for geometric nonlinearity:**
- δ/h < 0.5: Small deflection (linear FEA okay)
- δ/h > 0.5: Large deformation (geometric nonlinearity needed)

**For your materials:**
- Thickness h = 1 mm
- Pressure 1-10 psi → deflection δ ~ 0.5-5 mm (rough estimate)
- δ/h ~ 0.5-5 → **Expect nonlinearity**

**Experimental strategy:**
1. Start with **low pressure (1-3 psi)** to test algorithm in linear regime
2. Increase to **medium (5 psi)** for transition
3. Test **high (7-10 psi)** for full nonlinearity

## Validation with Uniform Membrane

**"Analytical solution" means:**
- Set all E_i = E_uniform (e.g., 100 MPa)
- Apply uniform pressure
- Check:
  1. Symmetric displacement (z should be symmetric)
  2. Maximum at center (for square/circular membrane)
  3. Order of magnitude reasonable (compare to δ ~ PL⁴/(ED·h³) scaling)

This is just a sanity check before learning.

## Next Steps

### Immediate (FEA Implementation):
1. **Choose FEA library**:
   - Recommend `solidspy` for simplicity
   - Need shell/membrane element capability
   
2. **Implement `solve_free()`**:
   - Set up stiffness matrix with patch-wise E
   - Apply pressure load vector
   - Apply fixed boundary conditions
   - Solve K·u = F
   
3. **Implement `solve_clamped()`**:
   - Same as free, but ADD displacement constraints at target nodes
   - Use Lagrange multipliers or penalty method

### Testing Sequence:
```bash
# Test 1: Uniform membrane (validation)
python experiments/test_uniform_membrane.py

# Test 2: Single target at (1,0)
python experiments/adaptive_membrane_experiment.py --target_patch 1,0 --iters 100

# Test 3: Retrain for (0,1)
python experiments/adaptive_membrane_experiment.py --target_patch 0,1 --iters 100

# Test 4: Pressure sweep
python experiments/pressure_sweep.py --pressure_range 1,3,7,10
```

### Analysis (After Learning):
1. Plot E evolution over time
2. Plot z_free convergence to target
3. Compare signal magnitude across patches
4. Check if target patch E decreases (softer → higher displacement)

## Questions Answered

### 1. Grid Size
**Answer:** User-configurable via `n_patches_x`, `n_patches_y` in backend constructor. Starting with 2×2 for simplicity.

### 2. Boundary Conditions  
**Answer:** Edges clamped (z=0, θ=0) to simulate housing constraint. Implemented in `_build_mesh()` via `self.fixed_nodes`.

### 3. Measurement Strategy
**Answer:** Phase 1 = patch centers only (implemented). Phase 2 = add edge measurements (easy extension).

### 4. Analytical Validation
**Answer:** Uniform E → check symmetry and center max. For circular membrane: δ_max ≈ 0.662·P·R⁴/(E·h³) can be used as order-of-magnitude check.

## File Structure

```
coupled-learning-membranes/
├── coupled_learning/
│   ├── backends/
│   │   └── pixelated_membrane_backend.py  ✓ Created
├── experiments/
│   └── adaptive_membrane_experiment.py    ✓ Created
├── CLAUDE.md                               ✓ Updated
└── README.md                               ✓ Updated
```

## Critical Implementation Note

The FEA solver is currently a **placeholder**. You need to:
1. Choose FEA library (solidspy recommended)
2. Implement membrane/shell element formulation
3. Handle nonlinearity if needed (for pressure > 5 psi)

Would you like me to:
- **A) Implement the solidspy integration now**
- **B) Create a minimal custom membrane solver**
- **C) Provide pseudocode for the FEA interface**

# Implementation Summary — Tier 1 “FEA‑ish” Membrane Surrogate (Variable‑Coefficient PDE)

This note documents a backend redesign intended to **look and behave more like a continuum membrane discretization** (closer to “FEA feel”) while staying compatible with the coupled‑learning loop (free vs clamped solves + local update signals).

---

## What we simplified (explicit, so it’s defensible in a meeting)

### We keep (important, advisor‑relevant)
- **Continuum-like spatial coupling**: neighbor influence is enforced by a PDE operator (variable‑coefficient Laplacian), not by ad‑hoc midpoint‑assigned springs.
- **Patch-wise learnable field**: each patch still owns a scalar parameter (`E_patches[i,j]`), updated by a local signal.
- **Two-solve learning structure**:
  - *Free* solve under uniform pressure
  - *Clamped* solve with target(s) enforced
  - Signal from a contrast between the two (same as before)

### We simplify (what makes this Tier 1)
- **Geometric nonlinearity is removed**.
  - We do **not** compute stretch‑dependent tension, and we do **not** update stiffness based on deformation geometry.
  - Instead we solve a *linear* surrogate:  
    \[
    -\nabla\cdot (K(x,y)\nabla w) = p
    \]
- **Material law is collapsed** into a single *effective* coefficient:
  - We treat `K_patch ≈ E_patch * thickness` as a **tension-like coupling strength**.
  - This is *not* a full membrane constitutive model; it’s a deliberately simple knob that gives:
    - monotonic stiffness effect
    - strong neighbor cascade
    - stable learning

### What this buys you (why it’s “significant” even as a simplification)
- The deformation field becomes **smooth and globally coupled** in the way a membrane/FE mesh typically looks.
- Patch boundaries influence each other naturally because the operator couples nodes via **face coefficients** (using harmonic averaging across patch boundaries).
- Sweeps should show clean scaling trends:
  - displacement increases with pressure
  - displacement decreases with stiffness field

---

## Tier 1 vs Tier 2 (geometric stiffening) — how much extra complexity?

### Tier 1 (implemented)
- **One sparse linear solve** per `solve_free()` and `solve_clamped()`.
- Runtime per iteration: roughly **2 linear solves**.
- Complexity: *O(N)* to assemble + sparse solve cost (depends on grid, but small for your patch sizes).

### Tier 2 (minimal geometric nonlinearity, not implemented here)
A practical reduced‑nonlinear approach is:
\[
K_{ij} = K_0 + \beta\,\langle |\nabla w|^2 \rangle_{ij}
\]
with a fixed‑point loop:
1) solve for `w` given `K`
2) update `K` using `|∇w|²`
3) repeat to convergence (5–20 inner iterations typical)

**What it costs**
- Each `solve_free()` becomes **5–20** linear solves (instead of 1)
- Same for `solve_clamped()`
- So learning iteration cost goes from:
  - Tier 1: ~2 solves
  - Tier 2: ~10–40 solves

**Rule-of-thumb estimate**
- Expect **~5× to 20× slower** wall‑time per learning iteration (depending on inner iterations).
- Engineering effort increase:
  - Tier 1: straightforward
  - Tier 2: moderate (needs convergence criteria, damping, stability tuning)

### Recommendation
- Use **Tier 1** to show your advisor:
  - the shape is “FEA-like”
  - the cascade exists
  - the learning pipeline works end-to-end
- Add Tier 2 only if you specifically need:
  - self-stiffening with deformation
  - more realistic pressure–deflection curves

---

## Files added

### 1) `membrane_pde_backend.py`
Implements the Tier‑1 backend:
- variable‑coefficient membrane PDE
- harmonic averaging across patch boundaries
- same API surface used by the experiment runner:
  - `solve_free()`
  - `solve_clamped(target_patches)`
  - `eval_signal(z_free, z_clamped)`
  - `get_patch_center_nodes()`
  - `get_full_displacement_field()`

### 2) `adaptive_membrane_experiment_tier1.py`
Sibling runner that mirrors the original experiment loop, but uses `MembranePDEBackend`.

### 3) `sweep_membrane_deflection_tier1.py`
Simple sweep script to quantify the achievable range vs pressure/thickness/E.

---

## How to demo this quickly to your advisor (suggested)

1) Run a sweep showing scaling trends (no learning):
```bash
python sweep_membrane_deflection_tier1.py --n_patches 2 --mesh_ref 6 --pressures 1,3,5,7,10 --thicknesses 0.5,1.0,2.0 --E_values 0.6,381.7 --out sweep_tier1.csv
```

2) Run a single-target learning demo:
```bash
python adaptive_membrane_experiment_tier1.py --n_patches 2 --pressure 5 --patch_size 2 --thickness 1 --mesh_ref 6 --target_patch 1,0 --target_height 0.25 --alpha 0.05 --iters 100
```

3) What to show (plots you can generate with your existing viz script with tiny edits):
- error vs iteration
- heatmaps of `E_patches` at iter 0 / mid / final
- surface plot of `w` (z) for free state, before/after learning

---

## Known limitations (so you can frame it correctly)
- This is a **linear surrogate**, so it will not capture:
  - pressure–deflection nonlinearity
  - stretch‑induced stiffening
  - wrinkling / slack regions
- It *will* capture:
  - smooth shapes
  - strong neighbor cascade
  - stable learning dynamics with patchwise stiffness


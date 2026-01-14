# Coupled Learning Framework — Project Context

## Goal
Replicate and extend Altman et al. (2024) Figure 2 (motion divider) using a modular
coupled-learning framework, then transfer the same learning engine to membrane systems.

## Current Status
- Motion divider replication mostly working
- Known bug: first target converges immediately due to missing initial state logging
- SimpleMechanicalBackend used for Altman Fig.2
- Learning variable: REST LENGTHS (not stiffness)

## Architecture
- Physics handled by backend (SimpleMechanicalBackend, SimpleMembraneBackend, etc.)
- Learning handled by CoupledLearningEngine
- Signals compare clamped vs free equilibria
- Update rules apply local parameter changes

## Physics Assumptions (Altman Fig.2)
- Nodes move in 2D (x,y)
- Springs with energy: (1/2) k (L - L0)^2
- Learning updates L0 (rest length)

## Known Bug
In test_figure2_full.py:
- Initial state not logged before convergence check
- Causes empty history and IndexError
- Fix: always log iteration 0 before loop

## Immediate Task
- Fully replicate Altman et al. Figure 2 (4 sequential targets)
- Verify qualitative behavior: L1 increases, L2 decreases
- Match convergence trends (not exact step count)

# Papers Index

- **Altman2024**  
  Motion divider via coupled learning in mechanical spring networks.  
  Used for: Figure 2 replication (rest-length learning).

- **Stern2021**  
  General mechanical learning in elastic networks.  
  Used for: stiffness-learning baseline and backend reference.

- **Dillavou2022**  
  Electrical / flow-network learning analogs.  
  Used for: SimpleFlowBackend comparison.

## Mechanical Network Replications (Altman et al.)

Citation: Altman et al. 2024, Physical Review Applied, Experimental demonstration of coupled learning in elastic networks.

Run:
- `python experiments/mechanical_network_replications.py --exp motion_divider --iters 200 --seed 0`
- `python experiments/mechanical_network_replications.py --exp fig4e_symmetry --iters 200 --seed 0`
- `python scripts/plot_run.py runs/<run_folder>`

Outputs:
- `runs/YYYYMMDD_HHMMSS_<experiment>/` with `meta.json`, `README.txt`, history files (`.npz`/`.csv`), and `plots/`.

Reproducibility:
- Random seed is stored in `meta.json`.

## Adaptive Membrane Experiments

**Physics**: 2D membrane with pixelated Young's modulus, uniform pressure actuation

**Learning variable**: Young's modulus E per patch (stiffness, not rest length)

**Validation sequence**:
1. Single target learning (2×2 patches, sequential targets)
2. Multi-target learning (3×3 patches, simultaneous targets)
3. Pressure regime comparison (1-3 psi vs 7-10 psi nonlinearity)

Run:
```bash
# Sequential single targets: (1,0) then (0,1)
python experiments/adaptive_membrane_experiment.py --exp single_target_2x2 --target_patch 1,0 --target_height 4.0 --pressure 5.0 --iters 100

# Multiple simultaneous targets
python experiments/adaptive_membrane_experiment.py --exp multi_target_3x3 --iters 100

# Pressure regime comparison
python experiments/adaptive_membrane_experiment.py --exp pressure_sweep --pressure_range 1,3,7,10 --iters 50
```

Outputs:
- Same structure as mechanical replications: `runs/YYYYMMDD_HHMMSS_<experiment>/`
- Includes E_patches history, z_displacement history, convergence plots

Material parameters:
- E_min = 0.6 MPa (Shore A50)
- E_max = 381.7 MPa (Shore A95)
- Random initialization: E_i ~ Uniform(E_min, E_max)

## Tier 1 PDE Membrane (FEA-ish surrogate)

Linear variable-coefficient PDE backend that produces smooth, globally-coupled shapes
while keeping the same free vs clamped learning loop.

Run (from `coupled-learning-membranes/`):
```bash
# Sweep deflection vs pressure/thickness/E
PYTHONPATH=coupled_learning/backends \
python scripts/sweep_membrane_deflection_tier1.py \
  --n_patches 2 --mesh_ref 6 \
  --pressures 1,3,5,7,10 \
  --thicknesses 0.5,1.0,2.0 \
  --E_values 0.6,381.7 \
  --out sweep_tier1.csv

# Single-target learning demo
PYTHONPATH=coupled_learning/backends \
python experiments/adaptive_membrane_experiment_tier1.py \
  --n_patches 2 --pressure 5 --patch_size 2 --thickness 1 --mesh_ref 6 \
  --target_patch 1,0 --target_height 0.25 \
  --alpha 0.05 --iters 100 --out runs_tier1
```

## Run Viewer UI

Install:
- `pip install streamlit`

Run:
- `streamlit run ui/run_viewer.py`

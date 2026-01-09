# Altman2024 Fig.2 (Motion Divider) Parameters

## Sequential trials (from Fig.3 table)
Trial a: a = 2.00 cm, alpha = 0.16
Trial b: a = -1.00 cm, alpha = 0.32
Trial c: a = 0.70 cm, alpha = 0.48
Trial d: a = -1.20 cm, alpha = 0.40

## Discrete update size (experiment)
Half-turn length change = 0.035 cm

## Task definition
Motion divider learns an additive constant a between input and output positions.
Define the output as:
- y1 = 0.5*y2 + a
- a = 0.5*(L1 - L2)
Each trial specifies a desired a (not a desired y1).

## Boundary conditions (paper setup)
- 1D vertical system: x is fixed for all nodes (no x DOF).
- y=0 is the fixed top anchor (node 0).
- y2 is the INPUT node position and is held fixed in both free and clamped states.
- y1 is the OUTPUT (middle node) and is the only node that changes between free and clamped states.

## Clamping strategy
- Compute y1_free from the free solve.
- Define y1_desired = 0.5*y2_input + a_trial.
- Apply clamp: y1_clamped = eta*y1_desired + (1-eta)*y1_free.
- For Fig.2 experiments, eta = 1.0 (full clamp).

## Learning update (rest length L0)
- Learning variable is rest length L0 (not stiffness).
- Update rule uses edge stretch difference:
  ΔLi = alpha * (s_i^C - s_i^F)
- For the 2-edge divider, enforce ΔL2 = -ΔL1.

## Baseline rest length
- Baseline rest length: L_BASE = 13.75 cm.
- Gravity/preload is absorbed into this baseline.
- Learning updates apply small deviations around L_BASE.

## Plotting conventions
- Top row: a_est = y1 - 0.5*y2 vs steps with dashed line at a_trial.
- Bottom-left combined: a_est = y1 - 0.5*y2 with dashed line at a_trial.
- Network configuration panel is schematic/normalized and does not represent cm-scale lengths.

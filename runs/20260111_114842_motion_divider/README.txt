Run contents

- meta.json: experiment metadata (seed, hyperparameters, git hash)
- plots/: output figures saved by scripts/plot_run.py
- history files:
  - motion_divider_run_a.npz: iteration (32,), y_middle (32,), y1_free (32,), y1_clamped (32,), error (32,), a_est (32,), L1 (32,), L2 (32,), delta_L1 (32,), delta_L2 (32,), cumulative_iter (32,)
  - motion_divider_run_b.npz: iteration (16,), y_middle (16,), y1_free (16,), y1_clamped (16,), error (16,), a_est (16,), L1 (16,), L2 (16,), delta_L1 (16,), delta_L2 (16,), cumulative_iter (16,)
  - motion_divider_run_c.npz: iteration (9,), y_middle (9,), y1_free (9,), y1_clamped (9,), error (9,), a_est (9,), L1 (9,), L2 (9,), delta_L1 (9,), delta_L2 (9,), cumulative_iter (9,)
  - motion_divider_run_d.npz: iteration (12,), y_middle (12,), y1_free (12,), y1_clamped (12,), error (12,), a_est (12,), L1 (12,), L2 (12,), delta_L1 (12,), delta_L2 (12,), cumulative_iter (12,)

Shapes are shown as (rows, columns) where applicable.

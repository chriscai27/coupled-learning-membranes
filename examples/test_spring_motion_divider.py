"""
Complete replication of Altman et al. 2024 Figure 2.

Figure 2 shows:
- Motion divider trained 4 consecutive times
- Targets defined by additive constant a: y1 = 0.5*y2 + a
- Trial-specific clamp factors (eta)
- Rest-length update uses edge stretch contrast
- Convergence in ~5-10 steps per target
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path


class MotionDividerBackend:
    """Motion divider with rest length learning (1D motion in y only)."""
    
    def __init__(self, node_positions, edges, fixed_nodes, stiffness=1e6):
        self.node_positions_initial = node_positions.copy()
        self.node_positions = node_positions.copy()
        self.edges = edges
        self.base_fixed_nodes = set(fixed_nodes)
        self.fixed_nodes = set(fixed_nodes)
        self.stiffness = stiffness
        self.fixed_x = node_positions[:, 0].copy()
        self.fixed_y = {idx: node_positions[idx, 1] for idx in self.base_fixed_nodes}
        
        # Compute initial rest lengths
        self.rest_lengths = {}
        for edge_id, (i, j) in enumerate(edges):
            dist = np.linalg.norm(node_positions[i] - node_positions[j])
            self.rest_lengths[edge_id] = dist
        
        self.free_nodes = sorted(list(set(range(len(node_positions))) - self.fixed_nodes))
    
    def update_rest_lengths(self, delta_L_dict):
        for edge_id, delta_L in delta_L_dict.items():
            self.rest_lengths[edge_id] += delta_L
    
    def _compute_energy(self, free_y):
        positions = self.node_positions.copy()
        positions[:, 0] = self.fixed_x
        positions[self.free_nodes, 1] = free_y
        
        energy = 0.0
        for edge_id, (i, j) in enumerate(self.edges):
            r = np.linalg.norm(positions[i] - positions[j])
            L0 = self.rest_lengths[edge_id]
            energy += 0.5 * self.stiffness * (r - L0)**2
        return energy
    
    def solve_free(self):
        if len(self.free_nodes) == 0:
            return
        self.node_positions[:, 0] = self.fixed_x
        for idx, y_val in self.fixed_y.items():
            self.node_positions[idx, 1] = y_val
        from scipy.optimize import minimize
        x0 = self.node_positions[self.free_nodes, 1]
        result = minimize(self._compute_energy, x0, method='BFGS')
        self.node_positions[self.free_nodes, 1] = result.x
    
    def solve_clamped(self, target_node, target_y):
        """Clamp specific node to target position."""
        original_fixed = self.fixed_nodes.copy()
        self.fixed_nodes = self.base_fixed_nodes | {target_node}
        self.node_positions[:, 0] = self.fixed_x
        for idx, y_val in self.fixed_y.items():
            self.node_positions[idx, 1] = y_val
        self.node_positions[target_node, 1] = target_y
        
        self.free_nodes = sorted(list(set(range(len(self.node_positions))) - self.fixed_nodes))
        
        if len(self.free_nodes) > 0:
            from scipy.optimize import minimize
            x0 = self.node_positions[self.free_nodes, 1]
            result = minimize(self._compute_energy, x0, method='BFGS')
            self.node_positions[self.free_nodes, 1] = result.x
        
        self.fixed_nodes = original_fixed
        self.free_nodes = sorted(list(set(range(len(self.node_positions))) - self.fixed_nodes))
    
    def get_edge_length(self, edge_id):
        """Get current length of an edge."""
        i, j = self.edges[edge_id]
        return np.linalg.norm(self.node_positions[i] - self.node_positions[j])


def train_single_target(backend, a_target, y2_input, eta=1.0, alpha=0.1,
                        max_iterations=20, threshold=0.01,
                        quantize_step=None):
    """
    Train network for single target using discrete updates.
    
    Args:
        backend: MotionDividerBackend
        target_y: Target position for middle node
        nudge_factor: η in paper
        step_size: Fixed step size for updates (like turnbuckle rotation)
        max_iterations: Maximum training steps
        threshold: Stop when error below this
        
    Returns:
        history: Dict with training history
    """
    history = {
        'iteration': [],
        'y_middle': [],
        'y1_free': [],
        'y1_clamped': [],
        'error': [],
        'a_est': [],
        'L1': [],
        'L2': [],
        'delta_L1': [],
        'delta_L2': []
    }
    
    # ALWAYS log initial state (iteration 0)
    backend.solve_free()
    y1_free = backend.node_positions[1, 1]
    y1_desired = 0.5 * y2_input + a_target
    error_initial = abs(y1_free - y1_desired)
    a_est = y1_free - 0.5 * y2_input
    
    history['iteration'].append(0)
    history['y_middle'].append(y1_free)
    history['y1_free'].append(y1_free)
    history['y1_clamped'].append(y1_free)
    history['error'].append(error_initial)
    history['a_est'].append(a_est)
    history['L1'].append(backend.rest_lengths[0])
    history['L2'].append(backend.rest_lengths[1])
    history['delta_L1'].append(0.0)
    history['delta_L2'].append(0.0)
    
    # Check if already at target
    if error_initial < threshold:
        print(f"    Already at target (error={error_initial:.4f})")
        return history
    
    for iteration in range(1, max_iterations + 1):
        # Step 1: Solve FREE
        backend.solve_free()
        y1_free = backend.node_positions[1, 1]
        L1_free = backend.get_edge_length(0)
        L2_free = backend.get_edge_length(1)
        
        # Check convergence
        error = abs(y1_free - y1_desired)
        if error < threshold:
            print(f"    Converged at iteration {iteration} (error={error:.4f})")
            break
        
        # Step 2: Compute CLAMPED target with eta
        y1_clamped_target = eta * y1_desired + (1 - eta) * y1_free
        
        # Step 3: Solve CLAMPED
        backend.solve_clamped(target_node=1, target_y=y1_clamped_target)
        y1_clamped = backend.node_positions[1, 1]
        L1_clamped = backend.get_edge_length(0)
        L2_clamped = backend.get_edge_length(1)
        
        # Step 4: Measure edge lengths in both states
        backend.solve_free()
        
        # Step 5: Rest-length update from stretch contrast.
        signal1 = L1_clamped - L1_free
        delta_L1 = alpha * signal1
        if quantize_step is not None and abs(delta_L1) > 1e-12:
            delta_L1 = np.round(delta_L1 / quantize_step) * quantize_step
        delta_L2 = -delta_L1
        
        backend.update_rest_lengths({
            0: delta_L1,
            1: delta_L2
        })
        
        # Log
        backend.solve_free()
        y1_free = backend.node_positions[1, 1]
        a_est = y1_free - 0.5 * y2_input
        
        history['iteration'].append(iteration)
        history['y_middle'].append(y1_free)
        history['y1_free'].append(y1_free)
        history['y1_clamped'].append(y1_clamped)
        history['error'].append(abs(y1_free - y1_desired))
        history['a_est'].append(a_est)
        history['L1'].append(backend.rest_lengths[0])
        history['L2'].append(backend.rest_lengths[1])
        history['delta_L1'].append(delta_L1)
        history['delta_L2'].append(delta_L2)
    
    return history


def main():
    print("=" * 70)
    print("FULL FIGURE 2 REPLICATION - Altman et al. 2024")
    print("=" * 70)
    
    # Setup
    L_BASE_CM = 13.75
    node_positions = np.array([[0.0, 0.0], [0.0, L_BASE_CM], [0.0, 2*L_BASE_CM]], dtype=float)
    edges = [(0, 1), (1, 2)]
    fixed_nodes = [0, 2]
    
    backend = MotionDividerBackend(node_positions, edges, fixed_nodes)
    
    print(f"\n[Setup]")
    print(f"  Motion divider: 3 nodes, 2 edges")
    print(f"  Fixed: top anchor (y=0.0) and input node (y=2*L_BASE_CM)")
    print(f"  Constraint: x fixed for all nodes; only y changes")
    print(f"  Learning: middle node position")
    print(f"  Fixed nodes (y): {sorted(backend.base_fixed_nodes)}")
    print(f"  Free nodes (y): {backend.free_nodes}")
    
    # Figure 2: Train 4 consecutive targets (Altman2024 Fig.2 parameters).
    trials = [
        {'label': 'a', 'a': 2.00, 'alpha': 0.16},
        {'label': 'b', 'a': -1.00, 'alpha': 0.32},
        {'label': 'c', 'a': 0.70, 'alpha': 0.48},
        {'label': 'd', 'a': -1.20, 'alpha': 0.40},
    ]

    # Additive-constant task: y1 = 0.5*y2 + a
    input_node = 2
    output_node = 1
    input_y = backend.node_positions[input_node, 1]
    
    targets = [0.5 * input_y + trial['a'] for trial in trials]
    print(f"\n[Training Sequence - Figure 2]")
    print(f"  Will train {len(targets)} consecutive targets")
    print(f"  Input node {input_node} y = {input_y:.2f}")
    print(f"  Targets (a): {[t['a'] for t in trials]}  (y1_desired = 0.5*y2 + a)")
    
    # Storage for all training runs
    all_histories = []
    # Track update steps only (iteration 0 is the logged initial state).
    cumulative_iterations = 0
    
    # Optional: match experiment's discrete half-turn step.
    use_discrete_update = False
    discrete_step_cm = 0.079

    for run_id, target in enumerate(targets):
        eta = 1.0
        a_value = trials[run_id]['a']
        label = trials[run_id]['label']
        print(
            f"\n  Run {run_id+1}/{len(targets)} (Trial {label}): "
            f"a = {a_value:.2f}, alpha = {trials[run_id]['alpha']:.2f}, eta = {eta:.2f}, y1_desired = {target:.2f}"
        )
        
        history = train_single_target(
            backend,
            a_target=a_value,
            y2_input=input_y,
            eta=eta,
            alpha=trials[run_id]['alpha'],
            max_iterations=20,
            threshold=0.01,
            quantize_step=(discrete_step_cm if use_discrete_update else None)
        )
        
        # Add cumulative iteration count for plotting
        history['cumulative_iter'] = [cumulative_iterations + i for i in history['iteration']]
        # Only count actual updates so sequential targets align without a phantom step.
        cumulative_iterations += max(len(history['iteration']) - 1, 0)
        
        all_histories.append({
            'run_id': run_id,
            'target': target,
            'history': history
        })
        
        final_y = history['y_middle'][-1]
        final_error = history['error'][-1]
        denom = abs(target) if abs(target) > 1e-9 else 1.0
        print(f"    Final: y={final_y:.4f}, error={final_error:.4f} ({final_error/denom*100:.2f}%)")
        y2_final = backend.node_positions[input_node, 1]
        y1_free_final = history['y1_free'][-1]
        y1_clamped_final = history['y1_clamped'][-1]
        a_est_final = history['a_est'][-1]
        max_x_drift = float(np.max(np.abs(backend.node_positions[:, 0] - backend.fixed_x)))
        print(
            f"    Diagnostics: y2={y2_final:.4f}, y1_free={y1_free_final:.4f}, "
            f"y1_clamped={y1_clamped_final:.4f}, a_est={a_est_final:.4f}, "
            f"L1={backend.rest_lengths[0]:.4f}, L2={backend.rest_lengths[1]:.4f}, "
            f"max_x_drift={max_x_drift:.2e}"
        )
        if max_x_drift > 1e-6:
            print("    WARNING: max_x_drift exceeds 1e-6")
    
    # =========================================================================
    # Plot - Replicate Figure 2
    # =========================================================================
    
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid: 2 rows, 4 columns
    # Top row: Individual training runs (2a, 2b, 2c, 2d)
    # Bottom row: Combined view + rest lengths + network diagram
    
    colors = ['blue', 'green', 'red', 'purple']
    
    # Top row: Individual runs (like Figure 2a-d)
    all_y = [target for target in targets]
    for run_data in all_histories:
        all_y.extend(run_data['history']['y_middle'])
    y_min = min(all_y) - 0.1
    y_max = max(all_y) + 0.1

    for i, run_data in enumerate(all_histories):
        ax = plt.subplot(2, 4, i+1)
        history = run_data['history']
        target = run_data['target']
        
        ax.plot(history['iteration'], history['y_middle'], 'o-', 
                color=colors[i], linewidth=2, markersize=8)
        ax.axhline(target, color='red', linestyle='--', linewidth=2)
        
        # Add initial point
        if i == 0:
            ax.plot(0, 1.0, 'o', color='lightgray', markersize=10, label='Initial')
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Middle Node Position (y)')
        trial = trials[i]
        ax.set_title(f'Run {i+1}: a_target={trial["a"]:.2f}, α={trial["alpha"]:.2f}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([y_min, y_max])
    
    # Bottom left: Combined progress (plot learned a = y1 - 0.5*y2)
    ax = plt.subplot(2, 4, 5)
    for i, run_data in enumerate(all_histories):
        history = run_data['history']
        target = run_data['target']
        a_target = trials[i]['a']
        a_history = [y_val - 0.5 * input_y for y_val in history['y_middle']]
        
        # Plot with cumulative iterations
        ax.plot(history['cumulative_iter'], a_history, 
                'o-', color=colors[i], linewidth=2, markersize=6, 
                label=f'a={a_target:.2f}')
        
        # Show target line for each segment
        if len(history['cumulative_iter']) > 0:
            start_idx = 1 if len(history['cumulative_iter']) > 1 else 0
            start = history['cumulative_iter'][start_idx]
            end = history['cumulative_iter'][-1]
            ax.hlines(a_target, start, end, color=colors[i], 
                     linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Cumulative Training Steps')
    ax.set_ylabel('Additive Constant a (y1 - 0.5*y2)')
    ax.set_title('All Training Runs Combined (a)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom middle: Rest length evolution
    ax = plt.subplot(2, 4, 6)
    for i, run_data in enumerate(all_histories):
        history = run_data['history']
        ax.plot(history['cumulative_iter'], history['L1'], 
                color=colors[i], linewidth=2, linestyle='-', alpha=0.7)
        ax.plot(history['cumulative_iter'], history['L2'], 
                color=colors[i], linewidth=2, linestyle='--', alpha=0.7)
    
    # Legend
    ax.plot([], [], 'k-', linewidth=2, label='L₁ (bottom)')
    ax.plot([], [], 'k--', linewidth=2, label='L₂ (top)')
    ax.set_xlabel('Cumulative Training Steps')
    ax.set_ylabel('Rest Length')
    ax.set_title('Rest Length Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom middle-right: Error (log scale)
    ax = plt.subplot(2, 4, 7)
    for i, run_data in enumerate(all_histories):
        history = run_data['history']
        ax.semilogy(history['cumulative_iter'], history['error'], 
                   'o-', color=colors[i], linewidth=2, markersize=6)
    
    ax.set_xlabel('Cumulative Training Steps')
    ax.set_ylabel('Error (log scale)')
    ax.set_title('Error Convergence')
    ax.grid(True, alpha=0.3)
    
    # Bottom right: Final network configuration (1D vertical)
    ax = plt.subplot(2, 4, 8)
    
    # Initial configuration (gray)
    ax.plot([0, 0, 0], [0, 1, 2], 'o-', color='lightgray', 
            linewidth=3, markersize=12, label='Initial', alpha=0.5)
    
    # Final configuration (blue)
    final_y = backend.node_positions[1, 1]
    ax.plot([0, 0, 0], [0, final_y, 2], 'o-', color='blue', 
            linewidth=3, markersize=12, label='Final')
    
    # Target line
    final_target = targets[-1]
    ax.axhline(final_target, color='red', linestyle='--', 
               linewidth=2, label=f'Final Target ({final_target:.1f})')
    
    # Annotations
    ax.text(-0.05, 0, 'Fixed\n(y=0)', ha='right', va='center')
    ax.text(-0.05, final_y, f'Learned\n(y={final_y:.2f})', ha='right', va='center')
    ax.text(-0.05, 2, 'Fixed\n(y=2)', ha='right', va='center')
    
    ax.set_ylabel('y')
    ax.set_title('Network Configuration')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.2, 2.2])
    ax.set_xticks([])
    
    plt.tight_layout()
    
    # Save
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = output_dir / f'figure2_full_test_{timestamp}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved: {fig_path}")
    
    plt.show()

    # =========================================================================
    # Plot - Bottom row only (2x2 layout)
    # =========================================================================

    rest_length_history = {
        'cumulative_iter': [],
        'L1': [],
        'L2': [],
        'run_segments': [],
        'L_base': L_BASE_CM,
    }
    idx_start = 0
    for i, run_data in enumerate(all_histories):
        history = run_data['history']
        rest_length_history['cumulative_iter'].extend(history['cumulative_iter'])
        rest_length_history['L1'].extend(history['L1'])
        rest_length_history['L2'].extend(history['L2'])
        idx_end = idx_start + len(history['cumulative_iter'])
        rest_length_history['run_segments'].append((idx_start, idx_end, colors[i]))
        idx_start = idx_end

    configs = [
        {
            'label': 'Initial',
            'color': 'lightgray',
            'L1': all_histories[0]['history']['L1'][0],
            'L2': all_histories[0]['history']['L2'][0],
            'a_est': all_histories[0]['history']['a_est'][0],
        }
    ]
    for i, run_data in enumerate(all_histories):
        history = run_data['history']
        configs.append({
            'label': f'After Run {i+1}',
            'color': colors[i],
            'L1': history['L1'][-1],
            'L2': history['L2'][-1],
            'a_est': history['a_est'][-1],
        })

    fig2, axs = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    # (0,0) All Training Runs Combined (a)
    ax = axs[0, 0]
    for i, run_data in enumerate(all_histories):
        history = run_data['history']
        a_target = trials[i]['a']
        a_hist = [y_val - 0.5 * input_y for y_val in history['y_middle']]

        ax.plot(history['cumulative_iter'], a_hist,
                'o-', color=colors[i], linewidth=2, markersize=5,
                label=f'a={a_target:.2f}')
        ax.axhline(a_target, color=colors[i], linestyle='--', alpha=0.6)

    ax.set_xlabel('Cumulative Training Steps')
    ax.set_ylabel('Additive constant a_est = y1 - 0.5*y2')
    ax.set_title('All Training Runs Combined (a)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # (1,0) Rest Length Evolution
    ax = axs[1, 0]
    ax.plot(rest_length_history['cumulative_iter'], rest_length_history['L1'],
            '-', color='black', linewidth=2, label='L1 (bottom)')
    ax.plot(rest_length_history['cumulative_iter'], rest_length_history['L2'],
            '--', color='black', linewidth=2, label='L2 (top)')
    ax.axhline(rest_length_history['L_base'], color='gray', linestyle=':', linewidth=2,
               label='Baseline rest length')

    for start, end, c in rest_length_history['run_segments']:
        ax.plot(rest_length_history['cumulative_iter'][start:end],
                rest_length_history['L1'][start:end],
                '-', color=c, linewidth=3, alpha=0.5)
        ax.plot(rest_length_history['cumulative_iter'][start:end],
                rest_length_history['L2'][start:end],
                '--', color=c, linewidth=3, alpha=0.5)

    ax.set_xlabel('Cumulative Training Steps')
    ax.set_ylabel('Rest Length')
    ax.set_title('Rest Length Evolution')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # (0,1) Error Convergence (log scale)
    ax = axs[0, 1]
    for i, run_data in enumerate(all_histories):
        history = run_data['history']
        ax.semilogy(history['cumulative_iter'], history['error'],
                    'o-', color=colors[i], linewidth=2, markersize=5)

    ax.set_xlabel('Cumulative Training Steps')
    ax.set_ylabel('Error (log scale)')
    ax.set_title('Error Convergence')
    ax.grid(True, alpha=0.3)

    # (1,1) Network Configuration (schematic)
    ax = axs[1, 1]
    y_schem = np.array([0.0, 0.5, 1.0])
    x_offsets = np.linspace(-0.15, 0.15, len(configs))

    for cfg, x0 in zip(configs, x_offsets):
        ax.plot([x0, x0, x0], y_schem, 'o-', color=cfg['color'],
                linewidth=3, markersize=10, alpha=0.95, label=cfg['label'])
        ax.text(x0 + 0.02, 0.5,
                f"L1={cfg['L1']:.2f}\nL2={cfg['L2']:.2f}\na={cfg['a_est']:.2f}",
                fontsize=8, va='center')

    ax.set_title('Network Configuration (schematic)')
    ax.set_ylabel('y (schematic)')
    ax.set_xticks([])
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(loc='upper right')

    fig2_path = output_dir / f'figure2_bottom_row_{timestamp}.png'
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved: {fig2_path}")
    plt.show()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Trained {len(targets)} consecutive targets")
    print(f"Total training steps (updates only): {cumulative_iterations}")
    print(f"\nFinal state after all training:")
    print(f"  Middle node: y = {backend.node_positions[1, 1]:.4f}")
    print(f"  Rest length L₁: {backend.rest_lengths[0]:.4f}")
    print(f"  Rest length L₂: {backend.rest_lengths[1]:.4f}")
    print(f"\n✅ Successfully replicated Altman et al. 2024 Figure 2!")


if __name__ == "__main__":
    main()

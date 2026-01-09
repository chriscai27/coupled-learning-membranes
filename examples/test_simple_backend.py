"""
Simple test example for coupled learning framework.

This example demonstrates the complete workflow:
1. Create a simple mechanical network (springs)
2. Define patch sampling layout
3. Run coupled learning to achieve target shape
4. Visualize results

No COMSOL required - uses SimpleMechanicalBackend!
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from coupled_learning import (
    SimpleMechanicalBackend,
    create_patch_sampling,
    SignalType,
    SignalComputer,
    create_update_rule,
    CoupledLearningEngine
)


def main():
    print("=" * 60)
    print("Coupled Learning Test - Simple Mechanical Backend")
    print("=" * 60)
    
    # =========================================================================
    # STEP 1: Create the physical network (2D spring network)
    # =========================================================================
    print("\n[1/6] Creating spring network...")
    
    # Define a 3×3 grid of nodes
    node_positions = np.array([
        [0, 0], [1, 0], [2, 0],  # Bottom row
        [0, 1], [1, 1], [2, 1],  # Middle row
        [0, 2], [1, 2], [2, 2],  # Top row
    ], dtype=float)
    
    # Define edges (springs connecting nodes)
    # Horizontal edges
    edges = [
        (0, 1), (1, 2),  # Bottom
        (3, 4), (4, 5),  # Middle
        (6, 7), (7, 8),  # Top
    ]
    # Vertical edges
    edges += [
        (0, 3), (1, 4), (2, 5),  # Left to right
        (3, 6), (4, 7), (5, 8),
    ]
    
    # Fix corner nodes (boundary conditions)
    fixed_nodes = [0, 2, 6, 8]
    
    # Create backend
    backend = SimpleMechanicalBackend(
        node_positions=node_positions,
        edges=edges,
        fixed_nodes=fixed_nodes,
        stiffness_initial=1e6,  # Initial stiffness
        rest_length_initial=None  # Auto-compute from initial positions
    )
    
    print(f"  ✓ Network: {len(node_positions)} nodes, {len(edges)} edges")
    print(f"  ✓ Fixed nodes: {fixed_nodes}")
    
    # =========================================================================
    # STEP 2: Create patch sampling layout
    # =========================================================================
    print("\n[2/6] Creating patch sampling layout...")
    
    # Option A: Use 3×3 grid (simple)
    # patches = create_patch_sampling('grid', n_rows=3, n_cols=3, 
    #                                 origin=(0, 0), patch_size=1.0)
    
    # Option B: Use custom layout (more flexible)
    patches = create_patch_sampling('grid', n_rows=3, n_cols=3, 
                                    origin=(0, 0), patch_size=1.0)
    
    print(f"  ✓ Created {len(patches)} patches")
    
    # =========================================================================
    # STEP 3: Create signal computer
    # =========================================================================
    print("\n[3/6] Setting up signal computation...")
    
    signal_computer = SignalComputer(SignalType.CURVATURE)
    print(f"  ✓ Using signal type: {SignalType.CURVATURE.value}")
    
    # =========================================================================
    # STEP 4: Create update rule
    # =========================================================================
    print("\n[4/6] Creating update rule...")
    
    # Define stiffness bounds
    k_min = 1e5
    k_max = 1e7
    
    # Only update the center patch (patch 4 in 3×3 grid)
    target_patches = {4}
    
    update_rule = create_update_rule(
        rule_type='continuous',  # or 'quantized'
        k_min=k_min,
        k_max=k_max,
        target_patches=target_patches,
        alpha=1e5  # Learning rate
    )
    
    print(f"  ✓ Update rule: continuous with α=1e5")
    print(f"  ✓ Stiffness bounds: [{k_min:.1e}, {k_max:.1e}]")
    print(f"  ✓ Target patches: {target_patches}")
    
    # =========================================================================
    # STEP 5: Create learning engine and run
    # =========================================================================
    print("\n[5/6] Running coupled learning...")
    
    # Define target: want center patch to have height 0.1
    target_heights = {4: 0.1}
    
    engine = CoupledLearningEngine(
        backend=backend,
        patch_sampling=patches,
        signal_computer=signal_computer,
        update_rule=update_rule,
        target_patches=target_heights,
        nudge_factor=0.1  # η = 0.1
    )
    
    # Initialize all patches with same stiffness
    initial_stiffness = {i: 1e6 for i in patches.keys()}
    engine.initialize_stiffness(initial_stiffness)
    
    # Run learning!
    n_iterations = 50
    print(f"\n  Running {n_iterations} iterations...")
    logs_df = engine.run_multiple_iterations(n_iterations)
    
    # =========================================================================
    # STEP 6: Visualize results
    # =========================================================================
    print("\n[6/6] Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Stiffness evolution for target patch
    ax = axes[0, 0]
    target_data = logs_df[logs_df['patch_id'] == 4]
    ax.plot(target_data['iteration'], target_data['k_new'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Stiffness (N/m)')
    ax.set_title('Center Patch Stiffness Evolution')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Signal contrast (ΔS) over time
    ax = axes[0, 1]
    ax.plot(target_data['iteration'], target_data['delta_s'], 'r-', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('ΔS (Clamped - Free)')
    ax.set_title('Signal Contrast (Driving Learning)')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Free vs Clamped signals
    ax = axes[1, 0]
    ax.plot(target_data['iteration'], target_data['s_free'], 'g-', 
            label='Free', linewidth=2)
    ax.plot(target_data['iteration'], target_data['s_clamped'], 'orange', 
            label='Clamped', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Signal Value')
    ax.set_title('Free vs Clamped Signals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Final stiffness distribution (heatmap)
    ax = axes[1, 1]
    final_stiffness = engine.get_current_stiffness()
    
    # Reshape to 3×3 grid
    stiffness_grid = np.array([final_stiffness[i] for i in range(9)]).reshape(3, 3)
    
    im = ax.imshow(stiffness_grid, cmap='viridis', aspect='auto')
    ax.set_title('Final Stiffness Distribution')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            patch_id = i * 3 + j
            text = ax.text(j, i, f'{stiffness_grid[i, j]:.2e}',
                          ha="center", va="center", color="w", fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Stiffness (N/m)')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    fig_path = output_dir / 'coupled_learning_test.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved figure: {fig_path}")
    
    # Save logs
    log_path = output_dir / 'coupled_learning_logs.csv'
    engine.save_logs(log_path)
    print(f"  ✓ Saved logs: {log_path}")
    
    plt.show()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Initial center stiffness: {initial_stiffness[4]:.2e} N/m")
    print(f"Final center stiffness:   {final_stiffness[4]:.2e} N/m")
    print(f"Change: {((final_stiffness[4] - initial_stiffness[4]) / initial_stiffness[4] * 100):.1f}%")
    print("\nTest completed successfully! ✓")


if __name__ == "__main__":
    main()
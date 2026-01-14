#!/usr/bin/env python
"""
Visualization script for adaptive membrane learning results.

Creates heatmaps showing:
1. E (Young's modulus) evolution over iterations
2. z-displacement at patch centers
3. Signal magnitude
4. Error convergence

Usage:
    python scripts/visualize_membrane_learning.py runs/<run_folder>
    python scripts/visualize_membrane_learning.py  # uses most recent run
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
import json
import argparse
import sys
from typing import List, Optional, Tuple


def load_results(run_dir: Path):
    """Load results from a run directory."""
    history = np.load(run_dir / 'history.npz')
    with open(run_dir / 'meta.json', 'r') as f:
        meta = json.load(f)
    return history, meta


def _get_targets(meta):
    if 'target_patches' in meta and 'target_heights_mm' in meta:
        patches = [tuple(p) for p in meta['target_patches']]
        heights = [float(h) for h in meta['target_heights_mm']]
        return patches, heights
    patch = meta.get('target_patch', [0, 0])
    if isinstance(patch, list):
        patch = tuple(patch)
    height = float(meta.get('target_height_mm', 0.0))
    return [patch], [height]


def _format_targets(meta) -> str:
    patches, heights = _get_targets(meta)
    return "; ".join([f"{patch}→{height:.2f} mm" for patch, height in zip(patches, heights)])


def plot_E_evolution(history, meta, save_dir: Path):
    """Plot E distribution as heatmaps at different iterations."""
    E_patches = history['E_patches']
    n_iters = len(E_patches)
    n_patches = E_patches.shape[1]

    # Select iterations to show
    if n_iters <= 5:
        iters_to_show = list(range(n_iters))
    else:
        iters_to_show = [0, n_iters//4, n_iters//2, 3*n_iters//4, n_iters-1]

    fig, axes = plt.subplots(1, len(iters_to_show), figsize=(3*len(iters_to_show), 3))
    if len(iters_to_show) == 1:
        axes = [axes]

    # Colorbar range (prefer meta if provided)
    vmin = float(meta.get('E_min_mpa', np.min(E_patches)))
    vmax = float(meta.get('E_max_mpa', np.max(E_patches)))

    for ax, iter_idx in zip(axes, iters_to_show):
        E = E_patches[iter_idx]
        im = ax.imshow(E, cmap='viridis', vmin=vmin, vmax=vmax, origin='upper')
        ax.set_title(f'Iter {iter_idx}')
        ax.set_xlabel('Patch j')
        ax.set_ylabel('Patch i')

        # Add text annotations
        for i in range(n_patches):
            for j in range(n_patches):
                color = 'white' if E[i, j] < (vmin + vmax) / 2 else 'black'
                ax.text(j, i, f'{E[i, j]:.1f}', ha='center', va='center',
                       color=color, fontsize=8)

    fig.colorbar(im, ax=axes, label='E (MPa)', shrink=0.8)

    # Mark target patch
    fig.suptitle(f'Young\'s Modulus Evolution\nTargets: {_format_targets(meta)}')

    plt.tight_layout()
    plt.savefig(save_dir / 'E_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'E_evolution.png'}")


def plot_E_snapshots(history, meta, save_dir: Path):
    """Plot E heatmaps for iteration 0, mid, and final."""
    E_patches = history['E_patches']
    n_iters = len(E_patches)

    if n_iters == 1:
        iters_to_show = [0]
    else:
        mid_idx = n_iters // 2
        iters_to_show = [0, mid_idx, n_iters - 1]

    fig, axes = plt.subplots(1, len(iters_to_show), figsize=(3.5 * len(iters_to_show), 3.5))
    if len(iters_to_show) == 1:
        axes = [axes]

    vmin = np.min(E_patches)
    vmax = np.max(E_patches)
    n_patches = E_patches.shape[1]

    for ax, iter_idx in zip(axes, iters_to_show):
        E = E_patches[iter_idx]
        im = ax.imshow(E, cmap='viridis', vmin=vmin, vmax=vmax, origin='upper')
        ax.set_title(f'Iter {iter_idx}')
        ax.set_xlabel('Patch j')
        ax.set_ylabel('Patch i')

        for i in range(n_patches):
            for j in range(n_patches):
                color = 'white' if E[i, j] < (vmin + vmax) / 2 else 'black'
                ax.text(j, i, f'{E[i, j]:.1f}', ha='center', va='center',
                       color=color, fontsize=8)

    fig.colorbar(im, ax=axes, label='E (MPa)', shrink=0.8)

    fig.suptitle(f'E Snapshots (0 / mid / final)\nTargets: {_format_targets(meta)}')

    plt.tight_layout()
    plt.savefig(save_dir / 'E_snapshots_0_mid_final.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'E_snapshots_0_mid_final.png'}")


def _select_patch_series(n_patches: int, target: Optional[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Select a subset of patches to plot when the grid is large."""
    max_all = 9
    if n_patches * n_patches <= max_all:
        return [(i, j) for i in range(n_patches) for j in range(n_patches)]

    if target is None:
        target = (n_patches // 2, n_patches // 2)

    i_t, j_t = target
    candidates = [
        (i_t, j_t),
        (i_t - 1, j_t),
        (i_t + 1, j_t),
        (i_t, j_t - 1),
        (i_t, j_t + 1),
        (i_t - 1, j_t - 1),
        (i_t - 1, j_t + 1),
        (i_t + 1, j_t - 1),
        (i_t + 1, j_t + 1),
    ]
    selected = []
    for i, j in candidates:
        if 0 <= i < n_patches and 0 <= j < n_patches and (i, j) not in selected:
            selected.append((i, j))
        if len(selected) >= max_all:
            break
    return selected


def plot_E_vs_iteration(history, meta, save_dir: Path):
    """Plot E versus iteration for each patch or a selected subset."""
    E_patches = history['E_patches']
    n_iters = E_patches.shape[0]
    n_patches = E_patches.shape[1]
    x = np.arange(n_iters)

    targets, _ = _get_targets(meta)
    target = targets[0] if targets else None
    patches = _select_patch_series(n_patches, target)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (i, j) in patches:
        label = f'E({i},{j})'
        ax.plot(x, E_patches[:, i, j], label=label)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('E (MPa)')
    ax.set_title('E vs Iteration per Patch')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', ncol=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_dir / 'E_vs_iteration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'E_vs_iteration.png'}")


def plot_error_vs_iteration(history, meta, save_dir: Path):
    """Plot target error versus iteration."""
    error = history['error']
    x = np.arange(len(error))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, error, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Error (mm)')
    ax.set_title('Target Error vs Iteration')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'error_vs_iteration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'error_vs_iteration.png'}")


def plot_z_evolution(history, meta, save_dir: Path):
    """Plot z-displacement as heatmaps at different iterations."""
    if 'z_free' not in history:
        print("Skipping z_evolution: z_free not found in history (Tier-1 runs omit it).")
        return
    z_free = history['z_free']
    n_iters = len(z_free)
    n_patches = z_free.shape[1]

    # Select iterations to show
    if n_iters <= 5:
        iters_to_show = list(range(n_iters))
    else:
        iters_to_show = [0, n_iters//4, n_iters//2, 3*n_iters//4, n_iters-1]

    fig, axes = plt.subplots(1, len(iters_to_show), figsize=(3*len(iters_to_show), 3))
    if len(iters_to_show) == 1:
        axes = [axes]

    # Find global min/max for consistent colorbar
    vmin = np.min(z_free)
    _, target_heights = _get_targets(meta)
    vmax = max(np.max(z_free), max(target_heights) if target_heights else np.max(z_free))

    for ax, iter_idx in zip(axes, iters_to_show):
        z = z_free[iter_idx]
        im = ax.imshow(z, cmap='plasma', vmin=vmin, vmax=vmax, origin='upper')
        ax.set_title(f'Iter {iter_idx}')
        ax.set_xlabel('Patch j')
        ax.set_ylabel('Patch i')

        # Add text annotations
        for i in range(n_patches):
            for j in range(n_patches):
                color = 'white' if z[i, j] < (vmin + vmax) / 2 else 'black'
                ax.text(j, i, f'{z[i, j]:.3f}', ha='center', va='center',
                       color=color, fontsize=8)

    fig.colorbar(im, ax=axes, label='z (mm)', shrink=0.8)

    fig.suptitle(f'Z-Displacement Evolution\nTargets: {_format_targets(meta)}')

    plt.tight_layout()
    plt.savefig(save_dir / 'z_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'z_evolution.png'}")


def plot_error_convergence(history, meta, save_dir: Path):
    """Plot error convergence over iterations."""
    error = history['error']

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(error, 'b-', linewidth=2, label='Error')
    ax.axhline(y=0, color='g', linestyle='--', alpha=0.5, label='Target')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Error (mm)')
    ax.set_title(f'Learning Convergence\n'
                f'Targets: {_format_targets(meta)}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Log scale if error spans orders of magnitude
    if error[0] > 0 and error[-1] > 0 and error[0] / error[-1] > 10:
        ax.set_yscale('log')

    # Add final error annotation
    ax.annotate(f'Final: {error[-1]:.4f} mm',
               xy=(len(error)-1, error[-1]),
               xytext=(len(error)*0.7, error[0]*0.5),
               arrowprops=dict(arrowstyle='->', color='red'),
               fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(save_dir / 'error_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'error_convergence.png'}")


def plot_signal_heatmap(history, meta, save_dir: Path):
    """Plot signal magnitude at first and last iteration."""
    signal = history['signal']
    n_patches = signal.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Find global min/max for consistent colorbar
    vmin = np.min(signal)
    vmax = np.max(signal)

    for ax, iter_idx, title in zip(axes, [0, -1], ['First Iteration', 'Last Iteration']):
        s = signal[iter_idx]
        im = ax.imshow(s, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='upper')
        ax.set_title(title)
        ax.set_xlabel('Patch j')
        ax.set_ylabel('Patch i')

        # Add text annotations
        for i in range(n_patches):
            for j in range(n_patches):
                ax.text(j, i, f'{s[i, j]:.4f}', ha='center', va='center',
                       color='black', fontsize=8)

    fig.colorbar(im, ax=axes, label='Signal (z_free² - z_clamped²)', shrink=0.8)

    fig.suptitle(f'Signal Contrast\nNegative = needs to soften, Positive = needs to stiffen')

    plt.tight_layout()
    plt.savefig(save_dir / 'signal_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'signal_heatmap.png'}")


def plot_combined_summary(history, meta, save_dir: Path):
    """Create a combined summary plot."""
    if 'z_free' not in history:
        print("Skipping combined_summary: z_free not found in history (Tier-1 runs omit it).")
        return
    E_patches = history['E_patches']
    z_free = history['z_free']
    error = history['error']
    signal = history['signal']
    n_patches = E_patches.shape[1]

    fig = plt.figure(figsize=(14, 10))

    # Create grid
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Row 1: E evolution (4 panels)
    E_vmin = float(meta.get('E_min_mpa', np.min(E_patches)))
    E_vmax = float(meta.get('E_max_mpa', np.max(E_patches)))
    iters_E = [0, len(E_patches)//3, 2*len(E_patches)//3, len(E_patches)-1]
    for idx, iter_i in enumerate(iters_E):
        ax = fig.add_subplot(gs[0, idx])
        im = ax.imshow(E_patches[iter_i], cmap='viridis', vmin=E_vmin, vmax=E_vmax, origin='upper')
        ax.set_title(f'E @ iter {iter_i}', fontsize=10)
        for i in range(n_patches):
            for j in range(n_patches):
                E = E_patches[iter_i][i, j]
                color = 'white' if E < 200 else 'black'
                ax.text(j, i, f'{E:.0f}', ha='center', va='center', color=color, fontsize=7)
        if idx == 0:
            ax.set_ylabel('E (MPa)')

    # Row 2: z evolution (4 panels)
    z_vmin = np.min(z_free)
    _, target_heights = _get_targets(meta)
    z_vmax = max(np.max(z_free), max(target_heights) if target_heights else np.max(z_free))
    iters_z = [0, len(z_free)//3, 2*len(z_free)//3, len(z_free)-1]
    for idx, iter_i in enumerate(iters_z):
        ax = fig.add_subplot(gs[1, idx])
        im = ax.imshow(z_free[iter_i], cmap='plasma', vmin=z_vmin, vmax=z_vmax, origin='upper')
        ax.set_title(f'z @ iter {iter_i}', fontsize=10)
        for i in range(n_patches):
            for j in range(n_patches):
                z = z_free[iter_i][i, j]
                color = 'white' if z < z_vmax/2 else 'black'
                ax.text(j, i, f'{z:.2f}', ha='center', va='center', color=color, fontsize=7)
        if idx == 0:
            ax.set_ylabel('z (mm)')

    # Row 3: Error convergence (left 2 panels) + Signal (right 2 panels)
    ax_error = fig.add_subplot(gs[2, :2])
    ax_error.plot(error, 'b-', linewidth=2)
    ax_error.axhline(y=0, color='g', linestyle='--', alpha=0.5)
    ax_error.set_xlabel('Iteration')
    ax_error.set_ylabel('Error (mm)')
    ax_error.set_title('Error Convergence')
    ax_error.grid(True, alpha=0.3)
    ax_error.annotate(f'Final: {error[-1]:.4f} mm',
                     xy=(len(error)-1, error[-1]),
                     xytext=(len(error)*0.6, error[0]*0.6),
                     arrowprops=dict(arrowstyle='->', color='red'),
                     fontsize=9, color='red')

    # Signal at start and end
    for idx, (iter_i, title) in enumerate([(0, 'Signal (start)'), (-1, 'Signal (end)')]):
        ax = fig.add_subplot(gs[2, 2+idx])
        s = signal[iter_i]
        vabs = max(abs(np.min(signal)), abs(np.max(signal)))
        im = ax.imshow(s, cmap='RdBu_r', vmin=-vabs, vmax=vabs, origin='upper')
        ax.set_title(title, fontsize=10)
        for i in range(n_patches):
            for j in range(n_patches):
                ax.text(j, i, f'{s[i, j]:.3f}', ha='center', va='center', color='black', fontsize=7)

    # Main title
    fig.suptitle(f'Adaptive Membrane Learning Summary\n'
                f'Targets: {_format_targets(meta)} | '
                f'Pressure: {meta["pressure_psi"]} psi | '
                f'α={meta["alpha"]} | Final error: {error[-1]:.4f} mm',
                fontsize=12, fontweight='bold')

    plt.savefig(save_dir / 'summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'summary.png'}")


def find_latest_run(runs_dir: Path) -> Path:
    """Find the most recent run directory."""
    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and 'adaptive_membrane' in d.name])
    if not run_dirs:
        raise FileNotFoundError(f"No adaptive_membrane runs found in {runs_dir}")
    return run_dirs[-1]


def main():
    parser = argparse.ArgumentParser(description='Visualize membrane learning results')
    parser.add_argument('run_dir', nargs='?', default=None,
                       help='Run directory (default: most recent)')
    args = parser.parse_args()

    # Find run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        runs_dir = Path('runs')
        if not runs_dir.exists():
            print("No 'runs' directory found. Run an experiment first.")
            sys.exit(1)
        run_dir = find_latest_run(runs_dir)
        print(f"Using most recent run: {run_dir}")

    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        sys.exit(1)

    # Load results
    print(f"\nLoading results from: {run_dir}")
    history, meta = load_results(run_dir)

    print(f"\nExperiment parameters:")
    print(f"  Targets: {_format_targets(meta)}")
    print(f"  Pressure: {meta['pressure_psi']} psi")
    print(f"  Learning rate (α): {meta['alpha']}")
    print(f"  Iterations: {meta['n_iters']}")
    print(f"  Final error: {meta['final_error_mm']:.4f} mm")

    # Create plots directory
    plots_dir = run_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    print(f"\nGenerating visualizations...")
    plot_E_evolution(history, meta, plots_dir)
    plot_E_snapshots(history, meta, plots_dir)
    plot_E_vs_iteration(history, meta, plots_dir)
    plot_error_vs_iteration(history, meta, plots_dir)
    plot_z_evolution(history, meta, plots_dir)
    plot_error_convergence(history, meta, plots_dir)
    plot_signal_heatmap(history, meta, plots_dir)
    plot_combined_summary(history, meta, plots_dir)

    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()

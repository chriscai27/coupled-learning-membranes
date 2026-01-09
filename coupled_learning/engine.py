"""
Coupled learning engine - orchestrates the learning process.
"""
from typing import Dict, List
import pandas as pd
from datetime import datetime
import numpy as np

from .backends import PhysicsBackend
from .signals import PatchSamplingPoints, SignalComputer
from .update_rules import UpdateRule


class CoupledLearningEngine:
    """
    Main engine for coupled learning.
    
    Orchestrates the iterative learning process:
    1. Solve free state
    2. Compute clamped targets with nudge
    3. Solve clamped state
    4. Compute signal contrast
    5. Update stiffness parameters
    """
    
    def __init__(self,
                 backend: PhysicsBackend,
                 patch_sampling: Dict[int, PatchSamplingPoints],
                 signal_computer: SignalComputer,
                 update_rule: UpdateRule,
                 target_patches: Dict[int, float],
                 nudge_factor: float = 0.1):
        """
        Args:
            backend: Physics simulation backend
            patch_sampling: Sampling points for each patch
            signal_computer: Computes signals from heights
            update_rule: Rule for updating stiffness
            target_patches: {patch_id: desired_height}
            nudge_factor: Nudge amplitude η ∈ (0, 1]
        """
        self.backend = backend
        self.patch_sampling = patch_sampling
        self.signal_computer = signal_computer
        self.update_rule = update_rule
        self.target_patches = target_patches
        self.nudge_factor = nudge_factor
        
        # Current stiffness values
        self.stiffness = {}
        
        # Logging
        self.logs = []
        
    def initialize_stiffness(self, k_initial: Dict[int, float]) -> None:
        """
        Initialize stiffness values for all patches.
        
        Args:
            k_initial: {patch_id: initial_stiffness}
        """
        self.stiffness = k_initial.copy()
        self.backend.set_stiffness_params(self.stiffness)
        
    def run_single_iteration(self, iteration: int) -> Dict:
        """
        Run one iteration of coupled learning.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Dictionary with iteration results
        """
        # Step 1: Solve FREE state
        self.backend.solve_free()
        
        # Measure signals in free state
        signals_free = self._measure_all_signals()
        
        # Step 2: Compute CLAMPED targets with nudge
        clamp_targets = {}
        for patch_id, h_desired in self.target_patches.items():
            h_free = signals_free[patch_id]
            # Nudge: h_clamped = h_free + η * (h_desired - h_free)
            h_clamped = h_free + self.nudge_factor * (h_desired - h_free)
            clamp_targets[patch_id] = h_clamped
        
        # Step 3: Solve CLAMPED state
        self.backend.solve_clamped(clamp_targets)
        
        # Measure signals in clamped state
        signals_clamped = self._measure_all_signals()
        
        # Step 4: Compute signal contrast and update stiffness
        for patch_id in self.patch_sampling.keys():
            s_free = signals_free[patch_id]
            s_clamped = signals_clamped[patch_id]
            k_old = self.stiffness[patch_id]
            
            # Update stiffness using update rule
            k_new = self.update_rule.compute_update(
                patch_id, s_free, s_clamped, k_old
            )
            
            self.stiffness[patch_id] = k_new
            
            # Log this update
            is_target = patch_id in self.target_patches
            self.logs.append({
                'iteration': iteration,
                'patch_id': patch_id,
                'k_old': k_old,
                'k_new': k_new,
                's_free': s_free,
                's_clamped': s_clamped,
                'delta_s': s_clamped - s_free,
                'is_target': is_target,
            })
        
        # Apply updated stiffness to backend
        self.backend.set_stiffness_params(self.stiffness)
        
        # Return summary statistics
        return {
            'iteration': iteration,
            'mean_delta_s': np.mean([s_clamped - signals_free[i] 
                                    for i, s_clamped in signals_clamped.items()]),
            'max_delta_s': np.max([abs(s_clamped - signals_free[i])
                                  for i, s_clamped in signals_clamped.items()]),
        }
    
    def _measure_all_signals(self) -> Dict[int, float]:
        """
        Measure signals for all patches in current backend state.
        
        Returns:
            {patch_id: signal_value}
        """
        patch_heights = {}
        
        for patch_id, sampling_points in self.patch_sampling.items():
            # Get all 5 measurement points for this patch
            points = sampling_points.get_all_points()
            
            # Query backend for heights at these points
            heights = self.backend.eval_height(points)
            
            patch_heights[patch_id] = heights
        
        # Compute signals from heights
        signals = self.signal_computer.compute_all_signals(patch_heights)
        
        return signals
    
    def run_multiple_iterations(self, n_iterations: int) -> pd.DataFrame:
        """
        Run multiple iterations of coupled learning.
        
        Args:
            n_iterations: Number of iterations to run
            
        Returns:
            DataFrame with all logged data
        """
        print(f"Starting coupled learning for {n_iterations} iterations...")
        
        for i in range(n_iterations):
            summary = self.run_single_iteration(i)
            
            if (i + 1) % 10 == 0:
                print(f"Iteration {i+1}/{n_iterations}: "
                      f"mean|Δs|={summary['mean_delta_s']:.6f}, "
                      f"max|Δs|={summary['max_delta_s']:.6f}")
        
        print("Learning complete!")
        
        return pd.DataFrame(self.logs)
    
    def save_logs(self, filename: str) -> None:
        """Save logs to CSV file."""
        df = pd.DataFrame(self.logs)
        df.to_csv(filename, index=False)
        print(f"  ✓ Saved logs: {filename}")
    
    def get_current_stiffness(self) -> Dict[int, float]:
        """Get current stiffness values."""
        return self.stiffness.copy()
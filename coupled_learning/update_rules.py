"""
Update rules for coupled learning.
Defines how learning parameters (stiffness values) evolve.
"""
from abc import ABC, abstractmethod
from typing import Dict, Set
import numpy as np


class UpdateRule(ABC):
    """Abstract base class for update rules."""
    
    @abstractmethod
    def compute_update(self, patch_id: int, signal_free: float, 
                      signal_clamped: float, k_current: float) -> float:
        """
        Compute new stiffness value for a patch.
        
        Args:
            patch_id: ID of the patch being updated
            signal_free: Signal value in free state
            signal_clamped: Signal value in clamped state
            k_current: Current stiffness value
            
        Returns:
            New stiffness value
        """
        pass


class QuantizedUpdateRule(UpdateRule):
    """
    Quantized update rule with fixed step size.
    
    Update: k_new = k_old + sign(Δs) * δk
    
    Bounded to [k_min, k_max].
    """
    
    def __init__(self, k_min: float, k_max: float, delta_k: float,
                 threshold: float = 1e-6):
        """
        Args:
            k_min: Minimum allowed stiffness
            k_max: Maximum allowed stiffness
            delta_k: Step size for updates
            threshold: Minimum |Δs| to trigger update
        """
        self.k_min = k_min
        self.k_max = k_max
        self.delta_k = delta_k
        self.threshold = threshold
    
    def compute_update(self, patch_id: int, signal_free: float,
                      signal_clamped: float, k_current: float) -> float:
        """Compute quantized update."""
        delta_s = signal_clamped - signal_free
        
        # Only update if signal difference exceeds threshold
        if abs(delta_s) < self.threshold:
            return k_current
        
        # Quantized step in direction of gradient
        if delta_s > 0:
            k_new = k_current + self.delta_k
        else:
            k_new = k_current - self.delta_k
        
        # Enforce bounds
        k_new = np.clip(k_new, self.k_min, self.k_max)
        
        return k_new


class ContinuousUpdateRule(UpdateRule):
    """
    Continuous gradient-based update rule.
    
    Update: k_new = k_old + α * Δs
    
    Bounded to [k_min, k_max].
    """
    
    def __init__(self, k_min: float, k_max: float, alpha: float = 0.01):
        """
        Args:
            k_min: Minimum allowed stiffness
            k_max: Maximum allowed stiffness
            alpha: Learning rate
        """
        self.k_min = k_min
        self.k_max = k_max
        self.alpha = alpha
    
    def compute_update(self, patch_id: int, signal_free: float,
                      signal_clamped: float, k_current: float) -> float:
        """Compute continuous gradient update."""
        delta_s = signal_clamped - signal_free
        
        # Gradient step
        k_new = k_current + self.alpha * delta_s
        
        # Enforce bounds
        k_new = np.clip(k_new, self.k_min, self.k_max)
        
        return k_new


class SelectiveUpdateRule(UpdateRule):
    """
    Wrapper that only updates specified patches.
    
    Other patches keep their current values.
    """
    
    def __init__(self, base_rule: UpdateRule, target_patches: Set[int]):
        """
        Args:
            base_rule: Underlying update rule to use
            target_patches: Set of patch IDs to update
        """
        self.base_rule = base_rule
        self.target_patches = target_patches
    
    def compute_update(self, patch_id: int, signal_free: float,
                      signal_clamped: float, k_current: float) -> float:
        """Only update if patch is in target set."""
        if patch_id in self.target_patches:
            return self.base_rule.compute_update(
                patch_id, signal_free, signal_clamped, k_current
            )
        else:
            return k_current  # No change


def create_update_rule(rule_type: str, k_min: float, k_max: float,
                       target_patches: Set[int] = None, **kwargs) -> UpdateRule:
    """
    Factory function to create update rules.
    
    Args:
        rule_type: 'quantized' or 'continuous'
        k_min: Minimum stiffness
        k_max: Maximum stiffness
        target_patches: Optional set of patches to update (others frozen)
        **kwargs: Additional parameters (delta_k, alpha, etc.)
        
    Returns:
        UpdateRule instance
    """
    if rule_type == 'quantized':
        delta_k = kwargs.get('delta_k', (k_max - k_min) / 100)
        threshold = kwargs.get('threshold', 1e-6)
        rule = QuantizedUpdateRule(k_min, k_max, delta_k, threshold)
        
    elif rule_type == 'continuous':
        alpha = kwargs.get('alpha', 0.01)
        rule = ContinuousUpdateRule(k_min, k_max, alpha)
        
    else:
        raise ValueError(f"Unknown rule type: {rule_type}")
    
    # Wrap with selective update if target patches specified
    if target_patches is not None:
        rule = SelectiveUpdateRule(rule, target_patches)
    
    return rule
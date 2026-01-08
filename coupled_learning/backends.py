"""
Abstract backend interface for physics simulation.
All backends (COMSOL, Simple, Hardware) must implement this interface.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import numpy as np


class PhysicsBackend(ABC):
    """
    Abstract base class for physics backends.
    
    A backend handles the physics simulation - it computes equilibrium states
    of the physical system (membrane network) given boundary conditions.
    """
    
    @abstractmethod
    def set_stiffness_params(self, k_dict: Dict[int, float]) -> None:
        """
        Set stiffness parameters for edges/patches.
        
        Args:
            k_dict: Dictionary mapping edge/patch IDs to stiffness values
        """
        pass
    
    @abstractmethod
    def solve_free(self) -> None:
        """
        Solve for equilibrium with only input boundary conditions.
        No output constraints applied.
        """
        pass
    
    @abstractmethod
    def solve_clamped(self, clamp_targets: Dict) -> None:
        """
        Solve for equilibrium with both input and output constraints.
        
        Args:
            clamp_targets: Dictionary specifying output constraints
                          (format depends on backend implementation)
        """
        pass
    
    @abstractmethod
    def eval_height(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """
        Evaluate displacement/height at specified points.
        
        Args:
            points: List of (x, y) coordinates to sample
            
        Returns:
            Array of height/displacement values at those points
        """
        pass
"""
Signal computation for coupled learning.
Defines how to extract local signals from physical states.
"""
from enum import Enum
from typing import List, Tuple, Dict
import numpy as np


class SignalType(Enum):
    """Types of signals that can be computed from patch measurements."""
    CENTER = "center"           # Just center height
    EDGE_MEAN = "edge_mean"     # Average of edge heights
    CURVATURE = "curvature"     # Center minus edge average (dome-ness)


class PatchSamplingPoints:
    """
    Defines sampling points for a patch.
    
    Each patch is sampled at:
    - 1 center point
    - 4 edge points (N, S, E, W)
    """
    
    def __init__(self, patch_id: int, center: Tuple[float, float], 
                 size: float):
        """
        Args:
            patch_id: Unique identifier for this patch
            center: (x, y) center coordinates
            size: Size of patch (assumes square)
        """
        self.patch_id = patch_id
        self.center = center
        self.size = size
        
        # Define 5 sampling points: center + 4 edges
        cx, cy = center
        half = size / 2.0
        
        self.points = {
            'center': (cx, cy),
            'north': (cx, cy + half),
            'south': (cx, cy - half),
            'east': (cx + half, cy),
            'west': (cx - half, cy),
        }
        
    def get_all_points(self) -> List[Tuple[float, float]]:
        """Get all 5 sampling points as a list."""
        return [
            self.points['center'],
            self.points['north'],
            self.points['south'],
            self.points['east'],
            self.points['west'],
        ]
    
    def get_center(self) -> Tuple[float, float]:
        """Get center point."""
        return self.points['center']
    
    def get_edges(self) -> List[Tuple[float, float]]:
        """Get the 4 edge points."""
        return [
            self.points['north'],
            self.points['south'],
            self.points['east'],
            self.points['west'],
        ]


class SignalComputer:
    """
    Computes local signals from height measurements.
    """
    
    def __init__(self, signal_type: SignalType):
        """
        Args:
            signal_type: Which signal definition to use
        """
        self.signal_type = signal_type
    
    def compute_signal(self, heights: np.ndarray) -> float:
        """
        Compute signal from height measurements.
        
        Args:
            heights: Array of 5 heights [center, north, south, east, west]
            
        Returns:
            Computed signal value
        """
        h_center = heights[0]
        h_edges = heights[1:]  # 4 edge heights
        
        if self.signal_type == SignalType.CENTER:
            return h_center
            
        elif self.signal_type == SignalType.EDGE_MEAN:
            return np.mean(h_edges)
            
        elif self.signal_type == SignalType.CURVATURE:
            # Discrete Laplacian: center - mean(edges)
            return h_center - np.mean(h_edges)
            
        else:
            raise ValueError(f"Unknown signal type: {self.signal_type}")
    
    def compute_all_signals(self, patch_heights: Dict[int, np.ndarray]) -> Dict[int, float]:
        """
        Compute signals for all patches.
        
        Args:
            patch_heights: {patch_id: array of 5 heights}
            
        Returns:
            {patch_id: signal_value}
        """
        signals = {}
        for patch_id, heights in patch_heights.items():
            signals[patch_id] = self.compute_signal(heights)
        return signals


# =============================================================================
# PATCH LAYOUT CREATION - Factory Pattern
# =============================================================================

def _create_grid_patches(n_rows: int, n_cols: int, 
                        origin: Tuple[float, float] = (0, 0),
                        patch_size: float = 1.0) -> Dict[int, PatchSamplingPoints]:
    """
    Internal function: Create rectangular grid of patches.
    
    Patch numbering (row-major, bottom to top):
    Example 3×3:
    6 7 8
    3 4 5
    0 1 2
    
    Args:
        n_rows: Number of rows
        n_cols: Number of columns
        origin: (x, y) bottom-left corner
        patch_size: Size of each square patch
        
    Returns:
        Dictionary mapping patch_id to PatchSamplingPoints
    """
    patches = {}
    patch_id = 0
    x0, y0 = origin
    
    for row in range(n_rows):
        for col in range(n_cols):
            center_x = x0 + (col + 0.5) * patch_size
            center_y = y0 + (row + 0.5) * patch_size
            
            patches[patch_id] = PatchSamplingPoints(
                patch_id=patch_id,
                center=(center_x, center_y),
                size=patch_size
            )
            patch_id += 1
    
    return patches


def _create_circular_patches(n_patches: int, radius: float,
                            center: Tuple[float, float] = (0, 0),
                            patch_size: float = 1.0,
                            start_angle: float = 0.0) -> Dict[int, PatchSamplingPoints]:
    """
    Internal function: Create patches arranged in a circle.
    
    Args:
        n_patches: Number of patches around the circle
        radius: Radius of the circle
        center: (x, y) center of the circle
        patch_size: Size of each patch
        start_angle: Starting angle in radians (0 = right, counterclockwise)
        
    Returns:
        Dictionary mapping patch_id to PatchSamplingPoints
    """
    patches = {}
    cx, cy = center
    
    for i in range(n_patches):
        angle = start_angle + 2 * np.pi * i / n_patches
        patch_x = cx + radius * np.cos(angle)
        patch_y = cy + radius * np.sin(angle)
        
        patches[i] = PatchSamplingPoints(
            patch_id=i,
            center=(patch_x, patch_y),
            size=patch_size
        )
    
    return patches


def _create_custom_patches(patch_specs: List[Tuple[int, Tuple[float, float], float]]) -> Dict[int, PatchSamplingPoints]:
    """
    Internal function: Create patches with fully custom positions.
    
    Args:
        patch_specs: List of (patch_id, center, size) tuples
                    Example: [(0, (0.5, 0.5), 1.0), (1, (2.0, 0.5), 0.8), ...]
        
    Returns:
        Dictionary mapping patch_id to PatchSamplingPoints
    """
    patches = {}
    
    for patch_id, center, size in patch_specs:
        patches[patch_id] = PatchSamplingPoints(
            patch_id=patch_id,
            center=center,
            size=size
        )
    
    return patches


def create_patch_sampling(layout_type: str, **kwargs) -> Dict[int, PatchSamplingPoints]:
    """
    Main factory function: Create patch sampling layout.
    
    This is the primary user-facing function for creating patch configurations.
    
    Args:
        layout_type: Type of layout - 'grid', 'circular', or 'custom'
        **kwargs: Layout-specific parameters (see examples below)
        
    Returns:
        Dictionary mapping patch_id to PatchSamplingPoints
        
    Examples:
        >>> # Create a 3×3 grid
        >>> patches = create_patch_sampling('grid', n_rows=3, n_cols=3, patch_size=1.0)
        
        >>> # Create a 5×7 grid with custom origin
        >>> patches = create_patch_sampling('grid', n_rows=5, n_cols=7, 
        ...                                 origin=(10, 20), patch_size=5.0)
        
        >>> # Create 8 patches in a circle
        >>> patches = create_patch_sampling('circular', n_patches=8, radius=10.0,
        ...                                 center=(50, 50), patch_size=3.0)
        
        >>> # Create custom layout (match your experimental setup)
        >>> specs = [(0, (10.5, 20.3), 5.0), (1, (25.8, 19.7), 5.0), ...]
        >>> patches = create_patch_sampling('custom', patch_specs=specs)
        
    Raises:
        ValueError: If layout_type is not recognized
        TypeError: If required parameters are missing
    """
    layout_type = layout_type.lower()
    
    if layout_type == 'grid':
        # Required: n_rows, n_cols
        # Optional: origin, patch_size
        required = ['n_rows', 'n_cols']
        for param in required:
            if param not in kwargs:
                raise TypeError(f"'grid' layout requires '{param}' parameter")
        
        return _create_grid_patches(
            n_rows=kwargs['n_rows'],
            n_cols=kwargs['n_cols'],
            origin=kwargs.get('origin', (0, 0)),
            patch_size=kwargs.get('patch_size', 1.0)
        )
    
    elif layout_type == 'circular':
        # Required: n_patches, radius
        # Optional: center, patch_size, start_angle
        required = ['n_patches', 'radius']
        for param in required:
            if param not in kwargs:
                raise TypeError(f"'circular' layout requires '{param}' parameter")
        
        return _create_circular_patches(
            n_patches=kwargs['n_patches'],
            radius=kwargs['radius'],
            center=kwargs.get('center', (0, 0)),
            patch_size=kwargs.get('patch_size', 1.0),
            start_angle=kwargs.get('start_angle', 0.0)
        )
    
    elif layout_type == 'custom':
        # Required: patch_specs
        if 'patch_specs' not in kwargs:
            raise TypeError("'custom' layout requires 'patch_specs' parameter")
        
        return _create_custom_patches(kwargs['patch_specs'])
    
    else:
        raise ValueError(
            f"Unknown layout_type '{layout_type}'. "
            f"Must be 'grid', 'circular', or 'custom'."
        )


def create_3x3_patch_sampling(origin=(0, 0), patch_size=1.0) -> Dict[int, PatchSamplingPoints]:
    """
    Convenience function: Create a 3×3 grid.
    
    This is a simple wrapper for the common 3×3 case.
    
    Args:
        origin: (x, y) bottom-left corner
        patch_size: Size of each square patch
        
    Returns:
        Dictionary with 9 patches (IDs 0-8)
    """
    return create_patch_sampling('grid', n_rows=3, n_cols=3, 
                                origin=origin, patch_size=patch_size)
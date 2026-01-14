"""
Simple physics backends for testing (no COMSOL required).

Contains multiple backend implementations from different papers:
1. SimpleMechanicalBackend - Stern/Altman spring networks (nodes move in xy)
2. SimpleMembraneBackend - Tensile membrane (fixed xy, varying z)
3. SimpleFlowBackend - Dillavou electrical networks (for comparison)

All use the same PhysicsBackend interface, making them interchangeable.
"""

import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple, Dict

from ..base import PhysicsBackend


# =============================================================================
# 1. MECHANICAL SPRING NETWORK (Stern/Altman Papers)
# =============================================================================

class SimpleMechanicalBackend(PhysicsBackend):
    """
    2D mechanical spring network from Stern/Altman papers.
    
    Physics:
    - Nodes can move in (x, y) plane
    - Some nodes fixed as boundary conditions
    - Energy = (1/2) * k * (L - L0)^2 where L is current length
    
    This replicates the spring networks from:
    - Stern et al. 2021 (elastic networks)
    - Altman et al. 2024 (experimental spring networks)
    """
    
    def __init__(self, node_positions: np.ndarray, edges: List[Tuple[int, int]],
                 fixed_nodes: List[int], stiffness_initial: float = 1e6,
                 rest_length_initial: float = None):
        """
        Args:
            node_positions: (N, 2) array of initial node positions
            edges: List of (node_i, node_j) tuples defining connectivity
            fixed_nodes: List of node indices that are fixed (boundary conditions)
            stiffness_initial: Initial stiffness for all edges
            rest_length_initial: Initial rest length (None = use initial distances)
        """
        self.node_positions_initial = node_positions.copy()
        self.edges = edges
        self.fixed_nodes = set(fixed_nodes)
        
        # Current state
        self.node_positions = node_positions.copy()
        
        # Edge parameters (learning parameters)
        self.stiffness = {i: stiffness_initial for i in range(len(edges))}
        
        # Compute rest lengths from initial configuration
        self.rest_lengths = {}
        if rest_length_initial is None:
            for edge_id, (i, j) in enumerate(edges):
                dist = np.linalg.norm(node_positions[i] - node_positions[j])
                self.rest_lengths[edge_id] = dist
        else:
            for edge_id in range(len(edges)):
                self.rest_lengths[edge_id] = rest_length_initial
        
        # Free nodes (those that can move)
        all_nodes = set(range(len(node_positions)))
        self.free_nodes = sorted(list(all_nodes - self.fixed_nodes))
        
    def set_stiffness_params(self, k_dict: Dict[int, float]) -> None:
        """Set stiffness for edges."""
        avg_k = np.mean(list(k_dict.values()))
        for edge_id in range(len(self.edges)):
            self.stiffness[edge_id] = avg_k
    
    def _compute_energy(self, free_positions: np.ndarray) -> float:
        """Compute total elastic energy."""
        # Reconstruct full position array
        positions = self.node_positions.copy()
        positions[self.free_nodes] = free_positions.reshape(-1, 2)
        
        energy = 0.0
        for edge_id, (i, j) in enumerate(self.edges):
            # Current length
            r_ij = np.linalg.norm(positions[i] - positions[j])
            # Spring energy: (1/2) * k * (r - l0)^2
            k = self.stiffness[edge_id]
            l0 = self.rest_lengths[edge_id]
            energy += 0.5 * k * (r_ij - l0)**2
        
        return energy
    
    def solve_free(self) -> None:
        """Solve for equilibrium with only fixed boundary conditions."""
        if len(self.free_nodes) == 0:
            return
            
        x0 = self.node_positions[self.free_nodes].flatten()
        result = minimize(self._compute_energy, x0, method='BFGS')
        self.node_positions[self.free_nodes] = result.x.reshape(-1, 2)
    
    def solve_clamped(self, clamp_targets: Dict[int, float]) -> None:
        """
        Solve with additional constraints.
        
        For mechanical network: clamps specific nodes at target positions.
        """
        # For this example, clamp center node to target y-position
        center_node = len(self.node_positions) // 2
        target_y = list(clamp_targets.values())[0] if clamp_targets else 0
        
        original_fixed = self.fixed_nodes.copy()
        self.fixed_nodes.add(center_node)
        self.node_positions[center_node, 1] = target_y
        
        all_nodes = set(range(len(self.node_positions)))
        self.free_nodes = sorted(list(all_nodes - self.fixed_nodes))
        
        if len(self.free_nodes) > 0:
            x0 = self.node_positions[self.free_nodes].flatten()
            result = minimize(self._compute_energy, x0, method='BFGS')
            self.node_positions[self.free_nodes] = result.x.reshape(-1, 2)
        
        self.fixed_nodes = original_fixed
        all_nodes = set(range(len(self.node_positions)))
        self.free_nodes = sorted(list(all_nodes - self.fixed_nodes))
    
    def eval_height(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """Evaluate y-coordinate at specified points (using nearest node)."""
        heights = []
        for x, y in points:
            distances = np.linalg.norm(self.node_positions - np.array([x, y]), axis=1)
            nearest_node = np.argmin(distances)
            heights.append(self.node_positions[nearest_node, 1])
        return np.array(heights)


# =============================================================================
# 2. TENSILE MEMBRANE (Your Research - Fixed XY, Varying Z)
# =============================================================================

class SimpleMembraneBackend(PhysicsBackend):
    """
    Tensile-dominated membrane (spring network in 3D).
    
    Physics:
    - Nodes at FIXED (x, y) positions
    - Only z-coordinate (height) varies
    - Energy = (1/2) * k * (L - L0)^2
      where L = sqrt(dx^2 + dy^2 + dz^2) (3D length)
            L0 = sqrt(dx^2 + dy^2) (rest length in xy plane)
    
    This is the correct physics for tensile-dominated membranes.
    """
    
    def __init__(self, grid_x: np.ndarray, grid_y: np.ndarray,
                 boundary_nodes: List[int], 
                 stiffness_initial: float = 1e6):
        """
        Args:
            grid_x: (N,) array of x-coordinates (FIXED)
            grid_y: (N,) array of y-coordinates (FIXED)
            boundary_nodes: List of node indices clamped at z=0
            stiffness_initial: Initial spring stiffness
        """
        self.n_nodes = len(grid_x)
        self.grid_x = grid_x.copy()  # FIXED
        self.grid_y = grid_y.copy()  # FIXED
        
        # Current z-coordinates (what we solve for!)
        self.node_z = np.zeros(self.n_nodes)
        
        # Boundary conditions
        self.boundary_nodes = set(boundary_nodes)
        self.free_nodes = sorted(list(set(range(self.n_nodes)) - self.boundary_nodes))
        
        # Build spring network
        self.edges = self._build_edges()
        
        # Compute rest lengths (xy distances)
        self.rest_lengths = {}
        for edge_id, (i, j) in enumerate(self.edges):
            dx = self.grid_x[i] - self.grid_x[j]
            dy = self.grid_y[i] - self.grid_y[j]
            L0 = np.sqrt(dx**2 + dy**2)
            self.rest_lengths[edge_id] = L0
        
        # Edge stiffness (learning parameters!)
        self.stiffness = {i: stiffness_initial for i in range(len(self.edges))}
        
    def _build_edges(self) -> List[Tuple[int, int]]:
        """Build edges connecting neighboring nodes in xy-plane."""
        edges = []
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                dx = self.grid_x[i] - self.grid_x[j]
                dy = self.grid_y[i] - self.grid_y[j]
                dist_xy = np.sqrt(dx**2 + dy**2)
                if dist_xy < 1.5:  # Neighbors
                    edges.append((i, j))
        return edges
    
    def set_stiffness_params(self, k_dict: Dict[int, float]) -> None:
        """Set spring stiffness based on patch parameters."""
        avg_k = np.mean(list(k_dict.values()))
        for edge_id in range(len(self.edges)):
            self.stiffness[edge_id] = avg_k
    
    def _compute_energy(self, free_z: np.ndarray) -> float:
        """
        Compute TENSILE (stretching) energy.
        
        E = sum_edges (1/2) * k * (L - L0)^2
        where L = sqrt(dx^2 + dy^2 + dz^2)
              L0 = sqrt(dx^2 + dy^2)
        """
        z = self.node_z.copy()
        z[self.free_nodes] = free_z
        
        energy = 0.0
        for edge_id, (i, j) in enumerate(self.edges):
            dx = self.grid_x[i] - self.grid_x[j]
            dy = self.grid_y[i] - self.grid_y[j]
            dz = z[i] - z[j]
            
            L = np.sqrt(dx**2 + dy**2 + dz**2)  # Current 3D length
            L0 = self.rest_lengths[edge_id]      # Rest length (xy)
            
            k = self.stiffness[edge_id]
            energy += 0.5 * k * (L - L0)**2
        
        return energy
    
    def solve_free(self) -> None:
        """Solve for equilibrium with only boundary conditions (z=0 at edges)."""
        if len(self.free_nodes) == 0:
            return
        
        z0 = self.node_z[self.free_nodes]
        result = minimize(self._compute_energy, z0, method='BFGS',
                         options={'gtol': 1e-8, 'ftol': 1e-10})
        self.node_z[self.free_nodes] = result.x
        self.node_z[list(self.boundary_nodes)] = 0.0
    
    def solve_clamped(self, clamp_targets: Dict[int, float]) -> None:
        """Solve with additional height constraints."""
        center_node = self.n_nodes // 2
        target_z = list(clamp_targets.values())[0] if clamp_targets else 0
        
        original_boundary = self.boundary_nodes.copy()
        self.boundary_nodes.add(center_node)
        self.node_z[center_node] = target_z
        
        self.free_nodes = sorted(list(set(range(self.n_nodes)) - self.boundary_nodes))
        
        if len(self.free_nodes) > 0:
            z0 = self.node_z[self.free_nodes]
            result = minimize(self._compute_energy, z0, method='BFGS',
                             options={'gtol': 1e-8, 'ftol': 1e-10})
            self.node_z[self.free_nodes] = result.x
        
        self.boundary_nodes = original_boundary
        self.free_nodes = sorted(list(set(range(self.n_nodes)) - self.boundary_nodes))
    
    def eval_height(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """Evaluate z at (x,y) points using nearest neighbor."""
        heights = []
        for x, y in points:
            distances = np.sqrt((self.grid_x - x)**2 + (self.grid_y - y)**2)
            nearest = np.argmin(distances)
            heights.append(self.node_z[nearest])
        return np.array(heights)


# =============================================================================
# 3. FLOW NETWORK (Dillavou - Electrical Analog)
# =============================================================================

class SimpleFlowBackend(PhysicsBackend):
    """
    Resistor network simulator (Kirchhoff's laws).
    
    Physics:
    - Electrical analog: nodes = voltages, edges = resistors
    - Solves: L * V = I (Laplacian system)
    - From Dillavou et al. 2022 (twin network implementation)
    
    Useful for testing coupled learning with electrical analog.
    """
    
    def __init__(self, n_nodes: int, edges: List[Tuple[int, int]],
                 source_nodes: Dict[int, float], ground_nodes: List[int],
                 conductance_initial: float = 1.0):
        """
        Args:
            n_nodes: Number of nodes
            edges: List of (node_i, node_j) tuples
            source_nodes: {node_id: current} current sources
            ground_nodes: Nodes fixed at V=0
            conductance_initial: Initial conductance for all edges
        """
        self.n_nodes = n_nodes
        self.edges = edges
        self.source_nodes = source_nodes
        self.ground_nodes = set(ground_nodes)
        
        self.conductance = {i: conductance_initial for i in range(len(edges))}
        self.voltages = np.zeros(n_nodes)
    
    def set_stiffness_params(self, g_dict: Dict[int, float]) -> None:
        """Set conductances (analogous to stiffness)."""
        for edge_id, g in g_dict.items():
            if edge_id < len(self.edges):
                self.conductance[edge_id] = g
    
    def _build_laplacian(self) -> np.ndarray:
        """Build graph Laplacian matrix."""
        L = np.zeros((self.n_nodes, self.n_nodes))
        for edge_id, (i, j) in enumerate(self.edges):
            g = self.conductance[edge_id]
            L[i, i] += g
            L[j, j] += g
            L[i, j] -= g
            L[j, i] -= g
        return L
    
    def solve_free(self) -> None:
        """Solve for voltages with only source currents."""
        L = self._build_laplacian()
        b = np.zeros(self.n_nodes)
        
        for node, current in self.source_nodes.items():
            b[node] = current
        
        for node in self.ground_nodes:
            L[node, :] = 0
            L[node, node] = 1
            b[node] = 0
        
        self.voltages = np.linalg.solve(L, b)
    
    def solve_clamped(self, clamp_targets: Dict[int, float]) -> None:
        """Solve with additional voltage constraints."""
        L = self._build_laplacian()
        b = np.zeros(self.n_nodes)
        
        for node, current in self.source_nodes.items():
            b[node] = current
        
        for node in self.ground_nodes:
            L[node, :] = 0
            L[node, node] = 1
            b[node] = 0
        
        for node, voltage in clamp_targets.items():
            L[node, :] = 0
            L[node, node] = 1
            b[node] = voltage
        
        self.voltages = np.linalg.solve(L, b)
    
    def eval_height(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """Return voltages at specified nodes."""
        heights = []
        for x, _ in points:
            node = int(round(x))
            if 0 <= node < self.n_nodes:
                heights.append(self.voltages[node])
            else:
                heights.append(0.0)
        return np.array(heights)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_3x3_membrane_grid() -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Create 3×3 membrane grid for testing.
    
    Returns:
        grid_x, grid_y: Fixed node positions
        boundary_nodes: Indices of boundary (z=0)
    """
    x = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=float)
    boundary = [0, 1, 2, 3, 5, 6, 7, 8]  # All except center
    return x, y, boundary


def create_3x3_mechanical_grid() -> Tuple[np.ndarray, List[Tuple[int, int]], List[int]]:
    """
    Create 3×3 mechanical spring network for testing.
    
    Returns:
        node_positions: (N, 2) array
        edges: List of edges
        fixed_nodes: Boundary nodes
    """
    positions = np.array([
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2],
    ], dtype=float)
    
    edges = [
        (0, 1), (1, 2),  # Bottom row
        (3, 4), (4, 5),  # Middle row
        (6, 7), (7, 8),  # Top row
        (0, 3), (1, 4), (2, 5),  # Vertical
        (3, 6), (4, 7), (5, 8),
    ]
    
    fixed_nodes = [0, 2, 6, 8]  # Corners
    
    return positions, edges, fixed_nodes
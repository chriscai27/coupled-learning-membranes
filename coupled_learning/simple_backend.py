"""
Simple mechanical physics backend for testing coupled learning.
Uses basic force balance to simulate spring networks without COMSOL.
"""
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple

from .backends import PhysicsBackend


class SimpleMechanicalBackend(PhysicsBackend):
    """
    A lightweight spring network simulator for testing coupled learning.
    
    Uses energy minimization to find equilibrium node positions.
    Much faster than COMSOL for small networks and perfect for prototyping.
    
    Network structure is defined by:
    - nodes: positions in 2D space
    - edges: list of (node_i, node_j, stiffness, rest_length) tuples
    """
    
    def __init__(self, 
                 node_positions: np.ndarray,
                 edges: List[Tuple[int, int]],
                 fixed_nodes: List[int],
                 stiffness_initial: float = 1.0,
                 rest_length_initial: float = 1.0):
        """
        Initialize simple mechanical backend.
        
        Args:
            node_positions: Initial (N, 2) array of node positions
            edges: List of (node_i, node_j) tuples defining connectivity
            fixed_nodes: List of node indices that don't move
            stiffness_initial: Initial stiffness for all edges
            rest_length_initial: Initial rest length for all edges
        """
        self.n_nodes = len(node_positions)
        self.n_edges = len(edges)
        self.edges = edges  # [(i, j), ...]
        self.fixed_nodes = set(fixed_nodes)
        self.free_nodes = [i for i in range(self.n_nodes) if i not in self.fixed_nodes]
        
        # Initial node positions
        self.node_positions = node_positions.copy()
        self.boundary_positions = {}  # Stores positions of fixed nodes
        
        # Learning degrees of freedom (one per edge)
        self.stiffness = {i: stiffness_initial for i in range(self.n_edges)}
        self.rest_length = {i: rest_length_initial for i in range(self.n_edges)}
        
        # Store current state
        self.current_positions = node_positions.copy()
        self.current_edge_lengths = np.zeros(self.n_edges)
        
    def set_stiffness_params(self, stiffness_dict: Dict[int, float]) -> None:
        """Set stiffness for edges."""
        self.stiffness.update(stiffness_dict)
        
    def set_rest_lengths(self, rest_length_dict: Dict[int, float]) -> None:
        """Set rest lengths for edges."""
        self.rest_length.update(rest_length_dict)
    
    def _compute_energy(self, free_node_positions: np.ndarray) -> float:
        """
        Compute total elastic energy of the network.
        
        Args:
            free_node_positions: Flattened array of free node positions
            
        Returns:
            Total elastic energy
        """
        # Reconstruct full position array
        positions = self.current_positions.copy()
        free_node_indices = self.free_nodes
        positions[free_node_indices] = free_node_positions.reshape(-1, 2)
        
        # Compute energy from all edges
        energy = 0.0
        for edge_idx, (i, j) in enumerate(self.edges):
            pos_i = positions[i]
            pos_j = positions[j]
            
            # Current length
            r_ij = np.linalg.norm(pos_j - pos_i)
            
            # Elastic energy: E = (1/2) * k * (r - l0)^2
            k = self.stiffness[edge_idx]
            l0 = self.rest_length[edge_idx]
            
            energy += 0.5 * k * (r_ij - l0)**2
            
        return energy
    
    def _solve_equilibrium(self, constrained_nodes: Dict[int, np.ndarray] = None) -> np.ndarray:
        """
        Find equilibrium positions by minimizing elastic energy.
        
        Args:
            constrained_nodes: Dict of {node_id: position} for nodes to fix
            
        Returns:
            Full array of node positions at equilibrium
        """
        # Update boundary conditions
        if constrained_nodes is not None:
            for node_id, pos in constrained_nodes.items():
                self.current_positions[node_id] = pos
        
        # Initial guess for free nodes
        x0 = self.current_positions[self.free_nodes].flatten()
        
        # Minimize energy
        result = minimize(
            self._compute_energy,
            x0,
            method='BFGS',
            options={'gtol': 1e-6, 'maxiter': 1000}
        )
        
        if not result.success:
            print(f"Warning: Optimization did not converge: {result.message}")
        
        # Update positions
        self.current_positions[self.free_nodes] = result.x.reshape(-1, 2)
        
        # Update edge lengths
        for edge_idx, (i, j) in enumerate(self.edges):
            pos_i = self.current_positions[i]
            pos_j = self.current_positions[j]
            self.current_edge_lengths[edge_idx] = np.linalg.norm(pos_j - pos_i)
        
        return self.current_positions.copy()
    
    def solve_free(self) -> None:
        """Solve for equilibrium with only boundary conditions (no output constraints)."""
        # Only fixed nodes are constrained
        constrained = {node_id: self.current_positions[node_id] 
                      for node_id in self.fixed_nodes}
        self._solve_equilibrium(constrained)
    
    def solve_clamped(self, clamp_targets: Dict[int, np.ndarray]) -> None:
        """
        Solve for equilibrium with output nodes clamped.
        
        Args:
            clamp_targets: {node_id: target_position} for nodes to clamp
        """
        # Both fixed nodes and clamped output nodes are constrained
        constrained = {node_id: self.current_positions[node_id] 
                      for node_id in self.fixed_nodes}
        constrained.update(clamp_targets)
        
        self._solve_equilibrium(constrained)
    
    def eval_height(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """
        Evaluate heights at specified points.
        
        For mechanical networks, this returns the node positions.
        Points should be node indices encoded as (node_id, 0).
        
        Args:
            points: List of (node_id, 0) tuples
            
        Returns:
            Array of node positions (2D)
        """
        heights = []
        for node_id, _ in points:
            node_id = int(node_id)
            heights.append(self.current_positions[node_id])
        
        return np.array(heights)
    
    def get_edge_lengths(self) -> Dict[int, float]:
        """Get current length of each edge."""
        return {i: self.current_edge_lengths[i] for i in range(self.n_edges)}
    
    def get_node_positions(self) -> np.ndarray:
        """Get current positions of all nodes."""
        return self.current_positions.copy()


class SimpleFlowBackend(PhysicsBackend):
    """
    Simple flow network simulator (voltage/current analogy).
    
    Uses Kirchhoff's laws to solve for node voltages.
    Equivalent to electrical resistor network.
    """
    
    def __init__(self,
                 n_nodes: int,
                 edges: List[Tuple[int, int]],
                 conductance_initial: float = 1.0):
        """
        Initialize simple flow backend.
        
        Args:
            n_nodes: Number of nodes
            edges: List of (node_i, node_j) tuples
            conductance_initial: Initial conductance for all edges
        """
        self.n_nodes = n_nodes
        self.n_edges = len(edges)
        self.edges = edges
        
        # Learning degrees of freedom (conductances)
        self.conductance = {i: conductance_initial for i in range(self.n_edges)}
        
        # Current state
        self.node_voltages = np.zeros(n_nodes)
        self.edge_currents = np.zeros(self.n_edges)
        
    def set_stiffness_params(self, conductance_dict: Dict[int, float]) -> None:
        """Set conductances (analogous to stiffness in mechanical networks)."""
        self.conductance.update(conductance_dict)
    
    def _build_laplacian(self) -> np.ndarray:
        """Build Laplacian matrix for the flow network."""
        L = np.zeros((self.n_nodes, self.n_nodes))
        
        for edge_idx, (i, j) in enumerate(self.edges):
            k = self.conductance[edge_idx]
            
            L[i, i] += k
            L[j, j] += k
            L[i, j] -= k
            L[j, i] -= k
            
        return L
    
    def _solve_voltages(self, fixed_voltages: Dict[int, float]) -> np.ndarray:
        """
        Solve for node voltages using Kirchhoff's laws.
        
        Args:
            fixed_voltages: {node_id: voltage} for nodes with fixed voltages
            
        Returns:
            Array of all node voltages
        """
        L = self._build_laplacian()
        
        # Separate fixed and free nodes
        free_nodes = [i for i in range(self.n_nodes) if i not in fixed_voltages]
        fixed_nodes = list(fixed_voltages.keys())
        
        if len(free_nodes) == 0:
            # All nodes fixed
            voltages = np.zeros(self.n_nodes)
            for node_id, v in fixed_voltages.items():
                voltages[node_id] = v
            return voltages
        
        # Extract submatrices
        L_ff = L[np.ix_(free_nodes, free_nodes)]
        L_fc = L[np.ix_(free_nodes, fixed_nodes)]
        
        # Known voltages
        v_c = np.array([fixed_voltages[i] for i in fixed_nodes])
        
        # Solve: L_ff * v_f = -L_fc * v_c
        v_f = np.linalg.solve(L_ff, -L_fc @ v_c)
        
        # Reconstruct full voltage array
        voltages = np.zeros(self.n_nodes)
        voltages[free_nodes] = v_f
        for node_id, v in fixed_voltages.items():
            voltages[node_id] = v
        
        self.node_voltages = voltages
        
        # Compute edge currents
        for edge_idx, (i, j) in enumerate(self.edges):
            k = self.conductance[edge_idx]
            self.edge_currents[edge_idx] = k * (voltages[i] - voltages[j])
        
        return voltages
    
    def solve_free(self) -> None:
        """Solve with only input voltages fixed."""
        # This needs to be set externally before calling
        pass  # Handled by setting node_voltages directly
    
    def solve_clamped(self, clamp_targets: Dict[int, float]) -> None:
        """
        Solve with input and output voltages fixed.
        
        Args:
            clamp_targets: {node_id: voltage} for clamped nodes
        """
        # This will include both inputs and outputs
        self._solve_voltages(clamp_targets)
    
    def eval_height(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """
        Evaluate voltages at specified nodes.
        
        Args:
            points: List of (node_id, 0) tuples
            
        Returns:
            Array of voltages
        """
        voltages = []
        for node_id, _ in points:
            node_id = int(node_id)
            voltages.append(self.node_voltages[node_id])
        
        return np.array(voltages)
    
    def get_edge_voltage_drops(self) -> Dict[int, float]:
        """Get voltage drop across each edge."""
        drops = {}
        for edge_idx, (i, j) in enumerate(self.edges):
            drops[edge_idx] = self.node_voltages[i] - self.node_voltages[j]
        return drops

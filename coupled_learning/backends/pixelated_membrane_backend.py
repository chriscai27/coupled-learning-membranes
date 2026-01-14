"""
Pixelated Membrane Backend for Coupled Learning

2D membrane divided into square patches, each with learnable Young's modulus.
Uses custom FEA solver for membrane mechanics.

Physics:
- Tensile-dominated membrane (fixed edges, bottom pressure)
- Each patch has independent E_i (learning parameter)
- Signal: (z_free)² - (z_clamped)² per patch

Solver:
- Uses energy minimization approach for geometric nonlinearity
- Membrane strain energy: E = (1/2) * k * (L - L0)² for each spring
- Spring stiffness derived from patch Young's modulus: k = E * h / L0
"""

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from typing import Dict, Tuple, List

from ..base import PhysicsBackend


class PixelatedMembraneBackend(PhysicsBackend):
    """
    Backend for pixelated membrane with learnable stiffness per patch.

    Coordinate system:
        x, y: in-plane (fixed at edges)
        z: out-of-plane displacement (solved)

    Grid indexing: (i, j) where i is row (y-direction), j is column (x-direction)
        Origin (0,0) at top-left

    Physics model:
        - Membrane modeled as network of springs in 3D
        - Springs have rest length = xy distance between nodes
        - Spring stiffness k = E * h / L0 (membrane stiffness)
        - Energy minimization finds equilibrium under pressure
    """

    def __init__(
        self,
        n_patches_x: int = 2,
        n_patches_y: int = 2,
        patch_size_mm: float = 2.0,
        thickness_mm: float = 1.0,
        poisson_ratio: float = 0.45,
        E_min_mpa: float = 0.6,
        E_max_mpa: float = 381.7,
        pressure_psi: float = 5.0,
        mesh_refinement: int = 4,
        use_nonlinear: bool = True,
    ):
        """
        Args:
            n_patches_x: Number of patches in x-direction
            n_patches_y: Number of patches in y-direction
            patch_size_mm: Size of each square patch (mm)
            thickness_mm: Membrane thickness (mm)
            poisson_ratio: Poisson's ratio (dimensionless)
            E_min_mpa: Minimum Young's modulus (MPa)
            E_max_mpa: Maximum Young's modulus (MPa)
            pressure_psi: Bottom pressure (psi)
            mesh_refinement: Number of FEA nodes per patch edge
            use_nonlinear: Use nonlinear solver (energy minimization) vs linear
        """
        self.n_patches_x = n_patches_x
        self.n_patches_y = n_patches_y
        self.patch_size = patch_size_mm
        self.thickness = thickness_mm
        self.poisson = poisson_ratio
        self.E_min = E_min_mpa
        self.E_max = E_max_mpa
        self.pressure_psi = pressure_psi
        self.mesh_refinement = mesh_refinement
        self.use_nonlinear = use_nonlinear

        # Total membrane size (mm)
        self.Lx = n_patches_x * patch_size_mm
        self.Ly = n_patches_y * patch_size_mm

        # Convert units: pressure psi -> N/mm²
        # 1 psi = 6894.76 Pa = 0.00689476 N/mm²
        self.pressure_nmm2 = pressure_psi * 0.00689476

        # Learnable parameters: E per patch (shape: n_patches_y × n_patches_x)
        # Initialize to middle of range
        E_mid = (E_min_mpa + E_max_mpa) / 2
        self.E_patches = np.full((n_patches_y, n_patches_x), E_mid)

        # Mesh arrays
        self.nodes = None          # Node coordinates (n_nodes, 3)
        self.edges = None          # Edge connectivity list of (n1, n2)
        self.edge_to_patch = None  # Map edges to patch (i, j) tuples
        self.fixed_nodes = None    # Edge nodes with z=0 (array)
        self.free_nodes = None     # Interior nodes (array)

        # Precomputed values
        self.rest_lengths = None   # Edge rest lengths (xy distance)
        self.edge_stiffness = None # Edge stiffness values

        # Current solution
        self.z_current = None      # Current z-displacements

        self._build_mesh()
        self._compute_edge_properties()

    def _build_mesh(self):
        """Create FEA mesh with nodes and spring edges."""
        # Number of nodes in each direction
        nx_nodes = self.n_patches_x * self.mesh_refinement + 1
        ny_nodes = self.n_patches_y * self.mesh_refinement + 1

        # Create node coordinates
        x = np.linspace(0, self.Lx, nx_nodes)
        y = np.linspace(0, self.Ly, ny_nodes)
        xx, yy = np.meshgrid(x, y)

        self.nodes = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
        n_nodes = len(self.nodes)

        self.nx_nodes = nx_nodes
        self.ny_nodes = ny_nodes
        self.n_nodes = n_nodes

        # Identify fixed edge nodes (clamped boundary z=0)
        edge_mask = (
            (np.abs(xx.ravel()) < 1e-6) |                    # Left edge
            (np.abs(xx.ravel() - self.Lx) < 1e-6) |         # Right edge
            (np.abs(yy.ravel()) < 1e-6) |                    # Bottom edge
            (np.abs(yy.ravel() - self.Ly) < 1e-6)           # Top edge
        )
        self.fixed_nodes = np.where(edge_mask)[0]
        self.free_nodes = np.where(~edge_mask)[0]

        # Create edges (horizontal, vertical, and diagonal springs)
        edges = []
        edge_to_patch = []

        for j in range(ny_nodes):
            for i in range(nx_nodes):
                node_idx = j * nx_nodes + i
                node_x, node_y = self.nodes[node_idx, :2]

                # Horizontal edge (right neighbor)
                if i < nx_nodes - 1:
                    neighbor = node_idx + 1
                    edges.append((node_idx, neighbor))
                    # Edge patch determined by midpoint
                    mid_x = (node_x + self.nodes[neighbor, 0]) / 2
                    mid_y = (node_y + self.nodes[neighbor, 1]) / 2
                    edge_patch_j = min(int(mid_x / self.patch_size), self.n_patches_x - 1)
                    edge_patch_i = min(int(mid_y / self.patch_size), self.n_patches_y - 1)
                    edge_to_patch.append((edge_patch_i, edge_patch_j))

                # Vertical edge (upper neighbor)
                if j < ny_nodes - 1:
                    neighbor = node_idx + nx_nodes
                    edges.append((node_idx, neighbor))
                    mid_x = (node_x + self.nodes[neighbor, 0]) / 2
                    mid_y = (node_y + self.nodes[neighbor, 1]) / 2
                    edge_patch_j = min(int(mid_x / self.patch_size), self.n_patches_x - 1)
                    edge_patch_i = min(int(mid_y / self.patch_size), self.n_patches_y - 1)
                    edge_to_patch.append((edge_patch_i, edge_patch_j))

                # Diagonal edges for better mesh quality (\ direction)
                if i < nx_nodes - 1 and j < ny_nodes - 1:
                    neighbor = node_idx + nx_nodes + 1
                    edges.append((node_idx, neighbor))
                    mid_x = (node_x + self.nodes[neighbor, 0]) / 2
                    mid_y = (node_y + self.nodes[neighbor, 1]) / 2
                    edge_patch_j = min(int(mid_x / self.patch_size), self.n_patches_x - 1)
                    edge_patch_i = min(int(mid_y / self.patch_size), self.n_patches_y - 1)
                    edge_to_patch.append((edge_patch_i, edge_patch_j))

        self.edges = edges
        self.edge_to_patch = edge_to_patch

        # Initialize z to zeros
        self.z_current = np.zeros(n_nodes)

    def _compute_edge_properties(self):
        """Compute rest lengths and stiffness for each edge."""
        n_edges = len(self.edges)
        self.rest_lengths = np.zeros(n_edges)
        self.edge_stiffness = np.zeros(n_edges)

        dx_mesh = self.Lx / (self.nx_nodes - 1)

        for edge_idx, (n1, n2) in enumerate(self.edges):
            # Rest length = xy-plane distance
            dx = self.nodes[n1, 0] - self.nodes[n2, 0]
            dy = self.nodes[n1, 1] - self.nodes[n2, 1]
            L0 = np.sqrt(dx**2 + dy**2)
            self.rest_lengths[edge_idx] = L0

            # Get patch for this edge
            patch_i, patch_j = self.edge_to_patch[edge_idx]
            E = self.E_patches[patch_i, patch_j]

            # Spring stiffness: k = E * h * w / L0
            # For membrane: effective spring stiffness ~ E * h / L0
            k = E * self.thickness * dx_mesh / L0
            self.edge_stiffness[edge_idx] = k

    def _update_edge_stiffness(self):
        """Update edge stiffness from current E_patches."""
        dx_mesh = self.Lx / (self.nx_nodes - 1)

        for edge_idx, (n1, n2) in enumerate(self.edges):
            L0 = self.rest_lengths[edge_idx]
            patch_i, patch_j = self.edge_to_patch[edge_idx]
            E = self.E_patches[patch_i, patch_j]
            self.edge_stiffness[edge_idx] = E * self.thickness * dx_mesh / L0

    def set_stiffness_params(self, E_patches: np.ndarray):
        """
        Set Young's modulus for each patch.

        Args:
            E_patches: Array of shape (n_patches_y, n_patches_x) with E values (MPa)
        """
        assert E_patches.shape == (self.n_patches_y, self.n_patches_x), \
            f"Expected shape ({self.n_patches_y}, {self.n_patches_x}), got {E_patches.shape}"
        self.E_patches = np.clip(E_patches, self.E_min, self.E_max)
        self._update_edge_stiffness()

    def get_patch_center_nodes(self) -> np.ndarray:
        """
        Find node closest to center of each patch.

        Returns:
            Array of shape (n_patches_y, n_patches_x) with node indices
        """
        center_nodes = np.zeros((self.n_patches_y, self.n_patches_x), dtype=int)

        for i in range(self.n_patches_y):
            for j in range(self.n_patches_x):
                # Patch center coordinates
                cx = (j + 0.5) * self.patch_size
                cy = (i + 0.5) * self.patch_size

                # Find closest node
                dist = np.linalg.norm(self.nodes[:, :2] - np.array([cx, cy]), axis=1)
                center_nodes[i, j] = np.argmin(dist)

        return center_nodes

    def _compute_energy(self, z_free_vals: np.ndarray, truly_free: List[int],
                        constrained_nodes: Dict[int, float] = None) -> float:
        """
        Compute total energy (elastic + potential).

        Args:
            z_free_vals: z-displacement of truly free nodes
            truly_free: List of truly free node indices
            constrained_nodes: Additional displacement constraints {node_idx: z_value}

        Returns:
            Total energy
        """
        # Reconstruct full z array
        z = np.zeros(self.n_nodes)

        # Set truly free nodes
        for i, node_idx in enumerate(truly_free):
            z[node_idx] = z_free_vals[i]

        # Set constrained nodes
        if constrained_nodes:
            for node_idx, z_val in constrained_nodes.items():
                z[node_idx] = z_val

        # Fixed nodes stay at 0

        # Elastic energy from springs
        elastic_energy = 0.0
        for edge_idx, (n1, n2) in enumerate(self.edges):
            dx = self.nodes[n1, 0] - self.nodes[n2, 0]
            dy = self.nodes[n1, 1] - self.nodes[n2, 1]
            dz = z[n1] - z[n2]

            L = np.sqrt(dx**2 + dy**2 + dz**2)  # Current 3D length
            L0 = self.rest_lengths[edge_idx]     # Rest length (xy)
            k = self.edge_stiffness[edge_idx]

            # Strain energy: (1/2) * k * (L - L0)²
            elastic_energy += 0.5 * k * (L - L0)**2

        # Potential energy from pressure (work done against pressure)
        # Pressure pushing up (+z direction), potential = -p * z * dA
        dx_mesh = self.Lx / (self.nx_nodes - 1)
        dy_mesh = self.Ly / (self.ny_nodes - 1)
        node_area = dx_mesh * dy_mesh

        potential_energy = 0.0
        for node_idx in range(self.n_nodes):
            if node_idx not in self.fixed_nodes:
                potential_energy -= self.pressure_nmm2 * z[node_idx] * node_area

        return elastic_energy + potential_energy

    def _compute_gradient(self, z_free_vals: np.ndarray, truly_free: List[int],
                          constrained_nodes: Dict[int, float] = None) -> np.ndarray:
        """
        Compute gradient of energy w.r.t. truly free node displacements.

        Args:
            z_free_vals: z-displacement of truly free nodes
            truly_free: List of truly free node indices
            constrained_nodes: Additional displacement constraints

        Returns:
            Gradient array of shape (len(truly_free),)
        """
        # Reconstruct full z array
        z = np.zeros(self.n_nodes)
        for i, node_idx in enumerate(truly_free):
            z[node_idx] = z_free_vals[i]
        if constrained_nodes:
            for node_idx, z_val in constrained_nodes.items():
                z[node_idx] = z_val

        # Initialize gradient for all nodes
        grad_full = np.zeros(self.n_nodes)

        # Gradient from elastic energy
        for edge_idx, (n1, n2) in enumerate(self.edges):
            dx = self.nodes[n1, 0] - self.nodes[n2, 0]
            dy = self.nodes[n1, 1] - self.nodes[n2, 1]
            dz = z[n1] - z[n2]

            L = np.sqrt(dx**2 + dy**2 + dz**2)
            L0 = self.rest_lengths[edge_idx]
            k = self.edge_stiffness[edge_idx]

            if L > 1e-10:  # Avoid division by zero
                # dE/dz_i = k * (L - L0) * (z_i - z_j) / L
                factor = k * (L - L0) / L
                grad_full[n1] += factor * dz
                grad_full[n2] -= factor * dz

        # Gradient from potential energy (pressure)
        dx_mesh = self.Lx / (self.nx_nodes - 1)
        dy_mesh = self.Ly / (self.ny_nodes - 1)
        node_area = dx_mesh * dy_mesh

        for node_idx in range(self.n_nodes):
            if node_idx not in self.fixed_nodes:
                grad_full[node_idx] -= self.pressure_nmm2 * node_area

        # Extract gradient for truly free nodes only
        grad = np.zeros(len(truly_free))
        for i, node_idx in enumerate(truly_free):
            grad[i] = grad_full[node_idx]

        return grad

    def _solve_nonlinear(self, constrained_nodes: Dict[int, float] = None) -> np.ndarray:
        """
        Solve using nonlinear energy minimization.

        Args:
            constrained_nodes: Dict mapping node indices to prescribed z-values

        Returns:
            z-displacement array for all nodes
        """
        # Determine which nodes are truly free (not boundary, not constrained)
        fixed_set = set(self.fixed_nodes)
        if constrained_nodes:
            extra_fixed = set(constrained_nodes.keys())
            truly_free = [n for n in range(self.n_nodes)
                         if n not in fixed_set and n not in extra_fixed]
        else:
            truly_free = [n for n in range(self.n_nodes) if n not in fixed_set]

        if len(truly_free) == 0:
            z = np.zeros(self.n_nodes)
            if constrained_nodes:
                for node_idx, z_val in constrained_nodes.items():
                    z[node_idx] = z_val
            return z

        def energy_func(z_vals):
            return self._compute_energy(z_vals, truly_free, constrained_nodes)

        def gradient_func(z_vals):
            return self._compute_gradient(z_vals, truly_free, constrained_nodes)

        # Initial guess: use current solution or zeros
        z0 = np.zeros(len(truly_free))
        if self.z_current is not None:
            for i, node_idx in enumerate(truly_free):
                z0[i] = self.z_current[node_idx]

        # Optimize
        result = minimize(
            energy_func,
            z0,
            method='L-BFGS-B',
            jac=gradient_func,
            options={'gtol': 1e-8, 'maxiter': 1000}
        )

        # Reconstruct full solution
        z = np.zeros(self.n_nodes)
        for i, node_idx in enumerate(truly_free):
            z[node_idx] = result.x[i]
        if constrained_nodes:
            for node_idx, z_val in constrained_nodes.items():
                z[node_idx] = z_val

        return z

    def _solve_linear(self, constrained_nodes: Dict[int, float] = None) -> np.ndarray:
        """
        Solve using linear approximation (small deformation).

        For small deformations, the membrane equation becomes approximately linear.

        Args:
            constrained_nodes: Dict mapping node indices to prescribed z-values

        Returns:
            z-displacement array for all nodes
        """
        n = self.n_nodes
        K = lil_matrix((n, n))
        F = np.zeros(n)

        # Convert fixed_nodes to set for O(1) lookup
        fixed_set = set(self.fixed_nodes)

        # Assemble stiffness matrix from springs
        for edge_idx, (n1, n2) in enumerate(self.edges):
            k = self.edge_stiffness[edge_idx]
            L0 = self.rest_lengths[edge_idx]

            # For linear membrane: k_eff represents the effective stiffness
            # For small out-of-plane deflections, use k directly
            k_eff = k

            K[n1, n1] += k_eff
            K[n2, n2] += k_eff
            K[n1, n2] -= k_eff
            K[n2, n1] -= k_eff

        # Load vector (pressure)
        dx_mesh = self.Lx / (self.nx_nodes - 1)
        dy_mesh = self.Ly / (self.ny_nodes - 1)
        node_area = dx_mesh * dy_mesh

        for node_idx in range(n):
            if node_idx not in fixed_set:
                F[node_idx] = self.pressure_nmm2 * node_area

        # Apply boundary conditions (z = 0 at fixed nodes)
        for node_idx in self.fixed_nodes:
            K[node_idx, :] = 0
            K[node_idx, node_idx] = 1
            F[node_idx] = 0

        # Apply additional constraints
        if constrained_nodes:
            for node_idx, z_val in constrained_nodes.items():
                K[node_idx, :] = 0
                K[node_idx, node_idx] = 1
                F[node_idx] = z_val

        # Convert to CSR and solve
        K_csr = K.tocsr()
        z = spsolve(K_csr, F)

        return z

    def solve_free(self) -> np.ndarray:
        """
        Solve for equilibrium with pressure BC only.

        Returns:
            z_displacement: z-displacement at all nodes
        """
        if self.use_nonlinear:
            z = self._solve_nonlinear(constrained_nodes=None)
        else:
            z = self._solve_linear(constrained_nodes=None)

        self.z_current = z.copy()
        return z

    def solve_clamped(self, target_patches: Dict[Tuple[int, int], float]) -> np.ndarray:
        """
        Solve with pressure BC + z-displacement constraints at target patches.

        Args:
            target_patches: Dict mapping (i, j) patch indices to target z-displacement (mm)

        Returns:
            z_displacement: z-displacement at all nodes
        """
        # Convert patch targets to node constraints
        center_nodes = self.get_patch_center_nodes()
        constrained_nodes = {}

        for (i, j), z_target in target_patches.items():
            node_idx = center_nodes[i, j]
            constrained_nodes[node_idx] = z_target

        if self.use_nonlinear:
            z = self._solve_nonlinear(constrained_nodes=constrained_nodes)
        else:
            z = self._solve_linear(constrained_nodes=constrained_nodes)

        return z

    def eval_signal(self, z_free: np.ndarray, z_clamped: np.ndarray) -> np.ndarray:
        """
        Compute signal contrast: (z_free)² - (z_clamped)² at patch centers.

        Args:
            z_free: Free-state z-displacements (all nodes)
            z_clamped: Clamped-state z-displacements (all nodes)

        Returns:
            signal: Array of shape (n_patches_y, n_patches_x)
        """
        signal = np.zeros((self.n_patches_y, self.n_patches_x))
        center_nodes = self.get_patch_center_nodes()

        for i in range(self.n_patches_y):
            for j in range(self.n_patches_x):
                node_idx = center_nodes[i, j]
                signal[i, j] = z_free[node_idx]**2 - z_clamped[node_idx]**2

        return signal

    def eval_height(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """
        Evaluate z-displacement at specified (x, y) points.

        Args:
            points: List of (x, y) coordinates

        Returns:
            Array of z-displacement values
        """
        heights = []
        for x, y in points:
            # Find nearest node
            dist = np.sqrt(
                (self.nodes[:, 0] - x)**2 +
                (self.nodes[:, 1] - y)**2
            )
            nearest = np.argmin(dist)
            heights.append(self.z_current[nearest] if self.z_current is not None else 0.0)
        return np.array(heights)

    def get_state_summary(self) -> Dict:
        """Return current state for logging."""
        center_nodes = self.get_patch_center_nodes()
        z_centers = np.zeros((self.n_patches_y, self.n_patches_x))

        if self.z_current is not None:
            for i in range(self.n_patches_y):
                for j in range(self.n_patches_x):
                    z_centers[i, j] = self.z_current[center_nodes[i, j]]

        return {
            'E_patches': self.E_patches.copy(),
            'z_centers': z_centers,
            'E_mean': np.mean(self.E_patches),
            'E_std': np.std(self.E_patches),
        }

    def get_full_displacement_field(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the full displacement field for visualization.

        Returns:
            x_coords, y_coords, z_displacements (all 1D arrays of length n_nodes)
        """
        return (
            self.nodes[:, 0],
            self.nodes[:, 1],
            self.z_current if self.z_current is not None else np.zeros(self.n_nodes)
        )

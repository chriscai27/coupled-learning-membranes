"""
Membrane PDE Backend (Tier 1 surrogate)

Goal
----
Provide a continuum-like, FEA-ish membrane surrogate that produces smooth "cascade" coupling
between neighboring patches, while staying lightweight and compatible with local update rules.

Model (linear surrogate)
------------------------
We solve a variable-coefficient Poisson / membrane equation on a 2D grid:

    -div( K(x,y) * grad w(x,y) ) = p

where:
  - w is out-of-plane displacement (mm)
  - p is uniform pressure load (N/mm^2)
  - K(x,y) is an *effective* tension-like stiffness (N/mm), derived from patch-wise E and thickness:
        K_patch = E_patch * thickness
    (This is a surrogate: it preserves monotonic trends and strong neighbor coupling.)

Boundary conditions:
  - default: clamped boundary (Dirichlet): w = 0 on the outer boundary
  - clamped targets: additional Dirichlet constraints at patch-center nodes (or nearest grid node)

Notes
-----
This backend is intentionally linear (Tier 1). It does *not* include geometric stiffening.
For a minimal nonlinear upgrade, see the README section "Tier 2" (not implemented here).

API
---
Matches the subset used by adaptive_membrane_experiment.py:
  - solve_free() -> z (flattened node array)
  - solve_clamped(target_patches: dict[(i,j)->w_mm]) -> z
  - eval_signal(z_free, z_clamped) -> (n_patches_y, n_patches_x)
  - get_patch_center_nodes() -> (n_patches_y, n_patches_x) indices into flattened z
  - get_full_displacement_field() -> (x, y, z_current) flattened
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


def psi_to_n_per_mm2(psi: float) -> float:
    # 1 psi = 6894.757 Pa = 0.006894757 N/mm^2
    return psi * 6894.757e-6


def harmonic_mean(a: float, b: float, eps: float = 1e-12) -> float:
    return 2.0 * a * b / (a + b + eps)


@dataclass
class MembraneGrid:
    nx: int  # nodes in x
    ny: int  # nodes in y
    Lx: float  # mm
    Ly: float  # mm

    @property
    def dx(self) -> float:
        return self.Lx / (self.nx - 1)

    @property
    def dy(self) -> float:
        return self.Ly / (self.ny - 1)

    def node_index(self, iy: int, ix: int) -> int:
        return iy * self.nx + ix

    def node_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.linspace(0.0, self.Lx, self.nx)
        y = np.linspace(0.0, self.Ly, self.ny)
        X, Y = np.meshgrid(x, y)
        return X.reshape(-1), Y.reshape(-1)


class MembranePDEBackend:
    """
    Tier-1 continuum surrogate backend.

    Parameters
    ----------
    n_patches_x, n_patches_y : int
        Patch grid size.
    patch_size_mm : float
        Patch side length (mm).
    thickness_mm : float
        Membrane thickness (mm).
    pressure_psi : float
        Uniform pressure (psi).
    mesh_refinement : int
        Number of sub-cells per patch edge (>=1). Nodes per patch edge = mesh_refinement.
        Total nodes = n_patches*mesh_refinement + 1 in each axis.
    E_min_mpa, E_max_mpa : float
        Bounds for E field (MPa). Stored in E_patches (same naming as previous backend).
    signal_mode : str
        Signal definition: "center" uses patch-center nodes; "patch_mean" averages all nodes in patch.
    """

    def __init__(
        self,
        n_patches_x: int,
        n_patches_y: int,
        patch_size_mm: float = 2.0,
        thickness_mm: float = 1.0,
        pressure_psi: float = 5.0,
        mesh_refinement: int = 4,
        E_min_mpa: float = 0.6,
        E_max_mpa: float = 381.7,
        signal_mode: str = "center",
        seed: int = 0,
    ):
        self.n_patches_x = int(n_patches_x)
        self.n_patches_y = int(n_patches_y)
        self.patch_size = float(patch_size_mm)
        self.thickness = float(thickness_mm)
        self.pressure_psi = float(pressure_psi)
        self.pressure = psi_to_n_per_mm2(self.pressure_psi)  # N/mm^2

        self.E_min = float(E_min_mpa)
        self.E_max = float(E_max_mpa)
        self.signal_mode = signal_mode

        rng = np.random.default_rng(seed)
        # initialize in the middle of the range (can be changed by experiment)
        self.E_patches = np.full((self.n_patches_y, self.n_patches_x), 0.5 * (self.E_min + self.E_max), dtype=float)

        # grid
        self.mesh_refinement = int(mesh_refinement)
        nx_nodes = self.n_patches_x * self.mesh_refinement + 1
        ny_nodes = self.n_patches_y * self.mesh_refinement + 1
        self.Lx = self.n_patches_x * self.patch_size
        self.Ly = self.n_patches_y * self.patch_size
        self.grid = MembraneGrid(nx=nx_nodes, ny=ny_nodes, Lx=self.Lx, Ly=self.Ly)

        self.n_nodes = self.grid.nx * self.grid.ny
        self.nodes_x, self.nodes_y = self.grid.node_coords()

        self.z_current: Optional[np.ndarray] = None  # flattened

        # cache patch center nodes
        self._center_nodes = self._compute_patch_center_nodes()
        self._patch_nodes = self._compute_patch_node_indices()
        self._validate_signal_mode()

    # -------------------------
    # Patch <-> node utilities
    # -------------------------
    def _compute_patch_center_nodes(self) -> np.ndarray:
        centers = np.zeros((self.n_patches_y, self.n_patches_x), dtype=int)
        for pi in range(self.n_patches_y):
            for pj in range(self.n_patches_x):
                cx = (pj + 0.5) * self.patch_size
                cy = (pi + 0.5) * self.patch_size
                # nearest node
                dist2 = (self.nodes_x - cx) ** 2 + (self.nodes_y - cy) ** 2
                centers[pi, pj] = int(np.argmin(dist2))
        return centers

    def _compute_patch_node_indices(self) -> List[List[np.ndarray]]:
        patch_nodes: List[List[np.ndarray]] = [
            [np.array([], dtype=int) for _ in range(self.n_patches_x)]
            for _ in range(self.n_patches_y)
        ]
        eps = 1e-9
        for pi in range(self.n_patches_y):
            y0 = pi * self.patch_size
            y1 = (pi + 1) * self.patch_size
            if pi < self.n_patches_y - 1:
                mask_y = (self.nodes_y >= y0 - eps) & (self.nodes_y < y1 - eps)
            else:
                mask_y = (self.nodes_y >= y0 - eps) & (self.nodes_y <= y1 + eps)
            for pj in range(self.n_patches_x):
                x0 = pj * self.patch_size
                x1 = (pj + 1) * self.patch_size
                if pj < self.n_patches_x - 1:
                    mask_x = (self.nodes_x >= x0 - eps) & (self.nodes_x < x1 - eps)
                else:
                    mask_x = (self.nodes_x >= x0 - eps) & (self.nodes_x <= x1 + eps)
                mask = mask_x & mask_y
                idx = np.nonzero(mask)[0]
                if idx.size == 0:
                    idx = np.array([int(self._center_nodes[pi, pj])], dtype=int)
                patch_nodes[pi][pj] = idx
        return patch_nodes

    def _validate_signal_mode(self) -> None:
        if self.signal_mode not in ("center", "patch_mean"):
            raise ValueError(f"Unknown signal_mode: {self.signal_mode}. Expected 'center' or 'patch_mean'.")

    def get_patch_center_nodes(self) -> np.ndarray:
        return self._center_nodes.copy()

    def _patch_of_xy(self, x: float, y: float) -> Tuple[int, int]:
        pj = min(int(x / self.patch_size), self.n_patches_x - 1)
        pi = min(int(y / self.patch_size), self.n_patches_y - 1)
        return pi, pj

    def _K_patch(self, pi: int, pj: int) -> float:
        # E in MPa = N/mm^2. Multiply by thickness (mm) gives N/mm.
        return float(self.E_patches[pi, pj]) * self.thickness

    def _K_face(self, n1: int, n2: int) -> float:
        # Face coupling: harmonic mean of the two endpoint patches' K
        pi1, pj1 = self._patch_of_xy(self.nodes_x[n1], self.nodes_y[n1])
        pi2, pj2 = self._patch_of_xy(self.nodes_x[n2], self.nodes_y[n2])
        K1 = self._K_patch(pi1, pj1)
        K2 = self._K_patch(pi2, pj2)
        return harmonic_mean(K1, K2)

    # -------------------------
    # Assembly / Solve
    # -------------------------
    def _assemble_system(self, constrained: Dict[int, float]) -> Tuple[csr_matrix, np.ndarray]:
        """
        Assemble sparse linear system A w = b with Dirichlet constraints.

        constrained: dict[node_idx -> prescribed w_mm]
        """
        nx, ny = self.grid.nx, self.grid.ny
        dx, dy = self.grid.dx, self.grid.dy
        n = self.n_nodes

        A = lil_matrix((n, n), dtype=float)
        b = np.zeros(n, dtype=float)

        # helper: whether boundary
        def is_boundary(iy, ix) -> bool:
            return iy == 0 or ix == 0 or iy == ny - 1 or ix == nx - 1

        # pressure load: distribute p * area to nodes (simple lumping)
        # cell area ~ dx*dy, each interior node gets ~p*dx*dy
        # units: p (N/mm^2) * area (mm^2) = N; in this surrogate, RHS uses "force-like" units
        # consistent scaling is absorbed by K; we're interested in relative shapes and sweep trends.
        f_node = self.pressure * dx * dy

        for iy in range(ny):
            for ix in range(nx):
                idx = self.grid.node_index(iy, ix)

                # Dirichlet constraints: boundary + user-specified
                if is_boundary(iy, ix) or idx in constrained:
                    A[idx, idx] = 1.0
                    b[idx] = 0.0 if is_boundary(iy, ix) else float(constrained[idx])
                    continue

                # interior: variable-coefficient Laplacian via fluxes to 4-neighbors
                diag = 0.0

                # neighbor up (iy-1, ix)
                n_up = self.grid.node_index(iy - 1, ix)
                K_up = self._K_face(idx, n_up)
                w = K_up / (dy * dy)
                A[idx, n_up] = -w
                diag += w

                # neighbor down (iy+1, ix)
                n_dn = self.grid.node_index(iy + 1, ix)
                K_dn = self._K_face(idx, n_dn)
                w = K_dn / (dy * dy)
                A[idx, n_dn] = -w
                diag += w

                # neighbor left (iy, ix-1)
                n_lt = self.grid.node_index(iy, ix - 1)
                K_lt = self._K_face(idx, n_lt)
                w = K_lt / (dx * dx)
                A[idx, n_lt] = -w
                diag += w

                # neighbor right (iy, ix+1)
                n_rt = self.grid.node_index(iy, ix + 1)
                K_rt = self._K_face(idx, n_rt)
                w = K_rt / (dx * dx)
                A[idx, n_rt] = -w
                diag += w

                A[idx, idx] = diag
                b[idx] = f_node

        return A.tocsr(), b

    def _solve(self, constrained_nodes: Dict[int, float]) -> np.ndarray:
        A, b = self._assemble_system(constrained_nodes)
        w = spsolve(A, b)
        return np.asarray(w, dtype=float)

    # -------------------------
    # Public API
    # -------------------------
    def solve_free(self) -> np.ndarray:
        z = self._solve(constrained_nodes={})
        self.z_current = z
        return z

    def solve_clamped(self, target_patches: Dict[Tuple[int, int], float]) -> np.ndarray:
        constrained: Dict[int, float] = {}
        centers = self.get_patch_center_nodes()
        for (pi, pj), w_mm in target_patches.items():
            pi = int(pi); pj = int(pj)
            node_idx = int(centers[pi, pj])
            constrained[node_idx] = float(w_mm)
        z = self._solve(constrained_nodes=constrained)
        self.z_current = z
        return z

    def eval_signal(self, z_free: np.ndarray, z_clamped: np.ndarray) -> np.ndarray:
        signal = np.zeros((self.n_patches_y, self.n_patches_x), dtype=float)
        if self.signal_mode == "center":
            centers = self.get_patch_center_nodes()
            for pi in range(self.n_patches_y):
                for pj in range(self.n_patches_x):
                    node = int(centers[pi, pj])
                    signal[pi, pj] = z_free[node] - z_clamped[node]
            return signal

        if self.signal_mode == "patch_mean":
            delta = z_free - z_clamped
            for pi in range(self.n_patches_y):
                for pj in range(self.n_patches_x):
                    nodes = self._patch_nodes[pi][pj]
                    signal[pi, pj] = float(np.mean(delta[nodes])) if nodes.size > 0 else 0.0
            return signal

        raise ValueError(f"Unknown signal_mode: {self.signal_mode}")

    def get_full_displacement_field(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.nodes_x.copy(),
            self.nodes_y.copy(),
            self.z_current.copy() if self.z_current is not None else np.zeros(self.n_nodes, dtype=float),
        )

    def get_state_summary(self) -> Dict[str, float]:
        return {
            "z_max": float(np.max(self.z_current)) if self.z_current is not None else 0.0,
            "z_min": float(np.min(self.z_current)) if self.z_current is not None else 0.0,
            "E_min": float(np.min(self.E_patches)),
            "E_max": float(np.max(self.E_patches)),
            "E_mean": float(np.mean(self.E_patches)),
            "E_std": float(np.std(self.E_patches)),
        }

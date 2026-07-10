"""
espol-cuboids — electrostatic polarizability of dielectric cuboids.

Provides DDA (discrete dipole approximation) and MoM (method of moments)
solvers for 2D and 3D rectangular geometries.
"""

# ── Data structures ──────────────────────────────────────────────
from .vector_analysis import V3, D3, Ori, Mesh

# ── Mesh generators ──────────────────────────────────────────────
from .gen_mesh_2d_rect import gen_mesh_2d_rect
from .gen_mesh_3d_rect import gen_mesh_3d_rect

# ── Solvers ──────────────────────────────────────────────────────
from .dda_core import (make_lattice, compute_polarizability_fft,
                        compute_polarizability, converged_polarizability,
                        pec_polarizability)
from .mom2d import comp_polarizability_2d
from .mom3d import comp_polarizability_3d

# ── Batch runners  ───────────────────────────────────────────────
from .gen_dda import (solve_single_d, converged_dda, d_levels_for_shape,
                       default_grid as dda_default_grid)
from .gen_mom_2d import (converged_2d, adaptive_nd_2d, fit_powerlaw as fit_powerlaw_2d,
                          default_grid as mom2d_default_grid)
from .gen_mom_3d import (converged_3d, adaptive_nd, fit_powerlaw as fit_powerlaw_3d,
                          default_grid as mom3d_default_grid)

__all__ = [
    # data structures
    'V3', 'D3', 'Ori', 'Mesh',
    # mesh generators
    'gen_mesh_2d_rect', 'gen_mesh_3d_rect',
    # DDA solver
    'make_lattice', 'compute_polarizability_fft', 'compute_polarizability',
    'converged_polarizability', 'pec_polarizability',
    # MoM solvers
    'comp_polarizability_2d', 'comp_polarizability_3d',
    # batch runners
    'solve_single_d', 'converged_dda', 'd_levels_for_shape',
    'converged_2d', 'adaptive_nd_2d',
    'converged_3d', 'adaptive_nd',
]

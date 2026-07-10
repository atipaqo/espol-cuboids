# espol-cuboids: Polarizability of Cuboids and Rectangular Cylinders

Electrostatic polarizability computation for dielectric cuboids and rectangular cylinders using three
independent numerical methods: **Method of Moments (3D)**, **Method of Moments (2D)**,
and **Discrete Dipole Approximation (DDA)**.

## Overview

For a homogeneous dielectric cuboid placed in a uniform static electric field,
the induced dipole moment **p** is related to the applied field **E₀** by the
polarizability tensor **α**:

**p** = **α** · **E₀**

This suite computes **α** (and the equivalent depolarization factors
*N<sub>x</sub>, N<sub>y</sub>, N<sub>z</sub>*) for rectangular shapes
spanning a wide range of aspect ratios and permittivities, including the
perfect-electric-conductor (PEC) limit.

### Methods

| Method | Type | Acceleration | Dimensionality |
|--------|------|-------------|----------------|
| **MoM 3D** | Surface integral equation (PMCHW) | Octant symmetry, ACA + GMRES | 3D cuboid |
| **MoM 2D** | Boundary integral equation | — | 2D infinite prism |
| **DDA**    | Volume integral equation | FFT-accelerated GMRES | 3D cuboid |

All methods use convergence extrapolation (power-law fit in discretization
density) to obtain continuum-limit results.

## Installation

```bash
git clone https://github.com/atipaqo/espol-cuboids.git
cd espol-cuboids
pip install -e .
```

Requirements: Python ≥ 3.9, NumPy ≥ 1.20, SciPy ≥ 1.7.

## Quick Start

### As a library

```python
from espol_cuboids import (
    # DDA
    make_lattice, compute_polarizability_fft,
    # MoM 3D
    gen_mesh_3d_rect, comp_polarizability_3d, Mesh,
    # MoM 2D
    gen_mesh_2d_rect, comp_polarizability_2d,
)

# DDA: 1×1×1 cube, εr = 10
alpha, info = compute_polarizability_fft(10.0, 1.0, 1.0, 1.0, d=0.2)
print(alpha[0, 0])  # α_xx ≈ 2.48

# MoM 3D: 1×1×1 cube, direct LU
r, n, ds, ne, V = gen_mesh_3d_rect(1.0, 1.0, 1.0, nd=30, octant=True)
P = comp_polarizability_3d(Mesh(r, n, ds, V=V, er=10.0, octant=True))
print(P.xx)  # ≈ 2.49

# MoM 2D: 1×1 square cross-section
r, n, ds, ne, A = gen_mesh_2d_rect(1.0, 1.0, nd=80)
P2 = comp_polarizability_2d(Mesh(r, n, ds, V=A, er=10.0))
print(P2.xx)  # ≈ 1.73
```

### Run the example script

```bash
python examples/basic_usage.py
```

### Run batch sweeps

```bash
# Preview what will be computed (no solves)
python examples/run_mom3d_sweep.py --dry-run
python examples/run_mom2d_sweep.py --dry-run
python examples/run_dda_sweep.py --mode converge --dry-run

# Run sweeps
python examples/run_mom3d_sweep.py                     # 3D, direct LU, er=10,20,100,PEC
python examples/run_mom3d_sweep.py --solver aca         # 3D with ACA compression
python examples/run_mom2d_sweep.py                      # 2D, sin-clustered mesh
python examples/run_dda_sweep.py --mode fixed           # DDA, single spacing
python examples/run_dda_sweep.py --mode converge --er 10  # DDA, d→0 extrapolation
```

All sweep scripts accept `--er` for a comma-separated list of permittivities
(e.g., `--er 2,5,10`).

## Project Structure

```
├── pyproject.toml
├── src/
│   └── espol_cuboids/
│       ├── __init__.py              # public API
│       ├── vector_analysis.py       # V3, D3, Ori, Mesh data structures
│       ├── dda_core.py              # DDA solver (FFT-GMRES, convergence)
│       ├── mom2d.py                 # MoM 2D solver
│       ├── mom3d.py                 # MoM 3D solver (LU, GMRES, ACA)
│       ├── gen_dda.py               # DDA batch runner (cache, convergence fit)
│       ├── gen_mom_2d.py            # MoM 2D batch runner
│       ├── gen_mom_3d.py            # MoM 3D batch runner
│       ├── gen_mesh_2d_rect.py      # 2D perimeter mesh generator
│       └── gen_mesh_3d_rect.py      # 3D surface mesh generator
├── tests/
│   ├── test_dda_core.py             # DDA unit & regression tests
│   ├── test_mom2d.py                # 2D MoM mesh, solver
│   ├── test_mom3d.py                # 3D MoM mesh, solver, ACA, convergence
│   └── test_vector.py               # V3, D3, Mesh unit tests
├── examples/
│   ├── basic_usage.py               # single-point demonstration of each solver
│   ├── run_dda_sweep.py             # DDA grid sweep (fixed or converge mode)
│   ├── run_mom2d_sweep.py           # MoM 2D grid sweep
│   └── run_mom3d_sweep.py           # MoM 3D grid sweep
└── data/                            # solver caches & CSV output (auto-created)
```

## Running Tests

```bash
python -m pytest tests/ -v                  # all 54 tests
python -m pytest tests/ -v -m "not slow"    # skip full MoM/DDA solves
```

## Documentation

- **[User Guide](docs/user_guide.pdf)** — how to run sweeps, interpret results.

## License

MIT

## References

If you use this code in academic work, the primary references are:

- D. Gomez-Garcia, J. Li, and C. J. Leuschen, "espol-cuboids: Electrostatic
  polarizability solvers for dielectric cuboids and rectangular cylinders,"
  2026, online. Available: <https://github.com/atipaqo/espol-cuboids>.
- R. F. Harrington, *Field Computation by Moment Methods*, Macmillan, 1968.
- B. T. Draine and P. J. Flatau, "Discrete-dipole approximation for
  scattering calculations," *J. Opt. Soc. Am. A*, 11(4), 1491–1499, 1994.


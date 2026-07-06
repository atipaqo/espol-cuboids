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

All methods use convergence extrapolation to obtain continuum-limit results.
See [`docs/`](docs/) for detailed implementation descriptions.

## Quick Start

```bash
# Preview what will be computed (no solves)
python run_mom3d_sweep.py --dry-run
python run_mom2d_sweep.py --dry-run
python run_dda_sweep.py --mode fixed --dry-run

# Run sweeps
python run_mom3d_sweep.py                    # 3D MoM, direct LU, er=10,20,100,PEC
python run_mom3d_sweep.py --solver aca        # 3D MoM with ACA compression
python run_mom2d_sweep.py                     # 2D MoM
python run_dda_sweep.py --mode fixed          # DDA, single lattice spacing
python run_dda_sweep.py --mode converge --er 10  # DDA, d→0 extrapolation
```

All three scripts accept `--er` to specify a comma-separated list of
relative permittivities (e.g., `--er 2,5,10`).

## Requirements

- Python ≥ 3.10
- NumPy, SciPy

## Installation

```bash
git clone https://github.com/atipaqo/espol-cuboids.git
cd espol-cuboids
pip install numpy scipy
```

## Project Structure

```
├── run_mom3d_sweep.py        # 3D MoM sweep entry point
├── run_mom2d_sweep.py        # 2D MoM sweep entry point
├── run_dda_sweep.py          # DDA sweep entry point
├── tools/
│   ├── gen_mom_3d.py         # 3D MoM: convergence, caching, power-law fit
│   ├── gen_mesh_3d_rect.py   # 3D surface mesh generator
│   ├── sm_comp_polarizability_3d.py  # 3D MoM core solver
│   ├── mom3d_kernel.py       # Matrix-free GMRES operator
│   ├── mom3d_aca.py          # ACA-compressed H-matrix solver
│   ├── gen_mesh_2d_rect.py   # 2D perimeter mesh generator
│   ├── sm_comp_polarizability_2d.py  # 2D MoM core solver
│   ├── dda_core.py           # DDA: lattice, FFT-GMRES, convergence
│   ├── vector_analysis.py    # V3, D3, Mesh data structures
│   └── _path_setup.py        # Python path bootstrapping
├── tests/
│   ├── conftest.py
│   ├── test_dda_core.py      # DDA unit & regression tests
│   ├── test_mom2d.py         # 2D MoM mesh, solver, regression
│   ├── test_mom3d.py         # 3D MoM mesh, solver, ACA, convergence
│   └── test_vector.py        # V3, D3, Mesh unit tests
├── docs/
│   ├── user_guide.pdf        # Full usage guide
│   ├── user_guide.tex
│   ├── implementation_mom3d.md
│   ├── implementation_mom2d.md
│   └── implementation_dda.md
├── data/                     # Solver caches & CSV output (auto-created)
```

## Running Tests

```bash
python -m pytest tests/ -q           # all 54 tests
python -m pytest tests/ -q -m "not slow"  # skip full MoM/DDA solves
```

## Documentation

- **[User Guide](docs/user_guide.pdf)** — how to run sweeps, interpret results,
  and manage caches.
- **[Implementation Notes](docs/)** — mathematical formulations, solver
  backends, acceleration techniques, and references for each method.

## License

MIT

## References

If you use this code in academic work, the primary references are:

- R. F. Harrington, *Field Computation by Moment Methods*, Macmillan, 1968.
- B. T. Draine and P. J. Flatau, "Discrete-dipole approximation for
  scattering calculations," *J. Opt. Soc. Am. A*, 11(4), 1491–1499, 1994.
- M. Bebendorf, "Approximation of boundary element matrices,"
  *Numer. Math.*, 86(4), 565–589, 2000.

See the implementation documents in [`docs/`](docs/) for the full bibliography.

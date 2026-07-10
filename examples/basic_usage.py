"""
basic_usage.py — single-point demonstration of each solver.

Computes the polarizability of a 1×1×1 cube (εr=10) using:
  - DDA (FFT-accelerated)
  - MoM 3D (direct LU, octant mode)
  - MoM 2D (square cross-section)

Usage:
    python examples/basic_usage.py
"""
import numpy as np
from espol_cuboids import (
    # DDA
    make_lattice, compute_polarizability_fft,
    # MoM 3D
    gen_mesh_3d_rect, comp_polarizability_3d, Mesh,
    # MoM 2D
    gen_mesh_2d_rect, comp_polarizability_2d,
)

ER = 10.0
print(f"=== Basic usage (εr = {ER}) ===\n")

# ── DDA ─────────────────────────────────────────────────────────
print("--- DDA: 1×1×1 cube ---")
r_dda, dx, dy, dz = make_lattice(1.0, 1.0, 1.0, d=0.2)
alpha, info = compute_polarizability_fft(ER, 1.0, 1.0, 1.0, d=0.2, verbose=False)
V = 1.0
for i, lbl in enumerate(['xx', 'yy', 'zz']):
    print(f"  α_{lbl} = {alpha[i,i]:.6f}  N_={lbl} = {V/alpha[i,i]:.6f}")
print(f"  dipoles: {info['N']},  d_actual: {info['d']:.4f},  iters: {info.get('iters','n/a')}")

# ── MoM 3D ──────────────────────────────────────────────────────
print("\n--- MoM 3D (octant LU): 1×1×1 cube ---")
r, n, ds, ne, Vo = gen_mesh_3d_rect(1.0, 1.0, 1.0, nd=30, beta=0.0, octant=True)
mesh = Mesh(r, n, ds, V=Vo, er=ER, octant=True)
P3 = comp_polarizability_3d(mesh)  # default: direct LU
for lbl in ['xx', 'yy', 'zz']:
    val = getattr(P3, lbl).real
    print(f"  P_{lbl} = {val:.6f}  N_{lbl} = {Vo/val:.6f}")
print(f"  elements: {ne}")

# ── MoM 2D ──────────────────────────────────────────────────────
print("\n--- MoM 2D: 1×1 square (per unit length) ---")
r, n, ds, ne, Ao = gen_mesh_2d_rect(1.0, 1.0, nd=80, beta=1.0)
mesh = Mesh(r, n, ds, V=Ao, er=ER)
P2 = comp_polarizability_2d(mesh)
for lbl in ['xx', 'yy']:
    val = getattr(P2, lbl).real
    print(f"  P_{lbl} = {val:.6f}  N_{lbl} = {Ao/val:.6f}")
print(f"  elements: {ne}")

print("\nDone.")

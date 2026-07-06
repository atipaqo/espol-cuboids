"""
run_mom3d_sweep.py — Run the full 3D MoM sweep (uniform mesh, beta=0.0).

Usage:
    python run_mom3d_sweep.py                              # default: er=10,20,100,PEC
    python run_mom3d_sweep.py --er 0.7,1.5,3,6             # custom er list
    python run_mom3d_sweep.py --solver aca                 # ACA-compressed (faster for large N)
    python run_mom3d_sweep.py --solver direct              # direct LU (default)
    python run_mom3d_sweep.py --dry-run                     # show what would be computed
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))

from gen_mom_3d import converged_3d, adaptive_nd

# ── Grid ────────────────────────────────────────────────────────────
UV = [(1.0, 1.0), (1.0, 0.7), (1.0, 0.5), (1.0, 0.3), (1.0, 0.2),
      (0.7, 0.7), (0.7, 0.5), (0.7, 0.3), (0.7, 0.2),
      (0.5, 0.5), (0.5, 0.3), (0.5, 0.2),
      (0.3, 0.3), (0.3, 0.2), (0.2, 0.2)]

def _parse_er():
    """Parse --er flag or return default list."""
    for i, arg in enumerate(sys.argv):
        if arg == '--er' and i+1 < len(sys.argv):
            return [float(x) for x in sys.argv[i+1].split(',')]
    return [10, 20, 100, 1e10]

def _parse_solver():
    """Parse --solver flag (returns True for aca, False for direct)."""
    for i, arg in enumerate(sys.argv):
        if arg == '--solver' and i+1 < len(sys.argv):
            return sys.argv[i+1].lower() == 'aca'
    return False  # direct by default

# ── Main ────────────────────────────────────────────────────────────
def main():
    ER = _parse_er()
    aca = _parse_solver()
    dry_run = '--dry-run' in sys.argv

    solver_name = 'ACA' if aca else 'direct LU'
    if dry_run:
        print(f"=== MoM 3D Sweep (dry-run) ===\n  Shapes: {len(UV)}, er: {ER}")
        print(f"  Solver: {solver_name}")
        for u, v in UV:
            sx, sy, sz = 1.0, u, v
            nd, net = adaptive_nd(sx, sy, sz)
            print(f"  {sx:.1f}x{sy:.2f}x{sz:.2f}  nd={[int(x) for x in nd]}")
        return

    print(f"=== MoM 3D Sweep ===  Shapes: {len(UV)}, er: {ER}, solver: {solver_name}")
    t_start = time.time()
    for u, v in UV:
        sx, sy, sz = 1.0, u, v
        nd, net = adaptive_nd(sx, sy, sz)
        print(f"\n--- {sx:.1f}x{sy:.2f}x{sz:.2f}  nd={[int(x) for x in nd]} ---")
        for er in ER:
            t1 = time.time()
            Pxx, Pyy, Pzz = converged_3d(er, sx, sy, sz, nd, net, aca=aca)
            dt = time.time() - t1
            er_s = 'PEC' if er > 1e9 else f'er={er:.0f}'
            Nx, Ny, Nz = 1.0 / Pxx, 1.0 / Pyy, 1.0 / Pzz
            print(f"  {er_s:>6s}  Nx={Nx:.6f} Ny={Ny:.6f} Nz={Nz:.6f}  "
                  f"sum={Nx+Ny+Nz:.6f}  [{dt:.1f}s]")
    elapsed = time.time() - t_start
    print(f"\nTotal: {elapsed:.0f}s  ({elapsed/3600:.1f}h)")


if __name__ == '__main__':
    main()

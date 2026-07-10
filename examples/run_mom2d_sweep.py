"""
run_mom2d_sweep.py — Run the full 2D MoM sweep (default: sin-clustered mesh).

Usage:
    python run_mom2d_sweep.py                              # default: er=0.7,1.5,3,6,1e3,1e5
    python run_mom2d_sweep.py --er 10,20,100,PEC           # custom er list
    python run_mom2d_sweep.py --dry-run                    # show what would be computed
"""

import sys, os, time

from espol_cuboids.gen_mom_2d import converged_2d, adaptive_nd_2d

# ── Grid ────────────────────────────────────────────────────────────
SRY_LIST = [0.1, 0.15, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.667, 0.8, 0.9,
            1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.33, 5.0, 7.0, 10.0]
# SRY_LIST = [1.0]


def _parse_er():
    """Parse --er flag or return default list."""
    for i, arg in enumerate(sys.argv):
        if arg == '--er' and i+1 < len(sys.argv):
            return [float(x) for x in sys.argv[i+1].split(',')]
    return [0.7, 1.5, 3.0, 6.0, 1e3, 1e5]

# ── Main ────────────────────────────────────────────────────────────
def main():
    ER = _parse_er()
    dry_run = '--dry-run' in sys.argv

    if dry_run:
        print(f"=== MoM 2D Sweep (dry-run) ===\n  Shapes: {len(SRY_LIST)}, er: {ER}")
        for sry in SRY_LIST:
            nd, net = adaptive_nd_2d(sry)
            print(f"  1.0x{sry:.2f}  nd={[int(x) for x in nd]}")
        return

    print(f"=== MoM 2D Sweep ===  Shapes: {len(SRY_LIST)}, er: {ER}")
    t_start = time.time()
    for sry in SRY_LIST:
        nd, net = adaptive_nd_2d(sry)
        print(f"\n--- 1.0x{sry:.2f} (∞)  nd={[int(x) for x in nd]} ---")
        for er in ER:
            t1 = time.time()
            Pxx_inf, Pyy_inf = converged_2d(er, 1.0, sry, nd, net)
            dt = time.time() - t1
            Nx = 1.0 / Pxx_inf if abs(Pxx_inf) > 1e-15 else float('inf')
            Ny = 1.0 / Pyy_inf if abs(Pyy_inf) > 1e-15 else float('inf')
            if er > 1e9:
                er_s = 'PEC'
            elif er >= 10:
                er_s = f'er={er:.0f}'
            else:
                er_s = f'er={er:.1f}'
            print(f"  {er_s:>6s}  Nx={Nx:.6f} Ny={Ny:.6f}  "
                  f"sum={Nx+Ny:.6f}  [{dt:.1f}s]")
    elapsed = time.time() - t_start
    print(f"\nTotal: {elapsed:.0f}s  ({elapsed/60:.1f}m)")


if __name__ == '__main__':
    main()

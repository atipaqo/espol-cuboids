"""
run_dda_sweep.py  ─  DDA shape-sweep for dielectric cuboids.

Two modes:
  --mode fixed      Single d = smin/10.        Output: data/csvdata/dda3d.csv
  --mode converge   Multiple d → d→0 fit.      Output: data/csvdata/dda_convergence_{raw,cnv}.csv

Usage:
    python run_dda_sweep.py --mode fixed
    python run_dda_sweep.py --mode converge --er 10
    python run_dda_sweep.py --mode fixed --dry-run
"""
import numpy as np
import os, sys, time, csv

_HERE = os.path.dirname(os.path.abspath(__file__))

from espol_cuboids.gen_dda import (solve_single_d, converged_dda, d_levels_for_shape,
                     default_grid, _CACHE_DIR, make_lattice)

# ═══ Shape grid ═══
UV, DEFAULT_ER = default_grid()
# UV = [(1.0, 1.0)]

def run_fixed_d(er_values, dry_run=False):
    rows = []
    for u, v in UV:
        sx, sy, sz = 1.0, u, v
        smin = min(sy, sz)
        d = smin / 10.0
        V = sx * sy * sz
        for er in er_values:
            if dry_run:
                r, _, _, _ = make_lattice(sx, sy, sz, d)
                print(f"  {sx:.1f}x{sy:.2f}x{sz:.2f}  er={er:.1f}  d={d:.4f}  N={len(r)}")
                continue
            (Pxx, Pyy, Pzz), info = solve_single_d(er, sx, sy, sz, d)
            Pxn, Pyn, Pzn = Pxx/V, Pyy/V, Pzz/V
            rows.append(dict(er=er, sx=sx, sy=sy, sz=sz, u=sy/sx, v=sz/sx, d=d,
                             Pxx=Pxn, Pyy=Pyn, Pzz=Pzn,
                             Nx=1/Pxn, Ny=1/Pyn, Nz=1/Pzn,
                             N_sum=1/Pxn+1/Pyn+1/Pzn))
            print(f"  {sx:.1f}x{sy:.2f}x{sz:.2f}  er={er:.1f}  "
                  f"Nx={1/Pxn:.4f} Ny={1/Pyn:.4f} Nz={1/Pzn:.4f}")
    if not dry_run and rows:
        out = os.path.join(_HERE, 'data', 'csvdata', 'dda3d.csv')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cols = ['er','sx','sy','sz','u (sy/sx)','v (sz/sx)','d',
                'Pxx','Pyy','Pzz','Nx','Ny','Nz','N_sum']
        rows.sort(key=lambda r: (r['er'], r['sz'], r['sy']))
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
        print(f"\nSaved {len(rows)} rows -> data/csvdata/dda3d.csv")


def run_converge(er_values, dry_run=False, verbose=True):
    raw_rows, cnv_rows = [], []
    for u, v in UV:
        sx, sy, sz = 1.0, u, v
        smin = min(sy, sz)
        V = sx * sy * sz
        d_levels = d_levels_for_shape(sx, sy, sz)
        if dry_run:
            print(f"  {sx:.1f}x{sy:.2f}x{sz:.2f}  d_levels={d_levels.tolist()}")
            continue
        for er in er_values:
            if verbose:
                print(f"\n--- {sx:.1f}x{sy:.2f}x{sz:.2f}  er={er:.3f} ---")

            Pxx, Pyy, Pzz = converged_dda(er, sx, sy, sz, d_levels, verbose=verbose)

            cnv_rows.append(dict(er=er, sx=sx, sy=sy, sz=sz, smin=smin,
                Pxx_inf=Pxx/V, Pyy_inf=Pyy/V, Pzz_inf=Pzz/V,
                Nx_inf=V/Pxx, Ny_inf=V/Pyy, Nz_inf=V/Pzz,
                N_sum_inf=V/Pxx+V/Pyy+V/Pzz))

            # Also collect raw d-level data for CSV export
            for ii, d_target in enumerate(d_levels):
                (Px, Py, Pz), info = solve_single_d(er, sx, sy, sz, d_target)
                Pxn, Pyn, Pzn = Px/V, Py/V, Pz/V
                raw_rows.append(dict(er=er, sx=sx, sy=sy, sz=sz, smin=smin,
                    d_target=d_target, d_actual=info.get('d_actual', d_target),
                    N_dipoles=info.get('N_dipoles', 0),
                    Pxx=Pxn, Pyy=Pyn, Pzz=Pzn,
                    Nx=1/Pxn if abs(Pxn)>1e-15 else np.inf,
                    Ny=1/Pyn if abs(Pyn)>1e-15 else np.inf,
                    Nz=1/Pzn if abs(Pzn)>1e-15 else np.inf))

            r = cnv_rows[-1]
            print(f"  converged: Nx={r['Nx_inf']:.4f} Ny={r['Ny_inf']:.4f} "
                  f"Nz={r['Nz_inf']:.4f}  sum={r['N_sum_inf']:.4f}")

    if not dry_run:
        out_dir = os.path.join(_HERE, 'data', 'csvdata')
        os.makedirs(out_dir, exist_ok=True)

        rc = ['er','sx','sy','sz','smin','d_target','d_actual','N_dipoles',
              'Pxx','Pyy','Pzz','Nx','Ny','Nz']
        raw_rows.sort(key=lambda r: (r['smin'], r['er'], r['d_target']))
        with open(os.path.join(out_dir, 'dda_convergence_raw.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rc); w.writeheader(); w.writerows(raw_rows)
        print(f"Saved {len(raw_rows)} rows -> data/csvdata/dda_convergence_raw.csv")

        cc = ['er','sx','sy','sz','smin','Pxx_inf','Pyy_inf','Pzz_inf',
              'Nx_inf','Ny_inf','Nz_inf','N_sum_inf']
        cnv_rows.sort(key=lambda r: (r['smin'], r['er']))
        with open(os.path.join(out_dir, 'dda_convergence_cnv.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cc); w.writeheader(); w.writerows(cnv_rows)
        print(f"Saved {len(cnv_rows)} rows -> data/csvdata/dda_convergence_cnv.csv")


def _parse_er():
    for i, arg in enumerate(sys.argv):
        if arg == '--er' and i+1 < len(sys.argv):
            return [float(x) for x in sys.argv[i+1].split(',')]
    return DEFAULT_ER


if __name__ == '__main__':
    mode = 'converge'
    for i, arg in enumerate(sys.argv):
        if arg == '--mode' and i+1 < len(sys.argv):
            mode = sys.argv[i+1]
    dry_run = '--dry-run' in sys.argv
    er_values = _parse_er()
    print(f"=== DDA Sweep - mode={mode}  er={er_values} ===")
    print(f"  Shapes: {len(UV)}  Cache: {_CACHE_DIR}")
    if mode == 'fixed':
        run_fixed_d(er_values, dry_run=dry_run)
    elif mode == 'converge':
        run_converge(er_values, dry_run=dry_run)
    else:
        print(f"Unknown mode: {mode}. Use --mode fixed|converge")

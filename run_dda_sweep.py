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
import os, sys, time, json, hashlib, csv

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, 'tools'))

from dda_core import (make_lattice, compute_polarizability_fft,
                       _fit_powerlaw_single)

# ═══ Cache ═══
_CACHE_DIR = os.path.join(_HERE, 'data', '_dda_cache')
os.makedirs(_CACHE_DIR, exist_ok=True)
_CACHE_VERSION = 'v2'

def _solve_key(er, sx, sy, sz, d):
    raw = (f"v={_CACHE_VERSION}_er={er:.16e}_"
           f"sx={sx:.16e}_sy={sy:.16e}_sz={sz:.16e}_d={d:.16e}")
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def _fit_key(er, sx, sy, sz, d_levels):
    d_str = '_'.join(f'{d:.6e}' for d in np.asarray(d_levels, dtype=float))
    raw = (f"v={_CACHE_VERSION}_fit_er={er:.16e}_"
           f"sx={sx:.16e}_sy={sy:.16e}_sz={sz:.16e}_d={d_str}")
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def _solve_load(er, sx, sy, sz, d):
    path = os.path.join(_CACHE_DIR, _solve_key(er, sx, sy, sz, d) + '.json')
    if os.path.exists(path):
        with open(path) as fh:
            return tuple(json.load(fh))
    return None

def _solve_save(er, sx, sy, sz, d, Pxx, Pyy, Pzz):
    path = os.path.join(_CACHE_DIR, _solve_key(er, sx, sy, sz, d) + '.json')
    with open(path, 'w') as fh:
        json.dump([float(Pxx), float(Pyy), float(Pzz)], fh)

def _fit_load(er, sx, sy, sz, d_levels):
    path = os.path.join(_CACHE_DIR, _fit_key(er, sx, sy, sz, d_levels) + '.json')
    if os.path.exists(path):
        with open(path) as fh:
            data = json.load(fh)
        return (data['Pxx_inf'], data['Pyy_inf'], data['Pzz_inf'])
    return None

def _fit_save(er, sx, sy, sz, d_levels, Pxx_inf, Pyy_inf, Pzz_inf):
    path = os.path.join(_CACHE_DIR, _fit_key(er, sx, sy, sz, d_levels) + '.json')
    with open(path, 'w') as fh:
        json.dump({'Pxx_inf': float(Pxx_inf), 'Pyy_inf': float(Pyy_inf),
                    'Pzz_inf': float(Pzz_inf)}, fh)

# ═══ Shape grid ═══
UV = [(1.0,1.0),(1.0,0.7),(1.0,0.5),(1.0,0.3),(1.0,0.2),
      (0.7,0.7),(0.7,0.5),(0.7,0.3),(0.7,0.2),
      (0.5,0.5),(0.5,0.3),(0.5,0.2),
      (0.3,0.3),(0.3,0.2),(0.2,0.2)]

DEFAULT_ER = [0.5, 1.25, 1.5, 1.75, 2.0, 3.0, 5.0, 8.0, 10.0, 12.5]


def d_levels_for_shape(sx, sy, sz):
    smin = min(sy, sz)
    if smin < 0.25:
        d_levels = smin / np.array([5.0, 8.0, 11.0, 14.0, 17.0])
    else:
        d_levels = smin / np.array([6.0, 10.0, 14.0, 18.0, 22.0])
    filtered = []
    for d in d_levels:
        r, _, _, _ = make_lattice(sx, sy, sz, d)
        if len(r) <= 20000:
            filtered.append(d)
    return np.array(filtered) if len(filtered) >= 4 else np.array(filtered)


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
            cached = _solve_load(er, sx, sy, sz, d)
            if cached is not None:
                Pxx, Pyy, Pzz = cached
            else:
                t0 = time.time()
                alpha, info = compute_polarizability_fft(er, sx, sy, sz, d, verbose=False)
                dt = time.time() - t0
                Pxx, Pyy, Pzz = alpha[0,0], alpha[1,1], alpha[2,2]
                _solve_save(er, sx, sy, sz, d, Pxx, Pyy, Pzz)
            Pxn, Pyn, Pzn = Pxx/V, Pyy/V, Pzz/V
            rows.append(dict(er=er,sx=sx,sy=sy,sz=sz,u=sy/sx,v=sz/sx,d=d,
                             Pxx=Pxn,Pyy=Pyn,Pzz=Pzn,
                             Nx=1/Pxn,Ny=1/Pyn,Nz=1/Pzn,N_sum=1/Pxn+1/Pyn+1/Pzn))
            print(f"  {sx:.1f}x{sy:.2f}x{sz:.2f}  er={er:.1f}  Nx={1/Pxn:.4f} Ny={1/Pyn:.4f} Nz={1/Pzn:.4f}")
    if not dry_run and rows:
        out = os.path.join(_HERE, 'data', 'csvdata', 'dda3d.csv')
        cols = ['er','sx','sy','sz','u (sy/sx)','v (sz/sx)','d','Pxx','Pyy','Pzz','Nx','Ny','Nz','N_sum']
        rows.sort(key=lambda r: (r['er'], r['sz'], r['sy']))
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
        print(f"\nSaved {len(rows)} rows -> {out}")


def run_converge(er_values, dry_run=False, verbose=True):
    raw_rows, cnv_rows = [], []
    for u, v in UV:
        sx, sy, sz = 1.0, u, v
        smin = min(sy, sz); V = sx*sy*sz
        d_levels = d_levels_for_shape(sx, sy, sz)
        if dry_run:
            print(f"  {sx:.1f}x{sy:.2f}x{sz:.2f}  d_levels={d_levels.tolist()}")
            continue
        for er in er_values:
            if verbose:
                print(f"\n--- {sx:.1f}x{sy:.2f}x{sz:.2f}  er={er:.1f} ---")
            cached_fit = _fit_load(er, sx, sy, sz, d_levels)
            if cached_fit is not None:
                Pxi, Pyi, Pzi = cached_fit
                cnv_rows.append(dict(er=er,sx=sx,sy=sy,sz=sz,smin=smin,
                    Pxx_inf=Pxi/V,Pyy_inf=Pyi/V,Pzz_inf=Pzi/V,
                    Nx_inf=V/Pxi,Ny_inf=V/Pyi,Nz_inf=V/Pzi,N_sum_inf=V/Pxi+V/Pyi+V/Pzi))
                if verbose:
                    print(f"  (cached) Nx={V/Pxi:.4f}")
                continue
            Pxx = np.zeros(len(d_levels)); Pyy = np.zeros(len(d_levels))
            Pzz = np.zeros(len(d_levels)); d_actual = np.zeros(len(d_levels))
            for ii, d_target in enumerate(d_levels):
                cached = _solve_load(er, sx, sy, sz, d_target)
                if cached is not None:
                    Pxx[ii], Pyy[ii], Pzz[ii] = cached
                    r, dx, dy, dz = make_lattice(sx, sy, sz, d_target)
                    d_actual[ii] = float(np.cbrt(dx*dy*dz))
                else:
                    t0 = time.time()
                    alpha, info = compute_polarizability_fft(er, sx, sy, sz, d_target, verbose=False)
                    dt_elapsed = time.time() - t0
                    Pxx[ii], Pyy[ii], Pzz[ii] = alpha[0,0], alpha[1,1], alpha[2,2]
                    d_actual[ii] = info['d']
                    _solve_save(er, sx, sy, sz, d_target, Pxx[ii], Pyy[ii], Pzz[ii])
                    if verbose:
                        print(f"  d={info['d']:.4f} N={info['N']:5d}  "
                              f"Nx={V/Pxx[ii]:.4f} Ny={V/Pyy[ii]:.4f} Nz={V/Pzz[ii]:.4f}  [{dt_elapsed:.1f}s]")
                Pxn, Pyn, Pzn = Pxx[ii]/V, Pyy[ii]/V, Pzz[ii]/V
                r,_,_,_ = make_lattice(sx, sy, sz, d_target)
                raw_rows.append(dict(er=er,sx=sx,sy=sy,sz=sz,smin=smin,
                    d_target=d_target,d_actual=d_actual[ii],N_dipoles=len(r),
                    Pxx=Pxn,Pyy=Pyn,Pzz=Pzn,
                    Nx=1/Pxn if abs(Pxn)>1e-15 else np.inf,
                    Ny=1/Pyn if abs(Pyn)>1e-15 else np.inf,
                    Nz=1/Pzn if abs(Pzn)>1e-15 else np.inf))
            try:
                Pxi, _, _ = _fit_powerlaw_single(Pxx, d_actual)
                Pyi, _, _ = _fit_powerlaw_single(Pyy, d_actual)
                Pzi, _, _ = _fit_powerlaw_single(Pzz, d_actual)
            except:
                print(f"  WARNING: fit failed; using finest-d")
                Pxi, Pyi, Pzi = Pxx[-1], Pyy[-1], Pzz[-1]
            _fit_save(er, sx, sy, sz, d_levels, Pxi, Pyi, Pzi)
            cnv_rows.append(dict(er=er,sx=sx,sy=sy,sz=sz,smin=smin,
                Pxx_inf=Pxi/V,Pyy_inf=Pyi/V,Pzz_inf=Pzi/V,
                Nx_inf=V/Pxi,Ny_inf=V/Pyi,Nz_inf=V/Pzi,N_sum_inf=V/Pxi+V/Pyi+V/Pzi))
            r = cnv_rows[-1]
            print(f"  converged: Nx={r['Nx_inf']:.4f} Ny={r['Ny_inf']:.4f} Nz={r['Nz_inf']:.4f}  sum={r['N_sum_inf']:.4f}")
    if not dry_run:
        out_dir = os.path.join(_HERE, 'data', 'csvdata')
        os.makedirs(out_dir, exist_ok=True)
        rc = ['er','sx','sy','sz','smin','d_target','d_actual','N_dipoles','Pxx','Pyy','Pzz','Nx','Ny','Nz']
        raw_rows.sort(key=lambda r: (r['smin'],r['er'],r['d_target']))
        with open(os.path.join(out_dir,'dda_convergence_raw.csv'),'w',newline='') as f:
            w = csv.DictWriter(f, fieldnames=rc); w.writeheader(); w.writerows(raw_rows)
        print(f"Saved {len(raw_rows)} rows -> data/csvdata/dda_convergence_raw.csv")
        cc = ['er','sx','sy','sz','smin','Pxx_inf','Pyy_inf','Pzz_inf','Nx_inf','Ny_inf','Nz_inf','N_sum_inf']
        cnv_rows.sort(key=lambda r: (r['smin'],r['er']))
        with open(os.path.join(out_dir,'dda_convergence_cnv.csv'),'w',newline='') as f:
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

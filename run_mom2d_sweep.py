"""
run_mom2d_sweep.py — MoM 2D sweep for rectangular prisms (infinite in z).

Computes converged 2D polarizability (per unit length) for a range of
aspect ratios and permittivities. Uses power-law convergence in nd.

Usage:
    python run_mom2d_sweep.py                        # default: er=10,20,100,PEC
    python run_mom2d_sweep.py --er 0.7,1.5,3,6       # custom er list
    python run_mom2d_sweep.py --dry-run               # show what would be computed
"""

import sys, os, time, json, hashlib, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))

from gen_mesh_2d_rect import gen_mesh_2d_rect
from sm_comp_polarizability_2d import comp_polarizability_2d
from vector_analysis import Mesh

# ── Cache ──────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(_HERE, 'data', '_mom2d_cache')
os.makedirs(_CACHE_DIR, exist_ok=True)
_CACHE_VERSION = 'v1'
_FIT_CACHE_VERSION = 'v1'

def _solve_key(er, srx, sry, nd, beta=1.0):
    raw = (f"v={_CACHE_VERSION}_er={er:.16e}_"
           f"srx={srx:.16e}_sry={sry:.16e}_nd={nd}_beta={beta:.3f}")
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def _fit_key(er, srx, sry, nd_levels, beta=1.0):
    nd_str = '_'.join(str(int(x)) for x in nd_levels)
    raw = (f"v={_FIT_CACHE_VERSION}_fit_er={er:.16e}_"
           f"srx={srx:.16e}_sry={sry:.16e}_nd={nd_str}_beta={beta:.3f}")
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def _solve_load(er, srx, sry, nd, beta=1.0):
    path = os.path.join(_CACHE_DIR, _solve_key(er, srx, sry, nd, beta) + '.json')
    if os.path.exists(path):
        with open(path) as fh:
            return tuple(json.load(fh))
    return None

def _solve_save(er, srx, sry, nd, Pxx, Pyy, beta=1.0):
    path = os.path.join(_CACHE_DIR, _solve_key(er, srx, sry, nd, beta) + '.json')
    with open(path, 'w') as fh:
        json.dump([Pxx, Pyy], fh)

def _fit_load(er, srx, sry, nd_levels, beta=1.0):
    path = os.path.join(_CACHE_DIR, _fit_key(er, srx, sry, nd_levels, beta) + '.json')
    if os.path.exists(path):
        with open(path) as fh:
            data = json.load(fh)
        return data['Pxx_inf'], data['Pyy_inf']
    return None

def _fit_save(er, srx, sry, nd_levels, Pxx_inf, Pyy_inf, beta=1.0):
    path = os.path.join(_CACHE_DIR, _fit_key(er, srx, sry, nd_levels, beta) + '.json')
    with open(path, 'w') as fh:
        json.dump({'Pxx_inf': Pxx_inf, 'Pyy_inf': Pyy_inf}, fh)


# ── Power-law fit ──────────────────────────────────────────────────
def _fit_powerlaw(P, nd):
    """Fit P(nd) = P_inf + C * nd^(-beta)."""
    # Guard: if data is flat (already converged at all nd), skip fit
    dP_range = np.max(np.abs(np.diff(P)))
    if dP_range < 1e-12 * max(abs(P[-1]), 1e-15):
        return P[-1], 0.0, 0.0

    from scipy.optimize import minimize
    nd2 = nd[1:]
    dnd = np.diff(nd)
    dP = np.diff(P) / dnd
    x, y = np.log(nd2), np.log(np.abs(dP) + 1e-300)
    slope, intercept = np.polyfit(x, y, 1)
    alpha = -slope
    D0 = np.exp(intercept)
    beta0 = max(alpha - 1.0, 0.3)
    C0 = -D0 / beta0
    P_inf0 = P[-1] - C0 * nd[-1]**(-beta0)
    
    def objective(params):
        P_inf, C, beta = params
        if beta <= 0.15 or beta > 5.0:
            return 1e10
        P_pred = P_inf + C * nd**(-beta)
        return np.sqrt(np.mean((P_pred - P)**2)) + 0.001 * (P_inf - P[-1])**2
    
    res = minimize(objective, [P_inf0, C0, beta0],
                   bounds=[(None, float(P[-1]) + abs(P[-1])*10), (None, None), (0.15, 5.0)],
                   method='L-BFGS-B', options={'maxiter': 200})
    P_inf, C, beta = res.x

    # Fallback: if fit drifted far from finest-nd, use finest-nd directly
    if abs(P_inf - P[-1]) > 0.5 * max(abs(P[-1]), 1e-15):
        return P[-1], 0.0, 0.0

    return P_inf, C, beta


# ── Adaptive nd ────────────────────────────────────────────────────
def adaptive_nd_2d(sry, nd_edge_target=None, nd_max=200):
    """Return nd sequence for a 2D rectangle (srx=1.0, sry variable)."""
    smin = min(1.0, sry)
    if nd_edge_target is None:
        if smin >= 0.3:
            nd_edge_target = np.array([40, 48, 56, 64, 72, 80], dtype=int)
        else:
            scale = 0.3 / smin
            base = np.array([40, 48, 56, 64, 72, 80]) / scale
            nd_edge_target = np.ceil(base).astype(int)
            nd_edge_target = np.maximum(nd_edge_target, 4)

    nd = np.ceil(nd_edge_target / smin).astype(int)
    nd = np.minimum(nd, nd_max)
    nd = np.unique(nd)
    if len(nd) < 4:

        nd_min = min(nd.min(), nd_max // 4)
        nd = np.unique(np.ceil(np.linspace(nd_min, nd_max, 6)).astype(int))
    return nd.astype(float), nd_edge_target


# ── Converged polarizability ───────────────────────────────────────
def converged_2d(er, srx, sry, nd=None, nd_edge_target=None, beta=1.0):
    """Compute converged (nd→∞) 2D polarizability for a prism."""
    nd_max = 200
    if nd is None:
        nd, nd_edge_target = adaptive_nd_2d(sry, nd_max=nd_max)
    elif nd_edge_target is None:
        _, nd_edge_target = adaptive_nd_2d(sry, nd_max=nd_max)

    # Check fit cache
    cached_fit = _fit_load(er, srx, sry, nd_edge_target, beta)
    if cached_fit is not None:
        return cached_fit

    Ao = srx * sry  # rectangle area
    Pxx = np.zeros(len(nd))
    Pyy = np.zeros(len(nd))

    for ii in range(len(nd)):
        nd_i = int(nd[ii])
        cached = _solve_load(er, srx, sry, nd_i, beta)
        if cached is not None:
            Pxx[ii], Pyy[ii] = cached
        else:
            r, n, ds, ne, A = gen_mesh_2d_rect(srx, sry, nd_i, beta=beta)
            mesh = Mesh(r, n, ds, V=Ao, er=er)
            P = comp_polarizability_2d(mesh)
            Pxx[ii] = P.xx.real / Ao
            Pyy[ii] = P.yy.real / Ao
            _solve_save(er, srx, sry, nd_i, Pxx[ii], Pyy[ii], beta)

    try:
        Pxx_inf, _, _ = _fit_powerlaw(Pxx, nd)
        Pyy_inf, _, _ = _fit_powerlaw(Pyy, nd)
    except Exception as e:
        print(f"  WARNING: fit failed: {e}; using finest-nd")
        Pxx_inf, Pyy_inf = Pxx[-1], Pyy[-1]

    _fit_save(er, srx, sry, nd_edge_target, Pxx_inf, Pyy_inf, beta)
    return Pxx_inf, Pyy_inf


# ── Grid ───────────────────────────────────────────────────────────
# 23 aspect ratios from data/csvdata/mom2d.csv
# r = sry/srx,  with srx=1.0 → sry is the aspect ratio
_R_LIST = [0.1, 0.15, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.667, 0.8, 0.9,
           1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.33, 5.0, 7.0, 10.0]
SRY_LIST = _R_LIST  # srx=1.0, sry = aspect ratio


def _parse_er():
    for i, arg in enumerate(sys.argv):
        if arg == '--er' and i+1 < len(sys.argv):
            return [float(x) for x in sys.argv[i+1].split(',')]
    return [0.7, 1.5, 3.0, 6.0, 1e3, 1e5]


# ── Main ───────────────────────────────────────────────────────────
def main():
    ER = _parse_er()
    dry_run = '--dry-run' in sys.argv

    if dry_run:
        print(f"=== MoM 2D Sweep (dry-run) ===")
        print(f"  Shapes: {len(SRY_LIST)}, er: {ER}")
        for sry in SRY_LIST:
            nd, net = adaptive_nd_2d(sry)
            print(f"  1.0x{sry:.2f} (∞)  nd={[int(x) for x in nd]}")
        return

    t_start = time.time()
    for sry in SRY_LIST:
        nd, net = adaptive_nd_2d(sry)
        print(f"\n--- 1.0x{sry:.2f} (∞)  nd={[int(x) for x in nd]} ---")
        for er in ER:
            t1 = time.time()
            Pxx_inf, Pyy_inf = converged_2d(er, 1.0, sry, nd, net)
            dt = time.time() - t1
            Nx = 1.0 / Pxx_inf if abs(Pxx_inf) > 1e-15 else np.inf
            Ny = 1.0 / Pyy_inf if abs(Pyy_inf) > 1e-15 else np.inf
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

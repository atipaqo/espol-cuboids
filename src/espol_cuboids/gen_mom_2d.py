"""
gen_mom_2d.py  —  Generate MoM polarizability data for 2D rectangular prisms.

Caches EVERY individual solve (not just converged).  The converged-
extrapolation step runs after all solves are done (or on re-load).

v1: mirrors gen_mom_3d.py structure.
"""

import sys, os, numpy as np, time, json, hashlib
from .gen_mesh_2d_rect import gen_mesh_2d_rect
from .mom2d import comp_polarizability_2d
from .vector_analysis import Mesh

_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', '_mom2d_cache')
os.makedirs(_CACHE_DIR, exist_ok=True)
_CACHE_VERSION = 'v2'
_FIT_CACHE_VERSION = 'v2'

# ---- cache keys ----
def _solve_key(er, srx, sry, nd, beta=1.0):
    raw = (f"v={_CACHE_VERSION}_er={er:.16e}_"
           f"srx={srx:.16e}_sry={sry:.16e}_nd={nd}_beta={beta:.3f}")
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def _fit_key(er, srx, sry, nd_edge_target, nd_max, beta=1.0):
    nd_str = '_'.join(str(int(x)) for x in nd_edge_target)
    raw = (f"v={_FIT_CACHE_VERSION}_fit_er={er:.16e}_"
           f"srx={srx:.16e}_sry={sry:.16e}_nd={nd_str}_nmax={nd_max}_beta={beta:.3f}")
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

# ---- raw solve cache ----
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

# ---- converged fit cache ----
def _fit_load(er, srx, sry, nd_edge_target, nd_max, beta=1.0):
    path = os.path.join(_CACHE_DIR, _fit_key(er, srx, sry, nd_edge_target, nd_max, beta) + '.json')
    if os.path.exists(path):
        with open(path) as fh:
            data = json.load(fh)
        return data['Pxx_inf'], data['Pyy_inf']
    return None

def _fit_save(er, srx, sry, nd_edge_target, nd_max, Pxx_inf, Pyy_inf, beta=1.0):
    path = os.path.join(_CACHE_DIR, _fit_key(er, srx, sry, nd_edge_target, nd_max, beta) + '.json')
    with open(path, 'w') as fh:
        json.dump({'Pxx_inf': Pxx_inf, 'Pyy_inf': Pyy_inf}, fh)

# ---- power-law fit ----
def fit_powerlaw(P, nd):
    """Fit P(nd) = P_inf + C * nd^(-beta)."""
    from scipy.optimize import minimize
    nd2 = nd[1:]; dnd = np.diff(nd); dP = np.diff(P) / dnd
    x = np.log(nd2); y = np.log(np.abs(dP) + 1e-300)
    slope, intercept = np.polyfit(x, y, 1)
    alpha0 = -slope; D0 = np.exp(intercept)
    beta0 = max(alpha0 - 1.0, 0.3); C0 = -D0 / beta0
    P_inf0 = P[-1] - C0 * nd[-1]**(-beta0)

    def objective(params):
        P_inf, C, beta = params
        if beta <= 0.15 or beta > 5.0:
            return 1e10
        P_pred = P_inf + C * nd**(-beta)
        return np.sqrt(np.mean((P_pred - P)**2)) + 0.001 * (P_inf - P[-1])**2

    res = minimize(objective, [P_inf0, C0, beta0],
                   bounds=[(None, float(P[-1]) + abs(P[-1]) * 10),
                           (None, None), (0.15, 5.0)],
                   method='L-BFGS-B', options={'maxiter': 200})
    P_inf, C, beta = res.x
    return P_inf, dict(alpha=beta + 1.0, D=C * beta, beta=beta)

# ---- adaptive nd ----
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

# ---- converged polarizability ----
def converged_2d(er, srx, sry, nd=None, nd_edge_target=None, beta=1.0):
    """Compute converged (nd→∞) 2D polarizability for a prism.

    Parameters
    ----------
    er : float
        Relative permittivity.
    srx, sry : float
        Rectangle dimensions (x, y).
    nd : array-like or None
        Discretization levels.  If None, auto-generated via adaptive_nd_2d.
    nd_edge_target : array-like or None
        Target edge counts for adaptive scaling.
    beta : float
        Mesh clustering parameter (0 = uniform, 1 = sin-clustered).

    Returns
    -------
    Pxx_inf, Pyy_inf : float
        Converged polarizability components (normalised by area).
    """
    nd_max = 200
    if nd is None:
        nd, nd_edge_target = adaptive_nd_2d(sry, nd_max=nd_max)
    elif nd_edge_target is None:
        _, nd_edge_target = adaptive_nd_2d(sry, nd_max=nd_max)

    # Check fit cache
    cached_fit = _fit_load(er, srx, sry, nd_edge_target, nd_max, beta)
    if cached_fit is not None:
        return cached_fit

    Ao = srx * sry
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

    Pxx_inf, _ = fit_powerlaw(Pxx, nd)
    Pyy_inf, _ = fit_powerlaw(Pyy, nd)
    _fit_save(er, srx, sry, nd_edge_target, nd_max, Pxx_inf, Pyy_inf, beta)
    return Pxx_inf, Pyy_inf

# ---- grid ----
def default_grid():
    """Default shape × er grid for 2D sweeps."""
    r_list = [0.1, 0.15, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.667, 0.8, 0.9,
              1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.33, 5.0, 7.0, 10.0]
    er_list = [0.7, 1.5, 3.0, 6.0, 1e3, 1e5]
    return r_list, er_list

# ---- main ----
def main():
    dry_run = '--dry-run' in sys.argv

    r_list, er_list = default_grid()
    nd_example, _ = adaptive_nd_2d(1.0)
    print(f"=== MoM 2D data generation (v2) ===")
    print(f"  Shapes: {len(r_list)}")
    print(f"  nd_max: 200, auto-scaled nd_edge_target for thin shapes")
    print(f"  Example nd (square): {[int(x) for x in nd_example]}")
    print(f"  Cache dir: {_CACHE_DIR}")

    if dry_run:
        for sry in r_list:
            nd, net = adaptive_nd_2d(sry)
            n_cached = sum(1 for nd_i in nd
                          if _solve_load(1e10, 1.0, sry, int(nd_i)) is not None)
            print(f"    1.0x{sry:.2f}  nd={[int(x) for x in nd]}  "
                  f"cached={n_cached}/{len(nd)}")
        return

    t_start = time.time()
    for sry in r_list:
        nd, net = adaptive_nd_2d(sry)
        if len(nd) < 4:
            print(f"  SKIP (1.0x{sry:.2f}) — only {len(nd)} distinct nd")
            continue
        print(f"\n--- 1.0x{sry:.2f} (∞)  nd={[int(x) for x in nd]} ---")
        for er in er_list:
            t0 = time.time()
            Pxx_inf, Pyy_inf = converged_2d(er, 1.0, sry, nd, net)
            dt = time.time() - t0
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

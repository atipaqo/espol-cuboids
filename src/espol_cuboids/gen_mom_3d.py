"""
gen_mom_3d.py  —  Generate MoM polarizability data for 3D cuboids.

Caches EVERY individual solve (not just converged).  The converged-
extrapolation step runs after all solves are done (or on re-load).

v9: auto-scaled nd_edge_target for thin shapes, nd_max=100.
"""

import sys, os, numpy as np, time, json, hashlib
from .gen_mesh_3d_rect import gen_mesh_3d_rect
from .mom3d import comp_polarizability_3d
from .vector_analysis import Mesh, D3

_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', '_mom3d_cache')
os.makedirs(_CACHE_DIR, exist_ok=True)
_CACHE_VERSION = 'v8'
_FIT_CACHE_VERSION = 'v9'

# ---- cache keys ----
def _solve_key(er, sx, sy, sz, nd, beta=0.0, aca=False):
    tag = '_aca' if aca else ''
    raw = (f"v={_CACHE_VERSION}_er={er:.16e}_"
           f"sx={sx:.16e}_sy={sy:.16e}_sz={sz:.16e}_nd={nd}_beta={beta:.3f}{tag}")
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def _fit_key(er, sx, sy, sz, nd_edge_target, nd_max, beta=0.0, aca=False):
    tag = '_aca' if aca else ''
    raw = (f"v={_FIT_CACHE_VERSION}_fit_er={er:.16e}_"
           f"sx={sx:.16e}_sy={sy:.16e}_sz={sz:.16e}_"
           f"net={nd_edge_target!r}_nmax={nd_max}_beta={beta:.3f}{tag}")
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

# ---- raw solve cache ----
def _solve_load(er, sx, sy, sz, nd, beta=0.0, aca=False):
    path = os.path.join(_CACHE_DIR, _solve_key(er, sx, sy, sz, nd, beta, aca) + '.json')
    if os.path.exists(path):
        with open(path) as fh:
            return tuple(json.load(fh))
    return None

def _solve_save(er, sx, sy, sz, nd, Pxx, Pyy, Pzz, beta=0.0, aca=False):
    path = os.path.join(_CACHE_DIR, _solve_key(er, sx, sy, sz, nd, beta, aca) + '.json')
    with open(path, 'w') as fh:
        json.dump([Pxx, Pyy, Pzz], fh)

# ---- converged fit cache ----
def _fit_load(er, sx, sy, sz, nd_edge_target, nd_max, beta=0.0, aca=False):
    path = os.path.join(_CACHE_DIR, _fit_key(er, sx, sy, sz, nd_edge_target, nd_max, beta, aca) + '.json')
    if os.path.exists(path):
        with open(path) as fh:
            data = json.load(fh)
        return data['Pxx_inf'], data['Pyy_inf'], data['Pzz_inf']
    return None

def _fit_save(er, sx, sy, sz, nd_edge_target, nd_max, Pxx_inf, Pyy_inf, Pzz_inf, beta=0.0, aca=False):
    path = os.path.join(_CACHE_DIR, _fit_key(er, sx, sy, sz, nd_edge_target, nd_max, beta, aca) + '.json')
    with open(path, 'w') as fh:
        json.dump({'Pxx_inf': Pxx_inf, 'Pyy_inf': Pyy_inf, 'Pzz_inf': Pzz_inf}, fh)

# ---- power-law fit ----
def fit_powerlaw(P, nd):
    from scipy.optimize import minimize
    nd2 = nd[1:]; dnd = np.diff(nd); dP = np.diff(P)/dnd
    x = np.log(nd2); y = np.log(np.abs(dP))
    slope, intercept = np.polyfit(x, y, 1)
    alpha0 = -slope; D0 = np.exp(intercept)
    beta0 = max(alpha0 - 1.0, 0.3); C0 = -D0 / beta0
    P_inf0 = P[-1] - C0 * nd[-1]**(-beta0)
    def objective(params):
        P_inf, C, beta = params
        if beta <= 0.15: return 1e10
        P_pred = P_inf + C * nd**(-beta)
        return np.sqrt(np.mean((P_pred-P)**2)) + 0.001*(P_inf-P[-1])**2
    res = minimize(objective, [P_inf0, C0, beta0],
                   bounds=[(None, P[-1]), (None, None), (0.15, None)],
                   method='L-BFGS-B', options={'maxiter': 200})
    P_inf, C, beta = res.x
    return P_inf, dict(alpha=beta+1.0, D=C*beta, beta=beta)

# ---- adaptive nd ----
def adaptive_nd(sx, sy, sz, nd_edge_target=None, nd_max=100):
    """Return nd sequence for a shape, auto-scaling for thin geometry."""
    smin = min(sy, sz)
    if nd_edge_target is None:
        if smin >= 0.3:
            nd_edge_target = np.array([20, 24, 28, 32, 36, 40], dtype=int)
        else:
            # For thin shapes, scale so max nd ~ 80-100
            scale = 0.3 / smin
            base = np.array([20, 24, 28, 32, 36, 40]) / scale
            nd_edge_target = np.ceil(base).astype(int)
            nd_edge_target = np.maximum(nd_edge_target, 4)

    nd = np.ceil(nd_edge_target / smin).astype(int)
    nd = np.minimum(nd, nd_max)
    nd = np.unique(nd)
    if len(nd) < 4:
        nd = np.unique(np.ceil(np.linspace(max(nd.min(), 20), nd_max, 6)).astype(int))
    return nd.astype(float), nd_edge_target

# ---- converged polarizability ----
def converged_3d(er, sx, sy, sz, nd=None, nd_edge_target=None, beta=0.0, aca=False):
    """Return converged P_inf.  Solves missing nd entries, caches solves, fits.

    Parameters
    ----------
    aca : bool
        If True, use ACA-compressed H-matrix + GMRES (faster for N > 2000).
        Default False uses direct LU solve.
    """
    nd_max = 100
    if nd is None:
        nd, nd_edge_target = adaptive_nd(sx, sy, sz, nd_max=nd_max)
    elif nd_edge_target is None:
        _, nd_edge_target = adaptive_nd(sx, sy, sz, nd_max=nd_max)

    # Check fit cache
    cached_fit = _fit_load(er, sx, sy, sz, nd_edge_target, nd_max, beta, aca)
    if cached_fit is not None:
        return cached_fit

    # Load cached solves, compute missing ones
    Pxx = np.zeros(len(nd)); Pyy = np.zeros(len(nd)); Pzz = np.zeros(len(nd))
    for ii in range(len(nd)):
        nd_i = int(nd[ii])
        cached = _solve_load(er, sx, sy, sz, nd_i, beta, aca)
        if cached is not None:
            Pxx[ii], Pyy[ii], Pzz[ii] = cached
        else:
            r, n, ds, ne, Vo = gen_mesh_3d_rect(sx, sy, sz, nd_i,
                                                 beta=beta, octant=True)
            mesh = Mesh(r, n, ds, V=Vo, er=er, octant=True)
            P = comp_polarizability_3d(mesh, iterative=aca, aca=aca)
            Pxx[ii] = P.xx.real / Vo
            Pyy[ii] = P.yy.real / Vo
            Pzz[ii] = P.zz.real / Vo
            _solve_save(er, sx, sy, sz, nd_i, Pxx[ii], Pyy[ii], Pzz[ii], beta, aca)

    Pxx_inf, _ = fit_powerlaw(Pxx, nd)
    Pyy_inf, _ = fit_powerlaw(Pyy, nd)
    Pzz_inf, _ = fit_powerlaw(Pzz, nd)
    _fit_save(er, sx, sy, sz, nd_edge_target, nd_max, Pxx_inf, Pyy_inf, Pzz_inf, beta, aca)
    return Pxx_inf, Pyy_inf, Pzz_inf

# ---- grid ----
def default_grid():
    uv_pairs = [
        (1.0,1.0), (1.0,0.7), (1.0,0.5), (1.0,0.3), (1.0,0.2),
        (0.7,0.7), (0.7,0.5), (0.7,0.3), (0.7,0.2),
        (0.5,0.5), (0.5,0.3), (0.5,0.2),
        (0.3,0.3), (0.3,0.2), (0.2,0.2),
    ]
    er_list = [10, 20, 100, 1e10]
    return uv_pairs, er_list

# ---- main ----
def main():
    dry_run = '--dry-run' in sys.argv

    uv_pairs, er_list = default_grid()
    nd_example, _ = adaptive_nd(1.0, 1.0, 1.0)
    print(f"=== MoM 3D data generation (v9) ===")
    print(f"  Shapes: {len(uv_pairs)}")
    print(f"  nd_max: 100, auto-scaled nd_edge_target for thin shapes")
    print(f"  Example nd (cube): {nd_example.tolist()}")
    print(f"  Cache dir: {_CACHE_DIR}")

    if dry_run:
        for u, v in uv_pairs:
            sx, sy, sz = 1.0, u, v
            nd, net = adaptive_nd(sx, sy, sz)
            n_cached = sum(1 for nd_i in nd
                          if _solve_load(1e10, sx, sy, sz, int(nd_i)) is not None)
            print(f"    {sx:.1f}x{sy:.2f}x{sz:.2f}  nd={nd.tolist()}  "
                  f"net={net.tolist()}  cached={n_cached}/{len(nd)}")
        return

    t_start = time.time()
    for u, v in uv_pairs:
        sx, sy, sz = 1.0, u, v
        nd, net = adaptive_nd(sx, sy, sz)
        if len(nd) < 4:
            print(f"  SKIP ({sx:.2f}x{sy:.2f}x{sz:.2f}) — only {len(nd)} distinct nd")
            continue
        for erj in er_list:
            t0 = time.time()
            Pxx, Pyy, Pzz = converged_3d(erj, sx, sy, sz, nd, net)
            dt = time.time() - t0
            Nx, Ny, Nz = 1/Pxx, 1/Pyy, 1/Pzz
            print(f"  ({sx:.2f}x{sy:.2f}x{sz:.2f}) er={erj:8.0e}  "
                  f"Nx={Nx:.4f} Ny={Ny:.4f} Nz={Nz:.4f} sum={Nx+Ny+Nz:.4f}  [{dt:.1f}s]")
    elapsed = time.time() - t_start
    print(f"\nTotal: {elapsed:.0f}s  ({elapsed/3600:.1f}h)")

if __name__ == '__main__':
    main()

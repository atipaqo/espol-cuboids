"""
gen_dda.py  —  Generate DDA polarizability data for dielectric cuboids.

Caches EVERY individual solve (not just converged).  The converged-
extrapolation step (d→0 power-law fit) runs after all solves are done
(or on re-load).

Mirrors gen_mom_3d.py / gen_mom_2d.py structure.
"""

import sys, os, numpy as np, time, json, hashlib
from .dda_core import (make_lattice, compute_polarizability_fft,
                       _fit_powerlaw_single)

_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', '_dda_cache')
os.makedirs(_CACHE_DIR, exist_ok=True)
_CACHE_VERSION = 'v3'
_FIT_CACHE_VERSION = 'v3'

# ═══════════════════════════════════════════════════════════════════
#  Cache keys  &  load / save
# ═══════════════════════════════════════════════════════════════════

def _solve_key(er, sx, sy, sz, d):
    raw = (f"v={_CACHE_VERSION}_er={er:.16e}_"
           f"sx={sx:.16e}_sy={sy:.16e}_sz={sz:.16e}_d={d:.16e}")
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def _fit_key(er, sx, sy, sz, d_levels):
    d_str = '_'.join(f'{x:.6e}' for x in np.asarray(d_levels, dtype=float))
    raw = (f"v={_FIT_CACHE_VERSION}_fit_er={er:.16e}_"
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
        return data['Pxx_inf'], data['Pyy_inf'], data['Pzz_inf']
    return None

def _fit_save(er, sx, sy, sz, d_levels, Pxx_inf, Pyy_inf, Pzz_inf):
    path = os.path.join(_CACHE_DIR, _fit_key(er, sx, sy, sz, d_levels) + '.json')
    with open(path, 'w') as fh:
        json.dump({'Pxx_inf': float(Pxx_inf), 'Pyy_inf': float(Pyy_inf),
                    'Pzz_inf': float(Pzz_inf)}, fh)

# ═══════════════════════════════════════════════════════════════════
#  Adaptive d-levels
# ═══════════════════════════════════════════════════════════════════

def d_levels_for_shape(sx, sy, sz, max_dipoles=20000):
    """Return d-spacing levels for a shape, auto-scaling for thin geometry.

    Filters out levels that would exceed max_dipoles.
    """
    smin = min(sy, sz)
    if smin < 0.25:
        d_levels = smin / np.array([5.0, 8.0, 11.0, 14.0, 17.0])
    else:
        d_levels = smin / np.array([6.0, 10.0, 14.0, 18.0, 22.0])
    filtered = []
    for d in d_levels:
        r, _, _, _ = make_lattice(sx, sy, sz, d)
        if len(r) <= max_dipoles:
            filtered.append(d)
    return np.array(filtered) if len(filtered) >= 4 else np.array(filtered)

# ═══════════════════════════════════════════════════════════════════
#  Single-d solve  (with cache)
# ═══════════════════════════════════════════════════════════════════

def solve_single_d(er, sx, sy, sz, d, verbose=False):
    """Compute polarizability at a single d level (uses cache).

    Returns
    -------
    (Pxx, Pyy, Pzz) : float
        Diagonal polarizability components.
    info : dict
        {d_actual, N_dipoles, time}
    """
    cached = _solve_load(er, sx, sy, sz, d)
    if cached is not None:
        r, _, _, _ = make_lattice(sx, sy, sz, d)
        return cached, dict(d_actual=None, N_dipoles=len(r), time=0.0, cached=True)

    t0 = time.time()
    alpha, info = compute_polarizability_fft(er, sx, sy, sz, d, verbose=verbose)
    dt = time.time() - t0
    Pxx, Pyy, Pzz = float(alpha[0, 0]), float(alpha[1, 1]), float(alpha[2, 2])
    _solve_save(er, sx, sy, sz, d, Pxx, Pyy, Pzz)
    return (Pxx, Pyy, Pzz), dict(d_actual=info['d'], N_dipoles=info['N'],
                                  time=dt, cached=False)

# ═══════════════════════════════════════════════════════════════════
#  Converged polarizability  (d → 0 power-law extrapolation)
# ═══════════════════════════════════════════════════════════════════

def converged_dda(er, sx, sy, sz, d_levels=None, verbose=True):
    """Compute d→0 extrapolated polarizability for a cuboid.

    Parameters
    ----------
    er : float
        Relative permittivity.
    sx, sy, sz : float
        Cuboid dimensions.
    d_levels : array-like or None
        Lattice spacings.  If None, auto-generated via d_levels_for_shape.
    verbose : bool

    Returns
    -------
    (Pxx_inf, Pyy_inf, Pzz_inf) : float
        Extrapolated polarizability components (d→0 limit).
    """
    V = sx * sy * sz
    if d_levels is None:
        d_levels = d_levels_for_shape(sx, sy, sz)
    d_levels = np.asarray(d_levels, dtype=float)

    # Check fit cache
    cached_fit = _fit_load(er, sx, sy, sz, d_levels)
    if cached_fit is not None:
        if verbose:
            Nx = V / cached_fit[0] if abs(cached_fit[0]) > 1e-15 else np.inf
            print(f"  (cached converged) Nx={Nx:.4f}")
        return cached_fit

    n_pts = len(d_levels)
    Pxx = np.zeros(n_pts); Pyy = np.zeros(n_pts); Pzz = np.zeros(n_pts)
    d_actual = np.zeros(n_pts)

    for ii, d_target in enumerate(d_levels):
        (Pxx[ii], Pyy[ii], Pzz[ii]), info = solve_single_d(
            er, sx, sy, sz, d_target, verbose=False)
        if info['d_actual'] is not None:
            d_actual[ii] = info['d_actual']
        else:
            r, dx, dy, dz = make_lattice(sx, sy, sz, d_target)
            d_actual[ii] = float(np.cbrt(dx * dy * dz))
        if verbose:
            Nx = V / Pxx[ii] if abs(Pxx[ii]) > 1e-15 else np.inf
            Ny = V / Pyy[ii] if abs(Pyy[ii]) > 1e-15 else np.inf
            Nz = V / Pzz[ii] if abs(Pzz[ii]) > 1e-15 else np.inf
            tag = ' (cached)' if info['cached'] else f' [{info["time"]:.1f}s]'
            print(f"  d={d_actual[ii]:.4f} N={info['N_dipoles']:5d}  "
                  f"Nx={Nx:.4f} Ny={Ny:.4f} Nz={Nz:.4f}{tag}")

    try:
        Pxx_inf, _, _ = _fit_powerlaw_single(Pxx, d_actual)
        Pyy_inf, _, _ = _fit_powerlaw_single(Pyy, d_actual)
        Pzz_inf, _, _ = _fit_powerlaw_single(Pzz, d_actual)
    except Exception:
        if verbose:
            print(f"  WARNING: fit failed; using finest-d")
        Pxx_inf, Pyy_inf, Pzz_inf = Pxx[-1], Pyy[-1], Pzz[-1]

    _fit_save(er, sx, sy, sz, d_levels, Pxx_inf, Pyy_inf, Pzz_inf)
    return Pxx_inf, Pyy_inf, Pzz_inf

# ═══════════════════════════════════════════════════════════════════
#  Default grid
# ═══════════════════════════════════════════════════════════════════

def default_grid():
    """Default shape × er grid for DDA sweeps."""
    uv_pairs = [
        (1.0, 1.0), (1.0, 0.7), (1.0, 0.5), (1.0, 0.3), (1.0, 0.2),
        (0.7, 0.7), (0.7, 0.5), (0.7, 0.3), (0.7, 0.2),
        (0.5, 0.5), (0.5, 0.3), (0.5, 0.2),
        (0.3, 0.3), (0.3, 0.2), (0.2, 0.2),
    ]
    er_list = [0.5, 1.25, 1.5, 1.75, 2.0, 3.0, 5.0, 8.0, 10.0, 12.5]
    return uv_pairs, er_list

# ═══════════════════════════════════════════════════════════════════
#  CLI main
# ═══════════════════════════════════════════════════════════════════

def main():
    dry_run = '--dry-run' in sys.argv

    uv_pairs, er_list = default_grid()
    print(f"=== DDA data generation (v3) ===")
    print(f"  Shapes: {len(uv_pairs)}")
    print(f"  Cache dir: {_CACHE_DIR}")

    if dry_run:
        for u, v in uv_pairs:
            sx, sy, sz = 1.0, u, v
            d_levels = d_levels_for_shape(sx, sy, sz)
            print(f"  {sx:.1f}x{sy:.2f}x{sz:.2f}  d_levels={d_levels.tolist()}")
        return

    t_start = time.time()
    for u, v in uv_pairs:
        sx, sy, sz = 1.0, u, v
        d_levels = d_levels_for_shape(sx, sy, sz)
        if len(d_levels) < 4:
            print(f"  SKIP ({sx:.2f}x{sy:.2f}x{sz:.2f}) — only {len(d_levels)} d-levels")
            continue
        V = sx * sy * sz
        for er in er_list:
            print(f"\n--- {sx:.1f}x{sy:.2f}x{sz:.2f}  er={er:.3f} ---")
            t0 = time.time()
            Pxx, Pyy, Pzz = converged_dda(er, sx, sy, sz, d_levels)
            dt = time.time() - t0
            Nx = V / Pxx if abs(Pxx) > 1e-15 else np.inf
            Ny = V / Pyy if abs(Pyy) > 1e-15 else np.inf
            Nz = V / Pzz if abs(Pzz) > 1e-15 else np.inf
            print(f"  converged: Nx={Nx:.4f} Ny={Ny:.4f} Nz={Nz:.4f}  "
                  f"sum={Nx+Ny+Nz:.4f}  [{dt:.1f}s]")
    elapsed = time.time() - t_start
    print(f"\nTotal: {elapsed:.0f}s  ({elapsed/3600:.1f}h)")

if __name__ == '__main__':
    main()

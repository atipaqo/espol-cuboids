"""
dda_core.py  ─  Discrete Dipole Approximation for dielectric cuboids
                (static / quasi-static limit, k→0).

Phase 1: dense solver for N < ~5000 dipoles.
Phase 2: FFT-accelerated GMRES (to be added).
Phase 3: convergence and PEC extrapolation.

Usage:
    from dda_core import compute_polarizability, converged_polarizability
    alpha = compute_polarizability(er=10, sx=1.0, sy=0.5, sz=0.5, d=0.05)
    alpha_inf = converged_polarizability(er=10, sx=1.0, sy=0.5, sz=0.5)
"""

import numpy as np
from scipy.optimize import minimize
import time

# ═══════════════════════════════════════════════════════════════════
#  Lattice
# ═══════════════════════════════════════════════════════════════════

def make_lattice(sx, sy, sz, d):
    """Fill cuboid [-sx/2,sx/2] × [-sy/2,sy/2] × [-sz/2,sz/2]
    with a cubic lattice of spacing ~d.

    Returns
    -------
    r : ndarray (N, 3)     dipole positions
    dx, dy, dz : float     actual spacings (preserve aspect ratio)
    """
    nx = max(1, round(sx / d))
    ny = max(1, round(sy / d))
    nz = max(1, round(sz / d))
    dx, dy, dz = sx / nx, sy / ny, sz / nz

    xs = np.linspace(-sx / 2 + dx / 2, sx / 2 - dx / 2, nx)
    ys = np.linspace(-sy / 2 + dy / 2, sy / 2 - dy / 2, ny)
    zs = np.linspace(-sz / 2 + dz / 2, sz / 2 - dz / 2, nz)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    r = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return r, dx, dy, dz


def lattice_info(sx, sy, sz, d):
    """Return summary of lattice for a given spacing."""
    r, dx, dy, dz = make_lattice(sx, sy, sz, d)
    N = len(r)
    print(f"  d_target={d:.4f}  actual=({dx:.4f},{dy:.4f},{dz:.4f})  "
          f"N={N}  unknowns={3*N}")
    return r, dx, dy, dz


# ═══════════════════════════════════════════════════════════════════
#  Polarizability of a single dipole (Clausius-Mossotti)
# ═══════════════════════════════════════════════════════════════════

def alpha_cm(er, d):
    """Clausius-Mossotti polarizability of a cube of edge d.

    alpha = 3 * eps0 * V_cell * (er - 1) / (er + 2)

    In units where eps0 = 1.
    """
    return 3.0 * d**3 * (er - 1.0) / (er + 2.0)


# ═══════════════════════════════════════════════════════════════════
#  Dense matrix assembly
# ═══════════════════════════════════════════════════════════════════

def build_matrix(r, dx, dy, dz, alpha, eps0=1.0):
    """Build the full 3N × 3N coupled-dipole system matrix.

    A_{ii}   = I / alpha                      (self block)
    A_{ij}   = -G(r_i - r_j)   (i != j)      (interaction)

    where G(R) = (3 Rhat Rhat^T - I) / (4 pi eps0 R^3)

    The matrix is real-symmetric for er > 1.

    Parameters
    ----------
    r : (N, 3) array
    dx, dy, dz : float   (unused; for future finite-size corrections)
    alpha : float         dipole polarizability (scalar, isotropic)
    eps0 : float          vacuum permittivity (default 1.0)

    Returns
    -------
    A : (3N, 3N) ndarray
    """
    N = len(r)
    dim = 3 * N
    A = np.zeros((dim, dim), dtype=float)

    inv_alpha = 1.0 / alpha
    factor = 1.0 / (4.0 * np.pi * eps0)

    for i in range(N):
        i3 = 3 * i
        ri = r[i]

        # self block (diagonal 3x3)
        A[i3:i3 + 3, i3:i3 + 3] = inv_alpha * np.eye(3)

        # interaction with j > i (symmetric)
        for j in range(i + 1, N):
            j3 = 3 * j
            R = ri - r[j]
            rr = np.dot(R, R)
            r_mag = np.sqrt(rr)
            Rhat = R / r_mag

            # Green's dyadic: (3 Rhat Rhat^T - I) / (4 pi R^3)
            G_block = factor * (3.0 * np.outer(Rhat, Rhat) - np.eye(3)) / (r_mag ** 3)

            A[i3:i3 + 3, j3:j3 + 3] = -G_block
            A[j3:j3 + 3, i3:i3 + 3] = -G_block

    return A


# ═══════════════════════════════════════════════════════════════════
#  Solve and extract polarizability tensor
# ═══════════════════════════════════════════════════════════════════

def compute_polarizability(er, sx, sy, sz, d, verbose=False):
    """Compute the 3x3 polarizability tensor for a dielectric cuboid.

    Parameters
    ----------
    er : float            relative permittivity (>= 1)
    sx, sy, sz : float    cuboid dimensions
    d : float             target lattice spacing
    verbose : bool

    Returns
    -------
    alpha : (3, 3) ndarray    polarizability tensor / eps0
    info : dict               {d, N, time, er, V}
    """
    V = sx * sy * sz
    r, dx, dy, dz = make_lattice(sx, sy, sz, d)
    N = len(r)
    alpha_val = alpha_cm(er, float(np.mean([dx, dy, dz])))

    t0 = time.time()
    A = build_matrix(r, dx, dy, dz, alpha_val)
    t_build = time.time() - t0

    if verbose:
        print(f"  dda: er={er:.1f} V={V:.3f} d~{d:.4f} N={N} "
              f"dim={3*N} build={t_build:.1f}s", end="")

    alpha_tensor = np.zeros((3, 3), dtype=float)

    e0_list = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    for k, e0 in enumerate(e0_list):
        b = np.tile(e0, N)
        x = np.linalg.solve(A, b)
        P = x.reshape(-1, 3).sum(axis=0)
        alpha_tensor[:, k] = P

    t_total = time.time() - t0

    if verbose:
        print(f"  solve={t_total-t_build:.1f}s  total={t_total:.1f}s")

    info = dict(d=float(np.cbrt(dx * dy * dz)),
                N=N, time=t_total, er=er, V=V,
                dx=dx, dy=dy, dz=dz)
    return alpha_tensor, info


# ═══════════════════════════════════════════════════════════════════
#  Convergence extrapolation
# ═══════════════════════════════════════════════════════════════════

def _fit_powerlaw_single(y, d_vals):
    """Fit  y(d) = y_inf + C * d^beta   to data (d_vals, y).

    Returns (y_inf, C, beta).
    """
    if len(d_vals) < 3:
        raise ValueError("Need at least 3 data points for power-law fit.")

    d_arr = np.asarray(d_vals, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    # initial guess from log-difference approach
    dy = np.diff(y_arr)
    dd = np.diff(d_arr)
    d_mid = 0.5 * (d_arr[1:] + d_arr[:-1])
    log_d = np.log(d_mid)
    log_dy = np.log(np.abs(dy / dd + 1e-300))
    slope, intercept = np.polyfit(log_d, log_dy, 1)
    beta0 = max(slope + 1.0, 0.3)
    C0 = np.exp(intercept) / beta0
    y_inf0 = y_arr[-1] - C0 * d_arr[-1] ** beta0

    def objective(params):
        y_inf, C, beta = params
        if beta <= 0.15 or beta > 5.0:
            return 1e10
        y_pred = y_inf + C * d_arr ** beta
        return np.sqrt(np.mean((y_pred - y_arr) ** 2))

    res = minimize(objective, [y_inf0, C0, beta0],
                   bounds=[(None, y_arr[-1] + abs(y_arr[-1]) * 10),
                           (None, None),
                           (0.15, 5.0)],
                   method='L-BFGS-B',
                   options={'maxiter': 200})
    return res.x[0], res.x[1], res.x[2]


def converged_polarizability(er, sx, sy, sz, d_levels=None, verbose=True):
    """Compute polarizability at several lattice spacings and
    extrapolate to the continuum limit d -> 0.

    Parameters
    ----------
    er : float
    sx, sy, sz : float
    d_levels : list of float or None
        If None, uses s_min / [6, 10, 14, 18, 22] (4-5 points).

    Returns
    -------
    alpha_inf : (3, 3) ndarray    extrapolated tensor
    alpha_raw : list of (3,3)     raw tensors at each d
    fit_info : dict               {Pxx_inf, C_xx, beta_xx, ...}
    """
    smin = min(sy, sz)
    if d_levels is None:
        d_levels = smin / np.array([6.0, 10.0, 14.0, 18.0, 22.0])

    d_levels = np.asarray(d_levels, dtype=float)
    V = sx * sy * sz

    if verbose:
        print(f"=== DDA convergence sweep ===")
        print(f"  shape: {sx:.2f}x{sy:.2f}x{sz:.2f}  er={er:.1f}")
        print(f"  d_levels: {d_levels.tolist()}")

    alpha_list = []
    d_actual = []

    for d in d_levels:
        alpha, info = compute_polarizability(er, sx, sy, sz, d, verbose=False)
        alpha_list.append(alpha)
        d_actual.append(info['d'])
        if verbose:
            Nx = V / alpha[0, 0] if abs(alpha[0, 0]) > 1e-15 else np.inf
            Ny = V / alpha[1, 1] if abs(alpha[1, 1]) > 1e-15 else np.inf
            Nz = V / alpha[2, 2] if abs(alpha[2, 2]) > 1e-15 else np.inf
            print(f"  d={info['d']:.4f}  N={info['N']:5d}  "
                  f"Nx={Nx:.4f}  Ny={Ny:.4f}  Nz={Nz:.4f}  "
                  f"[{info['time']:.1f}s]")

    d_arr = np.array(d_actual)
    alpha_inf = np.zeros((3, 3))
    fit_info = {}

    for k, axis in enumerate(['xx', 'yy', 'zz']):
        y_vals = np.array([a[k, k] for a in alpha_list])
        try:
            a_inf, C, beta = _fit_powerlaw_single(y_vals, d_arr)
        except Exception as e:
            if verbose:
                print(f"  WARNING: fit failed for {axis}: {e}; using finest-d value")
            a_inf = y_vals[-1]
            C, beta = np.nan, np.nan
        alpha_inf[k, k] = a_inf
        fit_info[f'P{axis}_inf'] = a_inf
        fit_info[f'C_{axis}'] = C
        fit_info[f'beta_{axis}'] = beta

    if verbose:
        print(f"  ---")
        Nx = V / alpha_inf[0, 0] if abs(alpha_inf[0, 0]) > 1e-15 else np.inf
        Ny = V / alpha_inf[1, 1] if abs(alpha_inf[1, 1]) > 1e-15 else np.inf
        Nz = V / alpha_inf[2, 2] if abs(alpha_inf[2, 2]) > 1e-15 else np.inf
        print(f"  converged: Nx={Nx:.4f}  Ny={Ny:.4f}  Nz={Nz:.4f}  "
              f"sum={Nx+Ny+Nz:.4f}")

    return alpha_inf, alpha_list, fit_info


# ═══════════════════════════════════════════════════════════════════
#  PEC limit by er extrapolation
# ═══════════════════════════════════════════════════════════════════

def pec_polarizability(sx, sy, sz, er_levels=None, d_levels=None, verbose=True):
    """Extrapolate polarizability to the PEC limit (er -> inf).

    For each er, the d -> 0 extrapolation is performed first.
    Then alpha(er) = alpha_PEC + A / er  is fitted.

    Parameters
    ----------
    sx, sy, sz : float
    er_levels : list of float or None
        If None, uses [2, 5, 10, 100, 10000].
    d_levels : list of float or None
        Passed to converged_polarizability.

    Returns
    -------
    alpha_pec : (3, 3) ndarray
    """
    if er_levels is None:
        er_levels = [2.0, 5.0, 10.0, 100.0, 10000.0]

    V = sx * sy * sz
    if verbose:
        print(f"=== PEC extrapolation (DDA) ===")
        print(f"  shape: {sx:.2f}x{sy:.2f}x{sz:.2f}")
        print(f"  er levels: {er_levels}")

    alpha_inf_list = []
    for er in er_levels:
        a_inf, _, _ = converged_polarizability(er, sx, sy, sz,
                                               d_levels=d_levels,
                                               verbose=False)
        alpha_inf_list.append(a_inf)
        if verbose:
            Nx = V / a_inf[0, 0] if abs(a_inf[0, 0]) > 1e-15 else np.inf
            Ny = V / a_inf[1, 1] if abs(a_inf[1, 1]) > 1e-15 else np.inf
            Nz = V / a_inf[2, 2] if abs(a_inf[2, 2]) > 1e-15 else np.inf
            print(f"  er={er:8.1f}  Nx={Nx:.4f}  Ny={Ny:.4f}  Nz={Nz:.4f}  "
                  f"sum={Nx+Ny+Nz:.4f}")

    # Fit alpha(er) = alpha_PEC + A / er  for each diagonal component
    inv_er = 1.0 / np.array(er_levels)
    alpha_pec = np.zeros((3, 3))

    for k in range(3):
        y_vals = np.array([a[k, k] for a in alpha_inf_list])
        coeffs = np.polyfit(inv_er, y_vals, 1)
        alpha_pec[k, k] = coeffs[1]  # intercept = alpha(er -> inf)

    if verbose:
        print(f"  ---")
        Nx = V / alpha_pec[0, 0] if abs(alpha_pec[0, 0]) > 1e-15 else np.inf
        Ny = V / alpha_pec[1, 1] if abs(alpha_pec[1, 1]) > 1e-15 else np.inf
        Nz = V / alpha_pec[2, 2] if abs(alpha_pec[2, 2]) > 1e-15 else np.inf
        print(f"  PEC limit: Nx={Nx:.4f}  Ny={Ny:.4f}  Nz={Nz:.4f}  "
              f"sum={Nx+Ny+Nz:.4f}")

    return alpha_pec


# ═══════════════════════════════════════════════════════════════════
#  Quick test
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=== dda_core self-test ===")
    print()

    # Small cube test
    print("--- Cube 1x1x1, er=10, d=0.2 (N=125 dipoles) ---")
    alpha, info = compute_polarizability(er=10.0, sx=1.0, sy=1.0, sz=1.0,
                                         d=0.2, verbose=True)
    V = 1.0
    for i, ax in enumerate(['x', 'y', 'z']):
        Ni = V / alpha[i, i]
        print(f"  N_{ax} = {Ni:.4f}")
    print(f"  N_sum = {V/alpha[0,0] + V/alpha[1,1] + V/alpha[2,2]:.4f}")


# ═══════════════════════════════════════════════════════════════════
#  Phase 2: FFT-accelerated GMRES
# ═══════════════════════════════════════════════════════════════════


def _build_green_kernel(nx, ny, nz, dx, dy, dz, eps0=1.0):
    """Green's dyadic on (2*nx-1, 2*ny-1, 2*nz-1) grid, zero at origin."""
    kx, ky, kz = 2*nx - 1, 2*ny - 1, 2*nz - 1
    xs = (np.arange(kx) - (nx - 1)) * dx
    ys = (np.arange(ky) - (ny - 1)) * dy
    zs = (np.arange(kz) - (nz - 1)) * dz
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    R2 = X**2 + Y**2 + Z**2
    R = np.sqrt(R2)
    cx, cy, cz = nx - 1, ny - 1, nz - 1
    R[cx, cy, cz] = 1.0
    R3 = R**3
    factor = 1.0 / (4.0 * np.pi * eps0)
    Xh, Yh, Zh = X / R, Y / R, Z / R
    Gxx = factor * (3.0*Xh*Xh - 1.0) / R3
    Gyy = factor * (3.0*Yh*Yh - 1.0) / R3
    Gzz = factor * (3.0*Zh*Zh - 1.0) / R3
    Gxy = factor * (3.0*Xh*Yh) / R3
    Gxz = factor * (3.0*Xh*Zh) / R3
    Gyz = factor * (3.0*Yh*Zh) / R3
    for G in [Gxx, Gyy, Gzz, Gxy, Gxz, Gyz]:
        G[cx, cy, cz] = 0.0
    return Gxx, Gxy, Gxz, Gyy, Gyz, Gzz


def _fft_pad_green(G, fft_shape):
    """Place Green's kernel (centred at origin) into FFT array.

    G has shape (2*nx-1, 2*ny-1, 2*nz-1) with centre at (cx,cy,cz).
    Positive shifts [0, nx-1] go to pad[0:nx], negative go to pad end.
    """
    kx, ky, kz = G.shape
    nx = (kx + 1) // 2
    ny = (ky + 1) // 2
    nz = (kz + 1) // 2
    Nx, Ny, Nz = fft_shape
    G_pad = np.zeros(fft_shape, dtype=float)

    c = G[nx-1:, ny-1:, nz-1:]      # centre + all positive
    G_pad[:nx, :ny, :nz] = c

    a = G[:nx-1, ny-1:, nz-1:]      # neg x, pos y,z
    G_pad[Nx-(nx-1):Nx, :ny, :nz] = a

    b = G[nx-1:, :ny-1, nz-1:]      # pos x, neg y, pos z
    G_pad[:nx, Ny-(ny-1):Ny, :nz] = b

    d = G[nx-1:, ny-1:, :nz-1]      # pos x,y, neg z
    G_pad[:nx, :ny, Nz-(nz-1):Nz] = d

    ab = G[:nx-1, :ny-1, nz-1:]     # neg x,y, pos z
    G_pad[Nx-(nx-1):Nx, Ny-(ny-1):Ny, :nz] = ab

    ac = G[:nx-1, ny-1:, :nz-1]     # neg x, pos y, neg z
    G_pad[Nx-(nx-1):Nx, :ny, Nz-(nz-1):Nz] = ac

    bc = G[nx-1:, :ny-1, :nz-1]     # pos x, neg y, neg z
    G_pad[:nx, Ny-(ny-1):Ny, Nz-(nz-1):Nz] = bc

    abc = G[:nx-1, :ny-1, :nz-1]    # all negative
    G_pad[Nx-(nx-1):Nx, Ny-(ny-1):Ny, Nz-(nz-1):Nz] = abc

    return np.fft.rfftn(G_pad)

def _make_matvec_fft(nx, ny, nz, dx, dy, dz, inv_alpha, eps0=1.0):
    """Efficient matvec: pre-FFTs G kernels, FFTs p once per matvec call."""
    Gxx, Gxy, Gxz, Gyy, Gyz, Gzz = _build_green_kernel(
        nx, ny, nz, dx, dy, dz, eps0
    )
    # FFT size for linear convolution: at least 3*nx-2
    fft_shape = (3*nx - 2, 3*ny - 2, 3*nz - 2)

    # Pre-FFT all G kernels
    Gxx_fft = _fft_pad_green(Gxx, fft_shape)
    Gxy_fft = _fft_pad_green(Gxy, fft_shape)
    Gxz_fft = _fft_pad_green(Gxz, fft_shape)
    Gyy_fft = _fft_pad_green(Gyy, fft_shape)
    Gyz_fft = _fft_pad_green(Gyz, fft_shape)
    Gzz_fft = _fft_pad_green(Gzz, fft_shape)

    def matvec(p_vec):
        N = nx * ny * nz
        px = p_vec[0::3].reshape(nx, ny, nz)
        py = p_vec[1::3].reshape(nx, ny, nz)
        pz = p_vec[2::3].reshape(nx, ny, nz)

        # Pad p to fft_shape
        px_pad = np.zeros(fft_shape, dtype=float)
        py_pad = np.zeros(fft_shape, dtype=float)
        pz_pad = np.zeros(fft_shape, dtype=float)
        px_pad[:nx, :ny, :nz] = px
        py_pad[:nx, :ny, :nz] = py
        pz_pad[:nx, :ny, :nz] = pz

        # FFT p components (once each)
        px_fft = np.fft.rfftn(px_pad)
        py_fft = np.fft.rfftn(py_pad)
        pz_fft = np.fft.rfftn(pz_pad)

        # Multiply in freq domain (6 G FFTs reused)
        rx_fft = Gxx_fft * px_fft + Gxy_fft * py_fft + Gxz_fft * pz_fft
        ry_fft = Gxy_fft * px_fft + Gyy_fft * py_fft + Gyz_fft * pz_fft
        rz_fft = Gxz_fft * px_fft + Gyz_fft * py_fft + Gzz_fft * pz_fft

        # IFFT and crop dipole region
        rx = np.fft.irfftn(rx_fft, s=fft_shape)[:nx, :ny, :nz]
        ry = np.fft.irfftn(ry_fft, s=fft_shape)[:nx, :ny, :nz]
        rz = np.fft.irfftn(rz_fft, s=fft_shape)[:nx, :ny, :nz]

        result = np.empty(3 * N, dtype=float)
        result[0::3] = inv_alpha * px.ravel() - rx.ravel()
        result[1::3] = inv_alpha * py.ravel() - ry.ravel()
        result[2::3] = inv_alpha * pz.ravel() - rz.ravel()
        return result

    return matvec
def compute_polarizability_fft(er, sx, sy, sz, d, verbose=False,
                               gmres_rtol=1e-6, gmres_maxiter=500):
    """Compute polarizability tensor using FFT-accelerated GMRES.

    Much faster than dense solver for N > ~1000 dipoles.

    Parameters
    ----------
    er : float
    sx, sy, sz : float    cuboid dimensions
    d : float             target lattice spacing
    verbose : bool
    gmres_rtol : float    GMRES relative tolerance
    gmres_maxiter : int   GMRES max iterations

    Returns
    -------
    alpha : (3, 3) ndarray
    info : dict
    """
    V = sx * sy * sz
    r, dx, dy, dz = make_lattice(sx, sy, sz, d)
    nx = int(round(sx / dx))
    ny = int(round(sy / dy))
    nz = int(round(sz / dz))
    N = nx * ny * nz

    alpha_val = alpha_cm(er, float(np.mean([dx, dy, dz])))
    inv_alpha = 1.0 / alpha_val

    t0 = time.time()
    matvec = _make_matvec_fft(nx, ny, nz, dx, dy, dz, inv_alpha)
    t_build = time.time() - t0

    if verbose:
        print(f"  dda_fft: er={er:.1f} V={V:.3f} d~{d:.4f} "
              f"N={N} dim={3*N} build={t_build:.1f}s", end="")

    from scipy.sparse.linalg import LinearOperator, gmres

    def make_rhs(e0):
        bb = np.zeros(3 * N, dtype=float)
        bb[0::3] = e0[0]
        bb[1::3] = e0[1]
        bb[2::3] = e0[2]
        return bb

    alpha_tensor = np.zeros((3, 3), dtype=float)
    t_solve = 0.0
    A_op = LinearOperator((3*N, 3*N), matvec=matvec, dtype=float)

    e0_list = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    for k, e0 in enumerate(e0_list):
        b = make_rhs(e0)
        t1 = time.time()
        x, info = gmres(A_op, b, rtol=gmres_rtol, maxiter=gmres_maxiter,
                        atol=1e-12)
        t_solve += time.time() - t1
        if info > 0 and verbose:
            print(f"  [WARN: gmres stagnated iters={info}]", end="")
        P = x.reshape(-1, 3).sum(axis=0)
        alpha_tensor[:, k] = P

    t_total = time.time() - t0

    if verbose:
        print(f"  solve={t_solve:.1f}s  total={t_total:.1f}s")

    info = dict(d=float(np.cbrt(dx * dy * dz)), N=N, time=t_total,
                er=er, V=V, dx=dx, dy=dy, dz=dz)
    return alpha_tensor, info

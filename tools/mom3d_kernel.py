"""
mom3d_kernel.py  —  Matrix-free kernel operator + GMRES for 3D MoM.

Precomputes the 8 octant kernel NxN arrays once, then uses them in
matvecs.  This is still O(N^2) memory but enables iterative GMRES.

For true O(N log N) scaling, replace the dense matmul with ACA or FMM.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres


def _diag_preconditioner(ns, Z1):
    """Return a LinearOperator for M^{-1} = (1/Z1) * I."""
    def matvec(v):
        return v / Z1
    return LinearOperator((ns, ns), matvec=matvec, dtype=complex)


def _meshgrid_all(r, n, ds):
    """Build the shared NxN arrays once (expensive, O(N^2) memory)."""
    ns = len(r.x)
    rn_x, rm_x = np.meshgrid(r.x, r.x)
    rn_y, rm_y = np.meshgrid(r.y, r.y)
    rn_z, rm_z = np.meshgrid(r.z, r.z)
    nn_x, _ = np.meshgrid(n.x, n.x)
    nn_y, _ = np.meshgrid(n.y, n.y)
    nn_z, _ = np.meshgrid(n.z, n.z)
    dn, _ = np.meshgrid(ds, ds)
    I_mat = np.eye(ns)
    In = 1.0 - I_mat
    return dict(rn_x=rn_x, rn_y=rn_y, rn_z=rn_z,
                rm_x=rm_x, rm_y=rm_y, rm_z=rm_z,
                nn_x=nn_x, nn_y=nn_y, nn_z=nn_z,
                dn=dn, I_mat=I_mat, In=In, ns=ns)


def _octant_matvec_operator(mg, er, direction):
    """LinearOperator for Z @ x in octant mode (vectorised matmul)."""
    ns = mg['ns']
    Z1 = (er + 1.0) / 2.0
    f_k = (er - 1.0) / (4.0 * np.pi)

    sx_all = np.array([1, -1, 1, 1, -1, -1, 1, -1])
    sy_all = np.array([1, 1, -1, 1, -1, 1, -1, -1])
    sz_all = np.array([1, 1, 1, -1, 1, -1, -1, -1])
    s_acc = {'x': sx_all, 'y': sy_all, 'z': sz_all}[direction]

    kernels = []
    for oct_idx in range(8):
        sx, sy, sz = sx_all[oct_idx], sy_all[oct_idx], sz_all[oct_idx]
        vx = mg['rm_x'] - sx * mg['rn_x']
        vy = mg['rm_y'] - sy * mg['rn_y']
        vz = mg['rm_z'] - sz * mg['rn_z']
        r2 = vx * vx + vy * vy + vz * vz
        r3 = np.abs(r2) ** 1.5
        dot_v = (vx * (sx * mg['nn_x']) +
                 vy * (sy * mg['nn_y']) +
                 vz * (sz * mg['nn_z']))
        K = mg['dn'] * dot_v / r3
        K = np.nan_to_num(K) * mg['In']
        kernels.append(K)

    def matvec(x):
        out = Z1 * x
        for oct_idx in range(8):
            out += f_k * s_acc[oct_idx] * (kernels[oct_idx] @ x)
        return out

    return LinearOperator((ns, ns), matvec=matvec, dtype=complex)


def _full_matvec_operator(mg, er):
    """LinearOperator for the full-mesh Z @ x."""
    ns = mg['ns']
    Z1 = (er + 1.0) / 2.0
    f_k = (er - 1.0) / (4.0 * np.pi)

    rho_x = mg['rm_x'] - mg['rn_x']
    rho_y = mg['rm_y'] - mg['rn_y']
    rho_z = mg['rm_z'] - mg['rn_z']
    rhomn = np.sqrt(rho_x**2 + rho_y**2 + rho_z**2)
    dot_rho_n = rho_x * mg['nn_x'] + rho_y * mg['nn_y'] + rho_z * mg['nn_z']
    rhomn3_safe = np.where(mg['In'] > 0, rhomn**3, 1.0)
    K = mg['dn'] * dot_rho_n / rhomn3_safe
    K = np.nan_to_num(K) * mg['In']

    def matvec(x):
        return Z1 * x + f_k * (K @ x)

    return LinearOperator((ns, ns), matvec=matvec, dtype=complex)


def solve_octant_gmres(r, n, ds, er, tol=1e-6, maxiter=None, verbose=False):
    """Solve octant MoM with GMRES (diagonally preconditioned)."""
    ns = len(r.x)
    if maxiter is None:
        maxiter = min(ns, 500)
    mg = _meshgrid_all(r, n, ds)
    Z1 = (er + 1.0) / 2.0
    M = _diag_preconditioner(ns, Z1)

    results = []
    for direction in ['x', 'y', 'z']:
        op = _octant_matvec_operator(mg, er, direction)
        rhs = {'x': -r.x, 'y': -r.y, 'z': -r.z}[direction]
        x_sol, info = gmres(op, rhs, M=M, rtol=tol, maxiter=maxiter, atol=1e-14)
        if verbose:
            if info > 0:
                print(f"    GMRES({direction}): {info} iters")
            elif info < 0:
                print(f"    GMRES({direction}): WARNING info={info}")
        results.append(x_sol)

    return results[0], results[1], results[2]


def solve_full_gmres(r, n, ds, er, tol=1e-6, maxiter=None, verbose=False):
    """Solve full-mesh MoM with GMRES."""
    ns = len(r.x)
    if maxiter is None:
        maxiter = min(ns, 500)
    mg = _meshgrid_all(r, n, ds)
    op = _full_matvec_operator(mg, er)
    Z1 = (er + 1.0) / 2.0
    M = _diag_preconditioner(ns, Z1)

    results = []
    for rhs in [-r.x, -r.y, -r.z]:
        x_sol, info = gmres(op, rhs, M=M, rtol=tol, maxiter=maxiter, atol=1e-14)
        if verbose and info > 0:
            print(f"    GMRES: {info} iters")
        results.append(x_sol)

    return results[0], results[1], results[2]

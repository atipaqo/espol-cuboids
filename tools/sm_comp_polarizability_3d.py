"""
sm_comp_polarizability_3d.py  —  3D MoM polarizability solver.

v2: scipy.lu_factor/lu_solve + multi-RHS (full mesh),
    optional ACA-compressed GMRES for large octant problems.

Supports full 6-face mesh and octant-symmetry mode.
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from vector_analysis import V3, D3, Mesh


def comp_polarizability_3d(mesh: Mesh, iterative=False, aca=False, tol=1e-6):
    """Compute the 3x3 polarizability tensor of a 3D dielectric object.

    Parameters
    ----------
    mesh : Mesh
        Surface mesh with r, n, ds, V, er, octant.
    iterative : bool
        If True, use dense-kernel GMRES instead of direct LU.
    aca : bool
        If True, use ACA-compressed H-matrix + GMRES (best for N > 2000).
        Overrides iterative.
    tol : float
        GMRES tolerance.

    Returns
    -------
    D3
        Polarizability tensor (.xx, .xy, ..., .zz).
    """
    r = mesh.r
    n = mesh.n
    ds = mesh.ds
    er = mesh.er
    Vo = mesh.V
    ns = mesh.ne

    if mesh.octant:
        # -----------------------------------------------------------
        #  OCTANT SOLVER
        # -----------------------------------------------------------
        if aca:
            from mom3d_aca import solve_octant_aca
            X1, X2, X3 = solve_octant_aca(r, n, ds, er, tol=tol,
                                          leaf_size=128, eta=0.5, eps_aca=1e-4)
        elif iterative:
            from mom3d_kernel import solve_octant_gmres
            X1, X2, X3 = solve_octant_gmres(r, n, ds, er, tol=tol)
        else:
            # --- direct LU (three different matrices) ---
            rn_x, rm_x = np.meshgrid(r.x, r.x)
            rn_y, rm_y = np.meshgrid(r.y, r.y)
            rn_z, rm_z = np.meshgrid(r.z, r.z)
            nn_x, _ = np.meshgrid(n.x, n.x)
            nn_y, _ = np.meshgrid(n.y, n.y)
            nn_z, _ = np.meshgrid(n.z, n.z)
            dn, _ = np.meshgrid(ds, ds)
            I_mat = np.eye(ns)
            In = 1.0 - I_mat
            Z1 = (er + 1.0) / 2.0
            f_k = (er - 1.0) / (4.0 * np.pi)

            sx_all = np.array([1, -1, 1, 1, -1, -1, 1, -1])
            sy_all = np.array([1, 1, -1, 1, -1, 1, -1, -1])
            sz_all = np.array([1, 1, 1, -1, 1, -1, -1, -1])

            Z2_x = np.zeros((ns, ns), dtype=complex)
            Z2_y = np.zeros((ns, ns), dtype=complex)
            Z2_z = np.zeros((ns, ns), dtype=complex)

            for sx, sy, sz in zip(sx_all, sy_all, sz_all):
                src_x = sx * rn_x; src_y = sy * rn_y; src_z = sz * rn_z
                src_nx = sx * nn_x; src_ny = sy * nn_y; src_nz = sz * nn_z
                vx = rm_x - src_x; vy = rm_y - src_y; vz = rm_z - src_z
                r2 = vx*vx + vy*vy + vz*vz
                r3 = np.abs(r2) ** 1.5
                dot_v = vx*src_nx + vy*src_ny + vz*src_nz
                Z2_x += sx * dn * dot_v / r3
                Z2_y += sy * dn * dot_v / r3
                Z2_z += sz * dn * dot_v / r3

            Z2_x = np.nan_to_num(Z2_x) * In
            Z2_y = np.nan_to_num(Z2_y) * In
            Z2_z = np.nan_to_num(Z2_z) * In

            Z_x = Z1 * I_mat + f_k * Z2_x
            Z_y = Z1 * I_mat + f_k * Z2_y
            Z_z = Z1 * I_mat + f_k * Z2_z

            X1 = np.linalg.solve(Z_x, -r.x)
            X2 = np.linalg.solve(Z_y, -r.y)
            X3 = np.linalg.solve(Z_z, -r.z)

        # Polarizability: factor 8 for all octants
        f_pol = -(er - 1.0)
        Px = f_pol * np.real(np.sum(X1 * n.x * ds))
        Py = f_pol * np.real(np.sum(X2 * n.y * ds))
        Pz = f_pol * np.real(np.sum(X3 * n.z * ds))

        P = D3.zeros()
        P.xx = 8.0 * Px
        P.yy = 8.0 * Py
        P.zz = 8.0 * Pz
        P.xy = 0.0; P.xz = 0.0
        P.yx = 0.0; P.yz = 0.0
        P.zx = 0.0; P.zy = 0.0

    else:
        # -----------------------------------------------------------
        #  FULL MESH SOLVER  —  Option 3: single LU, multi-RHS
        # -----------------------------------------------------------
        rn_x, rm_x = np.meshgrid(r.x, r.x)
        rn_y, rm_y = np.meshgrid(r.y, r.y)
        rn_z, rm_z = np.meshgrid(r.z, r.z)
        nn_x, _ = np.meshgrid(n.x, n.x)
        nn_y, _ = np.meshgrid(n.y, n.y)
        nn_z, _ = np.meshgrid(n.z, n.z)
        dn, _ = np.meshgrid(ds, ds)
        I_mat = np.eye(ns)
        In = 1.0 - I_mat

        rho_x = rm_x - rn_x
        rho_y = rm_y - rn_y
        rho_z = rm_z - rn_z
        rhomn = np.sqrt(rho_x**2 + rho_y**2 + rho_z**2)
        dot_rho_n = rho_x * nn_x + rho_y * nn_y + rho_z * nn_z
        rhomn3_safe = np.where(In > 0, rhomn**3, 1.0)

        Z1 = (er + 1.0) / 2.0
        Z = Z1 * I_mat + (er - 1.0) / (4.0 * np.pi) * dn * dot_rho_n / rhomn3_safe * In

        if iterative:
            from mom3d_kernel import solve_full_gmres
            X1, X2, X3 = solve_full_gmres(r, n, ds, er, tol=tol)
        else:
            # Option 3: factor once, solve 3 RHS
            Y = np.column_stack([-r.x, -r.y, -r.z])
            X = np.linalg.solve(Z, Y)
            X1, X2, X3 = X[:, 0], X[:, 1], X[:, 2]

        # Polarizability
        f_pol = -(er - 1.0)
        P = D3.zeros()
        P.xx = f_pol * np.real(np.sum(X1 * n.x * ds))
        P.xy = f_pol * np.real(np.sum(X2 * n.x * ds))
        P.xz = f_pol * np.real(np.sum(X3 * n.x * ds))
        P.yx = f_pol * np.real(np.sum(X1 * n.y * ds))
        P.yy = f_pol * np.real(np.sum(X2 * n.y * ds))
        P.yz = f_pol * np.real(np.sum(X3 * n.y * ds))
        P.zx = f_pol * np.real(np.sum(X1 * n.z * ds))
        P.zy = f_pol * np.real(np.sum(X2 * n.z * ds))
        P.zz = f_pol * np.real(np.sum(X3 * n.z * ds))

    return P

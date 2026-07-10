"""
mom2d.py  —  2D MoM polarizability solver.

Python equivalent of MATLAB EM/sm_compPolarizability2d.m
"""

import numpy as np
from .vector_analysis import V3, D3, Mesh


def comp_polarizability_2d(mesh: Mesh):
    """Compute the 2×2 polarizability tensor of a 2D prismatic object.

    Parameters
    ----------
    mesh : Mesh
        Perimeter mesh with r, n, ds, A, er.

    Returns
    -------
    D3
        Polarizability tensor (xx, xy, yx, yy, zz).
    """
    r = mesh.r       # V3
    n = mesh.n       # V3
    ds = mesh.ds     # 1D array
    er = mesh.er
    Ao = mesh.V      # area is stored in V field for 2D

    ns = mesh.ne

    # ---- build meshgrid matrices ----
    rn_x, rm_x = np.meshgrid(r.x, r.x)   # rn=source (1st), rm=obs (2nd)
    rn_y, rm_y = np.meshgrid(r.y, r.y)
    rn_z, rm_z = np.meshgrid(r.z, r.z)

    nn_x, _ = np.meshgrid(n.x, n.x)      # nn=source normal (1st)
    nn_y, _ = np.meshgrid(n.y, n.y)
    nn_z, _ = np.meshgrid(n.z, n.z)
    dn, _ = np.meshgrid(ds, ds)           # dn=source area (1st)

    rm = V3(rm_x, rm_y, rm_z)            # observation
    rn = V3(rn_x, rn_y, rn_z)            # source
    nn = V3(nn_x, nn_y, nn_z)

    # ---- MoM matrix ----
    I_mat = np.eye(ns)
    In = 1.0 - I_mat

    rho = rm - rn
    rhomn = np.abs(rho.mag())
    dot_rho_n = rho.dot(nn)

    # compute Z2 safely: mask diagonal to avoid 0/0
    rhomn_safe = np.where(In > 0, rhomn**2, 1.0)   # avoid /0 on diagonal
    Z2 = (er - 1.0) / (2.0 * np.pi) * dn * dot_rho_n / rhomn_safe

    Z1 = (er + 1.0) / 2.0
    Z = Z1 * I_mat + Z2 * In          # In zeros the diagonal of Z2

    # ---- RHS ----
    Y1 = -r.x
    Y2 = -r.y

    # ---- solve ----
    X1 = np.linalg.solve(Z, Y1)
    X2 = np.linalg.solve(Z, Y2)

    # ---- polarizability (take real part) ----
    P = D3.zeros()
    P.xx = -(er - 1.0) * np.sum(X1 * n.x * ds).real
    P.xy = -(er - 1.0) * np.sum(X2 * n.x * ds).real
    P.yx = -(er - 1.0) * np.sum(X1 * n.y * ds).real
    P.yy = -(er - 1.0) * np.sum(X2 * n.y * ds).real
    P.zz = -(er - 1.0) * (-Ao)          # 2D area response

    return P

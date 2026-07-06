"""
gen_mesh_3d_rect.py  —  3D cuboid surface mesh generator.

Python equivalent of MATLAB Geometry/genMesh3DRect.m
Supports full (6-face) and octant (3-patch) modes.
"""
import _path_setup

import numpy as np
from vector_analysis import V3, D3


def gen_mesh_3d_rect(lx, ly, lz, nd, beta=1.0, octant=True):
    """Generate a 3D surface mesh for a cuboid [lx × ly × lz].

    Parameters
    ----------
    lx, ly, lz : float
        Cuboid dimensions.
    nd : int
        Points per edge (x-direction, full face).
    beta : float
        Spacing bias: 0 = uniform, 1 = sin clustering.
    octant : bool
        If True, mesh only first octant (3 patches) for symmetry solver.

    Returns
    -------
    r  : V3   —  element centre positions
    n  : V3   —  outward unit normals
    ds : ndarray —  element areas
    ne : int  —  number of elements
    V  : float —  cuboid volume
    """
    def spacing(n):
        """Generate n midpoints and widths in [-0.5, 0.5]."""
        ii = np.linspace(-n / 2, n / 2, n + 1)
        xx_unif = ii / n
        xx_clust = np.sin(ii * np.pi / n) / 2
        xx = (1 - beta) * xx_unif + beta * xx_clust
        ddx = np.diff(xx)
        xx = xx[:-1] + ddx / 2
        return xx, ddx

    def build_face(aa, bb, dda, ddb, offset, normal):
        """Build a rectangular face patch.

        aa, bb  : 1D arrays of in-plane midpoints
        dda, ddb: 1D arrays of element widths
        offset  : float, constant out-of-plane coordinate
        normal  : 'x', 'y', or 'z'
        """
        na, nb = len(aa), len(bb)
        ne = na * nb
        r = V3.zeros(ne)
        n_arr = V3.zeros(ne)
        ds = np.zeros(ne)

        AA, BB = np.meshgrid(aa, bb)
        DDA, DDB = np.meshgrid(dda, ddb)
        ds[:] = (DDA * DDB).ravel()

        if normal == 'z':
            r.x = AA.ravel()
            r.y = BB.ravel()
            r.z = np.full(ne, offset)
            n_arr.z = np.ones(ne)
        elif normal == 'x':
            r.x = np.full(ne, offset)
            r.y = AA.ravel()
            r.z = BB.ravel()
            n_arr.x = np.ones(ne)
        elif normal == 'y':
            r.x = AA.ravel()
            r.y = np.full(ne, offset)
            r.z = BB.ravel()
            n_arr.y = np.ones(ne)

        return r, n_arr, ds

    if octant:
        # ---- first-octant patches only ----
        xx, ddx = spacing(nd)
        mask = xx > 1e-12
        xxp = xx[mask]
        ddxp = ddx[mask]
        ddxp = ddxp * (0.5 / np.sum(ddxp))   # renormalize to half-line
        xxp[0] = ddxp[0] / 2
        for ii in range(1, len(xxp)):
            xxp[ii] = xxp[ii - 1] + ddxp[ii - 1] / 2 + ddxp[ii] / 2

        r1, n1, ds1 = build_face(xxp, xxp, ddxp, ddxp, +0.5, 'z')
        r2, n2, ds2 = build_face(xxp, xxp, ddxp, ddxp, +0.5, 'x')
        r3, n3, ds3 = build_face(xxp, xxp, ddxp, ddxp, +0.5, 'y')

        r = V3(
            np.concatenate([r1.x, r2.x, r3.x]),
            np.concatenate([r1.y, r2.y, r3.y]),
            np.concatenate([r1.z, r2.z, r3.z]),
        )
        n = V3(
            np.concatenate([n1.x, n2.x, n3.x]),
            np.concatenate([n1.y, n2.y, n3.y]),
            np.concatenate([n1.z, n2.z, n3.z]),
        )
        ds_raw = np.concatenate([ds1, ds2, ds3])
    else:
        # ---- all 6 faces ----
        xx, ddx = spacing(nd)

        r1, n1, ds1 = build_face(xx, xx, ddx, ddx, +0.5, 'z')
        r2, n2, ds2 = build_face(xx, xx, ddx, ddx, +0.5, 'x')
        r3, n3, ds3 = build_face(xx, xx, ddx, ddx, +0.5, 'y')

        T180 = D3.rot180('z')
        r1m, n1m = T180 @ r1, T180 @ n1   # 180° about z → opposite face?
        # Actually in MATLAB, rotate180(r1,n1,'x') rotates about x-axis
        # Let me match the MATLAB exactly:
        Tx = D3.rot180('x')
        Ty = D3.rot180('y')
        Tz = D3.rot180('z')

        r1m, n1m = Tx @ r1, Tx @ n1    # opposite z-face via x-rotation
        r2m, n2m = Ty @ r2, Ty @ n2    # opposite x-face via y-rotation
        r3m, n3m = Tz @ r3, Tz @ n3    # opposite y-face via z-rotation

        r = V3(
            np.concatenate([r1.x, r1m.x, r2.x, r2m.x, r3.x, r3m.x]),
            np.concatenate([r1.y, r1m.y, r2.y, r2m.y, r3.y, r3m.y]),
            np.concatenate([r1.z, r1m.z, r2.z, r2m.z, r3.z, r3m.z]),
        )
        n = V3(
            np.concatenate([n1.x, n1m.x, n2.x, n2m.x, n3.x, n3m.x]),
            np.concatenate([n1.y, n1m.y, n2.y, n2m.y, n3.y, n3m.y]),
            np.concatenate([n1.z, n1m.z, n2.z, n2m.z, n3.z, n3m.z]),
        )
        ds_raw = np.concatenate([ds1, ds1, ds2, ds2, ds3, ds3])

    # scale to physical dimensions
    r.x = r.x * lx
    r.y = r.y * ly
    r.z = r.z * lz

    # element areas
    if octant:
        ds = np.concatenate([
            ds1 * lx * ly,
            ds2 * ly * lz,
            ds3 * lx * lz,
        ])
    else:
        ds = np.concatenate([
            ds1 * lx * ly, ds1 * lx * ly,
            ds2 * ly * lz, ds2 * ly * lz,
            ds3 * lx * lz, ds3 * lx * lz,
        ])

    ne = r.size
    V = lx * ly * lz

    return r, n, ds, ne, V

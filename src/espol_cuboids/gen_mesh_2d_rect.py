"""
gen_mesh_2d_rect.py  —  2D rectangular boundary mesh generator.

Python equivalent of MATLAB Geometry/genMesh2DRect.m
"""

import numpy as np
from .vector_analysis import V3, D3


def gen_mesh_2d_rect(lx, ly, nd, beta=1.0):
    """Generate a 2D perimeter mesh for a rectangle [lx × ly].

    Parameters
    ----------
    lx, ly : float
        Rectangle dimensions.
    nd : int
        Approximate number of points per edge (x-direction).
    beta : float
        Spacing bias: 0 = uniform, 1 = sin-clustering (dense at corners).

    Returns
    -------
    Mesh  (see vector_analysis.Mesh)
        r   : V3  —  element centre positions
        n   : V3  —  outward unit normals
        ds  : ndarray —  element lengths
        ne  : int —  number of elements
        A   : float —  rectangle area (lx * ly)
    """
    ar = ly / lx
    nd2 = max(2, round(nd * ar))          # vertical edge count

    # ---- spacing function (x-direction) ----
    def spacing(n):
        ii = np.linspace(-n/2, n/2, n + 1)
        xx_unif = ii / n                     # uniform, [-0.5, 0.5]
        xx_clust = np.sin(ii * np.pi / n) / 2  # sin-clustered
        xx = (1 - beta) * xx_unif + beta * xx_clust
        ddx = np.diff(xx)
        xx = xx[:-1] + ddx / 2               # element midpoints
        return xx, ddx

    xx, ddx = spacing(nd)
    yy, ddy = spacing(nd2)

    # ---- bottom edge (y = +ly/2, outward normal +y) ----
    r1 = V3.zeros(nd)
    r1.x = xx
    r1 = r1 + V3(0, 0.5, 0)
    n1 = V3.zeros(nd)
    n1.y = np.ones(nd)

    # ---- top edge (180° rotation about z) ----
    T180 = D3.rot180('z')
    r2 = T180 @ r1
    n2 = T180 @ n1

    # ---- left edge (x = +lx/2, outward normal +x) ----
    r3 = V3.zeros(nd2)
    r3.y = yy
    r3 = r3 + V3(0.5, 0, 0)
    n3 = V3.zeros(nd2)
    n3.x = np.ones(nd2)

    # ---- right edge (180° rotation about z) ----
    r4 = T180 @ r3
    n4 = T180 @ n3

    # ---- assemble all four edges ----
    r = V3(
        np.concatenate([r1.x, r2.x, r3.x, r4.x]),
        np.concatenate([r1.y, r2.y, r3.y, r4.y]),
        np.concatenate([r1.z, r2.z, r3.z, r4.z]),
    )

    n = V3(
        np.concatenate([n1.x, n2.x, n3.x, n4.x]),
        np.concatenate([n1.y, n2.y, n3.y, n4.y]),
        np.concatenate([n1.z, n2.z, n3.z, n4.z]),
    )

    # element lengths
    ds = np.concatenate([
        ddx * lx,       # bottom edge
        ddx * lx,       # top edge
        ddy * ly,       # left edge
        ddy * ly,       # right edge
    ])

    # scale to physical dimensions
    r.x = r.x * lx
    r.y = r.y * ly

    ne = r.size
    A = lx * ly

    return r, n, ds, ne, A

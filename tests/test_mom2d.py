"""Tests for MoM 2D mesh generation and solver."""
import pytest
import numpy as np
import sys, os
from espol_cuboids.gen_mesh_2d_rect import gen_mesh_2d_rect
from espol_cuboids.mom2d import comp_polarizability_2d
from espol_cuboids.vector_analysis import Mesh


class TestMesh2D:
    def test_square_area(self):
        r, n, ds, ne, Ao = gen_mesh_2d_rect(2.0, 3.0, nd=20, beta=0.0)
        assert Ao == pytest.approx(6.0) and ne > 0

    def test_ne_scales(self):
        _, _, _, ne1, _ = gen_mesh_2d_rect(1.0, 1.0, nd=20, beta=0.0)
        _, _, _, ne2, _ = gen_mesh_2d_rect(1.0, 1.0, nd=40, beta=0.0)
        assert ne2 > ne1

    def test_perimeter(self):
        lx, ly = 1.0, 2.0
        r, n, ds, ne, Ao = gen_mesh_2d_rect(lx, ly, nd=100, beta=0.0)
        assert np.sum(ds) == pytest.approx(2*(lx+ly), rel=0.02)

    def test_normals(self):
        """All normals should be unit (±1 in x or y direction)."""
        r, n, ds, ne, Ao = gen_mesh_2d_rect(1.0, 1.0, nd=40, beta=0.0)
        # Each normal is either (±1, 0) or (0, ±1)
        for elem in range(ne):
            assert (abs(n.x[elem]) > 0.9 and abs(n.y[elem]) < 0.1) or                    (abs(n.y[elem]) > 0.9 and abs(n.x[elem]) < 0.1)

    def test_normals_outward(self):
        """Right-face normals point +x, bottom-face normals point -y."""
        r, n, ds, ne, Ao = gen_mesh_2d_rect(1.0, 1.0, nd=40, beta=0.0)
        # Right face: n.x == +1
        right = n.x > 0.9
        assert np.any(right)
        assert np.all(r.x[right] > 0.4)
        # Bottom face: n.y == -1
        bottom = n.y < -0.9
        assert np.any(bottom)
        assert np.all(r.y[bottom] < -0.4)


class TestSolver2D:
    @pytest.fixture(autouse=True)
    def _skip(self):
        pytest.importorskip("scipy")

    def test_square_symmetry(self):
        r, n, ds, ne, Ao = gen_mesh_2d_rect(1.0, 1.0, nd=40, beta=0.0)
        P = comp_polarizability_2d(Mesh(r, n, ds, V=Ao, er=10.0))
        assert P.xx.real == pytest.approx(P.yy.real, rel=1e-10)

    def test_er_monotonic(self):
        r, n, ds, ne, Ao = gen_mesh_2d_rect(1.0, 1.0, nd=40, beta=0.0)
        P_low = comp_polarizability_2d(Mesh(r, n, ds, V=Ao, er=1.5))
        P_high = comp_polarizability_2d(Mesh(r, n, ds, V=Ao, er=10.0))
        assert P_high.xx.real > P_low.xx.real

    def test_rectangle_ordering(self):
        """Long direction has larger polarizability."""
        r, n, ds, ne, Ao = gen_mesh_2d_rect(1.0, 2.0, nd=40, beta=0.0)
        P = comp_polarizability_2d(Mesh(r, n, ds, V=Ao, er=10.0))
        assert P.yy.real > P.xx.real

    def test_nd_convergence(self):
        r1, n1, ds1, _, Ao1 = gen_mesh_2d_rect(1.0, 1.0, nd=20, beta=0.0)
        r2, n2, ds2, _, Ao2 = gen_mesh_2d_rect(1.0, 1.0, nd=60, beta=0.0)
        P1 = comp_polarizability_2d(Mesh(r1, n1, ds1, V=Ao1, er=10.0))
        P2 = comp_polarizability_2d(Mesh(r2, n2, ds2, V=Ao2, er=10.0))
        assert abs(P2.xx.real - P1.xx.real) / abs(P1.xx.real) < 0.05


class TestRegression2D:
    @pytest.fixture(autouse=True)
    def _skip(self):
        pytest.importorskip("scipy")

    @pytest.mark.parametrize("er,expected", [
        (2,    0.6732),
        (5,    1.3184),
        (10,   1.7145),
        (100,  2.1003),
        (1e10, 2.1884),
    ])
    def test_square_P_normalized(self, er, expected):
        """Normalized P_xx/Ao at nd=60 within 5% of converged values."""
        r, n, ds, ne, Ao = gen_mesh_2d_rect(1.0, 1.0, nd=60, beta=0.0)
        P = comp_polarizability_2d(Mesh(r, n, ds, V=Ao, er=er))
        Pxx_norm = P.xx.real / Ao
        assert Pxx_norm == pytest.approx(expected, rel=0.05)

    def test_pec_square_N_sum(self):
        """PEC square: N_sum = Nx + Ny = 2 * Ao/P_xx."""
        r, n, ds, ne, Ao = gen_mesh_2d_rect(1.0, 1.0, nd=60, beta=0.0)
        P = comp_polarizability_2d(Mesh(r, n, ds, V=Ao, er=1e10))
        Nx = Ao / P.xx.real; Ny = Ao / P.yy.real
        assert (Nx + Ny) == pytest.approx(2 * 0.457, rel=0.05)

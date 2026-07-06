"""Tests for MoM 3D mesh generation and solver."""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tools'))
from gen_mesh_3d_rect import gen_mesh_3d_rect
from sm_comp_polarizability_3d import comp_polarizability_3d
from gen_mom_3d import converged_3d, adaptive_nd, fit_powerlaw
from vector_analysis import Mesh


class TestMesh3D:
    def test_volume(self):
        r, n, ds, ne, Vo = gen_mesh_3d_rect(2.0, 2.0, 2.0, nd=20, beta=0.0, octant=True)
        assert Vo == pytest.approx(8.0)  # total volume is always reported
        assert ne > 0

    def test_ne_scales(self):
        _, _, _, ne1, _ = gen_mesh_3d_rect(1.0, 1.0, 1.0, nd=20, beta=0.0, octant=True)
        _, _, _, ne2, _ = gen_mesh_3d_rect(1.0, 1.0, 1.0, nd=40, beta=0.0, octant=True)
        assert ne2 > ne1

    def test_normals_negative_face(self):
        """Octant: x=0 face elements should have nx < 0 (pointing outward from octant)."""
        r, n, ds, ne, Vo = gen_mesh_3d_rect(1.0, 1.0, 1.0, nd=20, beta=0.0, octant=True)
        x0 = np.abs(r.x) < 1e-6
        if np.any(x0):
            assert np.all(n.x[x0] < 0)

    def test_mesh_integral(self):
        """For octant: sum(n_i dot r_i * ds_i) = 3*Vo/8 (1/8 of full surface)."""
        r, n, ds, ne, Vo = gen_mesh_3d_rect(1.0, 1.0, 1.0, nd=20, beta=0.0, octant=True)
        integral = np.sum((r.x*n.x + r.y*n.y + r.z*n.z) * ds)
        assert integral == pytest.approx(3 * Vo / 8, rel=0.2)


class TestSolver3D:
    @pytest.fixture(autouse=True)
    def _skip(self):
        pytest.importorskip("scipy")

    def test_cube_symmetry(self):
        r, n, ds, ne, Vo = gen_mesh_3d_rect(1.0, 1.0, 1.0, nd=20, beta=0.0, octant=True)
        P = comp_polarizability_3d(Mesh(r, n, ds, V=Vo, er=10.0, octant=True))
        assert P.xx.real == pytest.approx(P.yy.real, rel=1e-10)
        assert P.xx.real == pytest.approx(P.zz.real, rel=1e-10)

    def test_er_monotonic(self):
        r, n, ds, ne, Vo = gen_mesh_3d_rect(1.0, 1.0, 1.0, nd=20, beta=0.0, octant=True)
        P_low = comp_polarizability_3d(Mesh(r, n, ds, V=Vo, er=1.5, octant=True))
        P_high = comp_polarizability_3d(Mesh(r, n, ds, V=Vo, er=10.0, octant=True))
        assert P_high.xx.real > P_low.xx.real

    def test_nd_convergence(self):
        r1, n1, ds1, _, Vo1 = gen_mesh_3d_rect(1.0, 1.0, 1.0, nd=12, beta=0.0, octant=True)
        r2, n2, ds2, _, Vo2 = gen_mesh_3d_rect(1.0, 1.0, 1.0, nd=24, beta=0.0, octant=True)
        P1 = comp_polarizability_3d(Mesh(r1, n1, ds1, V=Vo1, er=10.0, octant=True))
        P2 = comp_polarizability_3d(Mesh(r2, n2, ds2, V=Vo2, er=10.0, octant=True))
        assert abs(P2.xx.real - P1.xx.real) / abs(P1.xx.real) < 0.05

    def test_bar_ordering(self):
        """1x0.5x0.5 bar: longer axis (x) has larger polarizability (smaller N)."""
        r, n, ds, ne, Vo = gen_mesh_3d_rect(1.0, 0.5, 0.5, nd=20, beta=0.0, octant=True)
        P = comp_polarizability_3d(Mesh(r, n, ds, V=Vo, er=10.0, octant=True))
        assert P.xx.real > P.yy.real
        assert P.yy.real == pytest.approx(P.zz.real, rel=1e-10)


class TestConverged3D:
    @pytest.fixture(autouse=True)
    def _skip(self):
        pytest.importorskip("scipy")

    def test_cube_er10(self):
        nd, net = adaptive_nd(1.0, 1.0, 1.0)
        P = converged_3d(10.0, 1.0, 1.0, 1.0, nd, net)
        assert 1/P[0] == pytest.approx(0.4013, rel=0.01)

    def test_bar_er10(self):
        nd, net = adaptive_nd(1.0, 0.5, 0.5)
        P = converged_3d(10.0, 1.0, 0.5, 0.5, nd, net)
        assert 1/P[0] == pytest.approx(0.2758, rel=0.01)
        assert 1/P[1] == pytest.approx(0.4645, rel=0.01)

    def test_pec_cube(self):
        nd, net = adaptive_nd(1.0, 1.0, 1.0)
        P = converged_3d(1e10, 1.0, 1.0, 1.0, nd, net)
        assert 1/P[0] == pytest.approx(0.2786, rel=0.01)

    def test_powerlaw_fit(self):
        nd = np.array([10, 20, 30, 40, 50], dtype=float)
        P = 2.0 + 5.0 * nd**(-1.5)
        P_inf, _ = fit_powerlaw(P, nd)
        assert P_inf == pytest.approx(2.0, rel=0.01)


class TestACA:
    @pytest.fixture(autouse=True)
    def _skip(self):
        pytest.importorskip("scipy")

    def test_aca_matches_direct(self):
        r, n, ds, ne, Vo = gen_mesh_3d_rect(1.0, 1.0, 1.0, nd=16, beta=0.0, octant=True)
        P_dir = comp_polarizability_3d(Mesh(r, n, ds, V=Vo, er=10.0, octant=True),
                                       iterative=False, aca=False)
        P_aca = comp_polarizability_3d(Mesh(r, n, ds, V=Vo, er=10.0, octant=True),
                                       iterative=True, aca=True)
        assert P_dir.xx.real == pytest.approx(P_aca.xx.real, rel=0.05)


class TestNonOctant:
    @pytest.fixture(autouse=True)
    def _skip(self):
        pytest.importorskip("scipy")

    def test_full_volume(self):
        r, n, ds, ne, Vo = gen_mesh_3d_rect(1.0, 1.0, 1.0, nd=20, beta=0.0, octant=False)
        assert Vo == pytest.approx(1.0) and ne > 2000  # full mesh has many elements

    def test_full_vs_octant(self):
        r1, n1, ds1, ne1, Vo1 = gen_mesh_3d_rect(1.0, 1.0, 1.0, nd=20, beta=0.0, octant=True)
        r2, n2, ds2, ne2, Vo2 = gen_mesh_3d_rect(1.0, 1.0, 1.0, nd=20, beta=0.0, octant=False)
        P1 = comp_polarizability_3d(Mesh(r1, n1, ds1, V=Vo1, er=10.0, octant=True))
        P2 = comp_polarizability_3d(Mesh(r2, n2, ds2, V=Vo2, er=10.0, octant=False))
        assert P1.xx.real == pytest.approx(P2.xx.real, rel=0.1)

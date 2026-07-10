"""Tests for vector_analysis -- V3, D3, Mesh."""
import pytest
import numpy as np
import sys, os
from espol_cuboids.vector_analysis import V3, D3, Mesh


class TestV3:
    def test_zeros(self):
        v = V3.zeros(5)
        assert np.all(v.x == 0.0) and np.all(v.y == 0.0) and np.all(v.z == 0.0)
        assert v.x.shape == (5,)

    def test_broadcast(self):
        v = V3(1.0, 2.0, 3.0)
        assert v.x == 1.0 and v.y == 2.0 and v.z == 3.0
        v2 = V3(5.0)
        assert v2.x == 5.0 and v2.y == 5.0 and v2.z == 5.0

    def test_add_sub(self):
        a = V3(1.0, 2.0, 3.0); b = V3(4.0, 5.0, 6.0)
        c = a + b; d = a - b
        assert c.x == 5.0 and c.y == 7.0 and c.z == 9.0
        assert d.x == -3.0 and d.y == -3.0 and d.z == -3.0

    def test_mul_div(self):
        a = V3(2.0, 4.0, 6.0)
        b = a * 2.0; c = a / 2.0
        assert b.x == 4.0 and b.y == 8.0 and b.z == 12.0
        assert c.x == 1.0 and c.y == 2.0 and c.z == 3.0

    def test_dot(self):
        d = V3(1.0, 2.0, 3.0).dot(V3(4.0, 5.0, 6.0))
        assert abs(d - 32.0) < 1e-12

    def test_cross(self):
        c = V3(1.0, 0.0, 0.0).cross(V3(0.0, 1.0, 0.0))
        assert abs(c.x) < 1e-12 and abs(c.y) < 1e-12 and abs(c.z - 1.0) < 1e-12

    def test_norm(self):
        u = V3(3.0, 4.0, 0.0).norm()
        assert u.x == pytest.approx(0.6) and u.y == pytest.approx(0.8) and u.z == pytest.approx(0.0)

    def test_indexing(self):
        a = V3(np.array([1.0, 4.0]), np.array([2.0, 5.0]), np.array([3.0, 6.0]))
        assert a[0].x == 1.0 and a[0].y == 2.0 and a[0].z == 3.0
        assert a[1].x == 4.0 and a[1].y == 5.0 and a[1].z == 6.0


class TestD3:
    def test_zeros(self):
        d = D3.zeros()
        assert d.xx == 0.0 and d.xy == 0.0 and d.xz == 0.0

    def test_eye(self):
        d = D3.eye()
        assert d.xx == 1.0 and d.yy == 1.0 and d.zz == 1.0
        assert d.xy == 0.0 and d.xz == 0.0

    def test_add(self):
        c = D3.eye() + D3.eye()
        assert c.xx == 2.0 and c.yy == 2.0

    def test_scalar_mul(self):
        b = D3.eye() * 3.0
        assert b.xx == 3.0 and b.yy == 3.0 and b.zz == 3.0


class TestMesh:
    def test_construction(self):
        n = 10
        mesh = Mesh(V3.zeros(n), V3.zeros(n), np.ones(n), V=1.0, er=10.0, octant=True)
        assert mesh.ne == n and mesh.V == 1.0 and mesh.er == 10.0 and mesh.octant

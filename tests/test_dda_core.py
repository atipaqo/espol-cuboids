"""
test_dda_core.py — tests for dda_core.py DDA solver.

Validates lattice construction, Clausius-Mossotti polarizability,
and a small-scale DDA solve on a cube for symmetry and sanity checks.
"""
import pytest
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# These tests import dda_core — if it's not available or imports fail,
# the test is skipped rather than hard-failing.
try:
    from dda_core import (
        make_lattice, lattice_info, alpha_cm,
        compute_polarizability_fft,
    )
    DDA_AVAILABLE = True
except ImportError as e:
    DDA_AVAILABLE = False
    DDA_IMPORT_ERROR = str(e)


# ── Skip marker if DDA unavailable ─────────────────────────────────────────

dda_required = pytest.mark.skipif(not DDA_AVAILABLE, reason=f"dda_core unavailable: {DDA_IMPORT_ERROR if not DDA_AVAILABLE else ''}")


# ── Lattice tests ──────────────────────────────────────────────────────────

@pytest.mark.skipif(not DDA_AVAILABLE, reason="dda_core unavailable")
class TestLattice:
    """make_lattice and lattice_info correctness."""

    def test_cube_lattice_count(self):
        """A 1×1×1 cube at d=0.5 should have ~8 dipoles."""
        r, dx, dy, dz = make_lattice(1.0, 1.0, 1.0, 0.5)
        # d=0.5, nx=2, ny=2, nz=2 → 8 dipoles
        assert r.shape[0] == 8

    def test_lattice_count_formula(self):
        """n_total = round(Lx/d) * round(Ly/d) * round(Lz/d)."""
        sx, sy, sz = 2.0, 1.0, 0.5
        d = 0.25
        r, dx, dy, dz = make_lattice(sx, sy, sz, d)
        expected_nx = round(sx / d)
        expected_ny = round(sy / d)
        expected_nz = round(sz / d)
        assert r.shape[0] == expected_nx * expected_ny * expected_nz

    def test_lattice_positions_in_bounds(self):
        """All dipole positions should be within the cuboid."""
        sx, sy, sz = 1.0, 0.7, 0.3
        r, dx, dy, dz = make_lattice(sx, sy, sz, 0.2)
        assert np.all(r[:, 0] >= -sx/2 - 1e-10)
        assert np.all(r[:, 0] <= +sx/2 + 1e-10)
        assert np.all(r[:, 1] >= -sy/2 - 1e-10)
        assert np.all(r[:, 1] <= +sy/2 + 1e-10)
        assert np.all(r[:, 2] >= -sz/2 - 1e-10)
        assert np.all(r[:, 2] <= +sz/2 + 1e-10)

    def test_lattice_info(self):
        """lattice_info returns (r, dx, dy, dz) tuple from make_lattice."""
        r, dx, dy, dz = lattice_info(1.0, 1.0, 1.0, 0.5)
        assert r.shape == (8, 3)
        assert dx == 0.5 and dy == 0.5 and dz == 0.5


# ── Alpha CM tests ─────────────────────────────────────────────────────────

@pytest.mark.skipif(not DDA_AVAILABLE, reason="dda_core unavailable")
class TestAlphaCM:
    """Clausius-Mossotti dipole polarizability."""

    def test_vacuum(self):
        """εr=1 → α=0."""
        assert alpha_cm(1.0, 0.1) == pytest.approx(0.0, abs=1e-12)

    def test_pec(self):
        """εr→∞ → α = 3V_d."""
        d = 0.1
        Vd = d**3
        assert alpha_cm(1e10, d) == pytest.approx(3.0 * Vd, rel=1e-8)

    def test_positive(self):
        """α_cm > 0 for εr > 1."""
        assert alpha_cm(2.0, 0.1) > 0
        assert alpha_cm(10.0, 0.05) > 0


# ── DDA solve tests (small scale) ──────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.skipif(not DDA_AVAILABLE, reason="dda_core unavailable")

@pytest.mark.slow
@pytest.mark.skipif(not DDA_AVAILABLE, reason="dda_core unavailable")
class TestDDA:
    """Small-scale DDA solves for sanity checks."""

    def test_cube_symmetry_fft(self):
        """DDA on a cube should give α_xx = α_yy = α_zz."""
        alpha, info = compute_polarizability_fft(10.0, 1.0, 1.0, 1.0, 0.4)
        # alpha is a 3x3 matrix; extract diagonal
        assert alpha.shape == (3, 3)
        axx, ayy, azz = alpha[0,0], alpha[1,1], alpha[2,2]
        assert axx == pytest.approx(ayy, rel=0.05)
        assert ayy == pytest.approx(azz, rel=0.05)

    def test_alpha_positive_fft(self):
        """DDA polarizability diagonal should be positive for er > 1."""
        alpha, info = compute_polarizability_fft(5.0, 1.0, 1.0, 1.0, 0.4)
        for i in range(3):
            assert alpha[i, i] > 0

    def test_pec_increases(self):
        """α(er=20) > α(er=2) for the same shape."""
        alpha_low, _ = compute_polarizability_fft(2.0, 1.0, 1.0, 1.0, 0.4)
        alpha_high, _ = compute_polarizability_fft(20.0, 1.0, 1.0, 1.0, 0.4)
        for i in range(3):
            assert alpha_high[i, i] > alpha_low[i, i]

    def test_cube_near_sihvola(self):
        """DDA cube at er=10 should be within 15% of Sihvola's value."""
        alpha, info = compute_polarizability_fft(10.0, 1.0, 1.0, 1.0, 0.3)
        for i in range(3):
            a = alpha[i, i]
            # Coarse d=0.3, so allow 15% tolerance around Sihvola's ~2.515
            assert 2.0 < a < 3.0, f"α[{i},{i}]={a:.3f} not in [2.0, 3.0]"


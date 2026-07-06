"""
conftest.py — pytest configuration for DDA core tests.

Adds tools/ to sys.path.
"""
import sys
import os
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOLS = os.path.join(_ROOT, 'tools')
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: tests that run DDA solves or iterate (deselect with '-m \"not slow\"')"
    )


@pytest.fixture(scope="session")
def dda_module():
    """Provide the dda_core module."""
    import dda_core
    return dda_core


# ── Reference values from Sihvola (1994) cube polarizability ───────────────
SIHVOLA_CUBE = {
    0.5:   0.23445,
    1.25:  0.47687,
    1.5:   0.53775,
    1.75:  0.58906,
    2.0:   0.63350,
    3.0:   0.76125,
    5.0:   0.86850,
    8.0:   0.92525,
    10.0:  0.94925,
    12.5:  0.96783,
    15.0:  0.97940,
    17.5:  0.98720,
    20.0:  0.99260,
}

# ── 2D MoM reference values (r=1 square) ──────────────────────────────
MOM_SQUARE = {
    2:    0.6732,
    5:    1.3184,
    10:   1.7145,
    100:  2.1003,
    1e10: 2.1884,
}

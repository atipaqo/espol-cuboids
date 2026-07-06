"""
vector_analysis.py  —  3D vector algebra for MoM polarizability.

Python equivalent of the MATLAB VectorAnalysis/ folder.
Provides V3 (3D vector), D3 (3×3 tensor), and Ori (orientation)
with operator overloading for clean, readable code.

Usage:
    from vector_analysis import V3, D3, Ori
"""

import numpy as np
from numbers import Number


# ═══════════════════════════════════════════════════════════════════
#  V3  —  3D vector  (MATLAB: va_r)
# ═══════════════════════════════════════════════════════════════════

class V3:
    """3D vector with components x, y, z as numpy arrays.

    Construction:
        V3(x, y, z)     — from three arrays (must be broadcast-compatible)
        V3.zeros(n)     — n zeros (1D)
        V3.zeros(n, m)  — n×m zeros
        V3.ones(n)      — n ones
        V3.linspace(a, b, n)  — n vectors linearly spaced from a to b
    """

    __slots__ = ('x', 'y', 'z')

    def __init__(self, x=0.0, y=None, z=None):
        if y is None and z is None:
            # single argument: broadcast to all components
            self.x = np.asarray(x, dtype=None)
            self.y = np.asarray(x, dtype=None)
            self.z = np.asarray(x, dtype=None)
        else:
            self.x = np.asarray(x, dtype=None)
            self.y = np.asarray(y, dtype=None)
            self.z = np.asarray(z, dtype=None)

    # ---- factories ----
    @classmethod
    def zeros(cls, *dims):
        if not dims:
            return cls(0.0, 0.0, 0.0)
        z = np.zeros(dims, dtype=None)
        return cls(z, z.copy(), z.copy())

    @classmethod
    def ones(cls, *dims):
        if not dims:
            return cls(1.0, 1.0, 1.0)
        o = np.ones(dims, dtype=None)
        return cls(o, o.copy(), o.copy())

    @classmethod
    def linspace(cls, a, b, n):
        """Linearly spaced vectors between V3 a and V3 b."""
        if not isinstance(a, V3):
            a = V3(a)
        if not isinstance(b, V3):
            b = V3(b)
        return cls(
            np.linspace(a.x, b.x, n),
            np.linspace(a.y, b.y, n),
            np.linspace(a.z, b.z, n),
        )

    # ---- properties ----
    @property
    def shape(self):
        return self.x.shape

    @property
    def size(self):
        return self.x.size

    # ---- operators ----
    def __add__(self, other):
        if isinstance(other, V3):
            return V3(self.x + other.x, self.y + other.y, self.z + other.z)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, V3):
            return V3(self.x - other.x, self.y - other.y, self.z - other.z)
        return NotImplemented

    def __neg__(self):
        return V3(-self.x, -self.y, -self.z)

    def __mul__(self, other):
        """Scalar multiplication: V3 * scalar  or  V3 * V3 (element-wise)."""
        if isinstance(other, V3):
            return V3(self.x * other.x, self.y * other.y, self.z * other.z)
        if isinstance(other, Number) or isinstance(other, np.ndarray):
            return V3(self.x * other, self.y * other, self.z * other)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, V3):
            return V3(self.x / other.x, self.y / other.y, self.z / other.z)
        if isinstance(other, Number) or isinstance(other, np.ndarray):
            return V3(self.x / other, self.y / other, self.z / other)
        return NotImplemented

    def __repr__(self):
        return f"V3(x={self.x!r}, y={self.y!r}, z={self.z!r})"

    def __getitem__(self, idx):
        """Index into the underlying arrays."""
        return V3(self.x[idx], self.y[idx], self.z[idx])

    def __setitem__(self, idx, val):
        if isinstance(val, V3):
            self.x[idx] = val.x
            self.y[idx] = val.y
            self.z[idx] = val.z
        else:
            self.x[idx] = val
            self.y[idx] = val
            self.z[idx] = val

    # ---- vector operations ----
    def dot(self, other):
        """Dot product: self · other."""
        if isinstance(other, V3):
            return self.x * other.x + self.y * other.y + self.z * other.z
        return NotImplemented

    def cross(self, other):
        """Cross product: self × other."""
        if isinstance(other, V3):
            return V3(
                self.y * other.z - self.z * other.y,
                self.z * other.x - self.x * other.z,
                self.x * other.y - self.y * other.x,
            )
        return NotImplemented

    def mag(self):
        """Euclidean magnitude."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def norm(self):
        """Return unit vector (same direction)."""
        m = self.mag()
        return V3(self.x / m, self.y / m, self.z / m)

    def copy(self):
        return V3(self.x.copy(), self.y.copy(), self.z.copy())


# ═══════════════════════════════════════════════════════════════════
#  D3  —  3×3 dyadic / tensor  (MATLAB: va_d)
# ═══════════════════════════════════════════════════════════════════

class D3:
    """3×3 tensor with components xx, xy, xz, yx, yy, yz, zx, zy, zz.

    Construction:
        D3(xx, xy, xz, yx, yy, yz, zx, zy, zz)
        D3.zeros(n)       — n zeros (1D)
        D3.zeros(n, m)    — n×m zeros
        D3.eye()          — identity tensor
        D3.rot(axis, angle)       — rotation about axis by angle [rad]
        D3.rot_a_t_p(a, t, p)     — from Euler angles (azimuth, theta, phi)
    """

    __slots__ = ('xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz')

    def __init__(self, xx=0.0, xy=0.0, xz=0.0,
                       yx=0.0, yy=0.0, yz=0.0,
                       zx=0.0, zy=0.0, zz=0.0):
        if isinstance(xx, Number) and not any(isinstance(a, np.ndarray)
                for a in [xx, xy, xz, yx, yy, yz, zx, zy, zz]):
            # all scalars — treat as single value broadcast
            pass
        self.xx = np.asarray(xx, dtype=None)
        self.xy = np.asarray(xy, dtype=None)
        self.xz = np.asarray(xz, dtype=None)
        self.yx = np.asarray(yx, dtype=None)
        self.yy = np.asarray(yy, dtype=None)
        self.yz = np.asarray(yz, dtype=None)
        self.zx = np.asarray(zx, dtype=None)
        self.zy = np.asarray(zy, dtype=None)
        self.zz = np.asarray(zz, dtype=None)

    # ---- factories ----
    @classmethod
    def zeros(cls, *dims):
        if not dims:
            return cls(0,0,0,0,0,0,0,0,0)
        z = np.zeros(dims, dtype=None)
        return cls(z, z.copy(), z.copy(),
                   z.copy(), z.copy(), z.copy(),
                   z.copy(), z.copy(), z.copy())

    @classmethod
    def eye(cls, *dims):
        """Identity tensor."""
        if not dims:
            return cls(1,0,0, 0,1,0, 0,0,1)
        z = np.zeros(dims, dtype=None)
        o = np.ones(dims, dtype=None)
        return cls(o, z.copy(), z.copy(),
                   z.copy(), o, z.copy(),
                   z.copy(), z.copy(), o)

    @classmethod
    def rot(cls, axis, angle):
        """Rotation matrix about 'x', 'y', or 'z' by angle [rad]."""
        c = np.cos(angle)
        s = np.sin(angle)
        z = np.zeros_like(c)
        o = np.ones_like(c)
        if axis == 'x':
            return cls(o, z, z,  z, c, -s,  z, s, c)
        elif axis == 'y':
            return cls(c, z, s,  z, o, z,  -s, z, c)
        elif axis == 'z':
            return cls(c, -s, z,  s, c, z,  z, z, o)
        else:
            raise ValueError(f"Unknown axis: {axis!r}")

    @classmethod
    def rot_a_t_p(cls, a, t, p):
        """Rotation from Euler angles (azimuth a, theta t, phi p)."""
        # roll
        ca, sa = np.cos(a), np.sin(a)
        d1 = cls(ca, -sa, 0,  sa, ca, 0,  0, 0, 1)
        # elevation
        ct, st = np.cos(t), np.sin(t)
        d2 = cls(ct, 0, st,  0, 1, 0,  -st, 0, ct)
        # azimuth
        cp, sp = np.cos(p), np.sin(p)
        d3 = cls(cp, -sp, 0,  sp, cp, 0,  0, 0, 1)
        return d3 @ d2 @ d1

    @classmethod
    def rot180(cls, axis):
        """180° rotation about the given axis (shorthand)."""
        return cls.rot(axis, np.pi)

    # ---- properties ----
    @property
    def shape(self):
        return self.xx.shape

    @property
    def T(self):
        """Transpose."""
        return D3(self.xx, self.yx, self.zx,
                  self.xy, self.yy, self.zy,
                  self.xz, self.yz, self.zz)

    # ---- operators ----
    def __add__(self, other):
        if isinstance(other, D3):
            return D3(
                self.xx + other.xx, self.xy + other.xy, self.xz + other.xz,
                self.yx + other.yx, self.yy + other.yy, self.yz + other.yz,
                self.zx + other.zx, self.zy + other.zy, self.zz + other.zz,
            )
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, D3):
            return D3(
                self.xx - other.xx, self.xy - other.xy, self.xz - other.xz,
                self.yx - other.yx, self.yy - other.yy, self.yz - other.yz,
                self.zx - other.zx, self.zy - other.zy, self.zz - other.zz,
            )
        return NotImplemented

    def __mul__(self, other):
        """Scalar multiplication: D3 * scalar."""
        if isinstance(other, Number) or isinstance(other, np.ndarray):
            return D3(
                self.xx * other, self.xy * other, self.xz * other,
                self.yx * other, self.yy * other, self.yz * other,
                self.zx * other, self.zy * other, self.zz * other,
            )
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        """Matrix multiplication: D3 @ D3  or  D3 @ V3."""
        if isinstance(other, D3):
            # C = A @ B
            return D3(
                self.xx*other.xx + self.xy*other.yx + self.xz*other.zx,
                self.xx*other.xy + self.xy*other.yy + self.xz*other.zy,
                self.xx*other.xz + self.xy*other.yz + self.xz*other.zz,

                self.yx*other.xx + self.yy*other.yx + self.yz*other.zx,
                self.yx*other.xy + self.yy*other.yy + self.yz*other.zy,
                self.yx*other.xz + self.yy*other.yz + self.yz*other.zz,

                self.zx*other.xx + self.zy*other.yx + self.zz*other.zx,
                self.zx*other.xy + self.zy*other.yy + self.zz*other.zy,
                self.zx*other.xz + self.zy*other.yz + self.zz*other.zz,
            )
        if isinstance(other, V3):
            # v' = D @ v   (rotate vector)
            return V3(
                self.xx*other.x + self.xy*other.y + self.xz*other.z,
                self.yx*other.x + self.yy*other.y + self.yz*other.z,
                self.zx*other.x + self.zy*other.y + self.zz*other.z,
            )
        return NotImplemented

    def __repr__(self):
        return (f"D3(xx={self.xx!r}, xy={self.xy!r}, xz={self.xz!r}, "
                f"yx={self.yx!r}, yy={self.yy!r}, yz={self.yz!r}, "
                f"zx={self.zx!r}, zy={self.zy!r}, zz={self.zz!r})")


# ═══════════════════════════════════════════════════════════════════
#  Ori  —  orientation / Euler angles  (MATLAB: va_o)
# ═══════════════════════════════════════════════════════════════════

class Ori:
    """Orientation via Euler angles (azimuth a, theta t, phi p).

    Construction:
        Ori(a, t, p)  — from three angles [rad]
        Ori.zeros(n)   — n zero orientations
    """

    __slots__ = ('a', 't', 'p')

    def __init__(self, a=0.0, t=0.0, p=0.0):
        self.a = np.asarray(a, dtype=None)
        self.t = np.asarray(t, dtype=None)
        self.p = np.asarray(p, dtype=None)

    @classmethod
    def zeros(cls, *dims):
        if not dims:
            return cls(0.0, 0.0, 0.0)
        z = np.zeros(dims, dtype=None)
        return cls(z, z.copy(), z.copy())

    def to_matrix(self):
        """Convert to D3 rotation matrix."""
        return D3.rot_a_t_p(self.a, self.t, self.p)


# ═══════════════════════════════════════════════════════════════════
#  Mesh  —  surface mesh structure  (MATLAB: struct with r, n, ds, …)
# ═══════════════════════════════════════════════════════════════════

class Mesh:
    """Surface mesh: positions r, normals n, areas ds, volume V, er.

    Attributes:
        r      : V3  —  element centre positions
        n      : V3  —  unit normals (outward)
        ds     : ndarray  —  element areas
        ne     : int  —  number of elements
        V      : float  —  object volume
        er     : complex  —  relative permittivity
        octant : bool  —  first-octant flag (for symmetry solver)
    """

    __slots__ = ('r', 'n', 'ds', 'ne', 'V', 'er', 'octant')

    def __init__(self, r, n, ds, V, er, octant=False):
        self.r = r
        self.n = n
        self.ds = np.asarray(ds, dtype=None)
        self.ne = r.size
        self.V = float(V)
        self.er = complex(er)
        self.octant = bool(octant)

    def __repr__(self):
        return (f"Mesh(ne={self.ne}, V={self.V:.4g}, er={self.er!r}, "
                f"octant={self.octant})")


# ═══════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== V3 tests ===")
    a = V3(1, 2, 3)
    b = V3(4, 5, 6)
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a - b = {a - b}")
    print(f"a * 2 = {a * 2}")
    print(f"a.dot(b) = {a.dot(b)}")
    print(f"a.mag() = {a.mag()}")
    print(f"V3.zeros(3) = {V3.zeros(3)}")

    print("\n=== D3 tests ===")
    I = D3.eye()
    R = D3.rot('z', np.pi/2)
    v = V3(1, 0, 0)
    print(f"I @ v = {I @ v}")
    print(f"Rz(90°) @ (1,0,0) = {R @ v}")
    print(f"D3.eye(2,2) shape = {D3.eye(2,2).shape}")

    print("\n=== Ori tests ===")
    o = Ori(0, 0, np.pi/2)
    print(f"Ori(0,0,π/2).to_matrix() @ (1,0,0) = {o.to_matrix() @ v}")

    print("\n=== Mesh tests ===")
    r = V3.zeros(10)
    n = V3.zeros(10)
    n.z = np.ones(10)
    m = Mesh(r, n, ds=np.ones(10)*0.1, V=1.0, er=1e10)
    print(m)

    print("\nAll tests passed.")

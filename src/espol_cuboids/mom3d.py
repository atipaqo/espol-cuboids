"""
mom3d.py  —  3D MoM polarizability solver for dielectric cuboids.

Provides three solver backends:
  1. Direct LU (default)          — dense n×n matrix, O(n³)
  2. Matrix-free GMRES            — precomputed N×N kernels, O(n²) mem
  3. ACA-compressed H-matrix      — O(n log n) mem + GMRES

Also includes octant-symmetry mode for centro-symmetric cuboids.

Usage:
    from mom3d import comp_polarizability_3d, Mesh
    P = comp_polarizability_3d(mesh)                     # direct LU
    P = comp_polarizability_3d(mesh, iterative=True)     # dense GMRES
    P = comp_polarizability_3d(mesh, aca=True)           # ACA + GMRES
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from .vector_analysis import V3, D3, Mesh


# ═══════════════════════════════════════════════════════════════════
#  1. Matrix-free kernel operators (dense-kernel GMRES)
# ═══════════════════════════════════════════════════════════════════

def _diag_preconditioner(ns, Z1):
    """Return a LinearOperator for M^{-1} = (1/Z1) * I."""
    def matvec(v):
        return v / Z1
    return LinearOperator((ns, ns), matvec=matvec, dtype=complex)


def _meshgrid_all(r, n, ds):
    """Build the shared NxN arrays once (expensive, O(N^2) memory)."""
    ns = len(r.x)
    rn_x, rm_x = np.meshgrid(r.x, r.x)
    rn_y, rm_y = np.meshgrid(r.y, r.y)
    rn_z, rm_z = np.meshgrid(r.z, r.z)
    nn_x, _ = np.meshgrid(n.x, n.x)
    nn_y, _ = np.meshgrid(n.y, n.y)
    nn_z, _ = np.meshgrid(n.z, n.z)
    dn, _ = np.meshgrid(ds, ds)
    I_mat = np.eye(ns)
    In = 1.0 - I_mat
    return dict(rn_x=rn_x, rn_y=rn_y, rn_z=rn_z,
                rm_x=rm_x, rm_y=rm_y, rm_z=rm_z,
                nn_x=nn_x, nn_y=nn_y, nn_z=nn_z,
                dn=dn, I_mat=I_mat, In=In, ns=ns)


def _octant_matvec_operator(mg, er, direction):
    """LinearOperator for Z @ x in octant mode (vectorised matmul)."""
    ns = mg['ns']
    Z1 = (er + 1.0) / 2.0
    f_k = (er - 1.0) / (4.0 * np.pi)

    sx_all = np.array([1, -1, 1, 1, -1, -1, 1, -1])
    sy_all = np.array([1, 1, -1, 1, -1, 1, -1, -1])
    sz_all = np.array([1, 1, 1, -1, 1, -1, -1, -1])
    s_acc = {'x': sx_all, 'y': sy_all, 'z': sz_all}[direction]

    kernels = []
    for oct_idx in range(8):
        sx, sy, sz = sx_all[oct_idx], sy_all[oct_idx], sz_all[oct_idx]
        vx = mg['rm_x'] - sx * mg['rn_x']
        vy = mg['rm_y'] - sy * mg['rn_y']
        vz = mg['rm_z'] - sz * mg['rn_z']
        r2 = vx * vx + vy * vy + vz * vz
        r3 = np.abs(r2) ** 1.5
        dot_v = (vx * (sx * mg['nn_x']) +
                 vy * (sy * mg['nn_y']) +
                 vz * (sz * mg['nn_z']))
        K = mg['dn'] * dot_v / r3
        K = np.nan_to_num(K) * mg['In']
        kernels.append(K)

    def matvec(x):
        out = Z1 * x
        for oct_idx in range(8):
            out += f_k * s_acc[oct_idx] * (kernels[oct_idx] @ x)
        return out

    return LinearOperator((ns, ns), matvec=matvec, dtype=complex)


def _full_matvec_operator(mg, er):
    """LinearOperator for the full-mesh Z @ x."""
    ns = mg['ns']
    Z1 = (er + 1.0) / 2.0
    f_k = (er - 1.0) / (4.0 * np.pi)

    rho_x = mg['rm_x'] - mg['rn_x']
    rho_y = mg['rm_y'] - mg['rn_y']
    rho_z = mg['rm_z'] - mg['rn_z']
    rhomn = np.sqrt(rho_x**2 + rho_y**2 + rho_z**2)
    dot_rho_n = rho_x * mg['nn_x'] + rho_y * mg['nn_y'] + rho_z * mg['nn_z']
    rhomn3_safe = np.where(mg['In'] > 0, rhomn**3, 1.0)
    K = mg['dn'] * dot_rho_n / rhomn3_safe
    K = np.nan_to_num(K) * mg['In']

    def matvec(x):
        return Z1 * x + f_k * (K @ x)

    return LinearOperator((ns, ns), matvec=matvec, dtype=complex)


def solve_octant_gmres(r, n, ds, er, tol=1e-6, maxiter=None, verbose=False):
    """Solve octant MoM with GMRES (diagonally preconditioned)."""
    ns = len(r.x)
    if maxiter is None:
        maxiter = min(ns, 500)
    mg = _meshgrid_all(r, n, ds)
    Z1 = (er + 1.0) / 2.0
    M = _diag_preconditioner(ns, Z1)

    results = []
    for direction in ['x', 'y', 'z']:
        op = _octant_matvec_operator(mg, er, direction)
        rhs = {'x': -r.x, 'y': -r.y, 'z': -r.z}[direction]
        x_sol, info = gmres(op, rhs, M=M, rtol=tol, maxiter=maxiter, atol=1e-14)
        if verbose:
            if info > 0:
                print(f"    GMRES({direction}): {info} iters")
            elif info < 0:
                print(f"    GMRES({direction}): WARNING info={info}")
        results.append(x_sol)

    return results[0], results[1], results[2]


def solve_full_gmres(r, n, ds, er, tol=1e-6, maxiter=None, verbose=False):
    """Solve full-mesh MoM with GMRES."""
    ns = len(r.x)
    if maxiter is None:
        maxiter = min(ns, 500)
    mg = _meshgrid_all(r, n, ds)
    op = _full_matvec_operator(mg, er)
    Z1 = (er + 1.0) / 2.0
    M = _diag_preconditioner(ns, Z1)

    results = []
    for rhs in [-r.x, -r.y, -r.z]:
        x_sol, info = gmres(op, rhs, M=M, rtol=tol, maxiter=maxiter, atol=1e-14)
        if verbose and info > 0:
            print(f"    GMRES: {info} iters")
        results.append(x_sol)

    return results[0], results[1], results[2]


# ═══════════════════════════════════════════════════════════════════
#  2. H-matrix with ACA compression
# ═══════════════════════════════════════════════════════════════════

class Cluster:
    """A group of elements with bounding box and child pointers."""
    __slots__ = ('indices', 'bbox', 'left', 'right', 'is_leaf')

    def __init__(self, indices, bbox):
        self.indices = np.asarray(indices, dtype=int)
        self.bbox = bbox
        self.left = None
        self.right = None
        self.is_leaf = True

    @property
    def children(self):
        return [c for c in (self.left, self.right) if c is not None]

    @property
    def size(self):
        return len(self.indices)

    @property
    def diameter(self):
        return np.linalg.norm(self.bbox[1] - self.bbox[0])

    @property
    def center(self):
        return 0.5 * (self.bbox[0] + self.bbox[1])


def _bbox(points, indices):
    if len(indices) == 0:
        return (np.zeros(3), np.zeros(3))
    pts = points[indices]
    return (pts.min(axis=0), pts.max(axis=0))


def _split_axis(points, indices):
    pts = points[indices]
    extents = pts.max(axis=0) - pts.min(axis=0)
    axis = int(np.argmax(extents))
    median = np.median(pts[:, axis])
    return axis, median


def build_tree(points, leaf_size=64):
    indices = np.arange(len(points), dtype=int)
    return _build_recursive(points, indices, leaf_size)


def _build_recursive(points, indices, leaf_size):
    bbox = _bbox(points, indices)
    node = Cluster(indices, bbox)
    if len(indices) <= leaf_size:
        return node
    axis, median = _split_axis(points, indices)
    pts = points[indices]
    left_mask = pts[:, axis] <= median
    right_mask = ~left_mask
    if left_mask.sum() == 0 or right_mask.sum() == 0:
        return node
    node.left = _build_recursive(points, indices[left_mask], leaf_size)
    node.right = _build_recursive(points, indices[right_mask], leaf_size)
    node.is_leaf = False
    return node


def is_admissible(obs, src, eta=1.0):
    d_obs = obs.diameter
    d_src = src.diameter
    d_min = min(d_obs, d_src)
    dist = np.linalg.norm(obs.center - src.center)
    return d_min <= eta * dist


# -- ACA kernel helpers --

def _kernel_vector(obs_idx, src_j, r, n, ds):
    vx = r.x[obs_idx] - r.x[src_j]
    vy = r.y[obs_idx] - r.y[src_j]
    vz = r.z[obs_idx] - r.z[src_j]
    r2 = vx * vx + vy * vy + vz * vz
    r3 = np.abs(r2) ** 1.5
    dot_v = vx * n.x[src_j] + vy * n.y[src_j] + vz * n.z[src_j]
    col = ds[src_j] * dot_v / r3
    col = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
    return col


def _kernel_row(src_idx, obs_i, r, n, ds):
    vx = r.x[obs_i] - r.x[src_idx]
    vy = r.y[obs_i] - r.y[src_idx]
    vz = r.z[obs_i] - r.z[src_idx]
    r2 = vx * vx + vy * vy + vz * vz
    r3 = np.abs(r2) ** 1.5
    dot_v = vx * n.x[src_idx] + vy * n.y[src_idx] + vz * n.z[src_idx]
    row = ds[src_idx] * dot_v / r3
    row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
    return row


def aca_block(obs_idx, src_idx, r, n, ds, eps=1e-4, max_rank=100):
    m = len(obs_idx)
    n_src = len(src_idx)
    rank_max = min(max_rank, m, n_src)
    U = np.zeros((m, rank_max))
    V = np.zeros((n_src, rank_max))

    n_samples = min(20, m, n_src)
    row_samples = np.random.choice(m, size=n_samples, replace=False)
    col_samples = np.random.choice(n_src, size=n_samples, replace=False)
    est_sq = 0.0
    for i in row_samples:
        row = _kernel_row(src_idx, obs_idx[i], r, n, ds)
        est_sq += np.sum(row ** 2)
    frob_est = np.sqrt(est_sq / n_samples * m) if est_sq > 0 else 1.0
    stop_tol = eps * frob_est

    rank = 0
    used_rows = set()
    used_cols = set()

    for k in range(rank_max):
        available_rows = [i for i in range(m) if i not in used_rows]
        if not available_rows:
            break
        i_k = available_rows[np.random.randint(len(available_rows))]
        row_exact = _kernel_row(src_idx, obs_idx[i_k], r, n, ds)
        row_approx = U[i_k, :k] @ V.T[:k, :]
        residual_row = row_exact - row_approx
        j_k = np.argmax(np.abs(residual_row))
        pivot = residual_row[j_k]
        if abs(pivot) < stop_tol:
            break
        col_exact = _kernel_vector(obs_idx, src_idx[j_k], r, n, ds)
        col_approx = U[:, :k] @ V[j_k, :k]
        residual_col = col_exact - col_approx
        U[:, k] = residual_col / pivot
        V[:, k] = residual_row
        used_rows.add(i_k)
        used_cols.add(j_k)
        rank = k + 1
        if k > 0 and k % 10 == 0:
            approx_norm = np.linalg.norm(U[:, :rank] @ V[:, :rank].T, 'fro')
            if approx_norm > 0 and abs(pivot) / approx_norm < eps:
                break

    return U[:, :rank], V[:, :rank]


class DenseBlock:
    __slots__ = ('obs_idx', 'src_idx', 'data')

    def __init__(self, obs_idx, src_idx, data):
        self.obs_idx = np.asarray(obs_idx, dtype=int)
        self.src_idx = np.asarray(src_idx, dtype=int)
        self.data = data

    def matvec(self, x):
        return self.data @ x[self.src_idx]


class LowRankBlock:
    __slots__ = ('obs_idx', 'src_idx', 'U', 'V')

    def __init__(self, obs_idx, src_idx, U, V):
        self.obs_idx = np.asarray(obs_idx, dtype=int)
        self.src_idx = np.asarray(src_idx, dtype=int)
        self.U = U
        self.V = V

    def matvec(self, x):
        return self.U @ (self.V.T @ x[self.src_idx])


def build_hmatrix(r, n, ds, er, leaf_size=64, eta=1.0, eps_aca=1e-4, verbose=False):
    centroids = np.column_stack([r.x, r.y, r.z])
    tree = build_tree(centroids, leaf_size=leaf_size)
    blocks = []
    _assemble_blocks(tree, tree, r, n, ds, eta, eps_aca, blocks, verbose)
    N = len(r.x)
    Z1 = (er + 1.0) / 2.0
    f_k = (er - 1.0) / (4.0 * np.pi)

    if verbose:
        dense_elems = sum(b.data.size for b in blocks if isinstance(b, DenseBlock))
        lr_elems = sum(b.U.size + b.V.size for b in blocks if isinstance(b, LowRankBlock))
        full_elems = N * N
        print(f"    H-matrix: {len(blocks)} blocks, "
              f"{dense_elems + lr_elems} stored / {full_elems} full "
              f"({(dense_elems+lr_elems)/full_elems*100:.1f}%)")

    class HMatrixOp:
        def __init__(self, blocks, Z1, f_k):
            self._blocks = blocks
            self.Z1 = Z1
            self.f_k = f_k
            self.n = N

        def matvec(self, x):
            y = np.zeros(self.n, dtype=complex)
            for blk in self._blocks:
                y[blk.obs_idx] += blk.matvec(x)
            return self.Z1 * x + self.f_k * y

        def as_linearoperator(self):
            return LinearOperator((self.n, self.n), matvec=self.matvec, dtype=complex)

    return HMatrixOp(blocks, Z1, f_k)


def _assemble_blocks(obs, src, r, n, ds, eta, eps_aca, blocks, verbose):
    if obs.size == 0 or src.size == 0:
        return
    if is_admissible(obs, src, eta):
        U, V = aca_block(obs.indices, src.indices, r, n, ds, eps=eps_aca)
        if U.shape[1] > 0:
            blocks.append(LowRankBlock(obs.indices, src.indices, U, V))
        return
    max_dense = 2500
    if obs.size * src.size <= max_dense:
        data = np.zeros((obs.size, src.size))
        for jj, sj in enumerate(src.indices):
            data[:, jj] = _kernel_vector(obs.indices, sj, r, n, ds)
        blocks.append(DenseBlock(obs.indices, src.indices, data))
        return
    if obs.is_leaf and src.is_leaf:
        data = np.zeros((obs.size, src.size))
        for jj, sj in enumerate(src.indices):
            data[:, jj] = _kernel_vector(obs.indices, sj, r, n, ds)
        blocks.append(DenseBlock(obs.indices, src.indices, data))
        return
    if obs.is_leaf:
        _assemble_blocks(obs, src.left, r, n, ds, eta, eps_aca, blocks, verbose)
        _assemble_blocks(obs, src.right, r, n, ds, eta, eps_aca, blocks, verbose)
    elif src.is_leaf:
        _assemble_blocks(obs.left, src, r, n, ds, eta, eps_aca, blocks, verbose)
        _assemble_blocks(obs.right, src, r, n, ds, eta, eps_aca, blocks, verbose)
    else:
        if obs.diameter >= src.diameter:
            _assemble_blocks(obs.left, src, r, n, ds, eta, eps_aca, blocks, verbose)
            _assemble_blocks(obs.right, src, r, n, ds, eta, eps_aca, blocks, verbose)
        else:
            _assemble_blocks(obs, src.left, r, n, ds, eta, eps_aca, blocks, verbose)
            _assemble_blocks(obs, src.right, r, n, ds, eta, eps_aca, blocks, verbose)


def solve_full_aca(r, n, ds, er, tol=1e-6, leaf_size=64, eta=1.0,
                   eps_aca=1e-4, verbose=False):
    if verbose:
        print(f"    Building H-matrix (leaf_size={leaf_size}, eta={eta}, "
              f"eps_aca={eps_aca})...")
    H = build_hmatrix(r, n, ds, er, leaf_size=leaf_size, eta=eta,
                      eps_aca=eps_aca, verbose=verbose)
    op = H.as_linearoperator()
    Z1 = H.Z1
    M_op = LinearOperator((H.n, H.n), matvec=lambda v: v / Z1, dtype=complex)

    results = []
    for direction, rhs in [('x', -r.x), ('y', -r.y), ('z', -r.z)]:
        if verbose:
            print(f"    GMRES({direction})...")
        x_sol, info = gmres(op, rhs, M=M_op, rtol=tol, maxiter=min(H.n, 500),
                            atol=1e-14)
        if verbose:
            if info == 0:
                print(f"    GMRES({direction}): converged")
            elif info > 0:
                print(f"    GMRES({direction}): maxiter reached (info={info})")
        results.append(x_sol)
    return results[0], results[1], results[2]


# -- Octant-mode ACA --

def _octant_kernel_vector(obs_idx, src_j, r, n, ds, sx, sy, sz):
    vx = r.x[obs_idx] - sx * r.x[src_j]
    vy = r.y[obs_idx] - sy * r.y[src_j]
    vz = r.z[obs_idx] - sz * r.z[src_j]
    r2 = vx * vx + vy * vy + vz * vz
    r3 = np.abs(r2) ** 1.5
    dot_v = vx * (sx * n.x[src_j]) + vy * (sy * n.y[src_j]) + vz * (sz * n.z[src_j])
    col = ds[src_j] * dot_v / r3
    col = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
    col[obs_idx == src_j] = 0.0
    return col


def _octant_kernel_row(src_idx, obs_i, r, n, ds, sx, sy, sz):
    vx = r.x[obs_i] - sx * r.x[src_idx]
    vy = r.y[obs_i] - sy * r.y[src_idx]
    vz = r.z[obs_i] - sz * r.z[src_idx]
    r2 = vx * vx + vy * vy + vz * vz
    r3 = np.abs(r2) ** 1.5
    dot_v = vx * (sx * n.x[src_idx]) + vy * (sy * n.y[src_idx]) + vz * (sz * n.z[src_idx])
    row = ds[src_idx] * dot_v / r3
    row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
    row[src_idx == obs_i] = 0.0
    return row


def aca_block_octant(obs_idx, src_idx, r, n, ds, sx, sy, sz, eps=1e-4, max_rank=100):
    m = len(obs_idx)
    n_src = len(src_idx)
    rank_max = min(max_rank, m, n_src)
    U = np.zeros((m, rank_max))
    V = np.zeros((n_src, rank_max))

    n_samples = min(20, m, n_src)
    row_samples = np.random.choice(m, size=n_samples, replace=False)
    est_sq = 0.0
    for i in row_samples:
        row = _octant_kernel_row(src_idx, obs_idx[i], r, n, ds, sx, sy, sz)
        est_sq += np.sum(row ** 2)
    frob_est = np.sqrt(est_sq / n_samples * m) if est_sq > 0 else 1.0
    stop_tol = eps * frob_est

    used_rows = set()
    rank = 0
    for k in range(rank_max):
        available_rows = [i for i in range(m) if i not in used_rows]
        if not available_rows:
            break
        i_k = available_rows[np.random.randint(len(available_rows))]
        row_exact = _octant_kernel_row(src_idx, obs_idx[i_k], r, n, ds, sx, sy, sz)
        row_approx = U[i_k, :k] @ V.T[:k, :]
        residual_row = row_exact - row_approx
        j_k = np.argmax(np.abs(residual_row))
        pivot = residual_row[j_k]
        if abs(pivot) < stop_tol:
            break
        col_exact = _octant_kernel_vector(obs_idx, src_idx[j_k], r, n, ds, sx, sy, sz)
        col_approx = U[:, :k] @ V[j_k, :k]
        residual_col = col_exact - col_approx
        U[:, k] = residual_col / pivot
        V[:, k] = residual_row
        used_rows.add(i_k)
        rank = k + 1

    return U[:, :rank], V[:, :rank]


def build_octant_hmatrix(r, n, ds, er, leaf_size=64, eta=1.0, eps_aca=1e-4,
                         verbose=False):
    centroids = np.column_stack([r.x, r.y, r.z])
    tree = build_tree(centroids, leaf_size=leaf_size)

    sx_all = np.array([1, -1, 1, 1, -1, -1, 1, -1])
    sy_all = np.array([1, 1, -1, 1, -1, 1, -1, -1])
    sz_all = np.array([1, 1, 1, -1, 1, -1, -1, -1])

    octant_blocks = []
    for oct_idx in range(8):
        sx, sy, sz = sx_all[oct_idx], sy_all[oct_idx], sz_all[oct_idx]
        blocks = []
        _assemble_octant_blocks(tree, tree, r, n, ds, sx, sy, sz,
                                eta, eps_aca, blocks, verbose and oct_idx == 0)
        octant_blocks.append(blocks)

    N = len(r.x)
    Z1 = (er + 1.0) / 2.0
    f_k = (er - 1.0) / (4.0 * np.pi)

    if verbose:
        total_stored = 0
        for blocks in octant_blocks:
            for blk in blocks:
                if isinstance(blk, DenseBlock):
                    total_stored += blk.data.size
                else:
                    total_stored += blk.U.size + blk.V.size
        full_elems = N * N * 8
        print(f"    Octant H-matrix: {total_stored} stored / {full_elems} full "
              f"({total_stored/full_elems*100:.1f}%)")

    class OctantHMatrixOp:
        def __init__(self, octant_blocks, Z1, f_k):
            self._octant_blocks = octant_blocks
            self.Z1 = Z1
            self.f_k = f_k
            self.n = N

        def matvec(self, x, direction):
            s_acc = {'x': sx_all, 'y': sy_all, 'z': sz_all}[direction]
            y = Z1 * x
            for oct_idx in range(8):
                y_oct = np.zeros(self.n, dtype=complex)
                for blk in self._octant_blocks[oct_idx]:
                    y_oct[blk.obs_idx] += blk.matvec(x)
                y += f_k * s_acc[oct_idx] * y_oct
            return y

    return OctantHMatrixOp(octant_blocks, Z1, f_k)


def _assemble_octant_blocks(obs, src, r, n, ds, sx, sy, sz, eta, eps_aca,
                            blocks, verbose):
    if obs.size == 0 or src.size == 0:
        return
    if is_admissible(obs, src, eta):
        U, V = aca_block_octant(obs.indices, src.indices, r, n, ds, sx, sy, sz,
                                eps=eps_aca)
        if U.shape[1] > 0:
            blocks.append(LowRankBlock(obs.indices, src.indices, U, V))
        return
    max_dense = 2500
    if obs.size * src.size <= max_dense:
        data = np.zeros((obs.size, src.size))
        for jj, sj in enumerate(src.indices):
            data[:, jj] = _octant_kernel_vector(obs.indices, sj, r, n, ds, sx, sy, sz)
        blocks.append(DenseBlock(obs.indices, src.indices, data))
        return
    if obs.is_leaf and src.is_leaf:
        data = np.zeros((obs.size, src.size))
        for jj, sj in enumerate(src.indices):
            data[:, jj] = _octant_kernel_vector(obs.indices, sj, r, n, ds, sx, sy, sz)
        blocks.append(DenseBlock(obs.indices, src.indices, data))
        return
    if obs.is_leaf:
        _assemble_octant_blocks(obs, src.left, r, n, ds, sx, sy, sz, eta, eps_aca, blocks, verbose)
        _assemble_octant_blocks(obs, src.right, r, n, ds, sx, sy, sz, eta, eps_aca, blocks, verbose)
    elif src.is_leaf:
        _assemble_octant_blocks(obs.left, src, r, n, ds, sx, sy, sz, eta, eps_aca, blocks, verbose)
        _assemble_octant_blocks(obs.right, src, r, n, ds, sx, sy, sz, eta, eps_aca, blocks, verbose)
    else:
        if obs.diameter >= src.diameter:
            _assemble_octant_blocks(obs.left, src, r, n, ds, sx, sy, sz, eta, eps_aca, blocks, verbose)
            _assemble_octant_blocks(obs.right, src, r, n, ds, sx, sy, sz, eta, eps_aca, blocks, verbose)
        else:
            _assemble_octant_blocks(obs, src.left, r, n, ds, sx, sy, sz, eta, eps_aca, blocks, verbose)
            _assemble_octant_blocks(obs, src.right, r, n, ds, sx, sy, sz, eta, eps_aca, blocks, verbose)


def solve_octant_aca(r, n, ds, er, tol=1e-6, leaf_size=64, eta=1.0,
                     eps_aca=1e-4, verbose=False):
    if verbose:
        print(f"    Building octant H-matrix (leaf_size={leaf_size}, eta={eta})...")
    H = build_octant_hmatrix(r, n, ds, er, leaf_size=leaf_size, eta=eta,
                             eps_aca=eps_aca, verbose=verbose)
    Z1 = H.Z1
    M_op = LinearOperator((H.n, H.n), matvec=lambda v: v / Z1, dtype=complex)

    results = []
    for direction in ['x', 'y', 'z']:
        rhs = {'x': -r.x, 'y': -r.y, 'z': -r.z}[direction]

        def make_matvec(d):
            return lambda v: H.matvec(v, d)
        op = LinearOperator((H.n, H.n), matvec=make_matvec(direction),
                            dtype=complex)
        x_sol, info = gmres(op, rhs, M=M_op, rtol=tol,
                            maxiter=min(H.n, 500), atol=1e-14)
        if verbose and info > 0:
            print(f"    GMRES({direction}): maxiter reached ({info} iters)")
        results.append(x_sol)

    return results[0], results[1], results[2]


# ═══════════════════════════════════════════════════════════════════
#  3. Main solver — comp_polarizability_3d
# ═══════════════════════════════════════════════════════════════════

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
            X1, X2, X3 = solve_octant_aca(r, n, ds, er, tol=tol,
                                          leaf_size=128, eta=0.5, eps_aca=1e-4)
        elif iterative:
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
        #  FULL MESH SOLVER
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
            X1, X2, X3 = solve_full_gmres(r, n, ds, er, tol=tol)
        elif aca:
            X1, X2, X3 = solve_full_aca(r, n, ds, er, tol=tol)
        else:
            # factor once, solve 3 RHS
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

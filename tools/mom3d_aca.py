"""
mom3d_aca.py  —  H-matrix with ACA compression for 3D MoM.

Builds a kd-tree over element centroids, then compresses well-separated
kernel blocks via Adaptive Cross Approximation.  Provides a LinearOperator
for GMRES solving.

For the full-mesh kernel:
    K[i,j] = ds[j] * n_j · (r_i - r_j) / |r_i - r_j|³
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator

# ---------------------------------------------------------------------------
#  Cluster / tree
# ---------------------------------------------------------------------------

class Cluster:
    """A group of elements with bounding box and child pointers."""
    __slots__ = ('indices', 'bbox', 'left', 'right', 'is_leaf')

    def __init__(self, indices, bbox):
        self.indices = np.asarray(indices, dtype=int)
        self.bbox = bbox           # (min_xyz, max_xyz)
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
    """Axis-aligned bounding box."""
    if len(indices) == 0:
        return (np.zeros(3), np.zeros(3))
    pts = points[indices]
    return (pts.min(axis=0), pts.max(axis=0))


def _split_axis(points, indices):
    """Return split axis (0,1,2) and median along that axis."""
    pts = points[indices]
    extents = pts.max(axis=0) - pts.min(axis=0)
    axis = int(np.argmax(extents))
    median = np.median(pts[:, axis])
    return axis, median


def build_tree(points, leaf_size=64):
    """Build a kd-tree of Clusters over point coordinates.

    Returns the root Cluster.
    """
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
        return node  # degenerate

    node.left = _build_recursive(points, indices[left_mask], leaf_size)
    node.right = _build_recursive(points, indices[right_mask], leaf_size)
    node.is_leaf = False
    return node


# ---------------------------------------------------------------------------
#  Admissibility
# ---------------------------------------------------------------------------

def is_admissible(obs, src, eta=1.0):
    """Well-separated admissibility criterion.

    min(diameter(obs), diameter(src)) <= eta * distance(obs, src)
    """
    d_obs = obs.diameter
    d_src = src.diameter
    d_min = min(d_obs, d_src)
    dist = np.linalg.norm(obs.center - src.center)
    return d_min <= eta * dist


# ---------------------------------------------------------------------------
#  ACA kernel block compression
# ---------------------------------------------------------------------------

def _kernel_vector(obs_idx, src_j, r, n, ds):
    """Kernel column: K[:, j] for a fixed source j, evaluated at obs_idx."""
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
    """Kernel row: K[i, :] for a fixed observation i, evaluated at src_idx."""
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
    """Adaptive Cross Approximation for the kernel block K[obs_idx, src_idx].

    Returns U (|obs| × rank), V (|src| × rank) such that K ≈ U @ V^T.
    """
    m = len(obs_idx)
    n_src = len(src_idx)
    rank_max = min(max_rank, m, n_src)

    U = np.zeros((m, rank_max))
    V = np.zeros((n_src, rank_max))

    # Estimate Frobenius norm via sampling
    n_samples = min(20, m, n_src)
    row_samples = np.random.choice(m, size=n_samples, replace=False)
    col_samples = np.random.choice(n_src, size=n_samples, replace=False)
    est_sq = 0.0
    for i in row_samples:
        row = _kernel_row(src_idx, obs_idx[i], r, n, ds)
        est_sq += np.sum(row ** 2)
    frob_est = np.sqrt(est_sq / n_samples * m) if est_sq > 0 else 1.0
    stop_tol = eps * frob_est

    # Residual tracking: we maintain R = K - U V^T implicitly
    # via pivot row/column updates
    # Simplified approach: store the last pivot column/row of the exact K
    rank = 0
    used_rows = set()
    used_cols = set()

    for k in range(rank_max):
        # Find pivot: random row not yet used, find max entry column
        available_rows = [i for i in range(m) if i not in used_rows]
        if not available_rows:
            break
        i_k = available_rows[np.random.randint(len(available_rows))]

        # Compute the full row of the residual
        row_exact = _kernel_row(src_idx, obs_idx[i_k], r, n, ds)
        # Subtract current approximation
        row_approx = U[i_k, :k] @ V.T[:k, :]
        residual_row = row_exact - row_approx

        j_k = np.argmax(np.abs(residual_row))
        pivot = residual_row[j_k]
        if abs(pivot) < stop_tol:
            break

        # Compute column of residual
        col_exact = _kernel_vector(obs_idx, src_idx[j_k], r, n, ds)
        col_approx = U[:, :k] @ V[j_k, :k]
        residual_col = col_exact - col_approx

        # Normalize
        u = residual_col / pivot
        v = residual_row.copy()

        U[:, k] = u
        V[:, k] = v
        used_rows.add(i_k)
        used_cols.add(j_k)
        rank = k + 1

        # Quick convergence check
        if k > 0 and k % 10 == 0:
            approx_norm = np.linalg.norm(U[:, :rank] @ V[:, :rank].T, 'fro')
            if approx_norm > 0 and abs(pivot) / approx_norm < eps:
                break

    return U[:, :rank], V[:, :rank]


# ---------------------------------------------------------------------------
#  H-matrix block storage
# ---------------------------------------------------------------------------

class DenseBlock:
    """Exact dense storage for a small block."""
    __slots__ = ('obs_idx', 'src_idx', 'data')

    def __init__(self, obs_idx, src_idx, data):
        self.obs_idx = np.asarray(obs_idx, dtype=int)
        self.src_idx = np.asarray(src_idx, dtype=int)
        self.data = data

    def matvec(self, x):
        return self.data @ x[self.src_idx]


class LowRankBlock:
    """ACA-compressed block: U (|obs|×r), V (|src|×r)."""
    __slots__ = ('obs_idx', 'src_idx', 'U', 'V')

    def __init__(self, obs_idx, src_idx, U, V):
        self.obs_idx = np.asarray(obs_idx, dtype=int)
        self.src_idx = np.asarray(src_idx, dtype=int)
        self.U = U
        self.V = V

    def matvec(self, x):
        # (U V^T) @ x[src] = U @ (V^T @ x[src])
        return self.U @ (self.V.T @ x[self.src_idx])


# ---------------------------------------------------------------------------
#  H-matrix assembly
# ---------------------------------------------------------------------------

def build_hmatrix(r, n, ds, er, leaf_size=64, eta=1.0, eps_aca=1e-4, verbose=False):
    """Build an H-matrix representation of the kernel K.

    Returns an object with a .matvec(x) method returning K @ x,
    plus Z1, f_k for the full Z = Z1*I + f_k*K.
    """
    centroids = np.column_stack([r.x, r.y, r.z])
    tree = build_tree(centroids, leaf_size=leaf_size)

    blocks = []  # list of DenseBlock or LowRankBlock
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
    """Recursively assemble block matrix for cluster pair (obs × src)."""
    if obs.size == 0 or src.size == 0:
        return

    if is_admissible(obs, src, eta):
        # ACA compress
        U, V = aca_block(obs.indices, src.indices, r, n, ds, eps=eps_aca)
        if U.shape[1] > 0:
            blocks.append(LowRankBlock(obs.indices, src.indices, U, V))
        return

    # Inadmissible — either compute dense (if small enough) or recurse
    max_dense = 2500  # max elements in a dense block (50×50)
    if obs.size * src.size <= max_dense:
        # Compute dense block
        data = np.zeros((obs.size, src.size))
        for jj, sj in enumerate(src.indices):
            data[:, jj] = _kernel_vector(obs.indices, sj, r, n, ds)
        blocks.append(DenseBlock(obs.indices, src.indices, data))
        return

    # Recurse — split the larger cluster
    if obs.is_leaf and src.is_leaf:
        # Both leaves but too large for dense — force dense anyway?
        data = np.zeros((obs.size, src.size))
        for jj, sj in enumerate(src.indices):
            data[:, jj] = _kernel_vector(obs.indices, sj, r, n, ds)
        blocks.append(DenseBlock(obs.indices, src.indices, data))
        return

    # Split the larger cluster
    if obs.is_leaf:
        _assemble_blocks(obs, src.left, r, n, ds, eta, eps_aca, blocks, verbose)
        _assemble_blocks(obs, src.right, r, n, ds, eta, eps_aca, blocks, verbose)
    elif src.is_leaf:
        _assemble_blocks(obs.left, src, r, n, ds, eta, eps_aca, blocks, verbose)
        _assemble_blocks(obs.right, src, r, n, ds, eta, eps_aca, blocks, verbose)
    else:
        # Split the one with larger diameter
        if obs.diameter >= src.diameter:
            _assemble_blocks(obs.left, src, r, n, ds, eta, eps_aca, blocks, verbose)
            _assemble_blocks(obs.right, src, r, n, ds, eta, eps_aca, blocks, verbose)
        else:
            _assemble_blocks(obs, src.left, r, n, ds, eta, eps_aca, blocks, verbose)
            _assemble_blocks(obs, src.right, r, n, ds, eta, eps_aca, blocks, verbose)


# ---------------------------------------------------------------------------
#  GMRES solve wrapper
# ---------------------------------------------------------------------------

def solve_full_aca(r, n, ds, er, tol=1e-6, leaf_size=64, eta=1.0,
                   eps_aca=1e-4, verbose=False):
    """Solve the full-mesh MoM system using ACA-compressed H-matrix + GMRES.

    Returns X1, X2, X3 (solutions for x, y, z incident fields).
    """
    from scipy.sparse.linalg import gmres

    if verbose:
        print(f"    Building H-matrix (leaf_size={leaf_size}, eta={eta}, "
              f"eps_aca={eps_aca})...")

    H = build_hmatrix(r, n, ds, er, leaf_size=leaf_size, eta=eta,
                      eps_aca=eps_aca, verbose=verbose)
    op = H.as_linearoperator()
    Z1 = H.Z1

    # Diagonal preconditioner
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

# ---------------------------------------------------------------------------
#  Octant-mode ACA solver
# ---------------------------------------------------------------------------

def _octant_kernel_vector(obs_idx, src_j, r, n, ds, sx, sy, sz):
    """Kernel column for octant (sx,sy,sz): K[i,j] with self-term zeroed."""
    vx = r.x[obs_idx] - sx * r.x[src_j]
    vy = r.y[obs_idx] - sy * r.y[src_j]
    vz = r.z[obs_idx] - sz * r.z[src_j]
    r2 = vx * vx + vy * vy + vz * vz
    r3 = np.abs(r2) ** 1.5
    dot_v = vx * (sx * n.x[src_j]) + vy * (sy * n.y[src_j]) + vz * (sz * n.z[src_j])
    col = ds[src_j] * dot_v / r3
    col = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
    # Zero self-term (handled by Z1*I)
    col[obs_idx == src_j] = 0.0
    return col


def _octant_kernel_row(src_idx, obs_i, r, n, ds, sx, sy, sz):
    """Kernel row for octant (sx,sy,sz) with self-term zeroed."""
    vx = r.x[obs_i] - sx * r.x[src_idx]
    vy = r.y[obs_i] - sy * r.y[src_idx]
    vz = r.z[obs_i] - sz * r.z[src_idx]
    r2 = vx * vx + vy * vy + vz * vz
    r3 = np.abs(r2) ** 1.5
    dot_v = vx * (sx * n.x[src_idx]) + vy * (sy * n.y[src_idx]) + vz * (sz * n.z[src_idx])
    row = ds[src_idx] * dot_v / r3
    row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
    # Zero self-term
    row[src_idx == obs_i] = 0.0
    return row


def aca_block_octant(obs_idx, src_idx, r, n, ds, sx, sy, sz, eps=1e-4, max_rank=100):
    """ACA for an octant kernel block."""
    m = len(obs_idx)
    n_src = len(src_idx)
    rank_max = min(max_rank, m, n_src)

    U = np.zeros((m, rank_max))
    V = np.zeros((n_src, rank_max))

    # Frobenius norm estimate
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
    """Build H-matrices for all 8 octant kernels.

    Returns an OctantHMatrixOp with .matvec(x, direction) for 'x','y','z'.
    """
    centroids = np.column_stack([r.x, r.y, r.z])
    tree = build_tree(centroids, leaf_size=leaf_size)

    sx_all = np.array([1, -1, 1, 1, -1, -1, 1, -1])
    sy_all = np.array([1, 1, -1, 1, -1, 1, -1, -1])
    sz_all = np.array([1, 1, 1, -1, 1, -1, -1, -1])

    # For each octant, build the compressed kernel
    octant_blocks = []   # list of 8 lists-of-blocks
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
            """Z_dir @ x = Z1*x + f_k * sum_k s_k * (K_k @ x)"""
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
    """Recursively assemble block matrix for octant cluster pair."""
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
    """Solve octant MoM using ACA-compressed H-matrix + GMRES.

    Returns X1, X2, X3.
    """
    from scipy.sparse.linalg import gmres, LinearOperator

    if verbose:
        print(f"    Building octant H-matrix (leaf_size={leaf_size}, eta={eta})...")

    H = build_octant_hmatrix(r, n, ds, er, leaf_size=leaf_size, eta=eta,
                             eps_aca=eps_aca, verbose=verbose)
    Z1 = H.Z1
    M_op = LinearOperator((H.n, H.n), matvec=lambda v: v / Z1, dtype=complex)

    results = []
    for direction in ['x', 'y', 'z']:
        rhs = {'x': -r.x, 'y': -r.y, 'z': -r.z}[direction]

        # Build direction-specific LinearOperator
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

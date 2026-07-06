# Implementation: Discrete Dipole Approximation (DDA) — 3D Electrostatics

> Part of the **espol-cuboids** project — electrostatic polarizability of cuboids
> and rectangular cylinders.

## Overview

The Discrete Dipole Approximation (DDA) computes the polarizability of a
dielectric object by replacing the continuous material with a cubic lattice
of point dipoles.  Each dipole responds to the local field, which is the sum
of the applied field and the contributions of all other dipoles.  This leads
to a coupled-dipole system that is solved for the dipole moments, from which
the bulk polarizability follows.

In the static (\(k \to 0\)) limit considered here, the DDA is equivalent to a
volume integral equation (VIE) discretised with the simplest possible basis
and testing functions (point dipoles on a regular grid) [1, 2].

## Geometry and Discretization

A rectangular cuboid of dimensions \(l_x \times l_y \times l_z\) is filled
with a cubic lattice of \(N = n_x n_y n_z\) dipoles at positions
\(\{\mathbf{r}_i\}\).  The lattice spacing \(d\) is chosen adaptively:

\[
n_\alpha = \max\left(1, \;{\rm round}(l_\alpha / d)\right),
\qquad
\Delta_\alpha = l_\alpha / n_\alpha  \quad (\alpha = x, y, z)
\]

The dipoles are centred in each voxel.  The spacing is uniform but may differ
slightly along the three axes to preserve the exact aspect ratio.

## Single-Dipole Polarizability

Each dipole responds to the local field via the Clausius–Mossotti relation
(also known as the Lorentz–Lorenz formula), which accounts for depolarization
of a cubic cell [3]:

\[
\alpha_{\rm CM} = 3 \varepsilon_0 V_{\rm cell}
\frac{\varepsilon_r - 1}{\varepsilon_r + 2}
\]

where \(V_{\rm cell} = \Delta_x \Delta_y \Delta_z\) is the volume of one
voxel.  In our units, \(\varepsilon_0 = 1\).

## Coupled-Dipole Equations

For dipole \(i\) at position \(\mathbf{r}_i\), the local field is

\[
\mathbf{E}_{\rm loc}(\mathbf{r}_i) =
\mathbf{E}_0 + \sum_{j \neq i} \mathbf{G}(\mathbf{r}_i - \mathbf{r}_j)
\; \mathbf{p}_j
\]

where \(\mathbf{p}_j = \alpha_{\rm CM} \mathbf{E}_{\rm loc}(\mathbf{r}_j)\)
is the dipole moment at site \(j\), \(\mathbf{E}_0\) is the uniform applied
field, and \(\mathbf{G}(\mathbf{R})\) is the static Green's dyadic

\[
\mathbf{G}(\mathbf{R}) =
\frac{1}{4\pi\varepsilon_0}
\frac{3\hat{\mathbf{R}}\hat{\mathbf{R}}^T - \mathbf{I}}{R^3}
\]

Substituting \(\mathbf{p}_i = \alpha_{\rm CM} \mathbf{E}_{\rm loc}(\mathbf{r}_i)\)
yields the linear system

\[
\sum_j \mathbf{A}_{ij} \mathbf{p}_j = \mathbf{E}_0
\]

where

\[
\mathbf{A}_{ij} =
\begin{cases}
\alpha_{\rm CM}^{-1} \, \mathbf{I}, & i = j \\[4pt]
-\mathbf{G}(\mathbf{r}_i - \mathbf{r}_j), & i \neq j
\end{cases}
\]

The system is of size \(3N \times 3N\) and is real-symmetric for
\(\varepsilon_r > 1\).  The total polarizability tensor is obtained by summing
the dipole moments for three orthogonal applied field directions:

\[
\alpha_{\mu\nu} = \sum_i p_{i,\mu}^{(\nu)},
\qquad \mu,\nu \in \{x,y,z\}
\]

where \(p_i^{(\nu)}\) is the dipole moment at site \(i\) when the applied
field is along \(\nu\).

## Solver Backends

### 1. Dense Direct Solver

The full \(3N \times 3N\) matrix is assembled and solved with LU decomposition.
**Cost:** \(O(N^3)\) time, \(O(N^2)\) memory.
Suitable for \(N \lesssim 2000\) dipoles (system size \(\lesssim 6000\)).

### 2. FFT-Accelerated GMRES (`compute_polarizability_fft`)

For larger systems, the matrix-vector product \(\mathbf{A} \mathbf{p}\) is
evaluated via fast Fourier transform (FFT) without forming the matrix [4, 5].
The key observation is that on a regular lattice, the Green's interaction
is a Toeplitz convolution: the field at site \(i\) depends only on
\(\mathbf{r}_i - \mathbf{r}_j\).

The procedure is:

1.  **Precompute** the Green's kernel on a grid of size
    \((2n_x-1) \times (2n_y-1) \times (2n_z-1)\) and FFT-transform each of the
    6 independent dyadic components \((\mathbf{G}_{xx}, \mathbf{G}_{xy},
    \mathbf{G}_{xz}, \mathbf{G}_{yy}, \mathbf{G}_{yz}, \mathbf{G}_{zz})\)
    to padded grids of size \((3n_x-2) \times (3n_y-2) \times (3n_z-2)\).

2.  **Per matvec:** pad the dipole vector components into the FFT grid,
    FFT, multiply pointwise by the pre-transformed Green's kernels (6 products),
    inverse-FFT, crop to the physical region, and add the self-term
    \(\alpha_{\rm CM}^{-1} \mathbf{p}_i\).

3.  Solve the resulting linear operator with GMRES (SciPy's
    `scipy.sparse.linalg.gmres`).

**Cost per matvec:** \(O(N \log N)\).  **Memory:** \(O(N)\).  
Suitable for \(N \sim 2000\text{–}20000\) dipoles.

## Convergence Extrapolation (\(d \to 0\))

The DDA converges to the exact continuum polarizability as the lattice spacing
\(d \to 0\).  For each shape, we compute the polarizability at 4–5 different
values of \(d\) (typically \(d = s_{\min} / \{6, 10, 14, 18, 22\}\)) and fit

\[
P(d) = P_\infty + C \, d^{\beta}
\]

via nonlinear least squares (L-BFGS-B).  The fitted intercept \(P_\infty\) is
the continuum-limit polarizability.  The exponent \(\beta\) is empirically
found to be in the range 0.3–2.0 depending on geometry and contrast.

## PEC Limit (\( \varepsilon_r \to \infty\))

For PEC objects, the DDA can extrapolate from finite \(\varepsilon_r\):
converged polarizabilities are computed at several \(\varepsilon_r\) values
(e.g., [2, 5, 10, 100, 10 000]), then fitted to

\[
\alpha(\varepsilon_r) = \alpha_{\rm PEC} + \frac{A}{\varepsilon_r}
\]

yielding the perfect-conductor limit.

## Verification

*   Symmetry: cube components are equal.
*   Convergence: decreasing \(d\) yields monotonically changing polarizability.
*   Cross-validation: DDA and MoM 3D agree to within 1–3% for moderate
    \(\varepsilon_r\).
*   Power-law fit recovers exact values from synthetic data.

## References

1.  E. M. Purcell and C. R. Pennypacker, "Scattering and absorption of light
    by nonspherical dielectric grains," *Astrophys. J.*, vol. 186,
    pp. 705–714, 1973.

2.  B. T. Draine and P. J. Flatau, "Discrete-dipole approximation for
    scattering calculations," *J. Opt. Soc. Am. A*, vol. 11, no. 4,
    pp. 1491–1499, 1994.

3.  J. D. Jackson, *Classical Electrodynamics*, 3rd ed.
    New York: Wiley, 1999, sec. 4.5.

4.  J. J. Goodman, B. T. Draine, and P. J. Flatau, "Application of
    fast-Fourier-transform techniques to the discrete-dipole approximation,"
    *Opt. Lett.*, vol. 16, no. 15, pp. 1198–1200, 1991.

5.  M. A. Yurkin and A. G. Hoekstra, "The discrete-dipole-approximation code
    ADDA: capabilities and known limitations," *J. Quant. Spectrosc. Radiat.
    Transfer*, vol. 112, no. 13, pp. 2234–2247, 2011.

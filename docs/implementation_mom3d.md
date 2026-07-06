# Implementation: Method of Moments (MoM) — 3D Electrostatics

> Part of the **espol-cuboids** project — electrostatic polarizability of cuboids
> and rectangular cylinders.

## Overview

We compute the full \(3 \times 3\) polarizability tensor of a dielectric cuboid
with dimensions \(l_x \times l_y \times l_z\) and relative permittivity
\(\varepsilon_r\).  The formulation is a 3D electrostatic surface integral
equation (SIE) discretised with the Method of Moments (MoM).

Three solver backends are available: direct LU, matrix-free iterative GMRES,
and ACA-compressed H-matrix + GMRES.

## Geometry and Mesh

### Mesh generation (`gen_mesh_3d_rect`)

The six faces of the cuboid are each subdivided into rectangular elements.
For the uniform mesh (\(\beta = 0\)), each face gets an equal number of
elements in each direction:

\[
n_{\rm edge} = \lceil {\rm nd} / \min(l_y, l_z) \rceil
\]

with a hard ceiling at nd_max = 100 to bound the total number of surface
elements.  The mesh returns element centres \(\mathbf{r}_i\), outward normals
\(\mathbf{n}_i\), areas \(\Delta s_i\), and total volume \(V = l_x l_y l_z\).

### Octant symmetry

For centro-symmetric cuboids, only one octant of the surface is meshed
(3 of the 6 faces, restricted to the first octant).  The Galerkin integrals
are summed over all 8 octants using symmetry transformations
\((s_x, s_y, s_z) \in \{\pm1\}^3\).  This yields an 8× reduction in the number
of unknowns with no loss of accuracy.

## Integral Equation (PMCHW Formulation)

The governing electrostatic SIE on the closed surface \(S\) of a homogeneous
dielectric is [1, 2]:

\[
\frac{\varepsilon_r + 1}{2} \sigma(\mathbf{r})
+ \frac{\varepsilon_r - 1}{4\pi}
\int_S \sigma(\mathbf{r}')
\frac{(\mathbf{r} - \mathbf{r}') \cdot \mathbf{n}(\mathbf{r}')}
        {|\mathbf{r} - \mathbf{r}'|^3}
\, dS'
= -E_n^{\rm inc}(\mathbf{r})
\]

where \(\sigma\) is the equivalent charge density, and
\(E_n^{\rm inc} = \mathbf{E}_0 \cdot \mathbf{n}\) for a uniform applied field.
This is the 3D analog of the PMCHW equation used in the 2D solver.

## Discretization

With collocation (point-matching) the system becomes

\[
\mathbf{Z} \mathbf{x}_k = \mathbf{y}_k, \quad k \in \{1,2,3\}
\]

where the three RHS vectors correspond to applied fields along
\(\hat{\mathbf{x}}, \hat{\mathbf{y}}, \hat{\mathbf{z}}\).  The system matrix is

\[
Z_{ij} =
\begin{cases}
\displaystyle \frac{\varepsilon_r + 1}{2}, & i = j \\[8pt]
\displaystyle \frac{\varepsilon_r - 1}{4\pi}
\frac{(\mathbf{r}_i - \mathbf{r}_j) \cdot \mathbf{n}_j}
        {|\mathbf{r}_i - \mathbf{r}_j|^3}
\Delta s_j, & i \neq j
\end{cases}
\]

For the **full** (non-octant) solver, a single system matrix is formed and
solved once for the three RHS simultaneously via `scipy.linalg.solve`.

For the **octant** solver, three distinct system matrices are built (one per
applied-field direction), each accounting for the 8 symmetry transformations
in the Green's function sum.  The diagonal blocks remain identical.

### Polarizability extraction

Once the charge density \(\sigma_k\) is known for applied field direction \(k\),

\[
\alpha_{kk} = -(\varepsilon_r - 1) \sum_i \sigma_{i,k} \, n_{i,k} \, \Delta s_i
\]

For the octant solver, the result is multiplied by 8 to account for all octants.
Off-diagonal terms vanish for rectangular cuboids by symmetry.

## Solver Backends

### 1. Direct LU (default)

The dense \(n \times n\) system matrix is factorised with LAPACK
(`scipy.linalg.lu_factor`).  Cost: \(O(n^2)\) memory, \(O(n^3)\) time.
Suitable for \(n \lesssim 2000\) (nd ≤ 40 for most shapes).

### 2. Matrix-free GMRES (`mom3d_kernel`)

The kernel operator \(\mathbf{Z} \mathbf{x}\) is evaluated on-the-fly without
forming the matrix, using the same 8-symmetry sum and the analytical
single-layer kernel.  Solved with SciPy's restarted GMRES.

**Cost:** \(O(n_{\rm iter} \cdot n^2)\) time, \(O(n)\) memory.
Useful for \(n \sim 1000\text{–}4000\) where LU memory becomes prohibitive.

### 3. ACA-compressed H-matrix + GMRES (`mom3d_aca`)

The system matrix is represented as a hierarchically block-separable (H-matrix)
structure using adaptive cross approximation (ACA) [3, 4].  Low-rank blocks
are compressed with tolerance \(\varepsilon_{\rm ACA} = 10^{-4}\) and leaf size
128.  The admissibility parameter \(\eta = 0.5\) controls the H-tree
partitioning.  The compressed system is solved with GMRES.

**Cost:** \(O(n \log n)\) memory and \(O(n \log n)\) per matvec.
Suitable for \(n \gtrsim 2000\).

## Convergence Extrapolation

For each shape, the polarizability is computed at a sequence of discretization
levels nd = [nd₁, …, ndₘ] (typically 4–6 levels).  The data are fitted to a
power law

\[
P({\rm nd}) = P_\infty + C \cdot {\rm nd}^{-\beta}
\]

via nonlinear least squares (`scipy.optimize.minimize` with L-BFGS-B).
The fitted intercept \(P_\infty\) is taken as the continuum limit.

## Verification

*   Symmetry: for a cube, \(\alpha_{xx} = \alpha_{yy} = \alpha_{zz}\).
*   Monotonicity: \(\alpha\) increases with \(\varepsilon_r\).
*   Aspect-ratio ordering: the polarizability along the longest axis is largest.
*   ACA vs. direct agreement (within 5%).
*   Full mesh vs. octant agreement (within 10%).
*   Power-law fit correctly recovers synthetic data.

## References

1.  A. J. Poggio and E. K. Miller, "Integral equation solutions of
    three-dimensional scattering problems," in *Computer Techniques for
    Electromagnetics*, R. Mittra, Ed. Oxford: Pergamon, 1973, ch. 4.

2.  R. F. Harrington, *Field Computation by Moment Methods*.
    New York: Macmillan, 1968.

3.  M. Bebendorf, "Approximation of boundary element matrices,"
    *Numerische Mathematik*, vol. 86, no. 4, pp. 565–589, 2000.

4.  K. Zhao, M. N. Vouvakis, and J.-F. Lee, "The adaptive cross approximation
    algorithm for accelerated method of moments computations of EMC problems,"
    *IEEE Trans. Electromagn. Compat.*, vol. 47, no. 4, pp. 763–773, 2005.

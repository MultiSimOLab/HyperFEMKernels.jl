# Changelog

## Unreleased

- **Functions of the coordinates compiled.** `CellField(f, Ω)`, a Julia function `f` used directly in the
  weak form (`f * v`, `Operation(g)(f, uh)`), on volume, boundary, skeleton and subdomain triangulations,
  are read as a new auxiliary leaf (`CoordArray`) and evaluated by the engine: `f(x)` at Gridap's physical
  quadrature points (read once per group after checking them against the group's geometry, so that a
  function discontinuous on a face gives Gridap's values), only on the cells or faces the group
  integrates, with the operations above them compiled into the integrand. Before, they were evaluated cell by cell through Gridap's lazy
  arrays (e.g. `∫(2 * cfun * H(uf) * v)dΩ` on 64 000 hexahedra, one thread: 47 MB and ~11 ms per assembly).
  A group whose geometry does not reproduce Gridap's points keeps Gridap's array (same values;
  `HFK_DEBUG_COORD=1` prints why). Not yet by the engine: `∇(CellField(f, Ω))` and functions of x
  passed to `update_states!` (still Gridap's arrays).
- `update`: the type of a function of the coordinates is part of the structure; the same function, or a
  closure of the same type with other captured values, is rebound (the new values are used), another
  type compiles again (documented in `update` and the README).
- **Geometry once per cell on affine cells.** At construction each cell is checked for an affine map (every
  node on the affine map through the Jacobian at one point: simplices of order 1, rectangles,
  parallelograms and parallelepipeds — also in unstructured meshes —, straight-sided higher-order cells);
  on those cells the kernels compute `Jt`, `inv(Jt)` and the measure once and reuse them at every
  quadrature point (residual, Jacobian, matrix-free `prepare!`, skeletons, functionals, FE functions in
  the form, `update_states!`). Curved or deformed cells keep the per-point computation; results change only
  by round-off. `HFK_AFFINE_GEOMETRY=0` (read when the package is loaded) disables it.

## 0.2.4

- **Gridap 0.20**: Gridap 0.19 is no longer supported (`[compat] Gridap = "0.20"`). H(div)
  (Raviart–Thomas) spaces are read from the cell basis of Gridap 0.20, where the sign flips are a diagonal
  change of basis per cell. The rest of the engine needed no change.

## 0.2.3

- Second derivatives (`∇∇`, `Δ`) on skeleton integrals: `jump(∇∇(u))`, `mean(∇∇(u))`, `mean(Δ(u))`, … of
  the unknown, test and trial functions are compiled by the facet kernels (C⁰ interior-penalty terms,
  Hessian-jump regularisations ⟦∇∇u⟧:⟦∇∇v⟧), with the Hessian push-forward of each side (its Jt⁻¹ and the
  curvature of its cell map: exact on non-affine quadrilaterals and hexahedra); residual, assembled and
  matrix-free Jacobian. They no longer send the skeleton to the Gridap fallback.
- `KernelFunctional` evaluates skeleton integrals with the facet kernels (energies of jump penalties such
  as `½ ∫ c ⟦∇u⟧:⟦∇u⟧ dΛ` and `½ ∫ c ⟦∇∇u⟧:⟦∇∇u⟧ dΛ`, used by the Armijo line search; `:min`/`:max` over the
  facet points). Before, those terms were summed by the Gridap fallback.

## 0.2.2

- p-multigrid preconditioner `PMultigrid` for `MatrixFreeSolver`: nested Lagrangian levels on the same mesh
  (Q2 → Q1, Q3 → Q1, Q3 → Q2 → Q1), Chebyshev smoothing (`Chebyshev`, default degree 2) on the matrix-free
  working order, assembled coarse level with its own solver (e.g. one GAMG V-cycle); the coarse operators are
  rediscretised from the Newton's forms at the injected state, and the levels follow load steps (`update`) and
  the Dirichlet projection.
- Transfers between nested spaces: `Prolongation`, `prolongate!`, `restrict!` (exact transpose), `inject!`.
- Nodal blocks from the stored point tangents (`block_diagonal!` route `:tangents`, ×3.5 cheaper than the
  element pass), also for `BlockJacobi`.
- `preconditioner_ok`: a non-finite tangent, a singular nodal block or an indefinite tangent makes the linear
  solve fail at once (no exception), so the Newton cuts back.
- Order-3 tensor-product tables and matrix-free product tested in CI.

## 0.2.1

- Second derivatives of the unknown, test and trial functions (`∇∇(u)`, `∇(∇(u))`, `Δ(u)`) compiled by
  the kernels, with Gridap's layout; exact on non-affine quadrilaterals and hexahedra (curvature of the
  cell map, where Gridap's `∇∇` is not defined); residual, assembled Jacobian and matrix-free Jacobian;
  `∇∇` of FE functions that are not the unknown read from their nodal values. `∇∇` on skeleton
  integrals and of H(div) fields goes to the Gridap fallback.
- Hybrid fallback by triangulation: an integral the engine cannot translate (e.g. `∇(u*w)`) sends only
  the integrals of its triangulation to Gridap, not the whole weak form.
- Cheap `update`: forms with the structure of the operator's (same measures, operations and constants
  up to their values, same FE functions) rebind the compiled groups — nothing is compiled or allocated
  per cell per load step; `HFK_CHECK_REBIND=1` checks the fast path against a full compilation. The
  same for `KernelFunctional`.
- Numbers of the integrand are read as parameters also on simplexified meshes; `HFK_DEBUG_LINEAR` is
  read at run time.

## 0.2.0

- `KernelOperator(…; jacobian = :matrixfree)`, `MatrixFreeSolver` (Krylov.jl) with the preconditioner
  slots `LaggedAssembled` and `BlockJacobi`, `KernelFunctional`; lazy sparsity pattern
  (`sparsity_pattern`, `nonzero_positions`; `op.A0`/`op.nzpos` removed).

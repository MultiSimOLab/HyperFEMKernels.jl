# Changelog

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

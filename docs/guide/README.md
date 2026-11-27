# HyperFEMKernels.jl — user guide

Detailed reference for HyperFEMKernels: every feature, every usage pattern and the current limits.
The [README](../../README.md) has the short overview.

## Features

- **Drop-in operator.** `KernelOperator(res, jac, U, V)` is a Gridap `NonlinearOperator`: it works
  with Gridap's and GridapSolvers' nonlinear solvers and with HyperFEM's Newton–Raphson.
- **Same results as Gridap.** The generated code calls the same Julia functions and `TensorValue`
  operations that Gridap applies pointwise; agreement is checked to machine precision (`check`).
- **Integrals:** volume integrals, boundary integrals (Neumann, Robin, follower loads with the
  normal) and skeleton integrals on interior facets (`plus`/`minus`, `jump`, `mean`).
- **Second derivatives.** `∇∇(u)`, `∇(∇(u))` and `Δ(u)` of the unknown, test and trial functions
  (Hessian regularisations `β ∫ ∇∇u ⊙ ∇∇v`, biharmonic and C⁰-interior-penalty terms) with Gridap's
  layout (`TensorValue{D,D}`, `ThirdOrderTensorValue{D,D,N}` with `H[a,b,k] = ∂²u_k/∂x_a∂x_b`),
  on simplices and Cartesian cells as Gridap's `∇∇` and, unlike it, also on non-affine
  quadrilaterals and hexahedra (the curvature of the cell map is included). Also on skeleton
  integrals: `jump(∇∇(u))`, `mean(Δ(u))`, … with the Hessian of each side.
- **Spaces:** Lagrangian spaces of any order and H(div)-conforming Raviart–Thomas spaces (contravariant
  Piola map, per-cell dof signs) on hexahedra, quadrilaterals, tetrahedra and triangles; single- and
  multi-field (e.g. Biot u–W, mixed Darcy); Dirichlet masks; subdomains. Lagrangian spaces also on
  manifold triangulations (curves in 2D/3D, surfaces in 3D, e.g. an interface
  `BoundaryTriangulation(Ωsolid, tags = "Interface")`), with tangential gradients as in Gridap.
- **Data in the form.** `FEFunction`s that are not the unknown (a previous state, a field interpolated
  from another physics or from boundary data), their gradients and their Hessians are evaluated by the engine from
  their nodal values at every assembly, with the operations above them compiled into the integrand.
  Functions of the coordinates (`CellField(f, Ω)`, or a Julia function `f` used directly in the form,
  also on boundaries, skeletons and subdomains) are evaluated by the engine as `f(x)` at the physical
  quadrature points (Gridap's own points, read once), without allocations. Other `CellField`s (and, for
  now, gradients of functions of x and functions of x passed to `update_states!`) are evaluated by Gridap
  once per assembly into dense buffers that are reused.
- **Affine cells.** Cells whose map is affine (simplices of order 1, rectangles, parallelograms,
  parallelepipeds, straight-sided higher-order cells, also in unstructured meshes) are detected at
  construction and their Jacobian, inverse and measure are computed once per cell instead of at every
  quadrature point (assembly, matrix-free, skeletons, functionals, FE functions in the form, internal
  variables); curved or deformed cells keep the per-point geometry (`HFK_AFFINE_GEOMETRY=0`, read when the
  package is loaded, disables the shortcut).
- **Factorised Jacobians.** When a term of the Jacobian is a sum of products of a factor that depends only on the
  test function and one that depends only on the trial function (by ⋅, ⊙ or *), under multipliers and divisors that
  depend on the state, the tangent at a quadrature point is built as T[i, j] = c·W(N(∂A/∂s_i, ∂B/∂r_j)) with one
  level of dual numbers instead of two (the stabilised SUPG/PSPG terms of the CFD formulations, for example). The generated
  code is checked at construction against the original integrand at random points; a group that does not take this
  form, or fails the check, keeps the nested dual numbers. `HFK_FACTORISE=0` disables it, `HFK_DEBUG_FACTORISE=1`
  prints why a group is not factorised, and `linear_maps = false` (nested dual numbers everywhere) turns it off too.
- **Threads, no allocations.** Cells are coloured so that the assembly loop is threaded without
  locks; element kernels allocate nothing (`allocations` reports it), which is also the
  requirement for GPU execution. The element buffers are kept in the operator, so a repeated
  `residual!`/`jacobian!` with one thread allocates nothing (with threads, only the tasks: a fixed
  amount per call); auxiliary data left to Gridap (other `CellField`s) still allocates in Gridap. An
  operator is not reentrant: do not assemble it, or operators `update` derived from it, from two tasks at
  once (each assembly is threaded internally).
- **Tuning switches** (read when the package is loaded; `HyperFEMKernels.read_env_settings!()` reads them
  again, for operators built afterwards): `HFK_MATRIX_PRODUCTS` (`1`/`0`, element residual and Jacobian as
  matrix products on node-contiguous fields), `HFK_WORK_PER_TASK` (minimum work per task, default 20000),
  `HFK_POINT_BLOCK` (points per block of the matrix-product Jacobian, default 12, 0 = all),
  `HFK_AFFINE_GEOMETRY` (`1`/`0`, geometry once per affine cell), `HFK_FACTORISE` (`1`/`0`, factorised Jacobians), `HFK_AUX_WORK_PER_TASK` (minimum quadrature
  points per task when the engine fills FE functions and functions of x on volume and boundary groups, default 512:
  small meshes are filled without tasks; skeleton groups and data evaluated by Gridap keep one task per thread).
- **Cheap updates.** Numbers in the weak form (load factors, time steps, material parameters) are
  run-time parameters: `update(op, res, jac)` with forms of the same structure (same measures,
  operations and constants up to their values, same FE functions, functions of the coordinates of
  the same type) rebinds the compiled groups —
  code, tables, sparsity pattern, colouring, linear maps and the dense auxiliary buffers are reused and
  only the values change; nothing is compiled or allocated per cell. A form of another structure is
  compiled again, still reusing tables, geometry and colouring. The same for `KernelFunctional`.
  The type of an operation or of a function of the coordinates is structure, its captured values are
  not: a closure created again with other captured values (`a -> (x -> a * x[1])`) is rebound and its
  new values are used (not with `cache_aux = true`, whose buffers are never refilled), while a function
  of another type (another closure, another named function) compiles again.
- **Matrix-free Jacobian.** `MatrixFreeJacobian` applies `J(x)·dx` without assembling, with sum
  factorisation on quadrilaterals and hexahedra; `KernelOperator(…; jacobian = :matrixfree)` hands
  it to a Newton solver and `MatrixFreeSolver` (Krylov.jl) solves the tangent systems on it, with
  preconditioners that assemble every k iterations (`LaggedAssembled`) or never (`BlockJacobi`).
- **Scalar functionals.** `KernelFunctional` evaluates energies and pointwise extrema (the injectivity
  step of a line search) with the same kernels, without assembling.
- **Internal variables.** Gridap's `update_state!` (visco-elasticity, plasticity, …) executed by
  threaded kernels with the same updater and arguments; `KernelState` stores the states as a dense
  `np × ncells` matrix.
- **Never blocks a simulation.** Integrals the engine cannot translate are assembled by Gridap and added
  to the result, with a warning naming the triangulation and the reason; the integrals on the other
  measures of the same weak form stay in the kernels.

## Usage

### A nonlinear problem with Gridap

The residual and Jacobian are written exactly as for Gridap's `FEOperator`; only the operator
changes.

```julia
using Gridap, HyperFEMKernels
using GridapSolvers.NonlinearSolvers: NewtonSolver

# St Venant–Kirchhoff hyperelasticity
const λ, μ = 2.0, 1.0
Fk(∇u)    = one(∇u) + ∇u
E(F)      = 0.5 * (F' ⋅ F - one(F))
S(F)      = λ * tr(E(F)) * one(F) + 2μ * E(F)
P(F)      = F ⋅ S(F)
dS(F, dF) = (dE = 0.5 * (dF' ⋅ F + F' ⋅ dF); λ * tr(dE) * one(F) + 2μ * dE)
dP(F, dF) = dF ⋅ S(F) + F ⋅ dS(F, dF)

model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (20, 10, 10))
Ω  = Triangulation(model)
dΩ = Measure(Ω, 4)
# face x = 0 clamped, face x = 1 stretched
V  = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 2), dirichlet_tags = ["tag_25", "tag_26"])
U  = TrialFESpace(V, [VectorValue(0.0, 0.0, 0.0), VectorValue(0.05, 0.0, 0.0)])

res(u, v)     = ∫(∇(v)' ⊙ (P ∘ (Fk ∘ ∇(u)')))dΩ
jac(u, du, v) = ∫(∇(v)' ⊙ (dP ∘ (Fk ∘ ∇(u)', ∇(du)')))dΩ

op = KernelOperator(res, jac, U, V)       # instead of FEOperator(res, jac, U, V)

nls = NewtonSolver(LUSolver(); maxiter = 20, rtol = 1e-10, verbose = true)
x = zeros(num_free_dofs(U))
solve!(x, nls, op)
uh = FEFunction(U, x)
```

Run Julia with threads (`julia -t auto`) to use them; `KernelOperator(...; nthreads = n)` sets
the number explicitly.

### Load stepping

Numbers inside the weak form are parameters of the compiled code. A new load factor or time step
only rebuilds the operator's terms, keeping tables, sparsity pattern, colouring and compiled code:

```julia
# face x = 0 clamped, traction on face x = 1
V  = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 2), dirichlet_tags = "tag_25")
U  = TrialFESpace(V, VectorValue(0.0, 0.0, 0.0))
dΓ = Measure(BoundaryTriangulation(model, tags = "tag_26"), 4)
t  = VectorValue(0.0, 0.0, -0.01)

res(Λ) = (u, v) -> ∫(∇(v)' ⊙ (P ∘ (Fk ∘ ∇(u)')))dΩ - Λ * ∫(t ⋅ v)dΓ
jac(Λ) = (u, du, v) -> ∫(∇(v)' ⊙ (dP ∘ (Fk ∘ ∇(u)', ∇(du)')))dΩ

op = KernelOperator(res(0.0), jac(0.0), U, V)
x  = zeros(num_free_dofs(U))
for Λ in 0.1:0.1:1.0
    op = update(op, res(Λ), jac(Λ))     # no recompilation
    solve!(x, nls, op)
end
```

### Checking against Gridap

`check` assembles with both backends at a given state and returns the relative differences of the
residual and the Jacobian. `allocations` reports the bytes allocated by each element kernel (0 is
expected; a nonzero value points to a pointwise function of the weak form that allocates).

```julia
c = check(res, jac, U, V, x)              # (res = …, jac = …), both ≈ 1e-15 or below
allocations(op, x)                        # [(kind = :residual, group = 1, bytes = 0), …]
```

### Multi-field problems

Multi-field weak forms are written as in Gridap. Electromechanics with HyperFEM's constitutive
models:

```julia
using HyperFEM
physmodel = ElectroMechModel(mechano = NeoHookean3D(λ = 10.0, μ = 1.0), electro = IdealDielectric(ε = 1.0))
Ψ, ∂ΨF, ∂ΨE, ∂ΨFF, ∂ΨEF, ∂ΨEE = physmodel()
F, _, _ = get_Kinematics(Kinematics(Mechano, Solid))
E       = get_Kinematics(Kinematics(Electro, Solid))

res((u, φ), (v, vφ)) = ∫(∇(v)' ⊙ (∂ΨF ∘ (F ∘ ∇(u)', E ∘ ∇(φ))))dΩ -
                       ∫(∇(vφ) ⋅ (∂ΨE ∘ (F ∘ ∇(u)', E ∘ ∇(φ))))dΩ
jac((u, φ), (du, dφ), (v, vφ)) =
    ∫(∇(v)' ⊙ ((∂ΨFF ∘ (F ∘ ∇(u)', E ∘ ∇(φ))) ⊙ ∇(du)'))dΩ +
    ∫(∇(vφ)' ⋅ ((∂ΨEE ∘ (F ∘ ∇(u)', E ∘ ∇(φ))) ⋅ ∇(dφ)))dΩ -
    ∫(∇(dφ) ⋅ ((∂ΨEF ∘ (F ∘ ∇(u)', E ∘ ∇(φ))) ⊙ ∇(v)'))dΩ -
    ∫(∇(vφ) ⋅ ((∂ΨEF ∘ (F ∘ ∇(u)', E ∘ ∇(φ))) ⊙ ∇(du)'))dΩ

op = KernelOperator(res, jac, U, V)       # U, V: MultiFieldFESpace
```

The Jacobian's sparsity pattern is Gridap's: a (test field, trial field) block is stored only on the cells (and
facets) of the integrals with a term that couples the two fields. In unstabilised Stokes the pressure–pressure
block is not stored, and a coupling integrated on part of the domain or on a boundary only reserves those cells.
The blocks follow from the structure of the form, not from values: a term multiplied by a number that is zero
keeps its block. `update` keeps the pattern (and the matrices already allocated) when the new form fits in it,
and builds a new one when it adds a block: allocate the matrix again (`jacobian!` checks the structure of a matrix
it has not seen and refuses one with another pattern). With a Jacobian part assembled by Gridap (hybrid fallback)
every block of every cell is stored, as before. As in Gridap, a block without a term has no stored diagonal
(the pressure block of unstabilised Stokes): ILU or SOR preconditioners that need the diagonal fail there.

### With HyperFEM's drivers

HyperFEM selects the assembly backend with a keyword; everything else in the simulation (models,
boundary conditions, solvers, post-processing) is unchanged:

```julia
comp_model = StaticNonlinearModel(res, jac, U, V, D_bc; nls = nls_, backend = :compiled)
x = solve!(comp_model; stepping = (nsteps = 5, maxbisec = 5), post = post_model)
```

`StaticLinearModel` accepts the same keyword, and `CellState(model, dΩ; backend = :compiled)` /
`update_state!(model, states, args...; backend = :compiled)` move the internal variables of HyperFEM's
visco-elastic models to the engine.

### Matrix-free Jacobian

```julia
mf = MatrixFreeJacobian(op)
prepare!(mf, x)                           # tangents at the state x (cheaper than assembling)
y  = mf * dx                              # J(x)·dx, or mul!(y, mf, dx), e.g. inside a Krylov solver
```

On quadrilaterals and hexahedra of order 2 and higher the matrix-free product is as fast as or
faster than a sparse matrix–vector product, with about a third of the memory (our measurement, Mac
M-series, Julia 1.13, 6 threads, Q2 electro-mechanics: 0.8 ms vs 4.1 ms at 57k dofs, 15 ms vs 77 ms
at 974k; on P1 tetrahedra and P2 triangles the dense path is slower than the sparse product).

### Matrix-free Newton

With `jacobian = :matrixfree` the operator gives a Newton solver the `MatrixFreeJacobian` in place
of the matrix (`allocate_jacobian`/`jacobian!` prepare it, the sparsity pattern is never built unless
a preconditioner assembles), and `MatrixFreeSolver` — a Gridap `LinearSolver` — solves the tangent
systems with Krylov.jl's GMRES, FGMRES or CG on the operator:

```julia
op  = KernelOperator(res, jac, U, V; jacobian = :matrixfree)
ls  = MatrixFreeSolver(; krylov = :gmres, rtol = 1e-8,
                        preconditioner = LaggedAssembled(LUSolver(); every = 3))
# in a Newton loop: A = allocate_jacobian(op, x); jacobian!(A, op, x); ns = numerical_setup(symbolic_setup(ls, A), A); solve!(dx, ns, b)
```

Preconditioner slots: `LaggedAssembled(solver; every = k)` assembles the tangent every `k` Newton
iterations and sets up any Gridap solver on it (LU, HyperFEM's `PETScSolver`/`AMGSolver`, MUMPS);
in between, the exact tangent is applied matrix-free, so the Newton trajectory is the exact one
while the factorisations or AMG hierarchies are paid `1/k` of the time (our measurement, same
machine: Q2 electro-mechanics, 218k dofs, MUMPS as the slot's solver, ×5 on the whole `solve!` with
`k = 3` and the same 20 Newton iterations as MUMPS at every iteration).
`BlockJacobi()` is the nodal block diagonal, computed from the point tangents the matrix-free
Jacobian already stores (no matrix, no second element pass). With HyperFEM,
`StaticNonlinearModel(…; backend = CompiledBackend(jacobian = :matrixfree),
nls = NewtonRaphson(MatrixFreeSolver(…)))` runs this inside the drivers (line searches included).

### p-multigrid

On quadrilaterals and hexahedra of order 2 and higher the tangent never needs to be assembled:
`PMultigrid` is a preconditioner slot that runs one V-cycle per Krylov iteration over lower-order
spaces on the same mesh — Chebyshev smoothing on the block-preconditioned tangent of every
matrix-free level, exact transfers between the nested Lagrangian spaces, and a coarse solver on the
assembled lowest-order level. The coarse operators are rediscretised from the same weak form at the
state interpolated onto each level, and follow the Newton's operator from load step to load step.

```julia
# spaces of every level, finest first: the working space and the Q1 space on the same mesh and boundary conditions
V1 = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), dirichlet_tags = tags)
spaces = [(U, V), (TrialFESpace(V1, g), V1)]          # with HyperFEM: spaces = level_spaces(U, V, D)
N1 = near_nullspace(spaces[2][1])                     # rigid-body modes of the Q1 space (HyperFEM)
ls = MatrixFreeSolver(; krylov = :gmres, rtol = 1e-8,
                       preconditioner = PMultigrid(spaces; coarse = PETScSolver(:preonly, :gamg; nullspace = N1),
                                                   coarse_every = 3))
```

`coarse` is any Gridap `LinearSolver`; one AMG V-cycle (HyperFEM's `PETScSolver(:preonly, :gamg)`
with the rigid-body modes of the coarse space) is the production choice, `LUSolver()` the reference.
`smoother = Chebyshev(; degree = 2, range = 20)` by default. The levels must have the same fields and
the same closed Dirichlet boundary (tags with their edges and vertices). Our measurement (Mac
M-series, Julia 1.13, 6 threads, PETSc serial; nonlinear elasticity on a unit cube, Q2, 5 load steps,
Krylov rtol 1e-10): the same Newton iterations as CG + GAMG on the assembled tangent, 25–28 GMRES
iterations per Newton iteration from 25k to 957k dofs, and per Newton iteration ×4.8 faster at 197k
and ×8.3 at 957k with a quarter of the memory (4.3 vs 16.8 GB); on Q3 at 227k dofs ×19. On slender
structures (few elements across the thickness) the point smoother limits it (×3 with 86 iterations
on a 50:5:1 Q2 cantilever, measured with a degree-4 smoother). The coarse spaces' Dirichlet values
are overwritten by the slot; an indefinite tangent (past a limit point) or a non-finite level makes
the linear solve fail, which the Newton handles as a failed iteration.

### Scalar functionals

```julia
E   = KernelFunctional(u -> ∫(Ψ ∘ (F ∘ ∇(u)'))dΩ, U)                 # energy: E(x) = Σ ∫ Ψ
α   = KernelFunctional(u -> ∫(g ∘ (∇(u)', ∇(duh)'))dΩ, U; reduce = :min)   # pointwise minimum
```

`reduce = :sum` integrates; `:min`/`:max` return the extremum of the integrand over the quadrature
points (no weights). `duh` is an FE function whose nodal values are read at every evaluation, so a
line search updates the increment in place without recompiling.

### Internal variables

Gridap's `update_state!(updater, states..., fields...)` is executed by the engine with the same
updater, arguments and semantics:

```julia
A = KernelState(I3, dΩ)                   # or Gridap's CellState(I3, dΩ); both usable in weak forms
upd(A, F, Fn) = (true, return_mapping(F, Fn, A))

update_states!(upd, A, F ∘ ∇(uh)', F ∘ ∇(uh⁻)')                 # engine
update_state!(KernelUpdater(upd), A, F ∘ ∇(uh)', F ∘ ∇(uh⁻)')   # same, through Gridap's name
```

## Current scope

- Lagrangian (`H1`) and Raviart–Thomas (`H(div)`) spaces in volume and boundary integrals and in the
  matrix-free Jacobian. Raviart–Thomas fields on skeleton integrals (the terms go to the Gridap fallback)
  and Nédélec (`H(curl)`) spaces are not supported yet.
- On manifold triangulations (cells of lower dimension than the space): Lagrangian spaces in cell
  integrals. Boundary and skeleton integrals of those spaces and Raviart–Thomas fields on them go to
  the Gridap fallback; internal variables on them are not supported yet.
- Second derivatives of Lagrangian fields in cell, boundary and skeleton integrals and in the matrix-free
  Jacobian (dense weights; sum factorisation is used for the fields without Hessian terms). `∇∇` of
  Raviart–Thomas fields goes to the Gridap fallback; third derivatives too.
- p-multigrid for single-field problems; multi-field (u, φ) saddle points need a block smoother and
  are not covered yet (use `LaggedAssembled` with HyperFEM's block preconditioners).
- Shared-memory parallelism (threads). Distributed runs with GridapDistributed and GPU execution
  are design targets: kernels are allocation-free and states use a GPU-friendly layout.
- Gridap is restricted to the 0.20 series, because the engine reads Gridap's internal expression tree.
- Boundary integrals on a `CartesianDiscreteModel` with a nonlinear `map` in 3D: Gridap builds the faces of
  that model with affine face maps while its cells are trilinear, so on curved hexahedra Gridap's boundary
  integrals differ from the kernel's (about 1 % in our tests), which integrate on the faces of the cells.
  The kernel agrees with Gridap on `UnstructuredDiscreteModel(model)`, whose faces are consistent with the
  cells; use that model there. On the Cartesian model itself, functions of x on those faces are evaluated by
  Gridap at its (affine-face) points while the measure is that of the faces of the cells, so the result
  matches neither. 2D (straight edges) and affine models are not affected.

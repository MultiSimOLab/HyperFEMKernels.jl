[![CI](https://github.com/MultiSimOLab/HyperFEMKernels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MultiSimOLab/HyperFEMKernels.jl/actions/workflows/CI.yml?branch=main)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

# HyperFEMKernels.jl

**Fused, threaded finite-element assembly for [Gridap.jl](https://github.com/gridap/Gridap.jl) weak forms.**

HyperFEMKernels keeps Gridap's way of writing a problem — the weak form as an explicit sum of
integrals `∫(...)dΩ` — and executes it with compiled, allocation-free element kernels that run in
parallel over the cells. Gridap still owns the mesh, the `FESpace`s, Dirichlet conditions, DOF
numbering, `Measure`s, `FEFunction`s and post-processing; HyperFEMKernels replaces only the
evaluation of the integrals and the assembly of residuals and Jacobians.

The weak form is read at run time from the expression tree that Gridap builds, and translated into
generated Julia code. No per-physics description, macro or annotation is needed: residuals and
Jacobians written for Gridap's `FEOperator` work as they are, including those built inside
functions, loops and conditionals, new constitutive models and new physics.

[HyperFEM.jl](https://github.com/MultiSimOLab/HyperFEM.jl) uses it as an alternative assembly
backend for its drivers (`backend = :compiled`).

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
  quadrilaterals and hexahedra (the curvature of the cell map is included).
- **Spaces:** Lagrangian spaces of any order and H(div)-conforming Raviart–Thomas spaces (contravariant
  Piola map, per-cell dof signs) on hexahedra, quadrilaterals, tetrahedra and triangles; single- and
  multi-field (e.g. Biot u–W, mixed Darcy); Dirichlet masks; subdomains. Lagrangian spaces also on
  manifold triangulations (curves in 2D/3D, surfaces in 3D, e.g. an interface
  `BoundaryTriangulation(Ωsolid, tags = "Interface")`), with tangential gradients as in Gridap.
- **Data in the form.** `FEFunction`s that are not the unknown (a previous state, a field interpolated
  from another physics or from boundary data), their gradients and their Hessians are evaluated by the engine from
  their nodal values at every assembly, with the operations above them compiled into the integrand;
  other `CellField`s are evaluated by Gridap once per assembly into dense buffers that are reused.
- **Threads, no allocations.** Cells are coloured so that the assembly loop is threaded without
  locks; element kernels allocate nothing (`allocations` reports it), which is also the
  requirement for GPU execution.
- **Cheap updates.** Numbers in the weak form (load factors, time steps, material parameters) are
  run-time parameters: `update` with new values reuses tables, sparsity pattern, colouring and the
  compiled code.
- **Matrix-free Jacobian.** `MatrixFreeJacobian` applies `J(x)·dx` without assembling, with sum
  factorisation on quadrilaterals and hexahedra; `KernelOperator(…; jacobian = :matrixfree)` hands
  it to a Newton solver and `MatrixFreeSolver` (Krylov.jl) solves the tangent systems on it, with
  preconditioners that assemble every k iterations (`LaggedAssembled`) or never (`BlockJacobi`).
- **Scalar functionals.** `KernelFunctional` evaluates energies and pointwise extrema (the injectivity
  step of a line search) with the same kernels, without assembling.
- **Internal variables.** Gridap's `update_state!` (visco-elasticity, plasticity, …) executed by
  threaded kernels with the same updater and arguments; `KernelState` stores the states as a dense
  `np × ncells` matrix.
- **Never blocks a simulation.** Terms the engine cannot translate are assembled by Gridap and added
  to the result, with a warning naming the term.

## Performance

Our measurements on an 18-thread Apple laptop; Jacobians equal to Gridap's to 1e-15.

| Case | Gridap | Kernel, 1 thread | Kernel, 18 threads |
|---|---|---|---|
| Mixed hyperelasticity, 7 700 Q1 hexahedra, 35 343 DOFs (Jacobian) | 0.382 s (AD) | 0.093 s | 0.010 s (×36) |
| Oseen–BDF2 Navier–Stokes, P2–P1 + SUPG/PSPG/LSIC, 468 926 DOFs | 19.2 s | 1.24 s | 0.137 s (×140) |
| Update of visco-elastic internal variables, Q2, 86 400 points | 0.064 s | 0.065 s | 0.0075 s (×8.5) |

The scripts that produce these figures are in [`benchmark/`](benchmark).

## Installation

From the Julia REPL:

```julia
pkg> add HyperFEMKernels
```

Until the package appears in the General registry, add it from the repository instead:

```julia
pkg> add https://github.com/MultiSimOLab/HyperFEMKernels.jl
```

It requires Julia ≥ 1.12 and Gridap 0.19.9 (the same version as HyperFEM).

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
`BlockJacobi()` is the nodal block diagonal gathered without a matrix (the smoother of the coming
p-multigrid). With HyperFEM, `StaticNonlinearModel(…; backend = CompiledBackend(jacobian = :matrixfree),
nls = NewtonRaphson(MatrixFreeSolver(…)))` runs this inside the drivers (line searches included).

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

## Examples

- [`examples/aorta_cfd`](examples/aorta_cfd) — pulsatile blood flow in a patient aorta
  (incompressible Navier–Stokes, Oseen–BDF2, Taylor–Hood P2–P1 with SUPG/PSPG/LSIC stabilisation,
  backflow and Windkessel outlets). `mode=compare` assembles with Gridap and with the kernels at every
  step and times both.
- [`examples/hmre_vacuum_2d`](examples/hmre_vacuum_2d) — hard-magnetic soft beam with explicit
  surrounding air, three staggered problems (magnetic, solid, mesh motion of the air) with HyperFEM's
  drivers and `backend = :lazy | :compiled`.

HyperFEM's simulation tests (electromechanics, hMRE, visco-elasticity, third-medium contact, Stokes)
run with both backends as part of the test suite.

## Current scope

- Lagrangian (`H1`) and Raviart–Thomas (`H(div)`) spaces in volume and boundary integrals and in the
  matrix-free Jacobian. Raviart–Thomas fields on skeleton integrals (the terms go to the Gridap fallback)
  and Nédélec (`H(curl)`) spaces are not supported yet.
- On manifold triangulations (cells of lower dimension than the space): Lagrangian spaces in cell
  integrals. Boundary and skeleton integrals of those spaces and Raviart–Thomas fields on them go to
  the Gridap fallback; internal variables on them are not supported yet.
- Second derivatives of Lagrangian fields in cell and boundary integrals and in the matrix-free
  Jacobian (dense weights; sum factorisation is used for the fields without Hessian terms). `∇∇` on
  skeleton integrals and of Raviart–Thomas fields goes to the Gridap fallback; third derivatives too.
- Shared-memory parallelism (threads). Distributed runs with GridapDistributed and GPU execution
  are design targets: kernels are allocation-free and states use a GPU-friendly layout.
- Gridap is pinned to 0.19.9, because the engine reads Gridap's internal expression tree.

## Testing

```julia
pkg> test HyperFEMKernels
```

Every feature is verified against Gridap's own assembly. The tests on HyperFEM's weak forms and
simulations are in [`test/hyperfem`](test/hyperfem).

## How to cite

If you use HyperFEMKernels, please cite HyperFEM:

C. Perez‐Garcia, R. Ortigosa, J. Martínez‐Frutos, and D. Garcia‐Gonzalez, **Topology and material
optimization in ultra-soft magnetoactive structures: making advantage of residual anisotropies.**
Adv. Mater. (2025): e18489. https://doi.org/10.1002/adma.202518489

along with the required citations for [Gridap](https://github.com/gridap/Gridap.jl).

## Project funded by

- Grants PID2022-141957OA-C22/PID2022-141957OB-C22 funded by MCIN/AEI/ 10.13039/501100011033 and by ''ERDF A way of making Europe''

<p align="center"><img src="docs/img/aei.png" width="70%"></p>

## Contact

Contact the project administrator [Jesús Martínez-Frutos](mailto:jesus.martinez@upct.es) for further
questions about licenses and terms of use.

## License

MIT.

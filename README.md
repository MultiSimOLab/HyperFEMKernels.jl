[![CI](https://github.com/MultiSimOLab/HyperFEMKernels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MultiSimOLab/HyperFEMKernels.jl/actions/workflows/CI.yml?branch=main)
[![Coverage](https://codecov.io/gh/MultiSimOLab/HyperFEMKernels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MultiSimOLab/HyperFEMKernels.jl)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

# HyperFEMKernels.jl

**Fused, threaded finite-element assembly for [Gridap.jl](https://github.com/gridap/Gridap.jl) weak forms.**

You write the weak form as in Gridap (`∫(...)dΩ`); HyperFEMKernels reads the expression tree that
Gridap builds, compiles it into allocation-free element kernels and assembles residuals and Jacobians
in parallel over the cells. Gridap still owns the mesh, the `FESpace`s, boundary conditions,
`Measure`s and post-processing. No macros or per-physics annotations are needed.

[HyperFEM.jl](https://github.com/MultiSimOLab/HyperFEM.jl) uses it as an assembly backend
(`backend = :compiled`).

## Features

- **Drop-in operator:** `KernelOperator(res, jac, U, V)` is a Gridap `NonlinearOperator`.
- **Same results as Gridap**, checked to machine precision (`check`).
- **Volume, boundary and skeleton integrals**, second derivatives (`∇∇`, `Δ`), Lagrangian and
  Raviart–Thomas spaces, multi-field problems, manifold triangulations.
- **Threads without allocations** in the element kernels; cheap `update` of load factors and
  parameters without recompiling.
- **Matrix-free Jacobian** with sum factorisation, matrix-free Newton and p-multigrid.
- **Scalar functionals** and **internal variables** (`update_state!`) on the same kernels.
- **Never blocks a simulation:** integrals the engine cannot translate are assembled by Gridap.

The [user guide](docs/guide/README.md) describes every feature, the tuning switches and the current limits.

## Performance

Our measurements on an 18-thread Apple laptop; Jacobians equal to Gridap's to 1e-15.

| Case | Gridap | Kernel, 1 thread | Kernel, 18 threads |
|---|---|---|---|
| Mixed hyperelasticity, 7 700 Q1 hexahedra, 35 343 DOFs (Jacobian) | 0.382 s (AD) | 0.093 s | 0.010 s (×36) |
| Oseen–BDF2 Navier–Stokes, P2–P1 + SUPG/PSPG/LSIC, 468 926 DOFs | 19.2 s | 1.24 s | 0.137 s (×140) |
| Update of visco-elastic internal variables, Q2, 86 400 points | 0.064 s | 0.065 s | 0.0075 s (×8.5) |

The scripts that produce these figures are in [`benchmark/`](benchmark).

## Installation

```julia
pkg> add HyperFEMKernels
```

It requires Julia ≥ 1.12 and Gridap 0.20.

## Quick start

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

With HyperFEM's drivers only the backend keyword changes:

```julia
comp_model = StaticNonlinearModel(res, jac, U, V, D_bc; nls = nls_, backend = :compiled)
```

Load stepping, multi-field problems, matrix-free solvers, functionals and internal variables are in
the [user guide](docs/guide/README.md#usage).

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

- Shared-memory parallelism (threads); distributed (GridapDistributed) and GPU runs are design targets.
- Nédélec (`H(curl)`) spaces are not supported yet; some integrals (e.g. Raviart–Thomas on skeletons)
  go to the Gridap fallback.
- Gridap is restricted to the 0.20 series, because the engine reads Gridap's internal expression tree.

Details in the [user guide](docs/guide/README.md#current-scope).

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

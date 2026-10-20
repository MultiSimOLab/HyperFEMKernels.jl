# HyperFEMKernels.jl

Fused, threaded finite-element assembly for [Gridap.jl](https://github.com/gridap/Gridap.jl)
weak forms, with [HyperFEM.jl](https://github.com/MultiSimOLab/HyperFEM.jl) as first user.

## Goal

Keep Gridap's philosophy — the weak form written explicitly as a sum of integrals —
and execute it with the efficiency of a fused, parallel element kernel.
Gridap keeps the mesh, `FESpace`s, Dirichlet conditions (masks included), DOF
numbering, `Measure`s, `FEFunction`s and post-processing. HyperFEMKernels only
replaces the evaluation of `∫(...)dΩ` and the assembly.

HyperFEM will use it through a thin adapter (`backend = :kernel | :gridap`).

## Motivation

Feasibility test with hand-written kernels (our measurement, 18-thread Mac, same
matrix as Gridap to 1e-15):

| Case | Gridap | Kernel, 1 thread | Kernel, 18 threads |
|---|---|---|---|
| Mixed hyperelasticity, 7 700 Q1 hexahedra, 35 343 DOFs (Jacobian) | 0.382 s (AD) | 0.093 s | 0.010 s (×36) |
| Oseen–BDF2 Navier–Stokes, P2–P1 + SUPG/PSPG/LSIC, 468 926 DOFs | 19.2 s | 1.24 s | 0.137 s (×140) |

## Status

Phase 1: engine working and verified against Gridap to machine precision on HyperFEM's
electro-mechanical, magneto-mechanical and visco-hyperelastic weak forms. Volume integrals on
Lagrangian spaces (single/multi-field, Dirichlet masks, subdomains); boundary integrals and
distributed (MPI) assembly are next.

## Roadmap

- **Phase 0** — profile HyperFEM EM 3D and hMRE 2D; continue only if assembly is > 50 % of runtime.
- **Phase 1** — minimal engine (volume integrals, Lagrange, single/multi-field, masked Dirichlet, threads) and HyperFEM adapter; continue if total speed-up ≥ ×5.
- **Phase 2** — `@weakform` macro, boundary integrals, subdomain fields, hybrid fallback to Gridap.
- **Phase 3** — GridapDistributed (MPI), DG, Piola spaces.

## Usage

```julia
using Gridap, HyperFEMKernels
# U, V, dΩ, res(u, v), jac(u, du, v) written as for Gridap's FEOperator
op = KernelOperator(res, jac, U, V)          # Gridap NonlinearOperator (threaded, fused kernels)
b  = residual(op, x); A = jacobian(op, x)
op = update(op, res2, jac2)                  # new load step: keeps tables, pattern, colouring
check(res, jac, U, V, x)                     # relative differences with Gridap (≈1e-15)
allocations(op, x)                           # bytes per element call (0 = GPU-ready weak form)

mf = MatrixFreeJacobian(op); prepare!(mf, x) # partial assembly (sum factorisation on n-cubes)
y  = mf * dx                                 # Jacobian–vector product without the matrix
```

With HyperFEM (branch `kernel-backend`): `StaticNonlinearModel(res, jac, U, V, D; backend = :kernel)`.

## License

MIT.

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

Phase 0 (profiling HyperFEM examples). No functionality yet. Internal progress log kept outside the repository.

## Roadmap

- **Phase 0** — profile HyperFEM EM 3D and hMRE 2D; continue only if assembly is > 50 % of runtime.
- **Phase 1** — minimal engine (volume integrals, Lagrange, single/multi-field, masked Dirichlet, threads) and HyperFEM adapter; continue if total speed-up ≥ ×5.
- **Phase 2** — `@weakform` macro, boundary integrals, subdomain fields, hybrid fallback to Gridap.
- **Phase 3** — GridapDistributed (MPI), DG, Piola spaces.

## Minimal example (planned API)

```julia
using Gridap, HyperFEMKernels
# U, V, dΩ, res, jac defined as usual in Gridap
op = FEOperator(res, jac, U, V; backend = :kernel)   # planned
check(res, jac, U, x)                                 # compares with Gridap
```

## License

MIT.

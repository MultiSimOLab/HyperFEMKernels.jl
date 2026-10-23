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

Phase 1 closed (v0.1.0): engine verified against Gridap to machine precision on HyperFEM's
electro-mechanical, magneto-mechanical and visco-hyperelastic weak forms. Volume integrals on
Lagrangian spaces (single/multi-field, Dirichlet masks, subdomains), threads, matrix-free.

Phase 2 (in progress): boundary integrals in the engine (Neumann, Robin, surface and follower
loads with the normal, on any `BoundaryTriangulation`), skeleton integrals (interior facets:
`plus`/`minus`, `jump`, `mean` of the unknown and of the test/trial functions, with facet kernels
that see both cells), a hybrid fallback — contributions the engine cannot translate (differential
operators on products with the unknown) are assembled by Gridap and added, with a warning, so
the engine never blocks a simulation — and the update of internal variables (`update_state!`) by
the same kind of kernels.

## Roadmap

- **Phase 0** — profile HyperFEM EM 3D and hMRE 2D; continue only if assembly is > 50 % of runtime.
- **Phase 1** — minimal engine (volume integrals, Lagrange, single/multi-field, masked Dirichlet, threads) and HyperFEM adapter; continue if total speed-up ≥ ×5.
- **Phase 2** — hybrid fallback to Gridap (done), boundary integrals (done), state writes (done), skeleton integrals (done), mixed meshes, iterative solver on the matrix-free Jacobian, KernelAbstractions (GPU).
- **Phase 3** — GridapDistributed (MPI), DG, Piola spaces.

## Usage

```julia
using Gridap, HyperFEMKernels
# U, V, dΩ, res(u, v), jac(u, du, v) written as for Gridap's FEOperator
op = KernelOperator(res, jac, U, V)          # Gridap NonlinearOperator (threaded, fused kernels)
b  = residual(op, x); A = jacobian(op, x)
op = update(op, res2, jac2)                  # new load step / time step: keeps tables, pattern, colouring
                                             # and the compiled code (numbers of the weak form are parameters)
check(res, jac, U, V, x)                     # relative differences with Gridap (≈1e-15)
allocations(op, x)                           # bytes per element call (0 = GPU-ready weak form)

mf = MatrixFreeJacobian(op); prepare!(mf, x) # partial assembly (sum factorisation on n-cubes)
y  = mf * dx                                 # Jacobian–vector product without the matrix
```

Internal variables (visco-elasticity, plasticity): Gridap's `update_state!(updater, states..., fields...)`
executed by the engine, with the same updater and arguments. The FE functions in the arguments are
evaluated from their dof values, the states are read and written in place and the loop over cells
is threaded; anything else is evaluated by Gridap.

```julia
A = CellState(I3, dΩ)                        # Gridap's storage (one vector per cell), or
A = KernelState(I3, dΩ)                      # dense np × ncells matrix (the layout for GPU/MPI); usable in weak forms
upd(A, F, Fn) = (true, return_mapping(F, Fn, A))
su = update_states!(upd, A, F ∘ ∇(uh)', F ∘ ∇(uh⁻)')   # = update_state!(upd, A, ...) with the engine
update_state!(KernelUpdater(upd), A, F ∘ ∇(uh)', F ∘ ∇(uh⁻)')   # same, through Gridap's name
update_state!(upd, A::KernelState, ...)      # a KernelState first also selects the engine
allocations(su)                              # bytes per cell (0 unless the updater allocates)
```

With HyperFEM (branch `kernel-backend`): `StaticNonlinearModel(res, jac, U, V, D; backend = :kernel)`.

## License

MIT.

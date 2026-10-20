# Patient aorta: pulsatile blood flow, Gridap vs HyperFEMKernels

Self-contained extract of `hemodynamics/examples/patient_aorta/flow.jl` (MultiSimO Lab hemodynamics
code): incompressible Navier–Stokes in a rigid patient aorta, Taylor–Hood P2–P1 on tetrahedra with
SUPG/PSPG/LSIC stabilization, BDF2 (first step backward Euler), Oseen linearization (one linear system per
step, LU), pulsatile inflow with the Poisson profile of the inlet and an eccentric valve jet, three-element
Windkessel outlets (implicit Robin coupling) and backflow stabilization. Units mm, s, g.

The script only depends on Gridap, GridapGmsh and HyperFEMKernels (the helpers `close_boundary_tags!`,
`windkessel_0d` and `inlet_profile` are copied verbatim). The same `a`/`l` forms are assembled by Gridap
(`collect_cell_matrix_and_vector` with the Dirichlet values lifted) and by a `KernelOperator` with
`r(x) = a(x, ·) - l(·)`, `J = a`: Gridap's system is `A = J`, `b = -r(0)`.

```
julia -t auto --project=benchmark examples/aorta_cfd/aorta_flow.jl mode=compare steps=10
julia -t auto --project=benchmark examples/aorta_cfd/aorta_flow.jl mode=kernel      # 2 cycles, 400 steps
julia -t auto --project=benchmark examples/aorta_cfd/aorta_flow.jl mode=gridap      # 2 cycles, 400 steps
julia -t auto --project=benchmark dev/prof_cfd.jl                                   # breakdown per group
```

Mesh: `aorta_fluido_h1p5_3p0.msh` (coarse, 33 643 tetrahedra, 99 141 free DOFs). The production mesh of
the hemodynamics project (`aorta_fluido_h0p8_2p0.msh`, 8 MB) is not versioned; copy it here and pass
`mesh=aorta_fluido_h0p8_2p0`.

Writing note: numbers in the weak form are folded into the generated code when the operator is built, so
values that change at every step (the Windkessel compliance pressures) are cell fields over a vector
updated in place; the BDF2 coefficients change once and the operator is rebuilt with `update` after the
first step. The weak form is also checked on a small channel in `test/test_cfd.jl`.

## Results (our measurement, Mac M5 Max, 18 threads, coarse mesh)

- Equality: matrix and vector equal to Gridap's at every step (relative differences ~2e-16, 1 and 18
  threads); everything is assembled by the engine (no hybrid fallback).
- Assembly per step: Gridap 4.48 s; kernel 0.37 s with 18 threads (x12), 3.0 s with 1 thread (x1.5).
- Step (assembly + LU + solve + post): Gridap 5.9 s, kernel 1.8–2.1 s (x3); LU (1.0–1.2 s) now dominates.
- Two cycles (400 steps) with the kernel: 2.11 s/step, about 16 min.
- 1 thread: the volume Jacobian group takes the dual-number path (SUPG terms are products of auxiliary
  inputs, so the linear-map shortcut does not apply): 2.2 s of the 3.0 s; auxiliary data 0.56 s.
- Compilation: operator 38 s, `update` after the first step 28 s (Gridap also compiles ~85 s on steps 1–2).

CSV files of each run are written to `results/` (git-ignored).

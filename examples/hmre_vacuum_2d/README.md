# hMRE beam with explicit surrounding air (vacuum), 2D, staggered

Local copy of `src/hMRE_Vacuum_2D.jl` of MultiSimOLab/Adv_Mater_Ultra_Soft_Magnet (commit 71a610e;
Perez-Garcia, Ortigosa, Martinez-Frutos, Garcia-Gonzalez, *Advanced Materials*): three staggered problems
on the mesh `test/models/Short_magnet.msh` (8 637 nodes, 25 375 DOFs in all):

1. magnetic potential on solid + vacuum (`HardMagnetic2D` in a `MagnetoMechModel`, `IdealMagnetic2D`);
2. mechanics of the solid beam (`NonlinearMooneyRivlin2D + TransverseIsotropy2D`), loaded on the interface by
   the Maxwell traction of the vacuum (an FE function on the interface, projected with a mass matrix);
3. fictitious mechanics of the vacuum (`HessianRegularization(NonlinearMooneyRivlin2D_CV)`), driven by the solid
   displacement interpolated on the interface (`BoundaryCondition("Interface", Coupling(τ -> field))`, the state of the solid as a function of the stage parameter).

Same weak forms, models, boundary conditions and stepping as the original; no DrWatson (mesh from `test/models`,
output to `results/`), the setup is a function so that it runs for each backend, and the drivers take
`backend = :gridap | :kernel` (HyperFEM branch `kernel-backend`).

```
julia -t auto --project=benchmark examples/hmre_vacuum_2d/hmre_vacuum_2d.jl mode=compare   # both backends, 20 × 2 steps
julia -t auto --project=benchmark examples/hmre_vacuum_2d/hmre_vacuum_2d.jl mode=kernel vtk=on
julia -t auto --project=benchmark dev/prof_hmre_vacuum.jl          # per problem: assembly times, allocations per element
julia -t auto --project=benchmark dev/prof_hmre_vacuum_solve.jl    # where the staggered solve spends its time
```

## Results (our measurement, Mac M5 Max, 18 threads)

- Every term of the three weak forms is assembled by the engine (no hybrid fallback, no dual numbers: all
  groups use the linear-map shortcut). Residual and Jacobian near the final state equal Gridap's to 1e-11
  (residual) and 1e-15 (Jacobian). Final solutions of the full staggered run (20 steps × 2 substeps), kernel
  vs Gridap: magnetic 1e-16, solid 9e-15, vacuum 8e-15.
- Assembly per problem (residual + Jacobian): magnetic 45 ms → 7 ms (×6), solid 23 → 14 ms (×1.6), vacuum
  131 → 59 ms (×2.2).
- Whole staggered solve: Gridap 29.7 s, kernel 25.2 s (×1.2 only). The assembly is a small part of the
  loop: `dirichlet_preconditioning!` (an `AffineFEOperator` assembled by Gridap at every load step, 41 %),
  the Dirichlet update of the vacuum space, the Maxwell-stress projection of `res_mech` (11 %), the LU (14 %)
  and GC (10 %). The kernel backend of HyperFEM covers only `StaticNonlinearModel`'s Newton operator so far
  (see ESTADO.md, "adaptar HyperFEM").
- Allocations in the element kernels come from HyperFEM's constitutive functions, not from the engine:
  `MagnetoMechModel(HardMagnetic2D + MR + TI)` 1.4–8 kB per call, `HessianRegularization` tangent 2.4 kB
  (`IdealMagnetic2D` and `MR2D + TI2D` allocate nothing). Measured with `dev/prof_hmre_vacuum.jl`.

## Notes

- HyperFEM regression (commit e161faa, legacy `DirichletCoupling`): the coupling became a function
  `x -> evaluate(uh, x)` that Gridap evaluates at every node of the vacuum, where `uh` (an FE function on the
  interface) is undefined, so the point search fails; and the search cache was rebuilt at every call (0.78 s and
  1.4 GB per Dirichlet update). Fixed in HyperFEM branch `fix-legacy-coupling` (a `DirichletApplicator` evaluates
  the prescriptions at the Dirichlet nodes of their tag with a cached point search; the update costs 4 ms), which
  also retires the legacy `DirichletBC`/`DirichletCoupling`/`InterpolableBC!`; the example needs HyperFEM with
  that branch merged with `kernel-backend`. Report: `docs/regresion_DirichletCoupling_HyperFEM.pdf`.
- New engine features exercised here (now covered by `test/test_subdomains.jl` and the hMRE + vacuum testset of
  `test/hyperfem/runtests.jl`): FE spaces on a sub-triangulation of the model with integrals on parts of it and
  different constitutive laws per part; the boundary of a subdomain as an internal interface (faces interior to
  the model); an auxiliary field that is an FE function of a space defined on that interface; Dirichlet data
  on the interface changed in place between assemblies.

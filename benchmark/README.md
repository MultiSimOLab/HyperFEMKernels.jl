# Benchmarks

Reproducible scripts that write timing tables (Gridap backend vs kernel backend,
1 thread and N threads). Every figure reported must state its origin
(our measurement, estimate or reference). Results go to `benchmark/results/`
(git-ignored).

## Setup (once)

```
julia benchmark/setup.jl
```

## Phase 0

```
julia -t auto --project=benchmark benchmark/phase0/em3d.jl              # test_static_EM.msh, P1
julia -t auto --project=benchmark benchmark/phase0/em3d.jl cart 40 8 8 2 # larger Cartesian case, Q2
```

## Assembly, matrix-free and roofline

`bench_cases.jl` holds the shared cases (EM box and plate) used by the scripts below.

```
julia -t auto --project=benchmark benchmark/bench1.jl          # Jacobian and residual assembly, Gridap vs kernel
julia -t auto --project=benchmark benchmark/bench_mf.jl        # matrix-free apply vs SpMV vs assembly
julia -t auto --project=benchmark benchmark/bench_visco.jl     # visco-hyperelastic dynamic weak form
julia -t auto --project=benchmark benchmark/bench_states.jl    # update of the visco internal variables: Gridap vs update_states!
julia -t auto --project=benchmark benchmark/roofline.jl        # efficiency vs STREAM bandwidth and FMA peak
```

## Compilation cost

Time to first assembly with each backend, every measurement in a fresh Julia process: building
the operator, first `residual!`, first `jacobian!`, and a warm `residual!`+`jacobian!`, with the
part spent compiling. A last process per backend assembles both cases in sequence to show the
marginal cost of a new weak form once the engine is compiled.

```
julia -t auto --project=benchmark benchmark/bench_compile.jl   # EM plate P1 and EM box Q2, Gridap vs kernel
```

## Profiling of the examples

```
julia -t auto --project=benchmark benchmark/prof_cfd.jl                  # aorta CFD, per compiled group
julia -t auto --project=benchmark benchmark/prof_hmre_vacuum.jl          # hMRE + vacuum, assembly and allocations
julia -t auto --project=benchmark benchmark/prof_hmre_vacuum_solve.jl    # hMRE + vacuum, staggered solve
```

## Solver baseline (design/solvers.md, stage S0 + R0)

Own environment (`benchmark/solvers/`, with GridapPETSc) so that PETSc stays out of the other benchmarks:

```
julia benchmark/solvers/setup.jl                                                 # once
julia --project=benchmark/solvers benchmark/solvers/petsc_check.jl 40            # preconditioners of the PETSc binary (GAMG, hypre, MUMPS, …)
julia --project=benchmark/solvers benchmark/meshes/generate.jl em 6              # EM plate n = 4, 6, 10
julia --project=benchmark/solvers benchmark/meshes/generate.jl hmre 14 2         # hMRE beam refined by r = 7, 14, 30; vacuum by 2
julia -t 6 --project=benchmark/solvers benchmark/solvers/baseline.jl em_box 52 11 11 compiled          # per Newton iteration
julia -t 6 --project=benchmark/solvers benchmark/solvers/baseline.jl hmre 7 lazy 300000 driver         # + full solve! (R0)
bash benchmark/solvers/run_all.sh iter 6 1 18                                    # the whole matrix; `small`, `large`, `driver` subsets
julia --project=benchmark/solvers benchmark/solvers/summary.jl                   # Markdown tables from results/solvers_baseline.csv
```

`baseline.jl` times residual, Jacobian, LU factorisation and triangular solve on the operator (both backends,
1/6/18 threads), the matrix and LU memory, and with `driver` the full `solve!` through `StaticNonlinearModel`
(time and allocations per step and per iteration). LU is skipped above `lu_max` DOFs (default 300 000).
Results and conclusions: project document `design/solvers.md`, section "S0 results".

## Distributed capture

`benchmark/dist_capture.jl` checks the capture with GridapDistributed (PartitionedArrays debug
mode, 4 parts, 1 process) against Gridap's cell vectors and matrices of each part.

## Refinable meshes

`benchmark/meshes/generate.jl` builds meshes with the same geometry and physical tags as the
examples, at any refinement level (output git-ignored):

```
julia --project=benchmark benchmark/meshes/generate.jl em 3     # EM plate, (7n, 2n, 19n) hexahedra
julia --project=benchmark benchmark/meshes/generate.jl hmre 4   # Short_magnet geometry, sizes / r
julia -t auto --project=benchmark benchmark/phase0/em3d.jl plate 3
julia -t auto --project=benchmark benchmark/phase0/hmre2d.jl sqrt 4
```

`em 1` reproduces `test_static_EM.msh` (266 hexahedra, 1 528 free DOFs, same Newton iterations);
`hmre 1` reproduces `Short_magnet.msh` to within ~3 % in element counts per region.

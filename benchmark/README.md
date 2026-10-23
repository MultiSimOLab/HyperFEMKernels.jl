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
julia -t auto --project=benchmark benchmark/bench_qblock.jl    # Jacobian assembly vs quadrature-point block size
julia -t auto --project=benchmark benchmark/bench_visco.jl     # visco-hyperelastic dynamic weak form
julia -t auto --project=benchmark benchmark/bench_states.jl    # update of the visco internal variables: Gridap vs update_states!
julia -t auto --project=benchmark benchmark/roofline.jl        # efficiency vs STREAM bandwidth and FMA peak
julia -t 1 --project=benchmark benchmark/gemm_bench.jl         # element matrix-product micro-benchmark
```

## Profiling of the examples

```
julia -t auto --project=benchmark benchmark/prof_cfd.jl                  # aorta CFD, per compiled group
julia -t auto --project=benchmark benchmark/prof_hmre_vacuum.jl          # hMRE + vacuum, assembly and allocations
julia -t auto --project=benchmark benchmark/prof_hmre_vacuum_solve.jl    # hMRE + vacuum, staggered solve
```

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

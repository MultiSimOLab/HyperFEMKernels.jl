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

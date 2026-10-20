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

# Environment of the solver baseline (stage S0 of design/solvers.md): Gridap, HyperFEM (local clone),
# HyperFEMKernels (this repository) and GridapPETSc, separate from benchmark/ so that PETSc is not
# pulled into the other benchmarks.
#
#   julia benchmark/solvers/setup.jl
using Pkg
Pkg.activate(@__DIR__)
root = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.develop([PackageSpec(path=root), PackageSpec(path=joinpath(root, "..", "HyperFEM.jl"))])
Pkg.add(["Gridap", "GridapGmsh", "GridapPETSc", "GridapSolvers", "SparseMatricesCSR", "SparseArrays", "LinearAlgebra",
         "Printf", "Dates", "DelimitedFiles"])
Pkg.compat("Gridap", "=0.19.9")
Pkg.instantiate()
Pkg.status()

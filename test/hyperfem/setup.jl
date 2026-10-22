# Environment for the HyperFEM-dependent tests: the HyperFEM.jl package must be cloned next to this
# package (CI does it; locally it already is in ~/Desktop/repos). Meshes are in test/models.
using Pkg
Pkg.activate(@__DIR__)
root = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.develop(path=root)
Pkg.develop(path=joinpath(root, "..", "HyperFEM.jl"))
Pkg.add(["Gridap", "GridapGmsh", "GridapSolvers", "Test", "LinearAlgebra", "Random", "SparseArrays", "TimerOutputs"])
Pkg.instantiate()
Pkg.status()

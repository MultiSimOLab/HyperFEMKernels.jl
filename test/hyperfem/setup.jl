# Environment for the HyperFEM-dependent tests: HyperFEM.jl and Adv_Mater_Ultra_Soft_Magnet must be
# cloned next to this package (CI does it; locally they already are in ~/Desktop/repos).
using Pkg
Pkg.activate(@__DIR__)
root = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.develop(path=root)
Pkg.develop(path=joinpath(root, "..", "HyperFEM.jl"))
Pkg.add(["Gridap", "GridapGmsh", "GridapSolvers", "Test", "LinearAlgebra", "Random"])
Pkg.instantiate()
Pkg.status()

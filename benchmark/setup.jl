# Instantiate the benchmark environment with the local HyperFEM checkout.
using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=normpath(joinpath(@__DIR__, "..", "..", "HyperFEM.jl")))
Pkg.add(["Gridap", "GridapGmsh", "GridapSolvers", "Printf", "Dates", "LinearAlgebra", "SparseArrays"])
Pkg.instantiate()
Pkg.status()

# Instantiate the benchmark environment with the local HyperFEM and HyperFEMKernels checkouts.
using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=normpath(joinpath(@__DIR__, "..", "..", "HyperFEM.jl")))
Pkg.develop(path=normpath(joinpath(@__DIR__, "..")))
Pkg.add(["Gridap", "GridapGmsh", "GridapSolvers", "GridapDistributed", "PartitionedArrays", "Printf", "Dates", "Random", "Test", "LinearAlgebra", "SparseArrays"])
Pkg.instantiate()
Pkg.status()

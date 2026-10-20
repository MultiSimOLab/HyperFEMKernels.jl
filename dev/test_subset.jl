# Run a subset of the package tests in the benchmark environment (via the watcher):
#   dev/test_subset.jl codegen linearmaps assembly cfd     (names of test/test_<name>.jl)
using Test
using HyperFEMKernels
using Gridap
using Gridap.FESpaces, Gridap.Algebra, Gridap.CellData, Gridap.Arrays, Gridap.TensorValues
using LinearAlgebra, SparseArrays, Random
using HyperFEMKernels.StaticArrays, HyperFEMKernels.ForwardDiff
const HK = HyperFEMKernels
const TOL = 1e-12
include(joinpath(@__DIR__, "..", "test", "common.jl"))
@testset "subset" begin
    for name in ARGS
        @testset "$name" begin include(joinpath(@__DIR__, "..", "test", "test_$name.jl")) end
    end
end

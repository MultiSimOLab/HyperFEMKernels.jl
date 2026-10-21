# Run only the CFD weak-form tests (test/test_cfd.jl) in the benchmark environment (via the watcher).
using Test
using HyperFEMKernels
using Gridap
using Gridap.FESpaces, Gridap.Algebra, Gridap.CellData, Gridap.Arrays, Gridap.TensorValues
using LinearAlgebra, SparseArrays, Random
using HyperFEMKernels.StaticArrays, HyperFEMKernels.ForwardDiff
const HK = HyperFEMKernels
const TOL = 1e-12
include(joinpath(@__DIR__, "..", "test", "common.jl"))
@testset "CFD (Oseen step)" begin include(joinpath(@__DIR__, "..", "test", "test_cfd.jl")) end

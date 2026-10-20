using Test
using HyperFEMKernels
using Gridap
using Gridap.FESpaces, Gridap.Algebra, Gridap.CellData, Gridap.Arrays, Gridap.TensorValues
using LinearAlgebra, SparseArrays, Random
using HyperFEMKernels.StaticArrays, HyperFEMKernels.ForwardDiff

const HK = HyperFEMKernels
const TOL = 1e-12

include("common.jl")

@testset "HyperFEMKernels" begin
    @testset "probes"           begin include("test_probes.jl") end
    @testset "tree"             begin include("test_tree.jl") end
    @testset "codegen"          begin include("test_codegen.jl") end
    @testset "tables"           begin include("test_tables.jl") end
    @testset "kernels"          begin include("test_kernels.jl") end
    @testset "linear maps"      begin include("test_linearmaps.jl") end
    @testset "assembly"         begin include("test_assembly.jl") end
    @testset "errors"           begin include("test_errors.jl") end
    @testset "Gridap equality"  begin include("test_gridap_equality.jl") end
end

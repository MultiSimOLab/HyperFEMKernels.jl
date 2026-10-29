using Test
using HyperFEMKernels
using Gridap
using Gridap.FESpaces, Gridap.Algebra, Gridap.CellData, Gridap.Arrays, Gridap.TensorValues
using LinearAlgebra, SparseArrays, Random
using HyperFEMKernels.StaticArrays, HyperFEMKernels.ForwardDiff

const HK = HyperFEMKernels
const TOL = 1e-12

include("common.jl")

# Test groups (`HFK_TEST_GROUP`): "all" (default), "hdiv" (H(div) fields only) or "core" (everything
# else). CI runs "core" and "hdiv" as separate jobs so that each stays well below the time limit.
const TEST_GROUP = get(ENV, "HFK_TEST_GROUP", "all")
TEST_GROUP in ("all", "core", "hdiv") || error("HFK_TEST_GROUP must be all, core or hdiv (got $TEST_GROUP)")
const TEST_FILES = [
    ("probes", "test_probes.jl"), ("tree", "test_tree.jl"), ("codegen", "test_codegen.jl"),
    ("tables", "test_tables.jl"), ("kernels", "test_kernels.jl"), ("linear maps", "test_linearmaps.jl"),
    ("assembly", "test_assembly.jl"), ("matrix-free", "test_matrixfree.jl"), ("errors", "test_errors.jl"),
    ("hybrid fallback", "test_fallback.jl"), ("boundary", "test_boundary.jl"), ("H(div) fields", "test_hdiv.jl"),
    ("CFD (Oseen step)", "test_cfd.jl"), ("subdomains, interface", "test_subdomains.jl"),
    ("skeleton", "test_skeleton.jl"), ("FE functions in the form", "test_fefunctions.jl"), ("states", "test_states.jl"), ("Gridap equality", "test_gridap_equality.jl"),
]
in_group(file) = TEST_GROUP == "all" || (TEST_GROUP == "hdiv") == (file == "test_hdiv.jl")

@testset "HyperFEMKernels ($TEST_GROUP)" begin
    for (name, file) in TEST_FILES
        in_group(file) || continue
        @testset "$name" begin
            include(file)
        end
    end
end

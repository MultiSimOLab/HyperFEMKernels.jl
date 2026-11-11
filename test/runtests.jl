using Test
using HyperFEMKernels
using Gridap
using Gridap.FESpaces, Gridap.Algebra, Gridap.CellData, Gridap.Arrays, Gridap.TensorValues
using LinearAlgebra, SparseArrays, Random
using HyperFEMKernels.StaticArrays, HyperFEMKernels.ForwardDiff

const HK = HyperFEMKernels
const TOL = 1e-12

include("common.jl")

# Test groups (`HFK_TEST_GROUP`): "all" (default), "hdiv" (H(div) fields only), "hessian" (second
# derivatives only) or "core" (everything else). CI runs "core", "hdiv" and "hessian" as separate jobs so
# that each stays well below the time limit.
const TEST_GROUP = get(ENV, "HFK_TEST_GROUP", "all")
TEST_GROUP in ("all", "core", "hdiv", "hessian") || error("HFK_TEST_GROUP must be all, core, hdiv or hessian (got $TEST_GROUP)")
const TEST_FILES = [
    ("probes", "test_probes.jl"), ("tree", "test_tree.jl"), ("codegen", "test_codegen.jl"),
    ("tables", "test_tables.jl"), ("kernels", "test_kernels.jl"), ("linear maps", "test_linearmaps.jl"),
    ("assembly", "test_assembly.jl"), ("matrix-free", "test_matrixfree.jl"), ("errors", "test_errors.jl"),
    ("hybrid fallback", "test_fallback.jl"), ("boundary", "test_boundary.jl"), ("H(div) fields", "test_hdiv.jl"),
    ("CFD (Oseen step)", "test_cfd.jl"), ("subdomains, interface", "test_subdomains.jl"),
    ("skeleton", "test_skeleton.jl"), ("Hessians (∇∇)", "test_hessian.jl"), ("cheap update", "test_update.jl"), ("solvers: operator mode", "solvers/test_operator.jl"), ("solvers: matrix-free solver", "solvers/test_matrixfree_solver.jl"), ("solvers: functionals", "solvers/test_functionals.jl"), ("solvers: p-multigrid transfers", "solvers/test_transfers.jl"), ("solvers: p-multigrid levels", "solvers/test_pmultigrid_levels.jl"), ("solvers: Chebyshev smoother", "solvers/test_chebyshev.jl"), ("solvers: p-multigrid", "solvers/test_pmultigrid.jl"), ("manifold spaces", "test_manifold.jl"), ("FE functions in the form", "test_fefunctions.jl"), ("states", "test_states.jl"), ("Gridap equality", "test_gridap_equality.jl"),
]
const GROUP_FILES = Dict("hdiv" => "test_hdiv.jl", "hessian" => "test_hessian.jl")
file_group(file) = something(findfirst(==(file), GROUP_FILES), "core")
in_group(file) = TEST_GROUP == "all" || TEST_GROUP == file_group(file)

@testset "HyperFEMKernels ($TEST_GROUP)" begin
    for (name, file) in TEST_FILES
        in_group(file) || continue
        @testset "$name" begin
            include(file)
        end
    end
end

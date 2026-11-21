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
# derivatives only), "core" (the other files up to and including `test_update.jl`) or "extended" (the
# other files after it: solvers, spaces, states, Gridap equality and later additions). CI runs the four
# groups as separate jobs so that each stays well below the time limit.
const TEST_GROUP = get(ENV, "HFK_TEST_GROUP", "all")
TEST_GROUP in ("all", "core", "extended", "hdiv", "hessian") ||
    error("HFK_TEST_GROUP must be all, core, extended, hdiv or hessian (got $TEST_GROUP)")
const TEST_FILES = [
    ("probes", "test_probes.jl"), ("tree", "test_tree.jl"), ("codegen", "test_codegen.jl"),
    ("tables", "test_tables.jl"), ("kernels", "test_kernels.jl"), ("linear maps", "test_linearmaps.jl"),
    ("assembly", "test_assembly.jl"), ("matrix-free", "test_matrixfree.jl"), ("errors", "test_errors.jl"),
    ("hybrid fallback", "test_fallback.jl"), ("boundary", "test_boundary.jl"), ("H(div) fields", "test_hdiv.jl"),
    ("CFD (Oseen step)", "test_cfd.jl"), ("subdomains, interface", "test_subdomains.jl"),
    ("skeleton", "test_skeleton.jl"), ("Hessians (∇∇)", "test_hessian.jl"), ("cheap update", "test_update.jl"), ("solvers: operator mode", "solvers/test_operator.jl"), ("solvers: matrix-free solver", "solvers/test_matrixfree_solver.jl"), ("solvers: functionals", "solvers/test_functionals.jl"), ("solvers: p-multigrid transfers", "solvers/test_transfers.jl"), ("solvers: p-multigrid levels", "solvers/test_pmultigrid_levels.jl"), ("solvers: Chebyshev smoother", "solvers/test_chebyshev.jl"), ("solvers: p-multigrid", "solvers/test_pmultigrid.jl"), ("manifold spaces", "test_manifold.jl"), ("FE functions in the form", "test_fefunctions.jl"), ("functions of the coordinates", "test_coordfunctions.jl"), ("affine cells: geometry once per cell", "test_affine.jl"), ("states", "test_states.jl"), ("Gridap equality", "test_gridap_equality.jl"), ("sparsity by field blocks", "test_block_pattern.jl"),
]
const GROUP_FILES = Dict("hdiv" => "test_hdiv.jl", "hessian" => "test_hessian.jl")
# Last file of the "core" group: the files listed after it (new files are usually appended) form "extended".
const LAST_CORE_FILE = "test_update.jl"
const LAST_CORE_INDEX = findfirst(t -> t[2] == LAST_CORE_FILE, TEST_FILES)
LAST_CORE_INDEX === nothing && error("LAST_CORE_FILE = $LAST_CORE_FILE is not in TEST_FILES")
function file_group(i, file)
    g = findfirst(==(file), GROUP_FILES)
    g === nothing || return g
    return i <= LAST_CORE_INDEX ? "core" : "extended"
end
in_group(i, file) = TEST_GROUP == "all" || TEST_GROUP == file_group(i, file)

@testset "HyperFEMKernels ($TEST_GROUP)" begin
    for (i, (name, file)) in enumerate(TEST_FILES)
        in_group(i, file) || continue
        t0 = time()
        @testset "$name" begin
            include(file)
        end
        # elapsed time per file, so that a job that hits the CI time limit shows where the time went
        println("[tests] ", file, ": ", round(time() - t0; digits=1), " s")
    end
end

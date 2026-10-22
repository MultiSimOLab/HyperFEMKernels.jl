# Jacobian assembly vs size of the quadrature-point block of the matrix-product path.
#   julia -t auto --project=benchmark benchmark/bench_qblock.jl
using HyperFEMKernels, HyperFEM, Gridap, GridapGmsh, Gridap.FESpaces, Gridap.Algebra
using LinearAlgebra, SparseArrays, Random, Printf
const HK = HyperFEMKernels
Random.seed!(1)
include(joinpath(@__DIR__, "bench_cases.jl"))
best(f; n=5) = (f(); minimum(@elapsed(f()) for _ in 1:n))
function run(name, res, jac, U, V)
    x = 1e-5 .* randn(num_free_dofs(U))
    op = KernelOperator(res, jac, U, V); op1 = KernelOperator(res, jac, U, V; nthreads=1)
    A = Algebra.jacobian(op, x); Aref = copy(A)
    @printf("%s — %d DOFs\n", name, length(x))
    for qb in (0, 2, 4, 6, 8, 9, 12, 16, 27)
        HK.POINT_BLOCK[] = qb
        t1 = best(() -> jacobian!(A, op1, x)); tN = best(() -> jacobian!(A, op, x))
        @printf("  block %2d: 1 thr %7.2f ms | %d thr %6.2f ms | diff %.1e\n", qb, 1e3t1, Threads.nthreads(), 1e3tN, norm(A - Aref) / norm(Aref))
        flush(stdout)
    end
    HK.POINT_BLOCK[] = 12
end
run("EM box 20x4x4 Q2", em_box(20, 4, 4, 2)...)
run("EM box 14x3x3 Q3", em_box(14, 3, 3, 3)...)
run("EM box 40x8x8 Q2", em_box(40, 8, 8, 2)...)
run("EM plate n=2 P1", em_case(GmshDiscreteModel(joinpath(@__DIR__, "..", "benchmark", "meshes", "em_plate_n2.msh")), 1)...)

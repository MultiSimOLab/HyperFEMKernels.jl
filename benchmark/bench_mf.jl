# Matrix-free apply vs assembled SpMV vs assembly (EM box Q2 / Q3, plate P1).
using HyperFEMKernels, HyperFEM, Gridap, GridapGmsh, Gridap.FESpaces, Gridap.Algebra
using LinearAlgebra, SparseArrays, Random, Printf
Random.seed!(1)
include(joinpath(@__DIR__, "bench_cases.jl"))
function bench(name, res, jac, U, V)
    x = 1e-5 .* randn(num_free_dofs(U)); dx = randn(num_free_dofs(U))
    op = KernelOperator(res, jac, U, V); op1 = KernelOperator(res, jac, U, V; nthreads=1)
    A = Algebra.jacobian(op, x)
    mf = MatrixFreeJacobian(op); mf1 = MatrixFreeJacobian(op1)
    prepare!(mf, x); prepare!(mf1, x); y = mf * dx; y1 = mf1 * dx
    best(f) = minimum(@elapsed(f()) for _ in 1:5)
    tasm = best(() -> jacobian!(A, op, x)); tasm1 = best(() -> jacobian!(A, op1, x))
    tprep = best(() -> prepare!(mf, x)); tprep1 = best(() -> prepare!(mf1, x))
    tapply = best(() -> mul!(y, mf, dx)); tapply1 = best(() -> mul!(y1, mf1, dx))
    err = norm(y - A * dx) / norm(A * dx)
    tspmv = best(() -> mul!(y, A, dx))
    @printf("\n%s — %d DOFs, nnz %d, %d threads\n", name, length(x), nnz(A), Threads.nthreads())
    @printf("  assembled Jacobian : %.4f s (1 thr %.4f s) ; SpMV %.5f s\n", tasm, tasm1, tspmv)
    @printf("  matrix-free prepare: %.4f s (1 thr %.4f s) ; apply %.5f s (1 thr %.5f s) ; rel.diff %.1e\n", tprep, tprep1, tapply, tapply1, err)
    @printf("  memory: assembled %.1f MB ; matrix-free tangents %.1f MB\n", (nnz(A) * 12) / 1e6, sum(sizeof(d.T) + sizeof(d.iJt) + sizeof(d.dV) for d in mf.data) / 1e6)
    flush(stdout)
end
bench("EM box 20x4x4 Q2", em_box(20, 4, 4, 2)...)
bench("EM box 14x3x3 Q3", em_box(14, 3, 3, 3)...)
bench("EM plate n=2 P1 (tets, dense fallback)", em_case(GmshDiscreteModel(joinpath(@__DIR__, "..", "benchmark", "meshes", "em_plate_n2.msh")), 1)...)

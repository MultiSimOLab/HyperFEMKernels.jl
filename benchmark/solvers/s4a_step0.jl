# S4-A, Step 0 (design/solvers-s4a-plan.md §5): the matrix-free operator at order 3 on the mechanical cases, and the
# sizes the p-multigrid levels will see. Q3 `mul!` against the assembled tangent, `prepare!` vs assembly, memory of the
# stored tangents vs the matrix, for the Q1 / Q2 / Q3 discretisations of the same mesh (the levels of one hierarchy).
#
#   julia -t 6 --project=benchmark/solvers benchmark/solvers/s4a_step0.jl mech_cube 8        # n³ cubes; Q1/Q2/Q3 on the same mesh
#   julia -t 6 --project=benchmark/solvers benchmark/solvers/s4a_step0.jl mech_box 26 6 6
#
# The GAMG measurements of Step 0 (Q1 discretisation of the box, assembled Q2 baselines) go through krylov.jl:
# bash benchmark/solvers/run_s4a0.sh.
include(joinpath(@__DIR__, "cases.jl"))
using HyperFEMKernels
using Gridap.Algebra
using LinearAlgebra, SparseArrays, Printf, Random
Random.seed!(1)

rss() = Sys.maxrss() / 2^30
best(f, n=5) = minimum(@elapsed(f()) for _ in 1:n)

function level(words, p)
    case, tag = build_case([words..., string(p)])
    U, V = case.U, case.V
    res = case.res(1.0); jac = case.jac(1.0)
    op = KernelOperator(res, jac, U, V)
    n = num_free_dofs(U)
    x = 1e-3 .* randn(n); dx = randn(n)
    A = Algebra.jacobian(op, x)
    tasm = best(() -> jacobian!(A, op, x))
    mf = MatrixFreeJacobian(op); prepare!(mf, x)
    tprep = best(() -> prepare!(mf, x))
    y = mf * dx; y2 = A * dx
    tmf = best(() -> mul!(y, mf, dx), 10); tsp = best(() -> mul!(y2, A, dx), 10)
    err = norm(y - y2) / norm(y2)
    mem_T = sum(sizeof(d.T) + sizeof(d.iJt) + sizeof(d.dV) for d in mf.data) / 2^30
    mem_A = (nnz(A) * 12) / 2^30
    @printf("%-22s p=%d  dofs %8d  nnz/row %5.0f  assembly %.4f s  prepare! %.4f s  mul! %.5f s  SpMV %.5f s  |mf-A|/|A| %.1e  tangents %.3f GB  matrix %.3f GB  rss %.2f GB\n",
            tag, p, n, nnz(A) / n, tasm, tprep, tmf, tsp, err, mem_T, mem_A, rss())
    flush(stdout)
    err
end

function main(args)
    words = args
    @printf("S4-A step 0: %s, threads=%d\n", join(words, " "), Threads.nthreads())
    for p in (1, 2, 3)
        level(words, p)
    end
end

main(ARGS)

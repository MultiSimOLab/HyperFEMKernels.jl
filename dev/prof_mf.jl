using HyperFEMKernels, HyperFEM, Gridap, Gridap.FESpaces, Gridap.Algebra
using LinearAlgebra, Random, Printf, Profile
Random.seed!(1)
include(joinpath(@__DIR__, "bench_cases.jl"))
order = parse(Int, get(ARGS, 1, "3"))
res, jac, U, V = em_box(14, 3, 3, order)
x = 1e-5 .* randn(num_free_dofs(U)); dx = randn(num_free_dofs(U))
op1 = KernelOperator(res, jac, U, V; nthreads=1)
mf1 = MatrixFreeJacobian(op1); prepare!(mf1, x); y = mf1 * dx
@printf("apply 1 thr %.4f s, alloc %d\n", @elapsed(mul!(y, mf1, dx)), @allocated(mul!(y, mf1, dx)))
Profile.clear(); @profile for _ in 1:3; mul!(y, mf1, dx); end
Profile.print(format=:flat, sortedby=:count, mincount=30, noisefloor=1.0)

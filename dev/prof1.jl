# Profile single-thread kernel Jacobian on the Q2 box.
using HyperFEMKernels, HyperFEM, Gridap
using Gridap.FESpaces, Gridap.Algebra
using Profile, Printf, Random
include(joinpath(@__DIR__, "..", "benchmark", "bench_cases.jl"))
res, jac, U, V = em_box(20, 4, 4, 2)
x = 0.01 .* randn(num_free_dofs(U))
t = @elapsed (kop = KernelOperator(res, jac, U, V; nthreads=1)); @printf("build %.2f s\n", t)
t = @elapsed (kop = KernelOperator(res, jac, U, V; nthreads=1)); @printf("build (2nd) %.2f s\n", t)
A = allocate_jacobian(kop, x); jacobian!(A, kop, x)
@printf("jac 1 thr %.4f s\n", @elapsed jacobian!(A, kop, x))
Profile.clear()
@profile for _ in 1:5; jacobian!(A, kop, x); end
Profile.print(format=:flat, sortedby=:count, mincount=30, maxdepth=60, noisefloor=1.0)

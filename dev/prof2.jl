using HyperFEMKernels, HyperFEM, Gridap, GridapGmsh
using Gridap.FESpaces, Gridap.Algebra
using Profile, Printf, Random
include(joinpath(@__DIR__, "bench_cases.jl"))
res, jac, U, V = em_case(GmshDiscreteModel(joinpath(@__DIR__, "..", "benchmark", "meshes", "em_plate_n2.msh")), 1)
x = 0.01 .* randn(num_free_dofs(U))
kop = KernelOperator(res, jac, U, V; nthreads=1)
A = allocate_jacobian(kop, x); jacobian!(A, kop, x)
@printf("jac 1 thr %.4f s\n", @elapsed jacobian!(A, kop, x))
Profile.clear()
@profile for _ in 1:20; jacobian!(A, kop, x); end
Profile.print(format=:flat, sortedby=:count, mincount=50, noisefloor=1.0)

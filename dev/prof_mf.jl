using HyperFEMKernels, HyperFEM, Gridap, Gridap.FESpaces, Gridap.Algebra
using LinearAlgebra, Random, Printf, Profile
const HK = HyperFEMKernels
Random.seed!(1)
include(joinpath(@__DIR__, "bench_cases.jl"))
order = parse(Int, get(ARGS, 1, "3"))
res, jac, U, V = em_box(14, 3, 3, order)
x = 1e-5 .* randn(num_free_dofs(U)); dx = randn(num_free_dofs(U))
op1 = KernelOperator(res, jac, U, V; nthreads=1)
mf1 = MatrixFreeJacobian(op1); prepare!(mf1, x); y = mf1 * dx
@printf("apply 1 thr %.4f s, alloc %d\n", @elapsed(mul!(y, mf1, dx)), @allocated(mul!(y, mf1, dx)))
g = op1.jgroups[1]; gd = mf1.data[1]; tps = mf1.tps[1]
b = HK.apply_buffers(g, tps, op1.ndl)
@printf("buffers alloc: %d\n", @allocated(HK.apply_buffers(g, tps, op1.ndl)))
HK.apply_cells!(y, gd, tps, g, op1, dx, 1:2)
@printf("apply_cells! 2 cells: %d bytes ; 20 cells: %d bytes\n", @allocated(HK.apply_cells!(y, gd, tps, g, op1, dx, 1:2)), @allocated(HK.apply_cells!(y, gd, tps, g, op1, dx, 1:20)))
t = g.tables[1]; tp = tps[1]; np = g.geo.np
HK.interpolate_slots!(b.su, t, 0, g.smu[1], tp, b.xl, b.uc[1], b.pts[1], b.Wb[1], b.tmp[1], gd, 1, np)
@printf("interpolate_slots!: %d bytes\n", @allocated(HK.interpolate_slots!(b.su, t, 0, g.smu[1], tp, b.xl, b.uc[1], b.pts[1], b.Wb[1], b.tmp[1], gd, 1, np)))
HK.contract_slots!(b.re, t, 0, g.smv[1], tp, b.sv, b.rc[1], b.coef[1], b.Wb[1], b.tmp[1], gd, 1, np)
@printf("contract_slots!: %d bytes\n", @allocated(HK.contract_slots!(b.re, t, 0, g.smv[1], tp, b.sv, b.rc[1], b.coef[1], b.Wb[1], b.tmp[1], gd, 1, np)))
HK.tp_interpolate!(b.pts[1], tp, b.uc[1], b.tmp[1])
@printf("tp_interpolate!: %d bytes ; tp_contract!: %d bytes\n", @allocated(HK.tp_interpolate!(b.pts[1], tp, b.uc[1], b.tmp[1])), @allocated(HK.tp_contract!(b.rc[1], tp, b.coef[1], b.tmp[1])))
Profile.clear(); @profile for _ in 1:40; mul!(y, mf1, dx); end
Profile.print(format=:flat, sortedby=:count, mincount=40, noisefloor=1.0)

# Breakdown of the kernel assembly of the aorta CFD example per compiled group: auxiliary data
# (Gridap evaluation into dense arrays) and element kernels, 1 thread and N threads.
#   julia -t auto --project=benchmark benchmark/prof_cfd.jl [mesh=...]
using Printf
const HK_ARGS = copy(ARGS)
empty!(ARGS); append!(ARGS, ["mode=kernel", "steps=3"]); append!(ARGS, HK_ARGS)
include(joinpath(@__DIR__, "..", "examples", "aorta_cfd", "aorta_flow.jl"))
const HK = HyperFEMKernels

function best(f, n=3)
    f(); minimum(@elapsed(f()) for _ in 1:n)
end

xs = copy(x)                           # last solution as state (the step is linear: any x works)
dvd = [get_dirichlet_dof_values(s) for s in HK.field_spaces(op.U)]
op1 = HK.KernelOperator(op.res, op.jac, op.U, op.V, op.fields, op.ndl, op.rgroups, op.jgroups, op.A0, op.nzpos,
                        op.colors, 1, op.cache_aux, op.linear_maps, op.fallback)
b = Gridap.Algebra.allocate_residual(op, xs); A = Gridap.Algebra.allocate_jacobian(op, xs)
println("\nPROFILE (best of 3, seconds)")
@printf("%-4s %-3s %-26s %6s %6s %4s %5s %4s | %8s %8s | %8s %8s\n", "kind", "grp", "triangulation", "cells", "np",
        "NSV", "NSU", "path", "aux 1t", "aux Nt", "all 1t", "all Nt")
tot = zeros(4)
for (kind, groups) in ((:res, op.rgroups), (:jac, op.jgroups)), (i, g) in enumerate(groups)
    ta1 = best(() -> HK.materialize_aux(g, 1)); taN = best(() -> HK.materialize_aux(g, op.nthreads))
    runk(o) = kind === :res ? HK.run_cells!(Val(:res), b, nothing, o, g, xs, dvd) : HK.run_cells!(Val(:jac), nothing, A, o, g, xs, dvd)
    t1 = best(() -> runk(op1)); tN = best(() -> runk(op))
    tot .+= (ta1, taN, t1, tN)
    nsv = g.pf.test.nslots; nsu = g.pf.trial.nslots
    @printf("%-4s %-3d %-26s %6d %6d %4d %5d %4s | %8.4f %8.4f | %8.4f %8.4f\n", kind, i, nameof(typeof(g.trian)),
            length(g.cellmap), g.geo.np, nsv, nsu, g.lin === nothing ? "dual" : "lin", ta1, taN, t1, tN)
end
@printf("%-62s | %8.4f %8.4f | %8.4f %8.4f\n", "total", tot...)
println("aux arrays per group: ", [length(g.aux) for g in op.jgroups], " (Jacobian), ", [length(g.aux) for g in op.rgroups], " (residual)")
tgm = best(() -> assemble_matrix(assem, Gridap.FESpaces.collect_cell_matrix(U⁺, V, a(du, dv))), 1)
tgv = best(() -> assemble_vector(assem, Gridap.FESpaces.collect_cell_vector(V, l(dv))), 1)
@printf("Gridap: matrix %.3f s, vector (l only) %.3f s\n", tgm, tgv)
println("FIN PROF")

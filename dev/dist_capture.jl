# Distributed capture test (GridapDistributed, PartitionedArrays debug mode: 4 parts, 1 process).
# For each part, the weak form is captured with distributed probes and the element residuals and
# Jacobians computed by the kernel are compared with Gridap's cell vectors/matrices of that part.
using HyperFEMKernels, Gridap, GridapDistributed, PartitionedArrays
using Gridap.FESpaces, Gridap.CellData
using LinearAlgebra, Printf, Random
const HK = HyperFEMKernels
Random.seed!(1)

E(F) = 0.5 * (transpose(F) ⋅ F - one(F))
S(F) = 2.0 * tr(E(F)) * one(F) + 2.0 * E(F)
P(F) = F ⋅ S(F)
dS(F, dF) = (dE = 0.5 * (transpose(dF) ⋅ F + transpose(F) ⋅ dF); 2.0 * tr(dE) * one(F) + 2.0 * dE)
dP(F, dF) = dF ⋅ S(F) + F ⋅ dS(F, dF)
Fk(∇u) = one(∇u) + ∇u

dprobe(U, role) = GridapDistributed.DistributedCellField(map(s -> HK.probes(s, role), local_views(U)), get_triangulation(U))

function compare(label, res, jac, U, V, uh)
    up, vp, dup = dprobe(U, :unknown), dprobe(V, :test), dprobe(U, :trial)
    rprobe, jprobe = res(up, vp), jac(up, dup, vp)
    rreal = res(uh, get_fe_basis(V)); jreal = jac(uh, get_trial_fe_basis(U), get_fe_basis(V))
    errs = map(local_views(rprobe), local_views(jprobe), local_views(rreal), local_views(jreal),
               local_views(U), local_views(uh)) do rp, jp, rr, jr, Ul, uhl
        er, ej = 0.0, 0.0
        for (dcp, dcr, kind) in ((rp, rr, :res), (jp, jr, :jac))
            for (trian, arr) in dcp.dict
                terms = HK.collect_terms(arr, 1.0)
                strian = get_triangulation(Ul)
                cmap = HK.integration_cells(trian, strian)
                g = HK.Group(terms, trian, [Ul], (0,), 3, cmap, [Int32.(1:length(cmap))])
                auxd = HK.materialize_aux(g, 1)
                nd = HK.ndofs(g.tables[1])
                xl = zeros(nd); re = zeros(nd); KeT = zeros(nd, nd)
                dNb = (zeros(g.tables[1].nn, 4),)
                cv = get_cell_dof_values(uhl)
                ref = dcr.dict[trian]
                for cell in 1:length(cmap)
                    xl .= cv[cmap[cell]]
                    if kind === :res
                        HK.element_residual!(re, g, xl, auxd, cell, dNb)
                        er = max(er, norm(re - ref[cell]) / max(norm(ref[cell]), 1e-300))
                    else
                        HK.element_jacobian!(KeT, g, xl, auxd, cell, dNb)
                        ej = max(ej, norm(transpose(KeT) - ref[cell]) / max(norm(ref[cell]), 1e-300))
                    end
                end
            end
        end
        (er, ej, num_cells(get_triangulation(Ul)))
    end
    for (p, (er, ej, n)) in enumerate(collect(errs))
        @printf("%-28s part %d (%3d local cells): max rel. diff element residual %.1e, Jacobian %.1e\n", label, p, n, er, ej)
    end
    flush(stdout)
end

with_debug() do distribute
    ranks = distribute(LinearIndices((4,)))
    model = CartesianDiscreteModel(ranks, (2, 2, 1), (0, 1, 0, 1, 0, 1), (6, 6, 3))
    V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), dirichlet_tags=[1, 3, 5, 7])
    U = TrialFESpace(V, x -> VectorValue(0.01, 0.0, 0.0))
    uh = interpolate(x -> VectorValue(0.01 * x[2], 0.02 * x[1] * x[3], -0.01 * x[1]), U)
    uold = interpolate(x -> VectorValue(0.0, 0.01 * x[1], 0.0), U)
    for (label, Ω) in (("owned cells (no_ghost)", Triangulation(model)),
                       ("owned + ghost (with_ghost)", Triangulation(GridapDistributed.with_ghost, model)))
        dΩ = Measure(Ω, 2)
        res(u, v) = ∫(∇(v)' ⊙ (P ∘ (Fk ∘ (∇(u)'))))dΩ + 0.3 * ∫(v ⋅ uold)dΩ
        jac(u, du, v) = ∫(∇(v)' ⊙ (dP ∘ (Fk ∘ (∇(u)'), ∇(du)')))dΩ
        compare(label, res, jac, U, V, uh)
    end
end

Random.seed!(1)
p = poisson_2d()
e = elasticity_3d()

function pointform(prob, kind)
    forms = kind === :res ? HK.capture(prob.res, HK.probes(prob.U, :unknown), HK.probes(prob.V, :test))[1] :
            HK.capture(prob.jac, HK.probes(prob.U, :unknown), HK.probes(prob.U, :trial), HK.probes(prob.V, :test))[1]
    HK.PointForm(forms[1].terms)
end

@testset "slot layout and slot maps" begin
    pf = pointform(p, :res)
    @test pf.test.nslots == 1 + 2        # v (1) + ∇v (2)
    @test pf.trial.nslots == 0
    sm = HK.slot_map(pf.test, 1, 1, 2)
    @test size(sm) == (1, 6) && all(>(0), sm[:, 1:3]) && allunique(sm[:, 1:3]) && all(==(0), sm[:, 4:6])   # no Hessian slots
    pe = pointform(e, :jac)
    @test pe.test.nslots == 9 && pe.trial.nslots == 9      # only gradients
    sm = HK.slot_map(pe.test, 1, 3, 3)
    @test sm[:, 1] == [0, 0, 0]                            # no value slots
    @test sort(vec(sm[:, 2:4])) == 1:9
    @test all(==(0), sm[:, 5:10])                          # no Hessian slots
end

@testset "Hessian slots: unique entries per component, tensor rebuilt in Gridap's layout" begin
    @test HK.hess_pairs(2) == [(1, 1), (1, 2), (2, 2)] && HK.hess_pairs(3) == [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3)]
    @test [HK.hess_index(a, b) for (a, b) in HK.hess_pairs(3)] == 1:6 && HK.hess_index(3, 1) == HK.hess_index(1, 3) == 4
    # scalar field in 2D: v, ∇v, ∇∇v
    res(u, v) = ∫(∇∇(v) ⊙ ∇∇(u) + ∇(v) ⋅ ∇(u) + v * u)p.dΩ
    pf = HK.PointForm(HK.capture(res, HK.probes(p.U, :unknown), HK.probes(p.V, :test))[1][1].terms)
    @test pf.test.nslots == 1 + 2 + 3
    sm = HK.slot_map(pf.test, 1, 1, 2)
    @test sort(vec(sm)) == 1:6
    hs = sm[1, 4:6]                                        # slots of (1,1), (1,2), (2,2)
    # the generated code rebuilds the symmetric Hessian from the unique slots: with slot values h,
    # the term ∇∇v ⊙ ∇∇u has gradient (w.r.t. the slots) = H_11, 2 H_12, H_22 of the unknown
    H = TensorValue(2.0, 1.0, 1.0, 6.0)
    FV = ((0.0, VectorValue(0.0, 0.0), H),)
    g = HK.ForwardDiff.gradient(s -> pf.core(pf.ops, pf.coefs, FV, (), s, SVector{0,Float64}()), zeros(SVector{6,Float64}))
    @test g[hs] ≈ [2.0, 2.0, 6.0]
    # vector field in 3D: ∇∇v is a ThirdOrderTensorValue{3,3,3} with H[a,b,k] = ∂²v_k/∂x_a∂x_b
    rese(u, v) = ∫(∇∇(v) ⊙ ∇∇(u))e.dΩ
    pfe = HK.PointForm(HK.capture(rese, HK.probes(e.U, :unknown), HK.probes(e.V, :test))[1][1].terms)
    @test pfe.test.nslots == 3 * 6 && pfe.test.leaves[1].T === ThirdOrderTensorValue{3,3,3,Float64,27}
    sme = HK.slot_map(pfe.test, 1, 3, 3)
    @test all(==(0), sme[:, 1:4]) && sort(vec(sme[:, 5:10])) == 1:18
    Hu = ThirdOrderTensorValue{3,3,3}(ntuple(i -> Float64(i), 27))
    Hs = ThirdOrderTensorValue{3,3,3}(ntuple(i -> begin
            a, b, k = Tuple(CartesianIndices((3, 3, 3))[i])
            0.5 * (Hu[a, b, k] + Hu[b, a, k])
        end, 27))
    FVe = ((VectorValue(0.0, 0.0, 0.0), zero(TensorValue{3,3,Float64}), Hu),)
    ge = HK.ForwardDiff.gradient(s -> pfe.core(pfe.ops, pfe.coefs, FVe, (), s, SVector{0,Float64}()), zeros(SVector{18,Float64}))
    for k in 1:3, (q, (a, b)) in enumerate(HK.hess_pairs(3))
        @test ge[sme[k, 4+q]] ≈ (a == b ? 1.0 : 2.0) * Hs[a, b, k]
    end
end

@testset "generated integrand reproduces the pointwise expression" begin
    # Poisson residual: ∇v⋅∇u - f v  with sv = (v, ∂1v, ∂2v)
    pf = pointform(p, :res)
    ∇u = VectorValue(0.3, -0.2)
    FV = ((0.0, ∇u),)
    A = (0.7,)                       # f at the point
    sv = SVector(1.5, 0.4, -0.1)
    val = pf.core(pf.ops, pf.coefs, FV, A, sv, SVector{0,Float64}())
    @test val ≈ VectorValue(0.4, -0.1) ⋅ ∇u - 0.7 * 1.5
    # linear in the test slots: gradient recovers (−f, ∂1u, ∂2u)
    g = HK.ForwardDiff.gradient(s -> pf.core(pf.ops, pf.coefs, FV, A, s, SVector{0,Float64}()), zero(sv))
    @test g ≈ [-0.7, 0.3, -0.2]
end

@testset "common subexpressions are emitted once" begin
    pf = pointform(e, :jac)
    body = string(Base.remove_linenums!(deepcopy(pf.expr)))
    # Fk ∘ ∇u' appears in the Jacobian once per term; ops are deduplicated
    fops = [o for (i, o) in enumerate(pf.ops) if !(i in pf.consts)]
    @test length(fops) == length(unique(objectid.(fops)))
    @test count("ops[", body) ≥ length(pf.ops)
end

@testset "show_code prints the expression" begin
    pf = pointform(p, :res)
    s = sprint(show, Base.remove_linenums!(deepcopy(pf.expr)))
    @test occursin("ops", s) && occursin("sv", s)
    @test HK.show_code(pf) === nothing
end

@testset "numbers of the integrand are parameters, not literals" begin
    r(Λ) = (u, v) -> ∫(∇(v) ⋅ ∇(u) + Λ * v * u - 2.0 * v)p.dΩ
    f(Λ) = HK.capture(r(Λ), HK.probes(p.U, :unknown), HK.probes(p.V, :test))[1][1]
    pf1 = HK.PointForm(f(1.5).terms); pf2 = HK.PointForm(f(-3.0).terms)
    @test length(pf1.consts) == 2 && sort([pf1.ops[i] for i in pf1.consts]) == [1.5, 2.0]
    @test pf1.expr == pf2.expr && typeof(pf1.core) === typeof(pf2.core)      # same code for other values
    @test !occursin("1.5", string(pf1.expr))
    FV = ((0.2, VectorValue(0.3, -0.2)),); sv = SVector(1.5, 0.4, -0.1)
    @test pf1.core(pf1.ops, pf1.coefs, FV, (), sv, SVector{0,Float64}()) ≈ 0.4 * 0.3 + 0.1 * 0.2 + 1.5 * 1.5 * 0.2 - 2.0 * 1.5
    @test pf2.core(pf2.ops, pf2.coefs, FV, (), sv, SVector{0,Float64}()) ≈ 0.4 * 0.3 + 0.1 * 0.2 - 3.0 * 1.5 * 0.2 - 2.0 * 1.5
    # equal values are not merged into one slot (the code would depend on the values)
    pf3 = HK.PointForm(f(2.0).terms)
    @test length(pf3.consts) == 2 && pf3.expr == pf1.expr
end

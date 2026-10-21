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
    @test size(sm) == (1, 3) && all(>(0), sm) && allunique(sm)
    pe = pointform(e, :jac)
    @test pe.test.nslots == 9 && pe.trial.nslots == 9      # only gradients
    sm = HK.slot_map(pe.test, 1, 3, 3)
    @test sm[:, 1] == [0, 0, 0]                            # no value slots
    @test sort(vec(sm[:, 2:4])) == 1:9
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

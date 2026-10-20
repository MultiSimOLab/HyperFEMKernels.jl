Random.seed!(1)
p = poisson_2d()
e = elasticity_3d()

function pointform(prob, kind)
    forms = kind === :res ? HK.capture(prob.res, HK.probes(prob.U, :unknown), HK.probes(prob.V, :test)) :
            HK.capture(prob.jac, HK.probes(prob.U, :unknown), HK.probes(prob.U, :trial), HK.probes(prob.V, :test))
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
    @test length(pf.ops) == length(unique(objectid.(collect(pf.ops))))
    @test count("ops[", body) ≥ length(pf.ops)
end

@testset "show_code prints the expression" begin
    pf = pointform(p, :res)
    s = sprint(io -> redirect_stdout(io) do; HK.show_code(pf); end)
    @test occursin("ops", s) && occursin("sv", s)
end

# Factorised Jacobians (src/factorforms.jl): a part c·W(N(A, B)) with A test-only and B trial-only gives
# T[i, j] = c·W(N(∂A/∂s_i, ∂B/∂r_j)). Checked against the nested dual-number path (the same operator with the
# factorisation disabled) and against Gridap, on forms that factorise and on forms that must not.

paths(op) = [(g.kind, g.domain, g.path) for g in HK.work_summary(op).groups]
"Operator with the factorisation on or off (read at construction)."
function with_factorise(f, flag::Bool)
    old = HK.FACTORISE[]
    HK.FACTORISE[] = flag
    try
        f()
    finally
        HK.FACTORISE[] = old
    end
end
jac_paths(op) = [g.path for g in HK.work_summary(op).groups if g.kind === :jacobian]

@testset "detection: parts, wrappers, signs; what must not factorise" begin
    p = elasticity_3d()
    dΩ = Measure(get_triangulation(p.U), 2)
    u0 = FEFunction(p.U, randx(p.U; scale=0.1))
    w = VectorValue(0.3, -0.2, 0.1)
    c = 2.5
    forms = [
        # products ⋅ ⊙ * under state multipliers, divisions and signs, sums inside a multiplier
        ("wrappers and signs", (u, du, v) -> ∫(c * ((∇(v) ⋅ w) ⋅ ((∇(du)' ⋅ ∇(u)) ⋅ w)) - (v ⋅ du) / (1.0 + norm ∘ u) +
                                               (norm ∘ u) * ((∇(v) ⊙ ∇(du)) + 3.0 * (v ⋅ du)))dΩ, true),
        # (a form the linear maps already take, without the division, stays on that path)
        ("test on the right of the product", (u, du, v) -> ∫(((∇(du) ⋅ w) ⋅ (∇(v) ⋅ w)) / (1.0 + norm ∘ u))dΩ, true),
        # a contraction with a state tensor around a mixed product: not in the accepted forms
        ("state contraction around the product", (u, du, v) -> ∫((∇(v)' ⋅ ∇(du)) ⊙ ∇(u))dΩ, false),
        # a function applied to test and trial together (a composed operation): not separable on the tree
        ("composed operation of test and trial", (u, du, v) -> ∫(((a, b) -> a ⋅ b) ∘ (v, du))dΩ, false),
    ]
    for (name, jac, expect) in forms
        res = (u, v) -> jac(u, u, v)
        op = KernelOperator(res, jac, p.U, p.V)
        @test (:factorised in jac_paths(op)) == expect
        x = randx(p.U; scale=0.1)
        cc = check(res, jac, p.U, p.V, x; op)
        @test cc.jac < 1e-12
    end
end

@testset "factorised tangent equals the nested dual-number path (whole Jacobian)" begin
    s = oseen_channel()
    cases = [(p.res, p.jac, p.U, p.V) for p in (elasticity_3d(), coupled_3d())]
    push!(cases, (s.res, s.jac, s.U, s.V))
    for (res, jac, U, V) in cases
        x = randx(U; scale=0.5)
        opf = with_factorise(() -> KernelOperator(res, jac, U, V), true)
        opn = with_factorise(() -> KernelOperator(res, jac, U, V), false)
        @test :factorised in jac_paths(opf)
        @test !(:factorised in jac_paths(opn))
        Af, An = jacobian(opf, x), jacobian(opn, x)
        @test norm(Af - An) <= 1e-13 * norm(An)
        # matrix-free: the prepared tangents give the same product
        # (matrix-free and assembled products differ by summation order, ~1e-12 relative on these cases)
        mff = MatrixFreeJacobian(with_factorise(() -> KernelOperator(res, jac, U, V; jacobian=:matrixfree), true))
        mfn = MatrixFreeJacobian(with_factorise(() -> KernelOperator(res, jac, U, V; jacobian=:matrixfree), false))
        prepare!(mff, x); prepare!(mfn, x)
        dx = randn(length(x))
        @test norm(mff * dx - mfn * dx) <= 1e-13 * norm(mfn * dx)
        @test norm(mff * dx - An * dx) <= 1e-11 * norm(An * dx)
        @test all(a -> a.bytes == 0, allocations(opf, x))
    end
end

@testset "construction check: a wrong factorisation falls back to the nested path" begin
    p = elasticity_3d()
    op = KernelOperator(p.res, p.jac, p.U, p.V)
    g = op.jgroups[1]
    @test g.lin isa HK.FactorForm
    terms = HK.terms_of(g, HK.capture(p.jac, HK.probes(p.U, :unknown), HK.probes(p.U, :trial), HK.probes(p.V, :test))[1])
    code = HK.factor_code(terms, g.pf)
    ex, ops, coefs, consts, st = code
    bad = HK.FactorForm(g.lin.fun, ops, coefs .* (1 + 1e-9), ex, consts)       # a part off by 1e-9
    @test HK.check_factorisation(bad, st, terms, g.pf, g.tables, g.aux) isa String
    @test HK.check_factorisation(g.lin, st, terms, g.pf, g.tables, g.aux) === true
end

@testset "update: the factorised groups are rebound without new code" begin
    s = oseen_channel()
    op = KernelOperator(s.res, s.jac, s.U, s.V)
    HK.reset_counters!()
    op2 = update(op, s.res, s.jac)
    @test HK.counters().update_fast == 1 && HK.counters().codegen == 0
    s.Pc[] = 1.3
    op3 = update(op2, s.res, s.jac)
    x = randx(s.U; scale=0.5)
    c = check(s.res, s.jac, s.U, s.V, x; op=op3)
    @test c.res < TOL && c.jac < TOL
    @test count(g -> g.lin isa HK.FactorForm, op3.jgroups) == count(g -> g.lin isa HK.FactorForm, op.jgroups)
end

@testset "the construction check leaves the random numbers of the script unchanged" begin
    p = elasticity_3d()
    Random.seed!(11)
    opn = with_factorise(() -> KernelOperator(p.res, p.jac, p.U, p.V), false)
    rn = rand(3)
    Random.seed!(11)
    opf = with_factorise(() -> KernelOperator(p.res, p.jac, p.U, p.V), true)
    @test :factorised in jac_paths(opf)          # the check ran
    @test rand(3) == rn                         # same stream as without the factorisation
end

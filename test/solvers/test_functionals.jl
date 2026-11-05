# S3 — `KernelFunctional`: energies and pointwise extrema evaluated by the kernels.
Random.seed!(5)
Ψ_svk(F) = (E = E_svk(F); 0.5 * λ_svk * tr(E)^2 + μ_svk * (E ⊙ E))
lazy_sum(f, U, x) = sum(f(FEFunction(U, x)))

"Extremum of a pointwise field over the quadrature points of `dΩ` with Gridap's lazy evaluation (as HyperFEM's injectivity step)."
function lazy_extremum(cf, dΩ, op)
    vals = cf(get_cell_points(dΩ.quad))
    cache = array_cache(vals)
    acc = op === min ? Inf : -Inf
    for c in 1:length(vals), v in getindex!(cache, vals, c)
        acc = op(acc, v)
    end
    acc
end

@testset "energy (reduce = :sum) equals the lazy sum of the integrals" begin
    e = elasticity_3d(; order=2); dΩe = e.dΩ
    energy_e(u) = ∫(Ψ_svk ∘ (Fk ∘ (∇(u)')))dΩe
    p = poisson_2d(; order=2); dΩp = p.dΩ
    fp(x) = sin(x[1]) * x[2]
    energy_p(u) = ∫(0.5 * (∇(u) ⋅ ∇(u)) - fp * u)dΩp
    c = coupled_3d(); dΩc = c.dΩ
    energy_c((u, φ)) = ∫(Ψ_svk ∘ (Fk ∘ (∇(u)')) + 0.5 * (∇(φ) ⋅ ∇(φ)) + 0.3 * φ * (tr ∘ (Fk ∘ (∇(u)'))))dΩc
    for (prob, en) in ((e, energy_e), (p, energy_p), (c, energy_c))
        fun = KernelFunctional(en, prob.U)
        for _ in 1:2
            x = randx(prob.U; scale=0.1)
            @test fun(x) ≈ lazy_sum(en, prob.U, x) rtol = 1e-12
        end
    end
    # the gradient of the energy is the residual (finite differences along a random direction)
    fun = KernelFunctional(energy_e, e.U)
    op = KernelOperator(e.res, e.jac, e.U, e.V)
    x = randx(e.U; scale=0.05); dx = randn(num_free_dofs(e.U)); h = 1e-6
    @test (fun(x + h * dx) - fun(x - h * dx)) / 2h ≈ dot(residual(op, x), dx) rtol = 1e-6
end

@testset "energy with a boundary term and a load factor; update reuses the tables" begin
    e = elasticity_3d(; order=2)
    Γ = BoundaryTriangulation(e.model; tags=[22]); dΓ = Measure(Γ, 4); dΩe = e.dΩ
    t = VectorValue(0.0, 0.0, 0.2)
    en(Λ) = u -> ∫(Ψ_svk ∘ (Fk ∘ (∇(u)')))dΩe - ∫(Λ * (u ⋅ t))dΓ
    fun = KernelFunctional(en(0.5), e.U)
    x = randx(e.U; scale=0.1)
    @test fun(x) ≈ lazy_sum(en(0.5), e.U, x) rtol = 1e-12
    fun2 = update(fun, en(1.0))
    @test fun2(x) ≈ lazy_sum(en(1.0), e.U, x) rtol = 1e-12
    @test fun2.groups[1].tables === fun.groups[1].tables && fun2.groups[1].geo === fun.groups[1].geo
    @test length(fun.groups) == 2 && length(fun.fallback) == 0
end

@testset "pointwise extrema (reduce = :min / :max) with an increment stored in an FE function" begin
    e = elasticity_3d(; order=2); dΩe = e.dΩ
    n = num_free_dofs(e.U)
    dxbuf = zeros(n)
    duh = FEFunction(e.U, dxbuf)
    # a pointwise quantity of the state and the increment (any deterministic function: the kernel and the
    # lazy evaluation must agree; the injectivity root itself is HyperFEM's and is tested there)
    g(∇u, ∇du) = det(one(∇u) + ∇u) / (1.0 + abs(tr(∇du)) + 0.1 * (∇du ⊙ ∇du))
    f_min(u) = ∫(g ∘ (∇(u)', ∇(duh)'))dΩe
    fmin = KernelFunctional(f_min, e.U; reduce=:min)
    fmax = KernelFunctional(f_min, e.U; reduce=:max)
    for _ in 1:3
        x = randx(e.U; scale=0.1)
        dxbuf .= randn(n)                                           # the increment changes in place: no update needed
        cf = g ∘ (∇(FEFunction(e.U, x))', ∇(duh)')
        @test fmin(x) ≈ lazy_extremum(cf, dΩe, min) rtol = 1e-13
        @test fmax(x) ≈ lazy_extremum(cf, dΩe, max) rtol = 1e-13
        @test fmin(x) < fmax(x)
    end
    # multi-field: the increment of the first field only
    c = coupled_3d(); dΩc = c.dΩ
    dxc = zeros(num_free_dofs(c.U)); duc = FEFunction(c.U, dxc)
    f_c((u, φ)) = ∫(g ∘ (∇(u)', ∇(duc[1])') + φ * φ)dΩc
    fc = KernelFunctional(f_c, c.U; reduce=:min)
    xc = randx(c.U; scale=0.1); dxc .= randn(length(dxc))
    uh = FEFunction(c.U, xc)
    @test fc(xc) ≈ lazy_extremum(g ∘ (∇(uh[1])', ∇(duc[1])') + uh[2] * uh[2], dΩc, min) rtol = 1e-13
end

@testset "errors and allocations" begin
    p = poisson_2d(; order=2); dΩp = p.dΩ
    @test_throws ArgumentError KernelFunctional(u -> ∫(u * u)dΩp, p.U; reduce=:mean)
    # a form the kernels cannot read (here: wrong arity) goes to Gridap with :sum, and is refused with :min
    bad(u, v) = ∫(u * v)dΩp
    @test_throws ErrorException KernelFunctional(bad, p.U; reduce=:min)
    fb = @test_logs (:warn, r"hybrid fallback") KernelFunctional(bad, p.U)
    @test fb.fallback == [:all] && isempty(fb.groups)
    x = randx(p.U)
    fun = KernelFunctional(u -> ∫(0.5 * (∇(u) ⋅ ∇(u)))dΩp, p.U)
    fun(x)
    # the cell loop does not allocate: the same bytes for one cell and for all of them (buffers only)
    g = fun.groups[1]
    dvals = [get_dirichlet_dof_values(s) for s in HK.field_spaces(p.U)]
    auxd = HK.aux_data(fun, g)
    one_cell, all_cells = Int32[1], g.colors[1]
    HK.functional_cells(Val(:sum), fun, g, x, dvals, auxd, one_cell)
    a1 = @allocated HK.functional_cells(Val(:sum), fun, g, x, dvals, auxd, one_cell)
    a2 = @allocated HK.functional_cells(Val(:sum), fun, g, x, dvals, auxd, all_cells)
    @test a1 == a2
end

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
    @test fun2.groups[1].pf.expr == fun.groups[1].pf.expr            # the load factor is a parameter: same generated code
    @test length(fun.groups) == 2 && length(fun.fallback) == 0
    # a functional whose first term is the boundary one, updated with a volume term: every cell integrated
    enb(Λ) = u -> ∫(-Λ * (u ⋅ t))dΓ + ∫(Ψ_svk ∘ (Fk ∘ (∇(u)')))dΩe
    fb0 = KernelFunctional(u -> ∫(-0.5 * (u ⋅ t))dΓ, e.U)
    fb1 = update(fb0, enb(0.5))
    @test fb1(x) ≈ lazy_sum(enb(0.5), e.U, x) rtol = 1e-12
    @test fb1(x) ≈ fun(x) rtol = 1e-12
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

# Skeleton integrals: the energy of the third-medium regularisation (jump penalties of ∇u and ∇∇u on the skeleton of a
# subdomain, facet size as data) and a mean term on the whole skeleton; non-affine quadrilaterals against hessian_nonaffine.
function skeleton_energy_problem(; simplex=true, mapped=false)
    model = cartesian_model(2, (4, 3); simplex, mapped)
    Ω = Triangulation(model); dΩ = Measure(Ω, 4)
    cc = [sum(c) / length(c) for c in get_cell_coordinates(model)]
    Ωc = Triangulation(model, [x[1] > 0.4 for x in cc]); dΩc = Measure(Ωc, 4)
    Λ = SkeletonTriangulation(Ωc); dΛ = Measure(Λ, 4)
    Λa = SkeletonTriangulation(model); dΛa = Measure(Λa, 4)
    hF = CellField(get_array(∫(1)dΛ), Λ)
    V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, 2), dirichlet_tags="boundary")
    U = TrialFESpace(V, x -> VectorValue(0.01x[2], -0.02x[1] * x[2]))
    mk(H) = β -> u -> ∫(Ψ_svk ∘ (Fk ∘ (∇(u)')))dΩ + (0.5β) * ∫(H(u) ⊙ H(u))dΩc +
                      ∫((0.5β / hF) * (jump(∇(u)) ⊙ jump(∇(u))) + (β * hF) * (jump(H(u)) ⊙ jump(H(u))))dΛ + ∫(0.1 * (mean(u) ⋅ mean(u)))dΛa
    # the residual of the energy (gradient), for the finite-difference check
    res(β) = (u, v) -> ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))))dΩ + β * ∫(∇∇(u) ⊙ ∇∇(v))dΩc +
                       ∫((β / hF) * (jump(∇(u)) ⊙ jump(∇(v))) + (2β * hF) * (jump(∇∇(u)) ⊙ jump(∇∇(v))))dΛ + ∫(0.2 * (mean(u) ⋅ mean(v)))dΛa
    jac(β) = (u, du, v) -> ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')))dΩ + β * ∫(∇∇(du) ⊙ ∇∇(v))dΩc +
                           ∫((β / hF) * (jump(∇(du)) ⊙ jump(∇(v))) + (2β * hF) * (jump(∇∇(du)) ⊙ jump(∇∇(v))))dΛ + ∫(0.2 * (mean(du) ⋅ mean(v)))dΛa
    (; model, Λ, dΛ, U, V, en=mk(∇∇), enref=mk(mapped ? hessian_nonaffine : ∇∇), res, jac)
end

@testset "skeleton integrals (jumps of ∇u and ∇∇u, mean): energy, update, threads, gradient, extrema" begin
    for (simplex, mapped) in ((true, false), (false, false), (false, true))
        s = skeleton_energy_problem(; simplex, mapped)
        fun = KernelFunctional(s.en(0.05), s.U)
        @test isempty(fun.fallback) && count(g -> g isa HK.SkeletonGroup, fun.groups) >= 2
        @test length(fun.skel) == 2 && all(sd -> length(sd.colors) == 1, fun.skel)   # a reduction: one colour per skeleton
        x = randx(s.U; scale=0.1)
        @test fun(x) ≈ lazy_sum(s.enref(0.05), s.U, x) rtol = 1e-12
        # update with another β: groups rebound, tables and skeleton data reused
        fun2 = update(fun, s.en(0.3))
        @test fun2(x) ≈ lazy_sum(s.enref(0.3), s.U, x) rtol = 1e-12
        @test fun2.skel === fun.skel && all(((g1, g2),) -> g1.tables === g2.tables, zip(fun.groups, fun2.groups))
        @test KernelFunctional(s.en(0.05), s.U; nthreads=1)(x) ≈ fun(x) rtol = 1e-13
        if !mapped
            # the gradient of the energy is the residual (finite differences along a random direction)
            op = KernelOperator(s.res(0.05), s.jac(0.05), s.U, s.V)
            dx = randn(num_free_dofs(s.U)); h = 1e-6
            @test (fun(x + h * dx) - fun(x - h * dx)) / 2h ≈ dot(residual(op, x), dx) rtol = 1e-6
        end
    end
    # pointwise extrema over the facet points
    s = skeleton_energy_problem()
    g(a, b) = a / (1.0 + b * b)
    f(u) = ∫(g ∘ (jump(∇(u)) ⊙ jump(∇(u)), mean(tr ∘ ∇(u))))s.dΛ
    x = randx(s.U; scale=0.1); uh = FEFunction(s.U, x)
    cf = g ∘ (jump(∇(uh)) ⊙ jump(∇(uh)), mean(tr ∘ ∇(uh)))
    @test KernelFunctional(f, s.U; reduce=:min)(x) ≈ lazy_extremum(cf, s.dΛ, min) rtol = 1e-13
    @test KernelFunctional(f, s.U; reduce=:max)(x) ≈ lazy_extremum(cf, s.dΛ, max) rtol = 1e-13
    # the facet loop does not allocate (same bytes for one facet and for all of them)
    fun = KernelFunctional(s.en(0.05), s.U)
    fun(x)
    gs = first(gg for gg in fun.groups if gg isa HK.SkeletonGroup)
    dvals = [get_dirichlet_dof_values(sp) for sp in HK.field_spaces(s.U)]
    auxd = HK.aux_data(fun, gs)
    one_facet, all_facets = Int32[1], gs.colors[1]
    HK.functional_cells(Val(:sum), fun, gs, x, dvals, auxd, one_facet)
    a1 = @allocated HK.functional_cells(Val(:sum), fun, gs, x, dvals, auxd, one_facet)
    a2 = @allocated HK.functional_cells(Val(:sum), fun, gs, x, dvals, auxd, all_facets)
    @test a1 == a2
end

@testset "errors and allocations" begin
    p = poisson_2d(; order=2); dΩp = p.dΩ
    @test_throws ArgumentError KernelFunctional(u -> ∫(u * u)dΩp, p.U; reduce=:mean)
    # a functional called with the wrong arguments is the caller's error, not a fallback to Gridap
    bad(u, v) = ∫(u * v)dΩp
    @test_throws MethodError KernelFunctional(bad, p.U; reduce=:min)
    @test_throws MethodError KernelFunctional(bad, p.U)
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

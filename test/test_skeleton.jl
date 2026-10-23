# Skeleton integrals in the engine: interior facets with a cell on each side (plus/minus, jump, mean),
# assembled by facet kernels that see both cells. Every case is checked against pure Gridap (tolerance TOL)
# and must not use the hybrid fallback.

nofallback_sk(op) = isempty(op.fallback.rtrians) && isempty(op.fallback.jtrians)
nskgroups(op) = count(g -> g isa HK.SkeletonGroup, op.jgroups)
noalloc(op, x) = all(a -> a.bytes == 0, allocations(op, x))

"Interior-penalty Poisson problem: volume + jump of the unknown on the skeleton."
function ip_poisson(; order=1, simplex=false, n=(5, 4), γ=10.0)
    p = poisson_2d(; n, order, simplex)
    Λ = SkeletonTriangulation(p.model); dΛ = Measure(Λ, 2order)
    res(u, v) = ∫(∇(v) ⋅ ∇(u) - 2.0 * v)p.dΩ + ∫(γ * jump(v) * jump(u))dΛ
    jac(u, du, v) = ∫(∇(v) ⋅ ∇(du))p.dΩ + ∫(γ * jump(v) * jump(du))dΛ
    (; p, Λ, dΛ, res, jac)
end

@testset "interior penalty on quads/triangles, orders 1–3: no fallback, equality, no allocations" begin
    for order in (1, 2, 3), simplex in (false, true)
        s = ip_poisson(; order, simplex)
        op = @test_logs KernelOperator(s.res, s.jac, s.p.U, s.p.V)
        @test nofallback_sk(op)
        @test length(op.skel) == 1 && nskgroups(op) >= 1
        x = randx(s.p.U)
        c = check(s.res, s.jac, s.p.U, s.p.V, x; op)
        @test c.res < TOL && c.jac < TOL
        @test noalloc(op, x)
        # the plus/minus leaves of the skeleton groups carry their side
        for g in op.jgroups
            g isa HK.SkeletonGroup || continue
            @test all(l -> l.side in (:plus, :minus), vcat(g.pf.test.leaves, g.pf.trial.leaves))
        end
    end
end

@testset "sparsity: the facets couple neighbouring cells; positions precomputed per facet" begin
    s = ip_poisson(; n=(6, 5), order=2)
    p = s.p
    op = KernelOperator(s.res, s.jac, p.U, p.V)
    op0 = KernelOperator(p.res, p.jac, p.U, p.V)
    x = randx(p.U)
    Ak = Algebra.jacobian(op, x)
    Ag = Algebra.jacobian(gridap_op((; s.res, s.jac, p.U, p.V)), x)
    @test nnz(op.A0) > nnz(op0.A0) && reldiff(Ak, Ag) < TOL
    @test HK.fallback_pattern(op.fallback, p.U, p.V, s.jac) === nothing       # nothing assembled by Gridap
    sd = op.skel[1]
    ndl = op.ndl
    @test size(sd.cells) == (2, num_cells(s.Λ)) && size(sd.nzpos) == ((2ndl)^2, num_cells(s.Λ))
    # every stored position points to the (row, column) of the facet entry
    rv = rowvals(op.A0); cp = SparseArrays.getcolptr(op.A0)
    colof = zeros(Int, nnz(op.A0))
    for c in 1:size(op.A0, 2), q in cp[c]:cp[c+1]-1
        colof[q] = c
    end
    fl = op.fields[1]
    for f in 1:num_cells(s.Λ), jj in 1:2ndl, ii in 1:2ndl
        ci, i = ii <= ndl ? (sd.cells[1, f], ii) : (sd.cells[2, f], ii - ndl)
        cj, j = jj <= ndl ? (sd.cells[1, f], jj) : (sd.cells[2, f], jj - ndl)
        r, c = fl.test_ids[i, ci], fl.trial_ids[j, cj]
        q = sd.nzpos[ii+(jj-1)*2ndl, f]
        if r > 0 && c > 0
            @test q > 0 && rv[q] == r && colof[q] == c
        else
            @test q == 0
        end
    end
end

@testset "colouring of the facets and threads" begin
    s = ip_poisson(; n=(7, 6), order=2)
    p = s.p
    x = randx(p.U)
    ref = nothing
    for nthreads in (1, Threads.nthreads())
        op = KernelOperator(s.res, s.jac, p.U, p.V; nthreads)
        sd = op.skel[1]
        for col in sd.colors      # two facets of a colour share no cell (hence no dof)
            @test allunique(vcat([sd.cells[:, f] for f in col]...))
        end
        @test sum(length, sd.colors) == num_cells(s.Λ)
        for g in op.jgroups       # the colours of a class partition its facets
            g isa HK.SkeletonGroup || continue
            @test sort(reduce(vcat, g.colors)) == 1:length(g.facets)
        end
        b = Algebra.residual(op, x); A = Algebra.jacobian(op, x)
        ref === nothing ? (ref = (b, A)) : (@test reldiff(b, ref[1]) < TOL && reldiff(A, ref[2]) < TOL)
    end
end

@testset "mean, normal and vector fields: 3D hexahedra and tetrahedra, orders 1–2" begin
    for simplex in (false, true), order in (1, 2)
        model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (2, 2, 2))
        simplex && (model = simplexify(model))
        Ω = Triangulation(model); dΩ = Measure(Ω, 2order)
        V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, order), dirichlet_tags=[1, 3, 5, 7])
        U = TrialFESpace(V, x -> VectorValue(0.01, 0.0, 0.0))
        Λ = SkeletonTriangulation(model); dΛ = Measure(Λ, 2order)
        n = get_normal_vector(Λ)
        res(u, v) = ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))))dΩ +
                    ∫(mean(v) ⋅ mean(u) + 0.3 * (jump(v ⊗ n) ⊙ jump(u ⊗ n)) + (mean(∇(v)) ⊙ mean(∇(u))))dΛ
        jac(u, du, v) = ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')))dΩ +
                        ∫(mean(v) ⋅ mean(du) + 0.3 * (jump(v ⊗ n) ⊙ jump(du ⊗ n)) + (mean(∇(v)) ⊙ mean(∇(du))))dΛ
        op = KernelOperator(res, jac, U, V)
        @test nofallback_sk(op)
        x = randx(U)
        c = check(res, jac, U, V, x; op)
        @test c.res < TOL && c.jac < TOL
        @test noalloc(op, x)
    end
end

# Weighted jump of the gradient as in third-medium contact (Bluhm, Sigmund, Poulios 2021): the weight is a
# nonlinear function of F = I + ∇u on both sides (mean), the term is the jump of ∇u, the tangent is consistent
# (and not symmetric). The regularisation parameter and the facet size enter as auxiliary data on the skeleton.
wtm(A) = exp(-2.0 * det(A)) + exp(0.5 * (A ⊙ A - 2det(A))) - 1.0
cof2(A) = TensorValue(A[2, 2], -A[1, 2], -A[2, 1], A[1, 1])
dwtm(A) = -2.0 * exp(-2.0 * det(A)) * cof2(A) + exp(0.5 * (A ⊙ A - 2det(A))) * (A - cof2(A))
F2(∇u) = one(∇u) + ∇u

function weighted_jump_problem(; order=1, simplex=true, n=(4, 3), linear_maps=true)
    model = CartesianDiscreteModel((0, 1, 0, 1), n)
    simplex && (model = simplexify(model))
    Ω = Triangulation(model); dΩ = Measure(Ω, 2order)
    V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, order), dirichlet_tags=[1, 2])
    U = TrialFESpace(V, x -> VectorValue(0.01, -0.02))
    Λ = SkeletonTriangulation(model); dΛ = Measure(Λ, 2order)
    hF = CellField(get_array(∫(1)dΛ), Λ)
    β = 0.05
    cJ = β / hF
    Fu(u) = F2 ∘ (∇(u)')
    qJ(a, b) = jump(∇(a)) ⊙ jump(∇(b))
    res(u, v) = ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))))dΩ + ∫(cJ * (mean(wtm ∘ Fu(u)) * qJ(u, v)))dΛ
    jac(u, du, v) = ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')))dΩ +
                    ∫(cJ * (mean(wtm ∘ Fu(u)) * qJ(du, v) + mean((dwtm ∘ Fu(u)) ⊙ ∇(du)') * qJ(u, v)))dΛ
    (; model, Ω, dΩ, Λ, dΛ, U, V, res, jac, linear_maps)
end

@testset "weighted jump of ∇u (third-medium regularisation): nonlinear weight on both sides, facet size as data" begin
    for simplex in (true, false), linear_maps in (true, false)
        w = weighted_jump_problem(; simplex, linear_maps)
        op = KernelOperator(w.res, w.jac, w.U, w.V; linear_maps)
        @test nofallback_sk(op)
        x = randx(w.U; scale=0.1)
        c = check(w.res, w.jac, w.U, w.V, x; op)
        @test c.res < TOL && c.jac < TOL
        @test noalloc(op, x)
        A = Algebra.jacobian(op, x)
        @test norm(A - A') > 1e-6 * norm(A)        # the consistent tangent is not symmetric
        # weight × jump is bilinear in the state inputs: the skeleton groups take the dual-number path
        for g in op.jgroups
            g isa HK.SkeletonGroup || continue
            @test g.lin === nothing
        end
    end
end

"Model (0,2)×(0,1); active region Ωd without the top row of cells; Ωa its left half."
function skeleton_subdomains(; n=(8, 4))
    model = CartesianDiscreteModel((0, 2, 0, 1), n)
    cc = [sum(c) / length(c) for c in get_cell_coordinates(model)]
    dom = [x[2] < 0.75 for x in cc]
    Ωd = Triangulation(model, dom)
    Ωa = Triangulation(model, [d && x[1] < 1 for (d, x) in zip(dom, cc)])
    (; model, Ωd, Ωa)
end
κsk(g) = (1 + g ⋅ g) * g
dκsk(g, dg) = (1 + g ⋅ g) * dg + 2 * (g ⋅ dg) * g

@testset "skeleton of a subdomain (composite triangulation), FE space on a sub-triangulation" begin
    # the space lives on the lower three rows (Ωd); the skeleton is that of the left half (Ωa)
    s = skeleton_subdomains()
    for order in (1, 2)
        dΩd = Measure(s.Ωd, 2order)
        Λa = SkeletonTriangulation(s.Ωa); dΛa = Measure(Λa, 2order)
        @test !(Λa isa SkeletonTriangulation)       # Gridap builds a composite triangulation; its glue is a SkeletonPair
        V = TestFESpace(s.Ωd, ReferenceFE(lagrangian, Float64, order); dirichlet_tags=[7])
        U = TrialFESpace(V, x -> x[2])
        res(φ, v) = ∫(∇(v) ⋅ (κsk ∘ ∇(φ)))dΩd + ∫(jump(∇(v)) ⋅ jump(∇(φ)) + mean(v) * mean(φ * φ))dΛa
        jac(φ, dφ, v) = ∫(∇(v) ⋅ (dκsk ∘ (∇(φ), ∇(dφ))))dΩd + ∫(jump(∇(v)) ⋅ jump(∇(dφ)) + 2.0 * mean(v) * mean(φ * dφ))dΛa
        op = KernelOperator(res, jac, U, V)
        @test nofallback_sk(op)
        @test size(op.skel[1].cells, 2) == num_cells(Λa)
        @test all(c -> 1 <= c <= num_cells(s.Ωd), op.skel[1].cells)
        c = check(res, jac, U, V, randx(U; scale=0.5); op)
        @test c.res < TOL && c.jac < TOL
    end
end

@testset "multi-field: jump of one field, coupling through the other" begin
    p = coupled_3d(; n=(2, 2, 2))
    Λ = SkeletonTriangulation(p.model); dΛ = Measure(Λ, 2)
    res((u, φ), (v, vφ)) = p.res((u, φ), (v, vφ)) + ∫(jump(vφ) * jump(φ) + 0.1 * (mean(v) ⋅ mean(u)) * mean(φ))dΛ
    jac((u, φ), (du, dφ), (v, vφ)) = p.jac((u, φ), (du, dφ), (v, vφ)) +
        ∫(jump(vφ) * jump(dφ) + 0.1 * (mean(v) ⋅ mean(du)) * mean(φ) + 0.1 * (mean(v) ⋅ mean(u)) * mean(dφ))dΛ
    op = KernelOperator(res, jac, p.U, p.V)
    @test nofallback_sk(op)
    x = randx(p.U)
    c = check(res, jac, p.U, p.V, x; op)
    @test c.res < TOL && c.jac < TOL
    @test noalloc(op, x)
end

"Same generated functions (no recompilation) in the groups of two operators, skeleton groups included."
same_code_sk(op, op2) = all(zip(vcat(op.rgroups, op.jgroups), vcat(op2.rgroups, op2.jgroups))) do (g, g2)
    typeof(g2) === typeof(g) && typeof(g2.pf.core) === typeof(g.pf.core) && (g.lin === nothing) == (g2.lin === nothing) &&
        (g.lin === nothing || (typeof(g2.lin.apply) === typeof(g.lin.apply) && typeof(g2.lin.state) === typeof(g.lin.state)))
end

@testset "update: numbers are parameters (same code, same pattern); a new skeleton recomputes the pattern" begin
    mk(γ, dΩ, dΛ) = ((u, v) -> ∫(∇(v) ⋅ ∇(u))dΩ + ∫(γ * jump(v) * jump(u) + γ * (jump(∇(v)) ⋅ jump(∇(u))))dΛ,
                     (u, du, v) -> ∫(∇(v) ⋅ ∇(du))dΩ + ∫(γ * jump(v) * jump(du) + γ * (jump(∇(v)) ⋅ jump(∇(du))))dΛ)
    pu = poisson_2d(; order=2, simplex=true)
    Λu = SkeletonTriangulation(pu.model); dΛu = Measure(Λu, 4)
    r1, j1 = mk(1.0, pu.dΩ, dΛu); r2, j2 = mk(2.5, pu.dΩ, dΛu)
    op1 = KernelOperator(r1, j1, pu.U, pu.V)
    op2 = HK.update(op1, r2, j2)
    @test op2.A0 === op1.A0 && op2.nzpos === op1.nzpos
    @test length(op2.skel) == 1 && op2.skel[1].nzpos === op1.skel[1].nzpos      # positions reused
    @test same_code_sk(op1, op2)
    # the tables of the facet classes are reused, not rebuilt (the Jacobian groups share those of the residual)
    all1 = vcat(op1.rgroups, op1.jgroups)
    for g2 in vcat(op2.rgroups, op2.jgroups)
        g2 isa HK.SkeletonGroup || continue
        g = all1[findfirst(g -> g isa HK.SkeletonGroup && g.key == g2.key, all1)]
        @test g2.tables === g.tables && g2.geo === g.geo && g2.geom === g.geom
    end
    for g in op1.jgroups
        g isa HK.SkeletonGroup || continue
        gr = op1.rgroups[findfirst(r -> r isa HK.SkeletonGroup && r.key == g.key, op1.rgroups)]
        @test g.tables === gr.tables
    end
    c = check(r2, j2, pu.U, pu.V, randx(pu.U); op=op2)
    @test c.res < TOL && c.jac < TOL
    # from an operator without skeleton: pattern recomputed (cross blocks), still equal to Gridap
    op3 = HK.update(KernelOperator(pu.res, pu.jac, pu.U, pu.V), r2, j2)
    @test op3.A0 !== op1.A0 && nnz(op3.A0) == nnz(op1.A0)
    c3 = check(r2, j2, pu.U, pu.V, randx(pu.U); op=op3)
    @test c3.res < TOL && c3.jac < TOL
    # the old operator keeps its own positions
    @test size(op1.skel[1].nzpos, 2) == num_cells(Λu)
end

@testset "skeleton term only in the residual; Dirichlet values re-read" begin
    e = elasticity_3d(; order=1)
    Λ = SkeletonTriangulation(e.model); dΛ = Measure(Λ, 2)
    res(u, v) = e.res(u, v) + ∫(mean(v) ⋅ mean(u))dΛ
    jac(u, du, v) = e.jac(u, du, v)
    op = KernelOperator(res, jac, e.U, e.V)
    @test nofallback_sk(op)
    @test any(g -> g isa HK.SkeletonGroup, op.rgroups) && nskgroups(op) == 0
    @test length(op.skel) == 1 && isempty(op.skel[1].nzpos)              # no Jacobian group: the facets are not in the pattern
    @test nnz(op.A0) == nnz(KernelOperator(e.res, e.jac, e.U, e.V).A0)
    x = randx(e.U)
    c = check(res, jac, e.U, e.V, x; op)
    @test c.res < TOL && c.jac < TOL
    # the Dirichlet values of the trial space enter the facet kernels through the gather
    get_dirichlet_dof_values(e.U) .*= 3.0
    c2 = check(res, jac, e.U, e.V, x; op)
    @test c2.res < TOL && c2.jac < TOL
end

@testset "errors: matrix-free with skeleton integrals; a trace without side" begin
    s = ip_poisson()
    op = KernelOperator(s.res, s.jac, s.p.U, s.p.V)
    @test_throws ErrorException MatrixFreeJacobian(op)
    # a function used on the skeleton without plus/minus: that integral goes to the fallback with a clear reason
    # (Gridap itself rejects it when it assembles with real bases)
    bad(u, v) = ∫(∇(v) ⋅ ∇(u))s.p.dΩ + ∫(v * u)s.dΛ
    op2 = @test_logs (:warn, r"without a side") KernelOperator(bad, s.jac, s.p.U, s.p.V)
    @test length(op2.fallback.rtrians) == 1 && op2.fallback.rtrians[1] === s.Λ && isempty(op2.fallback.jtrians)
end

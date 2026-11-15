# Geometry once per cell on affine cells: the kernels compute Jt, inv(Jt) and the measure once per
# cell when the cell map is affine (`GeoTable.cellaff`, `cell_geometry`) and per point otherwise.
# Checked: the detection (Cartesian, simplices, unstructured copies, mapped and partly deformed
# meshes) and equality with Gridap on affine, non-affine and mixed meshes (volume and boundary,
# P1/P2, residual, assembled and matrix-free Jacobian, 1 and 4 threads), and with the per-point path.

Random.seed!(13)

geo_groups(op) = [g for g in vcat(op.rgroups, op.jgroups) if g isa HK.Group]
geo_cellaff(model) = HK.GeoTable(Triangulation(model), quad_points(Measure(Triangulation(model), 2)), quad_weights(Measure(Triangulation(model), 2))).cellaff

"A copy of a Cartesian model with every `every`-th interior node displaced (non-affine cells around them)."
function deformed_model(D::Int, n; amount=0.2, every=1)
    model = Gridap.Geometry.UnstructuredDiscreteModel(CartesianDiscreteModel(D == 2 ? (0, 1, 0, 1) : (0, 1, 0, 1, 0, 1), n))
    grid = get_grid(model)
    coords = copy(Gridap.Geometry.get_node_coordinates(grid))
    rng = MersenneTwister(7)
    inner(c) = all(i -> 1e-9 < c[i] < 1 - 1e-9, 1:D)
    m = 0
    for (k, c) in enumerate(coords)
        inner(c) || continue
        m += 1
        (m - 1) % every == 0 || continue
        coords[k] = c + amount * VectorValue(ntuple(_ -> rand(rng) - 0.5, D)) / n[1]
    end
    pgrid = Gridap.Geometry.UnstructuredGrid(coords, Gridap.Geometry.get_cell_node_ids(grid),
        Gridap.Geometry.get_reffes(grid), Gridap.Geometry.get_cell_type(grid), Gridap.Geometry.Oriented())
    Gridap.Geometry.UnstructuredDiscreteModel(pgrid, Gridap.Geometry.get_grid_topology(model), Gridap.Geometry.get_face_labeling(model))
end

@testset "detection of affine cells" begin
    for D in (2, 3)
        n = D == 2 ? (4, 3) : (3, 2, 2)
        aff(model) = HK.GeoTable(Triangulation(model), quad_points(Measure(Triangulation(model), 2)), quad_weights(Measure(Triangulation(model), 2))).cellaff
        @test all(aff(cartesian_model(D, n)))                                           # Cartesian
        @test all(aff(cartesian_model(D, n; simplex=true)))                             # simplices
        @test all(aff(Gridap.Geometry.UnstructuredDiscreteModel(cartesian_model(D, n)))) # unstructured copy of a Cartesian grid
        @test all(aff(cartesian_model(D, n; mapped=true, simplex=true)))                # simplices of a curved map: still affine
        @test !any(aff(cartesian_model(D, n; mapped=true)))                             # n-cubes of a curved map
        a = aff(deformed_model(D, D == 2 ? (6, 6) : (4, 4, 4); every=1000))              # one interior node moved: its cells only
        @test any(a) && !all(a)
    end
    # a parallelogram (sheared, not a rectangle) is affine
    model = CartesianDiscreteModel((0, 1, 0, 1), (3, 3); map=x -> VectorValue(x[1] + 0.3x[2], 0.8x[2]))
    @test all(HK.GeoTable(Triangulation(model), quad_points(Measure(Triangulation(model), 2)), quad_weights(Measure(Triangulation(model), 2))).cellaff)
    # disabled: per point everywhere
    HK.AFFINE_GEOMETRY[] = false
    try
        @test !any(HK.GeoTable(Triangulation(cartesian_model(2, (2, 2))), quad_points(Measure(Triangulation(cartesian_model(2, (2, 2))), 2)),
                               quad_weights(Measure(Triangulation(cartesian_model(2, (2, 2))), 2))).cellaff)
    finally
        HK.AFFINE_GEOMETRY[] = true
    end
end

"Residual, assembled Jacobian and matrix-free action against Gridap."
function check_affine(res, jac, U, V; nthreads=1)
    for jacobian in (:assembled, :matrixfree)
        op = KernelOperator(res, jac, U, V; nthreads, jacobian)
        @test isempty(op.fallback.rtrians) && isempty(op.fallback.jtrians)
        x = randx(U; scale=0.05)
        c = check(res, jac, U, V, x; op)
        @test c.res < TOL && c.jac < TOL
        if jacobian === :matrixfree
            gop = get_algebraic_operator(FEOperator(res, jac, U, V))
            mf = MatrixFreeJacobian(op); prepare!(mf, x)
            dx = randn(num_free_dofs(U))
            @test reldiff(mf * dx, Algebra.jacobian(gop, x) * dx) < 1e-11    # applied without assembling (sum factorisation): other round-off
        end
    end
end

@testset "equality with Gridap: $(name) ($(D)D, order $order)" for D in (2, 3), order in (1, 2),
        (name, mk) in (("Cartesian", (D, n) -> cartesian_model(D, n)), ("simplices", (D, n) -> cartesian_model(D, n; simplex=true)),
                       ("mapped n-cubes", (D, n) -> cartesian_model(D, n; mapped=true)), ("partly deformed", (D, n) -> deformed_model(D, D == 2 ? (5, 4) : (3, 3, 3); every=3)))
    n = D == 2 ? (4, 3) : (2, 2, 2)
    model = mk(D, n)
    Ω = Triangulation(model); dΩ = Measure(Ω, 2order)
    name == "partly deformed" && @test any(geo_cellaff(model)) && !all(geo_cellaff(model))
    # boundary integrals on non-affine hexahedra of a mapped Cartesian model differ from Gridap already in `main`
    # (Gridap's face maps there, a separate issue): skipped only for that model
    with_boundary = !(D == 3 && name == "mapped n-cubes")
    Γ = BoundaryTriangulation(model); dΓ = Measure(Γ, 2order)
    V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{D,Float64}, order), dirichlet_tags=[1])
    U = TrialFESpace(V, x -> zero(VectorValue{D,Float64}))
    t = VectorValue(ntuple(i -> 0.1i, D))
    bnd(v) = with_boundary ? ∫(v ⋅ t)dΓ : ∫(0.0 * (v ⋅ t))dΩ
    res(u, v) = ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))))dΩ - bnd(v)
    jac(u, du, v) = ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')))dΩ
    for nthreads in (1, 4)
        check_affine(res, jac, U, V; nthreads)
    end
    # the same as the per-point geometry (round-off)
    op = KernelOperator(res, jac, U, V)
    HK.AFFINE_GEOMETRY[] = false
    oq = try
        KernelOperator(res, jac, U, V)
    finally
        HK.AFFINE_GEOMETRY[] = true
    end
    @test !any(g -> any(g.geo.cellaff), geo_groups(oq))
    x = randx(U; scale=0.05)
    @test reldiff(Algebra.residual(op, x), Algebra.residual(oq, x)) < 1e-13
    @test reldiff(Algebra.jacobian(op, x), Algebra.jacobian(oq, x)) < 1e-13
end

@testset "Hessians (affine cells, Gridap's ∇∇ needs them) and skeletons (affine and deformed cells)" begin
    for (model, hess) in ((cartesian_model(2, (3, 3)), true), (cartesian_model(3, (2, 2, 2)), true), (deformed_model(2, (4, 4); every=2), false))
        Ω = Triangulation(model); dΩ = Measure(Ω, 4)
        Λ = SkeletonTriangulation(model); dΛ = Measure(Λ, 4)
        V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 2), dirichlet_tags="boundary"); U = TrialFESpace(V, 0.0)
        β = hess ? 1e-2 : 0.0
        H(u, v) = hess ? β * (∇∇(u) ⊙ ∇∇(v)) : 0.0 * (u * v)
        res(u, v) = ∫(H(u, v) + ∇(v) ⋅ ∇(u) - v)dΩ + ∫(jump(∇(u)) ⋅ jump(∇(v)))dΛ
        jac(u, du, v) = ∫(H(du, v) + ∇(v) ⋅ ∇(du))dΩ + ∫(jump(∇(du)) ⋅ jump(∇(v)))dΛ
        op = KernelOperator(res, jac, U, V)
        c = check(res, jac, U, V, randx(U); op)
        @test c.res < TOL && c.jac < TOL
    end
end

@testset "functionals and FE functions in the form on deformed cells" begin
    model = deformed_model(2, (4, 4); every=2)
    Ω = Triangulation(model); dΩ = Measure(Ω, 2)
    V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1)); U = TrialFESpace(V)
    wh = interpolate(x -> x[1]^2 + x[2], V)
    en(u) = ∫(0.5 * (∇(u) ⋅ ∇(u)) + (∇(wh) ⋅ ∇(u)))dΩ
    fun = KernelFunctional(en, U)
    x = randx(U; scale=0.3)
    @test fun(x) ≈ sum(en(FEFunction(U, x))) rtol = 1e-12
end

@testset "the kernels use the cell geometry: flagging deformed cells as affine changes the result" begin
    model = deformed_model(2, (4, 4); every=1)
    Ω = Triangulation(model); dΩ = Measure(Ω, 2)
    V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1)); U = TrialFESpace(V)
    res(u, v) = ∫(∇(v) ⋅ ∇(u) + u * v)dΩ
    op = KernelOperator(res, (u, du, v) -> res(du, v), U, V)
    g = op.rgroups[1]
    @test !all(g.geo.cellaff)
    x = randx(U); b0 = Algebra.residual(op, x)
    saved = copy(g.geo.cellaff)
    try
        g.geo.cellaff .= true                                          # wrong on purpose: Jt of the first point everywhere
        @test reldiff(Algebra.residual(op, x), b0) > 1e-6
    finally
        g.geo.cellaff .= saved
    end
    @test Algebra.residual(op, x) == b0
end

@testset "∇∇, skeletons (assembled, matrix-free) and functionals on a mixed mesh, against the per-point path" begin
    model = deformed_model(2, (4, 4); every=3)
    @test any(geo_cellaff(model)) && !all(geo_cellaff(model))
    Ω = Triangulation(model); dΩ = Measure(Ω, 4)
    Λ = SkeletonTriangulation(model); dΛ = Measure(Λ, 4)
    V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 2), dirichlet_tags="boundary"); U = TrialFESpace(V, 0.0)
    res(u, v) = ∫(1e-2 * (∇∇(u) ⊙ ∇∇(v)) + ∇(v) ⋅ ∇(u) - v)dΩ + ∫(jump(∇(u)) ⋅ jump(∇(v)))dΛ
    jac(u, du, v) = ∫(1e-2 * (∇∇(du) ⊙ ∇∇(v)) + ∇(v) ⋅ ∇(du))dΩ + ∫(jump(∇(du)) ⋅ jump(∇(v)))dΛ
    en(u) = ∫(0.5 * (∇(u) ⋅ ∇(u)))dΩ + ∫(0.5 * (jump(∇(u)) ⋅ jump(∇(u))))dΛ
    build() = (KernelOperator(res, jac, U, V), KernelOperator(res, jac, U, V; jacobian=:matrixfree), KernelFunctional(en, U))
    op, mfop, fun = build()
    HK.AFFINE_GEOMETRY[] = false
    oq, mfq, funq = try
        build()
    finally
        HK.AFFINE_GEOMETRY[] = true
    end
    x = randx(U); dx = randn(num_free_dofs(U))
    @test reldiff(Algebra.residual(op, x), Algebra.residual(oq, x)) < 1e-13
    @test reldiff(Algebra.jacobian(op, x), Algebra.jacobian(oq, x)) < 1e-13
    mf = MatrixFreeJacobian(mfop); prepare!(mf, x)
    mq = MatrixFreeJacobian(mfq); prepare!(mq, x)
    @test reldiff(mf * dx, mq * dx) < 1e-13
    @test fun(x) ≈ funq(x) rtol = 1e-13
    @test fun(x) ≈ sum(en(FEFunction(U, x))) rtol = 1e-12
end

@testset "HFK_AFFINE_GEOMETRY is read when the package is loaded" begin
    code = "using HyperFEMKernels; print(HyperFEMKernels.AFFINE_GEOMETRY[])"
    run_with(v) = read(addenv(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`, "HFK_AFFINE_GEOMETRY" => v), String)
    @test run_with("0") == "false"
    @test run_with("1") == "true"
end

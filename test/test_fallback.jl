# Hybrid fallback: contributions the engine cannot translate are assembled by Gridap and added
# to the kernel's vector/matrix. Every case is checked against pure Gridap (tolerance TOL).

@testset "boundary integrals (Neumann and Robin) go to Gridap, volume to the kernel" begin
    for order in (1, 2), simplex in (false, true)
        p = poisson_2d(; order, simplex)
        Γ = BoundaryTriangulation(p.model; tags=[6, 8]); dΓ = Measure(Γ, 2order)
        g(x) = x[1] - x[2]; α = 3.0
        res(u, v) = ∫(∇(v) ⋅ ∇(u) - 2.0 * v)p.dΩ + ∫(α * v * u - g * v)dΓ
        jac(u, du, v) = ∫(∇(v) ⋅ ∇(du))p.dΩ + ∫(α * v * du)dΓ
        op = @test_logs (:warn, r"hybrid fallback") (:warn, r"hybrid fallback") match_mode = :any KernelOperator(res, jac, p.U, p.V)
        @test length(op.rgroups) == 1 && length(op.jgroups) == 1
        @test length(op.fallback.rtrians) == 1 && length(op.fallback.jtrians) == 1
        c = check(res, jac, p.U, p.V, randx(p.U); op)
        @test c.res < TOL && c.jac < TOL
    end
end

@testset "fallback terms enlarge the sparsity pattern when needed" begin
    # a Robin term on a boundary of a mesh where the volume pattern already covers it: same nnz;
    # the Gridap pattern is a subset. Check the kernel entries are not lost by the union.
    p = poisson_2d(; n=(6, 5), order=2)
    Γ = BoundaryTriangulation(p.model); dΓ = Measure(Γ, 4)
    res(u, v) = ∫(∇(v) ⋅ ∇(u))p.dΩ + ∫(v * u)dΓ
    jac(u, du, v) = ∫(∇(v) ⋅ ∇(du))p.dΩ + ∫(v * du)dΓ
    op = KernelOperator(res, jac, p.U, p.V)
    x = randx(p.U)
    Ak = Algebra.jacobian(op, x)
    Ag = Algebra.jacobian(gridap_op((; res, jac, p.U, p.V)), x)
    @test nnz(Ak) >= nnz(Ag) && reldiff(Ak, Ag) < TOL
    # only the fallback part of the Jacobian in a matrix with its own pattern
    Afb = HK.fallback_pattern(op.fallback, p.U, p.V, jac)
    HK.fallback_jacobian!(Afb, op, x)
    AΓ = Algebra.jacobian(gridap_op((; res, jac=(u, du, v) -> ∫(v * du)dΓ, p.U, p.V)), x)
    @test reldiff(Afb, AΓ) < TOL
end

@testset "multi-field with a surface load on one field" begin
    e = elasticity_3d(; order=1)
    Γ = BoundaryTriangulation(e.model; tags=[26]); dΓ = Measure(Γ, 2)
    t = VectorValue(0.0, 0.0, -0.1)
    res(u, v) = e.res(u, v) - (∫(v ⋅ t)dΓ)
    jac(u, du, v) = e.jac(u, du, v)
    op = KernelOperator(res, jac, e.U, e.V)
    @test length(op.fallback.rtrians) == 1 && isempty(op.fallback.jtrians)
    c = check(res, jac, e.U, e.V, randx(e.U); op)
    @test c.res < TOL && c.jac < TOL
    # residual without fallback terms and with: the difference is the surface load
    x = randx(e.U)
    b1 = Algebra.residual(op, x); b0 = Algebra.residual(KernelOperator(e.res, e.jac, e.U, e.V), x)
    bΓ = Algebra.residual(gridap_op((; res=(u, v) -> ∫(-1.0 * (v ⋅ t))dΓ, jac=e.jac, e.U, e.V)), x)
    @test reldiff(b1 - b0, bΓ) < TOL
end

@testset "unsupported operation in a volume integral: whole triangulation to Gridap" begin
    p = poisson_2d()
    bad(u, v) = ∫(∇(v) ⋅ ∇(u * u))p.dΩ       # gradient of a product with the unknown
    op = @test_logs (:warn, r"whole residual is assembled by Gridap") KernelOperator(bad, p.jac, p.U, p.V)
    @test isempty(op.rgroups) && op.fallback.rtrians == [:all] && isempty(op.fallback.jtrians)
    c = check(bad, p.jac, p.U, p.V, randx(p.U); op)
    @test c.res < TOL && c.jac < TOL
end

@testset "matrix-free Jacobian includes the Gridap part" begin
    p = poisson_2d(; n=(6, 5), order=2)
    Γ = BoundaryTriangulation(p.model); dΓ = Measure(Γ, 4)
    res(u, v) = ∫(∇(v) ⋅ ∇(u))p.dΩ + ∫(5.0 * v * u)dΓ
    jac(u, du, v) = ∫(∇(v) ⋅ ∇(du))p.dΩ + ∫(5.0 * v * du)dΓ
    op = KernelOperator(res, jac, p.U, p.V)
    x = randx(p.U); dx = randn(num_free_dofs(p.U))
    A = Algebra.jacobian(op, x)
    mf = MatrixFreeJacobian(op); prepare!(mf, x)
    @test reldiff(mf * dx, A * dx) < 1e-11
end

@testset "update keeps the pattern when the fallback triangulations do not change" begin
    p = poisson_2d(; order=2)
    Γ = BoundaryTriangulation(p.model); dΓ = Measure(Γ, 4)
    mk(α) = ((u, v) -> ∫(∇(v) ⋅ ∇(u))p.dΩ + ∫(α * v * u)dΓ, (u, du, v) -> ∫(∇(v) ⋅ ∇(du))p.dΩ + ∫(α * v * du)dΓ)
    r1, j1 = mk(1.0); r2, j2 = mk(2.0)
    op1 = KernelOperator(r1, j1, p.U, p.V)
    op2 = update(op1, r2, j2)
    @test op2.A0 === op1.A0 && op2.nzpos === op1.nzpos
    c = check(r2, j2, p.U, p.V, randx(p.U); op=op2)
    @test c.res < TOL && c.jac < TOL
    # a fallback appearing in update: pattern recomputed
    op3 = update(KernelOperator(p.res, p.jac, p.U, p.V), r2, j2)
    @test op3.A0 !== op1.A0
    c3 = check(r2, j2, p.U, p.V, randx(p.U); op=op3)
    @test c3.res < TOL && c3.jac < TOL
end

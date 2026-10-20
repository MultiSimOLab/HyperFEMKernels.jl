# Hybrid fallback: contributions the engine cannot translate are assembled by Gridap and added
# to the kernel's vector/matrix. Every case is checked against pure Gridap (tolerance TOL).

# Interior-penalty terms on the skeleton (jump of the unknown across faces) are not translated.
# As plus/minus need the cell-wise data of the probes, the whole weak form (not only the skeleton
# contribution) goes to Gridap: correct, with no speed-up. Finer granularity would need mock cell
# data for the probes (rejected: Gridap then builds trees the reader misparses).
function skeleton_problem(; order=1, simplex=false, n=(5, 4))
    p = poisson_2d(; n, order, simplex)
    Λ = SkeletonTriangulation(p.model); dΛ = Measure(Λ, 2order)
    γ = 10.0
    res(u, v) = ∫(∇(v) ⋅ ∇(u) - 2.0 * v)p.dΩ + ∫(γ * jump(v) * jump(u))dΛ
    jac(u, du, v) = ∫(∇(v) ⋅ ∇(du))p.dΩ + ∫(γ * jump(v) * jump(du))dΛ
    (; p, Λ, dΛ, res, jac, jacΛ=(u, du, v) -> ∫(γ * jump(v) * jump(du))dΛ)
end

@testset "skeleton (jump) terms: whole form to Gridap" begin
    for order in (1, 2), simplex in (false, true)
        s = skeleton_problem(; order, simplex)
        op = @test_logs (:warn, r"whole residual is assembled by Gridap") (:warn, r"whole Jacobian is assembled by Gridap") match_mode = :any KernelOperator(s.res, s.jac, s.p.U, s.p.V)
        @test isempty(op.rgroups) && isempty(op.jgroups)
        @test op.fallback.rtrians == [:all] && op.fallback.jtrians == [:all]
        c = check(s.res, s.jac, s.p.U, s.p.V, randx(s.p.U); op)
        @test c.res < TOL && c.jac < TOL
    end
end

@testset "fallback terms enlarge the sparsity pattern (skeleton couples neighbouring cells)" begin
    s = skeleton_problem(; n=(6, 5), order=2)
    p = s.p
    op = KernelOperator(s.res, s.jac, p.U, p.V)
    op0 = KernelOperator(p.res, p.jac, p.U, p.V)
    x = randx(p.U)
    Ak = Algebra.jacobian(op, x)
    Ag = Algebra.jacobian(gridap_op((; s.res, s.jac, p.U, p.V)), x)
    @test nnz(Ak) > nnz(op0.A0) && reldiff(Ak, Ag) < TOL
    # the fallback part of the Jacobian (here the whole Jacobian) in a matrix with its own pattern
    Afb = HK.fallback_pattern(op.fallback, p.U, p.V, s.jac)
    HK.fallback_jacobian!(Afb, op, x)
    @test reldiff(Afb, Ag) < TOL
end

@testset "residual-only fallback (skeleton term in the residual, none in the Jacobian)" begin
    e = elasticity_3d(; order=1)
    Λ = SkeletonTriangulation(e.model); dΛ = Measure(Λ, 2)
    res(u, v) = e.res(u, v) + ∫(mean(v) ⋅ mean(u))dΛ      # (jump(u) vanishes for conforming u)
    jac(u, du, v) = e.jac(u, du, v)
    op = KernelOperator(res, jac, e.U, e.V)
    @test length(op.fallback.rtrians) == 1 && isempty(op.fallback.jtrians)
    c = check(res, jac, e.U, e.V, randx(e.U); op)
    @test c.res < TOL && c.jac < TOL
    # residual without fallback terms and with: the difference is the skeleton term
    x = randx(e.U)
    b1 = Algebra.residual(op, x); b0 = Algebra.residual(KernelOperator(e.res, e.jac, e.U, e.V), x)
    bΛ = Algebra.residual(gridap_op((; res=(u, v) -> ∫(mean(v) ⋅ mean(u))dΛ, jac=e.jac, e.U, e.V)), x)
    @test reldiff(b1 - b0, bΛ) < TOL
end

@testset "unsupported operation in a volume integral: whole residual to Gridap" begin
    p = poisson_2d()
    bad(u, v) = ∫(∇(v) ⋅ ∇(u * u))p.dΩ       # gradient of a product with the unknown
    op = @test_logs (:warn, r"whole residual is assembled by Gridap") KernelOperator(bad, p.jac, p.U, p.V)
    @test isempty(op.rgroups) && op.fallback.rtrians == [:all] && isempty(op.fallback.jtrians)
    c = check(bad, p.jac, p.U, p.V, randx(p.U); op)
    @test c.res < TOL && c.jac < TOL
end

@testset "matrix-free Jacobian includes the Gridap part" begin
    s = skeleton_problem(; n=(6, 5), order=2)
    p = s.p; res = s.res; jac = s.jac
    op = KernelOperator(res, jac, p.U, p.V)
    x = randx(p.U); dx = randn(num_free_dofs(p.U))
    A = Algebra.jacobian(op, x)
    mf = MatrixFreeJacobian(op); prepare!(mf, x)
    @test reldiff(mf * dx, A * dx) < 1e-11
end

@testset "update keeps the pattern when the fallback triangulations do not change" begin
    p = poisson_2d(; order=2)
    Λ = SkeletonTriangulation(p.model); dΛ = Measure(Λ, 4)
    mk(α) = ((u, v) -> ∫(∇(v) ⋅ ∇(u))p.dΩ + ∫(α * jump(v) * jump(u))dΛ, (u, du, v) -> ∫(∇(v) ⋅ ∇(du))p.dΩ + ∫(α * jump(v) * jump(du))dΛ)
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

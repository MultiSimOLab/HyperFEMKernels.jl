# Hybrid fallback: contributions the engine cannot translate are assembled by Gridap and added
# to the kernel's vector/matrix. Every case is checked against pure Gridap (tolerance TOL).

# A differential operator applied to a product with the unknown (∇(u*u)) makes Gridap ask for the
# cell-wise data of the probes. There is none: the whole weak form (not only that integral) goes to
# Gridap: correct, with no speed-up. Returning mock cell data for the probes was tried and rejected
# (Gridap then builds trees the reader misparses). Skeleton integrals (jump/mean) are no longer a
# fallback case: see test_skeleton.jl.
"Builds an operator whose form is deliberately sent to Gridap, capturing the fallback warnings (they are expected here, not noise for the test log)."
fallback_op(f, args...) = @test_logs (:warn, r"assembled by Gridap") match_mode = :any f(args...)

function product_gradient_problem(; order=1, simplex=false, n=(5, 4))
    p = poisson_2d(; n, order, simplex)
    Λ = SkeletonTriangulation(p.model); dΛ = Measure(Λ, 2order)
    γ = 10.0
    # the skeleton term makes the Gridap-assembled pattern larger than the kernel's cell pattern
    res(u, v) = ∫(∇(v) ⋅ ∇(u) - 2.0 * v + ∇(v) ⋅ ∇(u * u))p.dΩ + ∫(γ * jump(v) * jump(u))dΛ
    jac(u, du, v) = ∫(∇(v) ⋅ ∇(du) + 2.0 * (∇(v) ⋅ ∇(du * u)))p.dΩ + ∫(γ * jump(v) * jump(du))dΛ
    (; p, Λ, dΛ, res, jac)
end

@testset "differential operator on a product with the unknown: whole form to Gridap" begin
    for order in (1, 2), simplex in (false, true)
        s = product_gradient_problem(; order, simplex)
        op = @test_logs (:warn, r"whole residual is assembled by Gridap") (:warn, r"whole Jacobian is assembled by Gridap") match_mode = :any KernelOperator(s.res, s.jac, s.p.U, s.p.V)
        @test isempty(op.rgroups) && isempty(op.jgroups)
        @test op.fallback.rtrians == [:all] && op.fallback.jtrians == [:all]
        c = check(s.res, s.jac, s.p.U, s.p.V, randx(s.p.U); op)
        @test c.res < TOL && c.jac < TOL
    end
end

@testset "fallback terms enlarge the sparsity pattern (the Gridap part couples neighbouring cells)" begin
    s = product_gradient_problem(; n=(6, 5), order=2)
    p = s.p
    op = fallback_op(KernelOperator, s.res, s.jac, p.U, p.V)
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

@testset "residual-only fallback (unsupported term in the residual, none in the Jacobian)" begin
    p = poisson_2d(; order=2)
    extra(u, v) = ∫(∇(v) ⋅ ∇(u * u))p.dΩ
    res(u, v) = p.res(u, v) + extra(u, v)
    jac = p.jac
    op = @test_logs (:warn, r"whole residual is assembled by Gridap") KernelOperator(res, jac, p.U, p.V)
    @test op.fallback.rtrians == [:all] && isempty(op.fallback.jtrians)
    @test isempty(op.rgroups) && !isempty(op.jgroups)
    c = check(res, jac, p.U, p.V, randx(p.U); op)
    @test c.res < TOL && c.jac < TOL
    # residual without the fallback term and with: the difference is that term
    x = randx(p.U)
    b1 = Algebra.residual(op, x); b0 = Algebra.residual(KernelOperator(p.res, p.jac, p.U, p.V), x)
    bx = Algebra.residual(gridap_op((; res=extra, jac=p.jac, p.U, p.V)), x)
    @test reldiff(b1 - b0, bx) < TOL
end

@testset "matrix-free Jacobian includes the Gridap part" begin
    s = product_gradient_problem(; n=(6, 5), order=2)
    p = s.p; res = s.res; jac = s.jac
    op = fallback_op(KernelOperator, res, jac, p.U, p.V)
    x = randx(p.U); dx = randn(num_free_dofs(p.U))
    A = Algebra.jacobian(op, x)
    mf = MatrixFreeJacobian(op); prepare!(mf, x)
    @test reldiff(mf * dx, A * dx) < 1e-11
end

@testset "update keeps the pattern when the fallback triangulations do not change" begin
    p = poisson_2d(; order=2)
    mk(α) = ((u, v) -> ∫(∇(v) ⋅ ∇(u) + α * (∇(v) ⋅ ∇(u * u)))p.dΩ, (u, du, v) -> ∫(∇(v) ⋅ ∇(du) + 2α * (∇(v) ⋅ ∇(du * u)))p.dΩ)
    r1, j1 = mk(1.0); r2, j2 = mk(2.0)
    op1 = fallback_op(KernelOperator, r1, j1, p.U, p.V)
    op2 = fallback_op(update, op1, r2, j2)
    @test op2.A0 === op1.A0 && op2.nzpos === op1.nzpos
    c = check(r2, j2, p.U, p.V, randx(p.U); op=op2)
    @test c.res < TOL && c.jac < TOL
    # a fallback appearing in update: pattern recomputed
    op3 = fallback_op(update, KernelOperator(p.res, p.jac, p.U, p.V), r2, j2)
    @test op3.A0 !== op1.A0
    c3 = check(r2, j2, p.U, p.V, randx(p.U); op=op3)
    @test c3.res < TOL && c3.jac < TOL
end

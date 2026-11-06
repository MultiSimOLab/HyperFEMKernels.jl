# S3 — operator mode: `KernelOperator(…; jacobian = :matrixfree)` inside a Newton loop.
Random.seed!(3)
const TOL_OP = 1e-11   # summation order of the matrix-free product differs from the assembled one

@testset "jacobian = :matrixfree: allocate_jacobian/jacobian! give the prepared operator" begin
    for (name, prob) in (("Poisson Q1", poisson_2d()), ("Poisson Q2", poisson_2d(; order=2)),
                         ("elasticity Q2", elasticity_3d(; order=2)), ("coupled u–φ Q1", coupled_3d()),
                         ("Poisson P1 (dense weights)", poisson_2d(; simplex=true)))
        op = KernelOperator(prob.res, prob.jac, prob.U, prob.V; jacobian=:matrixfree)
        ref = KernelOperator(prob.res, prob.jac, prob.U, prob.V)
        @test op.jacobian === :matrixfree && op.pattern.A0 === nothing     # no sparsity pattern yet
        x = randx(prob.U)
        A = allocate_jacobian(op, x)
        @test A isa MatrixFreeJacobian && size(A) == size(HK.sparsity_pattern(ref))
        @test length(allocate_in_domain(A)) == size(A, 2) && length(allocate_in_range(A)) == size(A, 1)
        jacobian!(A, op, x)
        dx = randn(num_free_dofs(prob.U))
        Aref = jacobian(ref, x)
        @test reldiff(A * dx, Aref * dx) < TOL_OP
        @test op.pattern.A0 === nothing                                     # still none after jacobian!
        @test A.x == x && A.x !== x                                          # the state is kept as a copy
        # new state: jacobian! re-prepares, no stale tangents
        x2 = randx(prob.U)
        jacobian!(A, op, x2)
        @test reldiff(A * dx, jacobian(ref, x2) * dx) < TOL_OP
        # residual_and_jacobian! with the operator; residual equal to the assembled operator's
        b = allocate_residual(op, x2)
        residual_and_jacobian!(b, A, op, x2)
        @test b == residual(ref, x2)
        @test reldiff(A * dx, jacobian(ref, x2) * dx) < TOL_OP
        # Gridap's non-mutating jacobian gives the operator too
        @test jacobian(op, x2) isa MatrixFreeJacobian
        # the pattern is built on demand and equals the assembled operator's
        Ap = HK.sparsity_pattern(op)
        @test op.pattern.A0 === Ap && Ap.colptr == HK.sparsity_pattern(ref).colptr && Ap.rowval == HK.sparsity_pattern(ref).rowval
        Ak = allocate_jacobian(ref, x2); jacobian!(Ak, op, x2)                # assembled with the matrix-free operator
        @test reldiff(Ak, jacobian(ref, x2)) < TOL
    end
end

@testset "jacobian! rebinds the operator after update (same structure) and reallocates otherwise" begin
    e = elasticity_3d(; order=2)
    op = KernelOperator(e.res, e.jac, e.U, e.V; jacobian=:matrixfree)
    x = randx(e.U); dx = randn(num_free_dofs(e.U))
    A = allocate_jacobian(op, x)
    jacobian!(A, op, x)
    # same structure, other coefficient (a runtime parameter): buffers reused
    dΩe = e.dΩ
    res2(u, v) = ∫(2.5 * (∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)')))))dΩe
    jac2(u, du, v) = ∫(2.5 * (∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)'))))dΩe
    op2 = update(op, res2, jac2)
    data = A.data
    jacobian!(A, op2, x)
    @test A.op === op2 && A.data === data && op2.jacobian === :matrixfree && op2.pattern.A0 === nothing
    ref2 = KernelOperator(res2, jac2, e.U, e.V)
    @test reldiff(A * dx, jacobian(ref2, x) * dx) < TOL_OP
    # a new term (more slots): buffers reallocated, product still right
    res3(u, v) = ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))) + 3.0 * (v ⋅ u))dΩe
    jac3(u, du, v) = ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')) + 3.0 * (v ⋅ du))dΩe
    op3 = update(op2, res3, jac3)
    jacobian!(A, op3, x)
    @test A.op === op3 && A.data !== data
    @test reldiff(A * dx, jacobian(KernelOperator(res3, jac3, e.U, e.V), x) * dx) < TOL_OP
    # an operator on a space with the same free dofs (the increment space of a Dirichlet projection) is accepted
    ΔU = TrialFESpace(e.V, x -> VectorValue(0.0, 0.0, 0.0))
    opΔ = KernelOperator(e.res, e.jac, ΔU, e.V; jacobian=:matrixfree)
    jacobian!(A, opΔ, x)
    @test A.op === opΔ && reldiff(A * dx, jacobian(KernelOperator(e.res, e.jac, ΔU, e.V), x) * dx) < TOL_OP
    jacobian!(A, op, x)
    @test A.op === op && reldiff(A * dx, jacobian(KernelOperator(e.res, e.jac, e.U, e.V), x) * dx) < TOL_OP
    # an operator on other spaces is refused
    e4 = elasticity_3d(; order=1)
    op4 = KernelOperator(e4.res, e4.jac, e4.U, e4.V; jacobian=:matrixfree)
    @test_throws ErrorException jacobian!(A, op4, randx(e4.U))
end

@testset "the apply does not allocate (buffers per task kept in the operator)" begin
    for prob in (elasticity_3d(; order=2), poisson_2d(; simplex=true))
        op = KernelOperator(prob.res, prob.jac, prob.U, prob.V; jacobian=:matrixfree)
        x = randx(prob.U); mf = jacobian(op, x)
        dx = randn(num_free_dofs(prob.U)); y = zeros(length(dx))
        g = op.jgroups[1]; cells = g.colors[1]
        HK.apply_cells!(y, mf.data[1], mf.tps[1], g, op, dx, cells, mf.bufs[1][1])
        @test (@allocated HK.apply_cells!(y, mf.data[1], mf.tps[1], g, op, dx, cells, mf.bufs[1][1])) == 0
        @test length(mf.bufs[1]) == op.nthreads
    end
end

@testset "errors: invalid option, direct solver on the operator" begin
    p = poisson_2d()
    @test_throws ArgumentError KernelOperator(p.res, p.jac, p.U, p.V; jacobian=:dense)
    op = KernelOperator(p.res, p.jac, p.U, p.V; jacobian=:matrixfree)
    x = randx(p.U)
    A = allocate_jacobian(op, x)
    @test_throws ErrorException symbolic_setup(LUSolver(), A)
end

# S3 — `MatrixFreeSolver` (Krylov.jl on the matrix-free Jacobian) and the `LaggedAssembled` slot.
Random.seed!(4)
const TOL_KRY = 1e-7    # solutions to rtol = 1e-10 compared with LU (error ≤ cond · residual; cond ~1e4 on the Q2 elasticity cube)

"Matrix-free operator and assembled reference at a random state; `b` the residual there."
function mf_setup(prob)
    op = KernelOperator(prob.res, prob.jac, prob.U, prob.V; jacobian=:matrixfree)
    ref = KernelOperator(prob.res, prob.jac, prob.U, prob.V)
    x = randx(prob.U)
    mf = allocate_jacobian(op, x)
    jacobian!(mf, op, x)
    A = jacobian(ref, x)
    b = residual(ref, x)
    (; op, ref, x, mf, A, b, n=length(b))
end

function mf_solve(s::MatrixFreeSolver, mf, b)
    ns = numerical_setup(symbolic_setup(s, mf), mf)
    dx = zeros(length(b))
    solve!(dx, ns, b)
    dx, ns
end

@testset "MatrixFreeSolver reproduces LU (GMRES, FGMRES, CG; without and with the slots)" begin
    for (name, prob, spd) in (("Poisson Q2", poisson_2d(; n=(8, 6), order=2), true),
                              ("elasticity Q2 3D", elasticity_3d(; order=2), false),
                              ("coupled u–φ Q1", coupled_3d(), false))
        t = mf_setup(prob)
        dx_ref = t.A \ t.b
        configs = Any[(:gmres, nothing), (:fgmres, nothing), (:gmres, LaggedAssembled(LUSolver())),
                      (:gmres, LaggedAssembled(LUSolver(); every=2))]
        spd && push!(configs, (:cg, nothing))
        for (kry, prec) in configs
            s = MatrixFreeSolver(; krylov=kry, rtol=1e-10, maxiter=4000, restart=t.n, preconditioner=prec)
            dx, ns = mf_solve(s, t.mf, t.b)
            @test HK.linear_converged(ns)
            @test reldiff(dx, dx_ref) < TOL_KRY
            its = HK.linear_iterations(ns)
            @test 0 < its ≤ t.n + 1
            if prec isa LaggedAssembled
                @test its ≤ 2                              # exact (right) preconditioner at the setup state
                @test HK.preconditioner_builds(ns) == 1
            else
                @test HK.preconditioner_builds(ns) == 0
            end
        end
    end
end

@testset "errors and reporting: wrong operator, options, non-convergence" begin
    p = poisson_2d(; order=2)
    t = mf_setup(p)
    @test_throws ArgumentError MatrixFreeSolver(; krylov=:bicgstab)
    @test_throws ArgumentError MatrixFreeSolver(; rtol=1.0)
    @test_throws ArgumentError MatrixFreeSolver(; maxiter=0)
    @test_throws ArgumentError LaggedAssembled(LUSolver(); every=0)
    @test_throws ErrorException symbolic_setup(MatrixFreeSolver(), t.A)
    # too few iterations: reported, not thrown
    s = MatrixFreeSolver(; rtol=1e-10, maxiter=3, restart=3)
    dx, ns = mf_solve(s, t.mf, t.b)
    @test !HK.linear_converged(ns) && HK.linear_iterations(ns) == 3
end

@testset "set_tolerance!/tolerance: the forcing-term hook" begin
    t = mf_setup(poisson_2d(; n=(8, 6), order=2))
    s = MatrixFreeSolver(; rtol=1e-10, maxiter=4000, restart=t.n)
    ns = numerical_setup(symbolic_setup(s, t.mf), t.mf)
    dx = zeros(t.n)
    @test HK.tolerance(ns) == 1e-10
    solve!(dx, ns, t.b); tight = HK.linear_iterations(ns)
    HK.set_tolerance!(ns, 1e-2)
    @test HK.tolerance(ns) == 1e-2
    solve!(dx, ns, t.b); loose = HK.linear_iterations(ns)
    @test loose < tight && HK.linear_converged(ns)
    @test norm(t.b - t.A * dx) ≤ 1e-2 * norm(t.b)          # true (unpreconditioned) residual
    HK.set_tolerance!(ns, nothing)
    @test HK.tolerance(ns) == 1e-10
    solve!(dx, ns, t.b)
    @test HK.linear_iterations(ns) == tight
    @test_throws ArgumentError HK.set_tolerance!(ns, 1.5)
end

@testset "LaggedAssembled: assembled every k setups, stale preconditioner still converges to the new solution" begin
    e = elasticity_3d(; order=2)
    op = KernelOperator(e.res, e.jac, e.U, e.V; jacobian=:matrixfree)
    ref = KernelOperator(e.res, e.jac, e.U, e.V)
    n = num_free_dofs(e.U)
    states = [randx(e.U; scale=0.05) for _ in 1:5]
    mf = allocate_jacobian(op, states[1]); jacobian!(mf, op, states[1])
    s = MatrixFreeSolver(; rtol=1e-10, maxiter=4000, restart=n, preconditioner=LaggedAssembled(LUSolver(); every=3))
    ns = numerical_setup(symbolic_setup(s, mf), mf)
    its = Int[]
    for (k, x) in enumerate(states)
        k > 1 && (jacobian!(mf, op, x); numerical_setup!(ns, mf))
        @test HK.preconditioner_builds(ns) == 1 + (k - 1) ÷ 3            # builds at setups 1 and 4
        b = residual(ref, x)
        dx = zeros(n); solve!(dx, ns, b)
        @test HK.linear_converged(ns) && reldiff(dx, jacobian(ref, x) \ b) < TOL_KRY
        push!(its, HK.linear_iterations(ns))
    end
    @test its[1] ≤ 2 && its[4] ≤ 2 && all(>(0), its)                     # exact where rebuilt
end

@testset "a Newton loop with the matrix-free solver follows the LU trajectory (nonlinear elasticity)" begin
    e = elasticity_3d(; order=2)
    bodyf = VectorValue(0.0, 0.0, -0.4); dΩe = e.dΩ
    res(u, v) = ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))) - v ⋅ bodyf)dΩe
    jac(u, du, v) = ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')))dΩe
    function newton(op, ls; maxiter=20, tol=1e-10)
        x = zero_initial_guess(op)
        b = allocate_residual(op, x); A = allocate_jacobian(op, x)
        residual!(b, op, x); r0 = norm(b)
        ss = symbolic_setup(ls, A); ns = nothing
        hist = Float64[]
        for k in 1:maxiter
            jacobian!(A, op, x)
            ns = ns === nothing ? numerical_setup(ss, A) : numerical_setup!(ns, A)
            dx = similar(x); solve!(dx, ns, b)
            x .-= dx
            residual!(b, op, x)
            push!(hist, norm(b) / r0)
            hist[end] < tol && break
        end
        x, hist
    end
    x_lu, h_lu = newton(KernelOperator(res, jac, e.U, e.V), LUSolver())
    n = num_free_dofs(e.U)
    for prec in (LaggedAssembled(LUSolver(); every=2),)
        s = MatrixFreeSolver(; rtol=1e-10, maxiter=4000, restart=n, preconditioner=prec)
        x_mf, h_mf = newton(KernelOperator(res, jac, e.U, e.V; jacobian=:matrixfree), s)
        @test length(h_mf) == length(h_lu)
        @test all(abs.(log10.(h_mf) .- log10.(h_lu)) .< 0.5)              # same Newton trajectory
        @test reldiff(x_mf, x_lu) < 1e-8
    end
end

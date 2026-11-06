# S3 — `MatrixFreeSolver` (Krylov.jl on the matrix-free Jacobian) and the preconditioner slots.
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
                      (:gmres, LaggedAssembled(LUSolver(); every=2)), (:gmres, BlockJacobi()), (:fgmres, BlockJacobi())]
        spd && push!(configs, (:cg, nothing), (:cg, BlockJacobi()))
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
                # Krylov.jl may stop on the Arnoldi breakdown of an exact preconditioner without its residual test:
                # the decision is then taken on the true residual (rnorm computed exactly in that case)
                @test isfinite(ns.rnorm) == !ns.workspace.stats.solved
            elseif prec isa BlockJacobi
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
    # the operator must be prepared before the setup (the slots assemble/gather at the state)
    op0 = KernelOperator(p.res, p.jac, p.U, p.V; jacobian=:matrixfree)
    mf0 = allocate_jacobian(op0, randx(p.U))
    @test_throws ErrorException numerical_setup(symbolic_setup(MatrixFreeSolver(), mf0), mf0)
    @test_throws ErrorException HK.preconditioner_setup(LaggedAssembled(LUSolver()), mf0)
    @test_throws ErrorException HK.preconditioner_setup(BlockJacobi(), mf0)
    # too few iterations: reported, not thrown
    s = MatrixFreeSolver(; rtol=1e-10, maxiter=3, restart=3)
    dx, ns = mf_solve(s, t.mf, t.b)
    @test !HK.linear_converged(ns) && HK.linear_iterations(ns) == 3
end

@testset "set_tolerance!/tolerance: the forcing-term hook" begin
    t = mf_setup(poisson_2d(; n=(8, 6), order=2))
    s = MatrixFreeSolver(; rtol=1e-10, maxiter=4000, restart=t.n, preconditioner=BlockJacobi())
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

"Elasticity with one component fixed on some faces (partially constrained nodes)."
function elasticity_masked(; n=(3, 2, 2))
    model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), n)
    Ω = Triangulation(model); dΩ = Measure(Ω, 4)
    V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 2), dirichlet_tags=[21, 22, 25],
                    dirichlet_masks=[(true, true, true), (true, false, false), (false, false, true)])
    U = TrialFESpace(V, [x -> VectorValue(0.0, 0.0, 0.0), x -> VectorValue(0.02, 0.0, 0.0), x -> VectorValue(0.0, 0.0, 0.01)])
    res(u, v) = ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))))dΩ
    jac(u, du, v) = ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')))dΩ
    (; model, Ω, dΩ, U, V, res, jac)
end

@testset "BlockJacobi: blocks equal the inverse of the assembled nodal blocks (free components)" begin
    for prob in (elasticity_3d(; order=2), coupled_3d(), poisson_2d(; order=2), elasticity_masked())
        t = mf_setup(prob)
        st = HK.preconditioner_setup(BlockJacobi(), t.mf)
        ncheck = 0
        for fb in st.fields
            nc, nb = size(fb.dofs)
            for blk in 1:nb
                free = [c for c in 1:nc if fb.dofs[c, blk] != 0]
                isempty(free) && continue
                ids = fb.dofs[free, blk]
                D = Matrix(t.A[ids, ids])
                @test fb.Binv[free, free, blk] ≈ inv(D) atol = 1e-10 * max(1.0, norm(inv(D)))
                ncheck += 1
            end
        end
        @test ncheck > 0
        # nodes with every component fixed have no block; partially fixed ones keep their free components
        for fb in st.fields
            @test all(blk -> any(!=(0), fb.dofs[:, blk]), 1:size(fb.dofs, 2))
        end
        # the extraction is deterministic: one thread or many give the same blocks (no shared nodes within a colour)
        op1 = KernelOperator(prob.res, prob.jac, prob.U, prob.V; jacobian=:matrixfree, nthreads=1)
        mf1 = jacobian(op1, t.x)
        st1 = HK.preconditioner_setup(BlockJacobi(), mf1)
        @test all(((a, b),) -> a.B == b.B && a.dofs == b.dofs, zip(st1.fields, st.fields))
        # ldiv! applies the inverse blocks: on a block-diagonal rhs it is exact
        r = randn(t.n); y = zeros(t.n)
        ldiv!(y, st, r)
        Dfull = zeros(t.n, t.n)
        for fb in st.fields, blk in 1:size(fb.dofs, 2)
            ids = [fb.dofs[c, blk] for c in 1:size(fb.dofs, 1) if fb.dofs[c, blk] != 0]
            Dfull[ids, ids] = t.A[ids, ids]
        end
        @test reldiff(Dfull * y, r) < 1e-10
        # the update at a new state follows the tangent there
        x2 = randx(prob.U); jacobian!(t.mf, t.op, x2); HK.preconditioner_update!(st, t.mf)
        @test HK.preconditioner_builds(st) == 2
        A2 = jacobian(t.ref, x2)
        fb = st.fields[1]; blk = findfirst(b -> all(!=(0), fb.dofs[:, b]), 1:size(fb.dofs, 2))
        ids = fb.dofs[:, blk]
        @test fb.Binv[:, :, blk] ≈ inv(Matrix(A2[ids, ids])) atol = 1e-10 * norm(inv(Matrix(A2[ids, ids])))
    end
end

@testset "eigenvalue_bounds: Arnoldi estimate of the preconditioned spectrum vs the dense one" begin
    t = mf_setup(poisson_2d(; n=(8, 6), order=2))
    st = HK.preconditioner_setup(BlockJacobi(), t.mf)
    Dinv = zeros(t.n, t.n)
    fb = st.fields[1]
    for blk in 1:size(fb.dofs, 2)
        id = fb.dofs[1, blk]; id == 0 && continue
        Dinv[id, id] = fb.Binv[1, 1, blk]
    end
    λ = eigvals(Matrix(t.A) * Dinv)
    λmin, λmax = eigenvalue_bounds(t.mf, st; k=min(30, t.n))
    @test 0.9 * maximum(real, λ) ≤ λmax ≤ 1.1 * maximum(real, λ)              # within 10 %
    @test λmin > 0
    # without preconditioner: the operator's own spectrum
    λmin0, λmax0 = eigenvalue_bounds(t.mf, nothing; k=min(30, t.n))
    λA = maximum(real, eigvals(Matrix(t.A)))
    @test 0.9λA ≤ λmax0 ≤ 1.1λA
    # deterministic with a given start vector; an exact preconditioner (J P⁻¹ = I) breaks down → (1, 1)
    b0 = randn(t.n)
    @test eigenvalue_bounds(t.mf, st; k=10, b=b0) == eigenvalue_bounds(t.mf, st; k=10, b=b0)
    stlu = HK.preconditioner_setup(LaggedAssembled(LUSolver()), t.mf)
    λe = eigenvalue_bounds(t.mf, stlu; k=10)
    @test all(isfinite, λe)
    @test λe[1] ≈ 1.0 atol = 1e-6
    @test λe[2] ≈ 1.0 atol = 1e-6
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
    for prec in (LaggedAssembled(LUSolver(); every=2), BlockJacobi())
        s = MatrixFreeSolver(; rtol=1e-10, maxiter=4000, restart=n, preconditioner=prec)
        x_mf, h_mf = newton(KernelOperator(res, jac, e.U, e.V; jacobian=:matrixfree), s)
        @test length(h_mf) == length(h_lu)
        @test all(abs.(log10.(h_mf) .- log10.(h_lu)) .< 0.5)              # same Newton trajectory
        @test reldiff(x_mf, x_lu) < 1e-8
    end
end

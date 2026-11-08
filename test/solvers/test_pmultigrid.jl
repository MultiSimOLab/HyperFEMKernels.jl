# S4-A step 4: the p-multigrid V-cycle as a preconditioner slot of MatrixFreeSolver. Same solution as LU, Krylov
# iterations nearly independent of the mesh on linear elasticity (Q2 → Q1, Q3 → Q1, Q3 → Q2 → Q1), symmetric cycle
# (CG), the Newton loop following LU, lifecycle through numerical_setup!, NaN guard, allocations of the cycle.
Random.seed!(7)

const λ_pmg, μ_pmg = 2.0, 1.0
pmg_stress(G) = λ_pmg * tr(G) * one(G) + μ_pmg * (G + G')

"Cube of n³ hexahedra, vector Lagrangian spaces of `orders` (finest first), Dirichlet on the whole boundary."
function pmg_cube(n, orders; g=x -> VectorValue(0.01x[1], -0.02x[2] * x[3], 0.005), body=VectorValue(0.3, -0.2, 0.5))
    model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (n, n, n))
    T = VectorValue{3,Float64}
    spaces = map(enumerate(orders)) do (l, p)
        V = TestFESpace(model, ReferenceFE(lagrangian, T, p), dirichlet_tags="boundary")
        (l == 1 ? TrialFESpace(V, g) : TrialFESpace(V, x -> zero(T)), V)
    end
    dΩ = Measure(Triangulation(model), 2first(orders))
    lres(u, v) = ∫(∇(v)' ⊙ (pmg_stress ∘ (∇(u)')) - v ⋅ body)dΩ
    ljac(u, du, v) = ∫(∇(v)' ⊙ (pmg_stress ∘ (∇(du)')))dΩ
    nres(u, v) = ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))) - v ⋅ body)dΩ
    njac(u, du, v) = ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')))dΩ
    (; model, spaces=collect(Any, spaces), dΩ, lres, ljac, nres, njac)
end

"Solve the linear tangent system at state `x` with `ls`; returns the solution, the setup and the LU reference."
function pmg_solve(c, ls, x; res=c.lres, jac=c.ljac)
    U, V = c.spaces[1]
    op = KernelOperator(res, jac, U, V; jacobian=:matrixfree)
    mf = allocate_jacobian(op, x); jacobian!(mf, op, x)
    ref = KernelOperator(res, jac, U, V)
    A = jacobian(ref, x); b = residual(ref, x)
    ns = numerical_setup(symbolic_setup(ls, mf), mf)
    dx = zeros(length(b)); solve!(dx, ns, b)
    dx, ns, A \ b
end

@testset "PMultigrid: same solution as LU, iterations nearly mesh-independent (Q2 → Q1)" begin
    its = Int[]
    for n in (2, 4, 8)
        c = pmg_cube(n, (2, 1))
        ls = MatrixFreeSolver(; krylov=:gmres, rtol=1e-10, maxiter=200, preconditioner=PMultigrid(c.spaces))
        x = zeros(num_free_dofs(c.spaces[1][1]))
        dx, ns, dx_lu = pmg_solve(c, ls, x)
        @test HK.linear_converged(ns)
        @test reldiff(dx, dx_lu) < 1e-8
        push!(its, HK.linear_iterations(ns))
    end
    @info "PMultigrid Q2→Q1 GMRES iterations (rtol 1e-10), n = 2, 4, 8" its
    @test all(≤(30), its)
    @test its[3] ≤ 1.4 * its[2] + 2                   # nearly h-independent
end

@testset "PMultigrid: Q3 → Q1 and Q3 → Q2 → Q1; CG with the symmetric cycle" begin
    for orders in ((3, 1), (3, 2, 1))
        its = Int[]
        for n in (2, 4)
            c = pmg_cube(n, orders)
            ls = MatrixFreeSolver(; krylov=:gmres, rtol=1e-10, maxiter=200, preconditioner=PMultigrid(c.spaces))
            dx, ns, dx_lu = pmg_solve(c, ls, zeros(num_free_dofs(c.spaces[1][1])))
            @test HK.linear_converged(ns)
            @test reldiff(dx, dx_lu) < 1e-7                # Q3: Gridap's basis round-off (step 0)
            push!(its, HK.linear_iterations(ns))
        end
        @info "PMultigrid $(join(orders, "→")) GMRES iterations, n = 2, 4" its
        @test all(≤(40), its) && its[2] ≤ 1.5 * its[1] + 3
    end
    c = pmg_cube(4, (2, 1))
    ls = MatrixFreeSolver(; krylov=:cg, rtol=1e-10, maxiter=200, preconditioner=PMultigrid(c.spaces))
    dx, ns, dx_lu = pmg_solve(c, ls, zeros(num_free_dofs(c.spaces[1][1])))
    @test HK.linear_converged(ns)
    @test reldiff(dx, dx_lu) < 1e-8
end

@testset "PMultigrid: the V-cycle is a symmetric operator on an SPD tangent" begin
    c = pmg_cube(2, (2, 1))
    U, V = c.spaces[1]
    op = KernelOperator(c.lres, c.ljac, U, V; jacobian=:matrixfree)
    x = zeros(num_free_dofs(U)); mf = allocate_jacobian(op, x); jacobian!(mf, op, x)
    st = HK.preconditioner_setup(PMultigrid(c.spaces), mf)
    n = size(mf, 1)
    B = reduce(hcat, [ldiv!(zeros(n), st, Float64.(1:n .== j)) for j in 1:n])
    @test norm(B - B') ≤ 1e-10 * norm(B)
    @test minimum(eigvals(Symmetric((B + B') / 2))) > 0
end

@testset "PMultigrid inside a Newton loop follows the LU trajectory (SVK)" begin
    c = pmg_cube(3, (2, 1))
    U, V = c.spaces[1]
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
        x, hist, ns
    end
    x_lu, h_lu, _ = newton(KernelOperator(c.nres, c.njac, U, V), LUSolver())
    for every in (1, 3)
        ls = MatrixFreeSolver(; rtol=1e-11, maxiter=200, preconditioner=PMultigrid(c.spaces; coarse_every=every))
        x_mf, h_mf, ns = newton(KernelOperator(c.nres, c.njac, U, V; jacobian=:matrixfree), ls)
        @test length(h_mf) == length(h_lu)
        @test all(abs.(log10.(h_mf) .- log10.(h_lu)) .< 0.5)
        @test reldiff(x_mf, x_lu) < 1e-8
        @test HK.preconditioner_builds(ns) == cld(length(h_mf), every)    # one coarse build per `every` Newton iterations
    end
end

@testset "PMultigrid: a NaN state makes the solve fail at once, a finite one recovers" begin
    c = pmg_cube(2, (2, 1))
    U, V = c.spaces[1]
    op = KernelOperator(c.nres, c.njac, U, V; jacobian=:matrixfree)
    x = randx(U); mf = allocate_jacobian(op, x); jacobian!(mf, op, x)
    ls = MatrixFreeSolver(; rtol=1e-8, preconditioner=PMultigrid(c.spaces))
    ns = numerical_setup(symbolic_setup(ls, mf), mf)
    b = randn(size(mf, 1)); dx = zeros(length(b))
    solve!(dx, ns, b); @test HK.linear_converged(ns)
    xb = copy(x); xb[3] = NaN
    jacobian!(mf, op, xb); numerical_setup!(ns, mf)
    solve!(dx, ns, b)
    @test !HK.linear_converged(ns) && HK.linear_iterations(ns) == 0
    jacobian!(mf, op, x); numerical_setup!(ns, mf)
    solve!(dx, ns, b); @test HK.linear_converged(ns)
end

@testset "PMultigrid: the cycle allocates nothing beyond its products, block applications and coarse solve" begin
    c = pmg_cube(4, (2, 1))
    U, V = c.spaces[1]
    op = KernelOperator(c.lres, c.ljac, U, V; jacobian=:matrixfree, nthreads=1)
    x = zeros(num_free_dofs(U)); mf = allocate_jacobian(op, x); jacobian!(mf, op, x)
    p = PMultigrid(c.spaces)
    st = HK.preconditioner_setup(p, mf)
    n = size(mf, 1); r = randn(n); y = zeros(n)
    ldiv!(y, st, r)
    S = st.smoothers[1]; w = st.work[1]; P = st.lv.P[1]
    amul = @allocated mul!(y, mf, r)
    adiv = @allocated ldiv!(w.r, S.blocks, r)
    acoarse = @allocated solve!(w.xc, st.lv.coarse.ns, w.bc)
    atr = (@allocated prolongate!(w.e, P, w.xc)) + (@allocated restrict!(w.bc, P, w.r))
    a = @allocated ldiv!(y, st, r)
    k = p.smoother.degree
    @test a ≤ 2k * amul + 2(k + 1) * adiv + acoarse + atr
    @test_throws ArgumentError PMultigrid(c.spaces[1:1])
    @test_throws ArgumentError PMultigrid(c.spaces; coarse_every=0)
end

# S4-A step 3: nodal blocks from the stored point tangents, and the Chebyshev smoother on the block-preconditioned
# tangent (damping of every eigencomponent in its interval within the Chebyshev bound, reproducible interval).
Random.seed!(6)

"Elasticity with one component fixed on some faces (partially constrained nodes; as in test_matrixfree_solver.jl)."
function cheb_masked(; n=(3, 2, 2))
    model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), n)
    Ω = Triangulation(model); dΩ = Measure(Ω, 4)
    V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 2), dirichlet_tags=[21, 22, 25],
                    dirichlet_masks=[(true, true, true), (true, false, false), (false, false, true)])
    U = TrialFESpace(V, [x -> VectorValue(0.0, 0.0, 0.0), x -> VectorValue(0.02, 0.0, 0.0), x -> VectorValue(0.0, 0.0, 0.01)])
    res(u, v) = ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))))dΩ
    jac(u, du, v) = ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')))dΩ
    (; model, Ω, dΩ, U, V, res, jac)
end

function cheb_setup(prob; nthreads=Threads.nthreads())
    op = KernelOperator(prob.res, prob.jac, prob.U, prob.V; jacobian=:matrixfree, nthreads)
    ref = KernelOperator(prob.res, prob.jac, prob.U, prob.V)
    x = randx(prob.U)
    mf = allocate_jacobian(op, x)
    jacobian!(mf, op, x)
    (; op, ref, x, mf, A=jacobian(ref, x), n=size(mf, 1))
end

"Dense matrix of a matrix-free operator (columns from unit vectors)."
dense_op(mf) = reduce(hcat, [mf * Float64.(1:size(mf, 2) .== j) for j in 1:size(mf, 2)])

"Dense block diagonal (the matrix M the smoother inverts) from the accumulated blocks."
function dense_block_diagonal(bj, n)
    M = zeros(n, n)
    for fb in bj.fields, blk in 1:size(fb.dofs, 2)
        free = [c for c in 1:size(fb.dofs, 1) if fb.dofs[c, blk] != 0]
        ids = fb.dofs[free, blk]
        M[ids, ids] = fb.B[free, free, blk]
    end
    M
end

@testset "nodal blocks from the stored tangents equal the element-tangent route" begin
    for prob in (elasticity_3d(; order=2), coupled_3d(), poisson_2d(; order=2), cheb_masked(), elasticity_3d(; n=(2, 2, 1), order=3))
        t = cheb_setup(prob)
        @test HK.blocks_from_tangents_ok(t.op)
        st1 = HK.preconditioner_setup(BlockJacobi(), t.mf)          # :auto → :tangents
        st2 = HK.preconditioner_setup(BlockJacobi(), t.mf)
        HK.block_diagonal!(st2, t.mf; route=:elements)
        for (a, b) in zip(st1.fields, st2.fields)
            @test norm(a.B - b.B) ≤ 1e-12 * max(1.0, norm(b.B))
            @test norm(a.Binv - b.Binv) ≤ 1e-10 * max(1.0, norm(b.Binv))
        end
        # and they are the assembled diagonal blocks
        M = dense_block_diagonal(st1, t.n)
        mask = M .!= 0
        @test norm(M[mask] - Matrix(t.A)[mask]) ≤ 1e-12 * norm(Matrix(t.A)[mask])
        # deterministic across threads (cells of a colour share no node)
        t1 = cheb_setup(prob; nthreads=1)
        jacobian!(t1.mf, t1.op, t.x)
        st3 = HK.preconditioner_setup(BlockJacobi(), t1.mf)
        @test all(((a, b),) -> a.B == b.B, zip(st3.fields, st1.fields))
    end
    @test_throws ArgumentError HK.block_diagonal!(HK.preconditioner_setup(BlockJacobi(), cheb_setup(poisson_2d()).mf), cheb_setup(poisson_2d()).mf; route=:dense)
end

@testset "Chebyshev: every eigencomponent of M⁻¹A in the interval damped within 1/T_k(θ/δ)" begin
    for (prob, cfg) in ((poisson_2d(; n=(6, 5), order=2), HK.Chebyshev()),
                        (poisson_2d(; n=(6, 5), order=2), HK.Chebyshev(; degree=3, range=10.0)),
                        (elasticity_3d(; n=(2, 2, 2), order=2), HK.Chebyshev(; degree=5, range=20.0)))
        t = cheb_setup(prob)
        st = HK.smoother_setup(cfg, t.mf)
        A = dense_op(t.mf)
        @test norm(A - A') ≤ 1e-10 * norm(A)                         # SPD model problems
        A = (A + A') / 2
        M = Symmetric(dense_block_diagonal(st.blocks, t.n))
        F = eigen(Symmetric(A), M)
        λtrue = maximum(F.values)
        @test minimum(F.values) > 0
        @test st.λest ≤ λtrue * (1 + 1e-8)                          # Ritz values from below
        @test st.λest ≥ 0.8λtrue
        @test st.b ≥ λtrue                                           # the ×1.1 inflation covers the estimate's deficit
        damp = HK.chebyshev_damping(cfg)
        rhs0 = zeros(t.n); y = zeros(t.n)
        worst_in, worst_below = 0.0, 0.0
        for j in eachindex(F.values)
            λ = F.values[j]; v = F.vectors[:, j]
            copyto!(y, v)
            HK.smooth!(y, st, rhs0)                                  # error propagation: y = p(M⁻¹A) v
            p = dot(y, M * v) / dot(v, M * v)
            @test norm(y - p * v) ≤ 1e-8 * norm(v)                   # an eigenvector stays one
            if st.a ≤ λ ≤ st.b
                worst_in = max(worst_in, abs(p))
            elseif λ < st.a
                worst_below = max(worst_below, abs(p))
            end
        end
        @test worst_in ≤ damp + 1e-8
        @test worst_below ≤ 1 + 1e-10
    end
end

@testset "Chebyshev: zero guess, residual output, reproducible interval, update" begin
    t = cheb_setup(elasticity_3d(; order=2))
    cfg = HK.Chebyshev()
    st = HK.smoother_setup(cfg, t.mf)
    st2 = HK.smoother_setup(cfg, t.mf)
    @test st.λest == st2.λest                                       # fixed start vector
    b = randn(t.n)
    x1 = zeros(t.n); HK.smooth!(x1, st, b)
    x2 = randn(t.n); HK.smooth!(x2, st, b; zero_guess=true)
    @test reldiff(x1, x2) < 1e-13
    r = zeros(t.n); x3 = zeros(t.n)
    HK.smooth!(x3, st, b; zero_guess=true, residual=r)
    @test x3 ≈ x1
    @test reldiff(r, b - t.mf * x3) < 1e-12
    # the residual decreases in the M⁻¹-norm on an SPD problem (the smoother is convergent)
    @test norm(r) < norm(b)
    # update at a new state: new blocks and interval
    x_new = randx(t.op.U)
    jacobian!(t.mf, t.op, x_new)
    HK.smoother_update!(st, t.mf)
    @test HK.preconditioner_builds(st) == 2
    st3 = HK.smoother_setup(cfg, t.mf)
    @test st.λest == st3.λest
    @test HK.chebyshev_damping(HK.Chebyshev(; degree=3, range=20.0)) ≈ 1 / cosh(3acosh(21 / 19))
    @test_throws ArgumentError HK.Chebyshev(; degree=0)
    @test_throws ArgumentError HK.Chebyshev(; range=1.0)
end

@testset "Chebyshev: the smoothing loop allocates nothing beyond its product and block application" begin
    # the product and the block application spawn one task per colour chunk (≈ 17 kB and 0.8 kB per call on this case,
    # the cell kernels themselves allocate nothing — test_operator.jl); the smoother adds no allocation of its own
    t = cheb_setup(elasticity_3d(; order=2); nthreads=1)
    cfg = HK.Chebyshev()
    st = HK.smoother_setup(cfg, t.mf)
    b = randn(t.n); x = zeros(t.n); r = zeros(t.n); y = zeros(t.n)
    HK.smooth!(x, st, b; zero_guess=true, residual=r); mul!(y, t.mf, b); ldiv!(y, st.blocks, b)
    amul = @allocated mul!(y, t.mf, b)
    adiv = @allocated ldiv!(y, st.blocks, b)
    a = @allocated HK.smooth!(x, st, b; zero_guess=true, residual=r)
    @test a ≤ cfg.degree * amul + (cfg.degree + 1) * adiv            # degree products (zero guess + residual), degree + 1 block applications
    a2 = @allocated HK.smooth!(x, st, b)
    @test a2 ≤ cfg.degree * amul + (cfg.degree + 1) * adiv           # initial residual + degree − 1 products
end

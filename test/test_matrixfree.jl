Random.seed!(1)

@testset "1D Lagrange basis: partition of unity, Kronecker property, derivative sum" begin
    for Q in 1:4, ξ in (0.0, 0.3, 1.0)
        N, dN = HK.lagrange1d(Q, ξ)
        @test sum(N) ≈ 1.0
        @test abs(sum(dN)) < 1e-12
    end
    N, _ = HK.lagrange1d(3, 0.5)
    @test N ≈ [0.0, 1.0, 0.0]
end

@testset "tensor-product tables reproduce the ND tables (Q1/Q2, 2D/3D)" begin
    for (prob, D) in ((poisson_2d(), 2), (poisson_2d(; order=2), 2), (elasticity_3d(), 3), (elasticity_3d(; order=2), 3))
        xq = quad_points(prob.dΩ)
        t = HK.FieldTable(prob.U, xq, D)
        tp = HK.TPTable(t, xq)
        @test tp !== nothing
        Q, P = HK.nnodes1d(tp), HK.npoints1d(tp)
        @test Q^D == t.nn && P^D == length(xq)
        # N and dN of the ND table from products of the 1D ones
        for (lin, idx) in enumerate(CartesianIndices(ntuple(_ -> Q, D))), (pl, pidx) in enumerate(CartesianIndices(ntuple(_ -> P, D)))
            n = tp.nodeperm[lin]; q = tp.pointperm[pl]
            ii, aa = Tuple(idx), Tuple(pidx)
            @test t.N[q, n] ≈ prod(d -> tp.N1[aa[d], ii[d]], 1:D) atol = 1e-12
            for d in 1:D
                g = prod(e -> e == d ? tp.dN1[aa[e], ii[e]] : tp.N1[aa[e], ii[e]], 1:D)
                @test t.dN[q, n][d] ≈ g atol = 1e-10
            end
        end
    end
    # simplices: no tensor-product structure
    s = elasticity_3d()
    ms = simplexify(s.model); Ωs = Triangulation(ms); dΩs = Measure(Ωs, 2)
    Vs = TestFESpace(ms, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1))
    @test HK.TPTable(HK.FieldTable(Vs, quad_points(dΩs), 3), quad_points(dΩs)) === nothing
end

@testset "sum-factorised interpolation and contraction agree with the dense tables" begin
    for (prob, D) in ((poisson_2d(; order=2), 2), (elasticity_3d(; order=2), 3))
        xq = quad_points(prob.dΩ)
        t = HK.FieldTable(prob.U, xq, D)
        tp = HK.TPTable(t, xq)
        np = length(xq)
        u = randn(t.nn)                                   # nodal values, table order
        ut = [u[tp.nodeperm[l]] for l in 1:t.nn]          # tensor order
        pts = zeros(np, D + 1); tmp = HK.tp_scratch(tp)
        HK.tp_interpolate!(pts, tp, ut, tmp)
        for pl in 1:np
            q = tp.pointperm[pl]
            @test pts[pl, 1] ≈ sum(u[n] * t.N[q, n] for n in 1:t.nn)
            for d in 1:D
                @test pts[pl, 1+d] ≈ sum(u[n] * t.dN[q, n][d] for n in 1:t.nn)
            end
        end
        coef = randn(np, D + 1); rc = zeros(t.nn)
        HK.tp_contract!(rc, tp, coef, tmp)
        for l in 1:t.nn
            n = tp.nodeperm[l]
            ref = sum(coef[pl, 1] * t.N[tp.pointperm[pl], n] + sum(coef[pl, 1+d] * t.dN[tp.pointperm[pl], n][d] for d in 1:D) for pl in 1:np)
            @test rc[l] ≈ ref
        end
    end
end

@testset "matrix-free Jacobian equals the assembled one" begin
    for (name, prob) in (("Poisson Q1", poisson_2d()), ("Poisson Q2", poisson_2d(; order=2)),
                         ("elasticity Q2", elasticity_3d(; order=2)), ("coupled u–φ Q1", coupled_3d()),
                         ("Poisson P1 (dense fallback)", poisson_2d(; simplex=true)))
        x = randx(prob.U)
        op = KernelOperator(prob.res, prob.jac, prob.U, prob.V)
        A = Algebra.jacobian(op, x)
        mf = MatrixFreeJacobian(op)
        prepare!(mf, x)
        dx = randn(num_free_dofs(prob.U))
        y = mf * dx
        @test size(mf) == size(A)
        @test reldiff(y, A * dx) < TOL
        y2 = similar(y); LinearAlgebra.mul!(y2, mf, dx)
        @test y2 == y
        # new state → prepare! again
        x2 = randx(prob.U)
        prepare!(mf, x2)
        @test reldiff(mf * dx, Algebra.jacobian(op, x2) * dx) < TOL
    end
end

@testset "matrix-free with a non-symmetric form and value+gradient slots" begin
    model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (3, 2, 2))
    Ω = Triangulation(model); dΩ = Measure(Ω, 4)
    V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 2), dirichlet_tags=[1, 2])
    U = TrialFESpace(V, x -> VectorValue(0.0, 0.0, 0.0))
    bv = VectorValue(1.0, 0.3, -0.2)
    res(u, v) = ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))) + v ⋅ u + ∇(v)' ⊙ (u ⊗ bv))dΩ
    jac(u, du, v) = ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')) + v ⋅ du + ∇(v)' ⊙ (du ⊗ bv))dΩ
    x = randx(U)
    op = KernelOperator(res, jac, U, V)
    mf = MatrixFreeJacobian(op); prepare!(mf, x)
    dx = randn(num_free_dofs(U))
    @test reldiff(mf * dx, Algebra.jacobian(op, x) * dx) < TOL
end

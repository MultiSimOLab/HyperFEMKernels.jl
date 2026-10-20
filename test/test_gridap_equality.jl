Random.seed!(1)
begin

    @testset "Poisson 2D ($(name))" for (name, model) in (
            ("Q1", CartesianDiscreteModel((0, 1, 0, 1), (5, 4))),
            ("P1", simplexify(CartesianDiscreteModel((0, 1, 0, 1), (4, 3)))))
        Ω = Triangulation(model); dΩ = Measure(Ω, 2)
        V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), dirichlet_tags="boundary")
        U = TrialFESpace(V, x -> x[1] + 2x[2])
        f(x) = sin(x[1]) * x[2]
        res(u, v) = ∫(∇(v) ⋅ ∇(u) - f * v)dΩ
        jac(u, du, v) = ∫(∇(v) ⋅ ∇(du))dΩ
        c = check(res, jac, U, V, rand(num_free_dofs(U)))
        @test c.res < TOL
        @test c.jac < TOL
    end

    @testset "Hyperelasticity 3D Q$order" for order in (1, 2)
        model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (3, 2, 2))
        Ω = Triangulation(model); dΩ = Measure(Ω, 2order)
        V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, order), dirichlet_tags=[1, 3, 5, 7])
        U = TrialFESpace(V, x -> VectorValue(0.01, 0.0, 0.0))
        res(u, v) = ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))))dΩ
        jac(u, du, v) = ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')))dΩ
        x = 0.01 .* randn(num_free_dofs(U))
        c = check(res, jac, U, V, x)
        @test c.res < TOL
        @test c.jac < TOL
        @test all(a -> a.bytes == 0, allocations(KernelOperator(res, jac, U, V), x))
        cd = check(res, jac, U, V, x; op=KernelOperator(res, jac, U, V; linear_maps=false))  # dual-number path
        @test cd.res < TOL
        @test cd.jac < TOL
        op = KernelOperator(res, jac, U, V)
        @test all(g -> g.lin !== nothing, op.rgroups)   # residual: linear map of P(F)
        @test all(g -> g.lin === nothing, op.jgroups)   # dP(F, dF) is not linear in F: dual numbers
    end

    @testset "Hyperelasticity 3D P1 (tetrahedra)" begin
        model = simplexify(CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (2, 2, 2)))
        Ω = Triangulation(model); dΩ = Measure(Ω, 2)
        V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), dirichlet_tags=[1, 3, 5, 7])
        U = TrialFESpace(V, x -> VectorValue(0.0, 0.01, 0.0))
        res(u, v) = ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))))dΩ
        jac(u, du, v) = ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')))dΩ
        c = check(res, jac, U, V, 0.01 .* randn(num_free_dofs(U)))
        @test c.res < TOL
        @test c.jac < TOL
    end

    @testset "Multi-field u–φ with value and gradient couplings" begin
        model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (2, 2, 2))
        Ω = Triangulation(model); dΩ = Measure(Ω, 2)
        Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), dirichlet_tags=[1, 2])
        Vφ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), dirichlet_tags=[7, 8])
        V = MultiFieldFESpace([Vu, Vφ])
        U = MultiFieldFESpace([TrialFESpace(Vu, x -> VectorValue(0.0, 0.0, 0.0)), TrialFESpace(Vφ, x -> 0.1)])
        β = 0.3
        res((u, φ), (v, vφ)) = ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))) + β * φ * tr(∇(v)) + ∇(vφ) ⋅ ∇(φ) +
                                 β * vφ * (tr ∘ (Fk ∘ (∇(u)'))))dΩ
        jac((u, φ), (du, dφ), (v, vφ)) = ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')) + β * dφ * tr(∇(v)) +
                                           ∇(vφ) ⋅ ∇(dφ) + β * vφ * tr(∇(du)))dΩ
        x = 0.01 .* randn(num_free_dofs(U))
        c = check(res, jac, U, V, x)
        @test c.res < TOL
        @test c.jac < TOL
        @test all(a -> a.bytes == 0, allocations(KernelOperator(res, jac, U, V), x))
    end

    @testset "Dirichlet masks, auxiliary FE functions and scaled integrals (2D)" begin
        model = CartesianDiscreteModel((0, 2, 0, 1), (6, 3))
        Ω = Triangulation(model); dΩ = Measure(Ω, 2)
        V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, 1),
                        dirichlet_tags=["tag_7", "tag_8"], dirichlet_masks=[(true, true), (true, false)])
        U = TrialFESpace(V, [x -> VectorValue(0.0, 0.0), x -> VectorValue(0.02, 0.0)])
        Uold = TrialFESpace(V, [x -> VectorValue(0.0, 0.0), x -> VectorValue(0.0, 0.0)])
        uold = FEFunction(Uold, 0.01 .* randn(num_free_dofs(Uold)))
        N = interpolate_everywhere(x -> VectorValue(cos(x[2]), sin(x[2])), V)
        Λ = 0.4
        g(F, N) = (F ⋅ N) ⊗ N
        res(u, v) = ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))))dΩ - Λ * ∫((g ∘ (Fk ∘ (∇(uold)'), N)) ⊙ ∇(v)')dΩ +
                    ∫(v ⋅ uold)dΩ
        jac(u, du, v) = ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')))dΩ
        x = 0.01 .* randn(num_free_dofs(U))
        c = check(res, jac, U, V, x)
        @test c.res < TOL
        @test c.jac < TOL
        @test all(a -> a.bytes == 0, allocations(KernelOperator(res, jac, U, V), x))
        c2 = check(res, jac, U, V, x; op=KernelOperator(res, jac, U, V; cache_aux=true))
        @test c2.res < TOL
        @test c2.jac < TOL
    end

    @testset "Non-symmetric Jacobians (convection, different test/trial slot sets)" begin
        model = CartesianDiscreteModel((0, 1, 0, 1), (6, 5))
        Ω = Triangulation(model); dΩ = Measure(Ω, 2)
        V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), dirichlet_tags="boundary")
        U = TrialFESpace(V, x -> x[1])
        b = VectorValue(1.0, 0.3)
        # test: value + gradient; trial: gradient only → nsv ≠ nsu, K not symmetric
        res(u, v) = ∫(∇(v) ⋅ ∇(u) + v * (b ⋅ ∇(u)) + v * u * u)dΩ
        jac(u, du, v) = ∫(∇(v) ⋅ ∇(du) + v * (b ⋅ ∇(du)))dΩ
        x = rand(num_free_dofs(U))
        for lm in (true, false)
            op = KernelOperator(res, jac, U, V; linear_maps=lm)
            c = check(res, jac, U, V, x; op)
            @test c.res < TOL
            @test c.jac < TOL
        end
        A = Algebra.jacobian(KernelOperator(res, jac, U, V), x)
        @test norm(A - A') / norm(A) > 1e-2                    # really non-symmetric
        # vector field with a non-symmetric 4th-order tensor (test gradient, trial value+gradient)
        Vv = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, 1), dirichlet_tags="boundary")
        Uv = TrialFESpace(Vv, x -> VectorValue(0.0, 0.0))
        Cns(dF) = TensorValue(1.0dF[1, 1] + 0.5dF[1, 2], 0.2dF[2, 1], 0.7dF[2, 2] + 0.1dF[1, 1], 2.0dF[2, 2] + 0.3dF[2, 1])
        resv(u, v) = ∫(∇(v)' ⊙ (Cns ∘ (∇(u)')) + v ⋅ u)dΩ
        jacv(u, du, v) = ∫(∇(v)' ⊙ (Cns ∘ (∇(du)')) + ∇(v)' ⊙ (du ⊗ b))dΩ
        xv = rand(num_free_dofs(Uv))
        for lm in (true, false)
            cv = check(resv, jacv, Uv, Vv, xv; op=KernelOperator(resv, jacv, Uv, Vv; linear_maps=lm))
            @test cv.res < TOL
            @test cv.jac < TOL
        end
    end

    @testset "Linear-map shortcut with a state-dependent coefficient tensor" begin
        model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (3, 3, 2))
        Ω = Triangulation(model); dΩ = Measure(Ω, 2)
        V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), dirichlet_tags="boundary")
        U = TrialFESpace(V, x -> x[1])
        K(u) = TensorValue(1 + u^2, 0.1u, 0.0, 0.1u, 2 + u, 0.0, 0.0, 0.0, 1.0)
        res(u, v) = ∫(∇(v) ⋅ ((K ∘ u) ⋅ ∇(u)) + 0.5 * v)dΩ
        jac(u, du, v) = ∫(∇(v) ⋅ ((K ∘ u) ⋅ ∇(du)))dΩ
        x = rand(num_free_dofs(U))
        op = KernelOperator(res, jac, U, V)
        @test all(g -> g.lin !== nothing, op.rgroups)
        @test all(g -> g.lin !== nothing, op.jgroups)
        c = check(res, jac, U, V, x; op)
        @test c.res < TOL
        @test c.jac < TOL
        @test all(a -> a.bytes == 0, allocations(op, x))
    end

    @testset "Integration on a subdomain of the FE space (and two measures)" begin
        model = CartesianDiscreteModel((0, 2, 0, 1), (8, 4))
        left = Bool[sum(c)[1] / length(c) < 1.0 for c in vec(collect(get_cell_coordinates(model)))]
        sub = Triangulation(model, left)
        Ω = Triangulation(model); dΩ = Measure(Ω, 2); dΩs = Measure(sub, 2)
        V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), dirichlet_tags="boundary")
        U = TrialFESpace(V, x -> x[1])
        res(u, v) = ∫(∇(v) ⋅ ∇(u))dΩ + ∫(3.0 * v * u * u)dΩs
        jac(u, du, v) = ∫(∇(v) ⋅ ∇(du))dΩ + ∫(6.0 * v * u * du)dΩs
        c = check(res, jac, U, V, rand(num_free_dofs(U)))
        @test c.res < TOL
        @test c.jac < TOL
    end

    @testset "update keeps the structure and follows new parameters" begin
        model = CartesianDiscreteModel((0, 1, 0, 1), (4, 4))
        Ω = Triangulation(model); dΩ = Measure(Ω, 2)
        V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), dirichlet_tags="boundary")
        U = TrialFESpace(V, 0.0)
        res(Λ) = (u, v) -> ∫(∇(v) ⋅ ∇(u) - Λ * v)dΩ
        jac(Λ) = (u, du, v) -> ∫(∇(v) ⋅ ∇(du))dΩ
        op = KernelOperator(res(1.0), jac(1.0), U, V)
        op2 = update(op, res(2.0), jac(2.0))
        x = rand(num_free_dofs(U))
        c = check(res(2.0), jac(2.0), U, V, x; op=op2)
        @test c.res < TOL
        @test c.jac < TOL
        @test op2.A0 === op.A0
    end
end

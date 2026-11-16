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

    @testset "Linear elasticity with the symmetric gradient, $(D)D, $(coef)" for D in (2, 3), coef in (:constant, :density)
        # ε(u) is a SymTensorValue (D(D+1)/2 stored entries): the residual is compiled, not sent to Gridap.
        # With a constant modulus it is linear in its state input σ(ε(u)) and gets a linear map; with a density
        # coefficient, as in SIMP, the term is the product of two state inputs, E(ρh) and σ(ε(u)), so it is not
        # linear in them and goes through dual numbers
        model = CartesianDiscreteModel(D == 2 ? (0, 1, 0, 1) : (0, 1, 0, 1, 0, 1), D == 2 ? (6, 4) : (3, 2, 2))
        Ω = Triangulation(model); dΩ = Measure(Ω, 2)
        V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{D,Float64}, 1), dirichlet_tags="boundary")
        U = TrialFESpace(V, x -> zero(VectorValue{D,Float64}))
        Vρ = FESpace(model, ReferenceFE(lagrangian, Float64, 1))
        ρh = FEFunction(Vρ, rand(num_free_dofs(Vρ)))
        σ(e) = 0.6 * tr(e) * one(e) + 0.8 * e
        E(ρ) = 1e-3 + ρ^3
        c = coef === :constant ? 2.0 : (E ∘ ρh)
        res(u, v) = ∫(c * (ε(v) ⊙ (σ ∘ ε(u))))dΩ
        jac(u, du, v) = ∫(c * (ε(v) ⊙ (σ ∘ ε(du))))dΩ
        x = rand(num_free_dofs(U))
        op = @test_logs KernelOperator(res, jac, U, V)          # no fallback warning
        @test op.fallback === nothing || (isempty(op.fallback.rtrians) && isempty(op.fallback.jtrians))
        @test all(g -> (g.lin !== nothing) == (coef === :constant), op.rgroups)
        r = check(res, jac, U, V, x; op)
        @test r.res < TOL
        @test r.jac < TOL
        @test all(a -> a.bytes == 0, allocations(op, x))
    end

    @testset "Symmetric tensors in the form, $(D)D" for D in (2, 3)
        # SymTensorValue / SymTracelessTensorValue store fewer entries than D²: state inputs of the Jacobian,
        # off-diagonal entries against a full tensor, auxiliary data, constants and `update`
        model = CartesianDiscreteModel(D == 2 ? (0, 1, 0, 1) : (0, 1, 0, 1, 0, 1), D == 2 ? (5, 4) : (3, 2, 2))
        Ω = Triangulation(model); dΩ = Measure(Ω, 2)
        V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{D,Float64}, 1), dirichlet_tags="boundary")
        U = TrialFESpace(V, x -> 0.01 * x)
        L = D * (D + 1) ÷ 2
        σ(e) = 0.6 * tr(e) * one(e) + 0.8 * e
        x = rand(num_free_dofs(U))
        nofallback(op) = op.fallback === nothing || (isempty(op.fallback.rtrians) && isempty(op.fallback.jtrians))
        function same(res, jac; jlin = nothing)
            op = KernelOperator(res, jac, U, V)
            r = check(res, jac, U, V, x; op)
            ok = nofallback(op) && r.res < TOL && r.jac < TOL
            jlin === nothing || (ok &= all(g -> (g.lin !== nothing) == jlin, op.jgroups))
            ok
        end
        # Jacobian with ε(u) as a state input: linear map
        @test same((u, v) -> ∫(0.5 * (ε(u) ⊙ ε(u)) * tr(ε(v)))dΩ, (u, du, v) -> ∫((ε(u) ⊙ ε(du)) * tr(ε(v)))dΩ; jlin = true)
        # off-diagonal entries of ε(u) against a full tensor and a vector
        w = VectorValue(ntuple(i -> Float64(i), D))
        @test same((u, v) -> ∫(∇(v) ⊙ ε(u) + (ε(u) ⋅ v) ⋅ w)dΩ, (u, du, v) -> ∫(∇(v) ⊙ ε(du) + (ε(du) ⋅ v) ⋅ w)dΩ)
        # auxiliary data: a SymTensorValue FE function, ε of an FE function, a function of x, a constant
        Vs = FESpace(model, ReferenceFE(lagrangian, SymTensorValue{D,Float64}, 1))
        σh = FEFunction(Vs, rand(num_free_dofs(Vs)))
        uh0 = FEFunction(V, rand(num_free_dofs(V)))
        @test same((u, v) -> ∫(ε(v) ⊙ σh + ε(v) ⊙ (σ ∘ ε(uh0)) + ε(v) ⊙ ε(u))dΩ,
                   (u, du, v) -> ∫((σh ⊙ ε(du)) * (σh ⊙ ε(v)) + ε(v) ⊙ ε(du))dΩ)
        fx(p) = SymTensorValue{D,Float64,L}(ntuple(i -> p[1] + i, L))
        S0 = SymTensorValue{D,Float64,L}(ntuple(i -> 0.3 + i, L))
        @test same((u, v) -> ∫(ε(v) ⊙ fx + ε(v) ⊙ S0 + (S0 ⊙ ε(u)) * tr(ε(v)))dΩ,
                   (u, du, v) -> ∫((S0 ⊙ ε(du)) * tr(ε(v)) + (fx ⊙ ε(du)) * (S0 ⊙ ε(v)))dΩ)
        # a traceless constant (its random parameters must have the independent entries only)
        Q0 = SymTracelessTensorValue{D,Float64,L}(ntuple(i -> 0.3 + i, L - 1))
        @test same((u, v) -> ∫(ε(v) ⊙ Q0 + (Q0 ⊙ ε(u)) * tr(ε(v)))dΩ, (u, du, v) -> ∫((Q0 ⊙ ε(du)) * tr(ε(v)))dΩ)
        # update with symmetric state inputs and new constants
        function forms(a, b)
            s(e) = a * tr(e) * one(e) + b * e
            r(u, v) = ∫(b * (ε(v) ⊙ (s ∘ ε(u))) + a * ((S0 ⊙ ε(u)) * tr(ε(v))))dΩ
            j(u, du, v) = ∫(b * (ε(v) ⊙ (s ∘ ε(du))) + a * ((ε(u) ⊙ ε(du)) * tr(ε(v))))dΩ
            r, j
        end
        r1, j1 = forms(0.6, 0.8); r2, j2 = forms(1.7, 0.3)
        op = KernelOperator(r1, j1, U, V)
        op2 = update(op, r2, j2)
        c = check(r2, j2, U, V, x; op=op2)
        @test c.res < TOL && c.jac < TOL && nofallback(op2)
        @test all(((g2, g),) -> g2.pf.core === g.pf.core, zip(vcat(op2.rgroups, op2.jgroups), vcat(op.rgroups, op.jgroups)))
        @test all(a -> a.bytes == 0, allocations(op2, x))
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
        @test op2.pattern === op.pattern
    end
end

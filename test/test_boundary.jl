# Boundary integrals in the engine: faces of a BoundaryTriangulation glued to the cells of the
# spaces, with the normal and volume fields on the faces as auxiliary data. Every case is checked
# against pure Gridap (tolerance TOL) and must not use the hybrid fallback.

nofallback(op) = isempty(op.fallback.rtrians) && isempty(op.fallback.jtrians)
ngroups_on(op, T) = count(g -> g.trian isa T, op.jgroups)

@testset "Neumann and Robin on quads/triangles, orders 1–3" begin
    for order in (1, 2, 3), simplex in (false, true)
        p = poisson_2d(; n=(4, 3), order, simplex)
        Γ = BoundaryTriangulation(p.model; tags=[6, 8]); dΓ = Measure(Γ, 2order)
        g(x) = x[1] - x[2]; α = 3.0
        res(u, v) = ∫(∇(v) ⋅ ∇(u) - 2.0 * v)p.dΩ + ∫(α * v * u - g * v)dΓ
        jac(u, du, v) = ∫(∇(v) ⋅ ∇(du))p.dΩ + ∫(α * v * du)dΓ
        op = @test_logs KernelOperator(res, jac, p.U, p.V)
        @test nofallback(op)
        c = check(res, jac, p.U, p.V, randx(p.U); op)
        @test c.res < TOL && c.jac < TOL
    end
end

@testset "whole boundary: several faces per cell, threads" begin
    p = poisson_2d(; n=(3, 2), order=2)       # corner cells have two boundary faces
    Γ = BoundaryTriangulation(p.model); dΓ = Measure(Γ, 4)
    res(u, v) = ∫(∇(v) ⋅ ∇(u))p.dΩ + ∫(v * u * u)dΓ
    jac(u, du, v) = ∫(∇(v) ⋅ ∇(du))p.dΩ + ∫(2.0 * v * u * du)dΓ
    for nthreads in (1, Threads.nthreads())
        op = KernelOperator(res, jac, p.U, p.V; nthreads)
        @test nofallback(op)
        # faces of the same cell are never in the same colour of a group
        for g in op.jgroups, color in g.colors
            @test allunique(g.cellmap[color])
        end
        c = check(res, jac, p.U, p.V, randx(p.U; scale=1.0); op)
        @test c.res < TOL && c.jac < TOL
    end
end

@testset "normal vector and gradient of the unknown on the boundary" begin
    for order in (1, 2)
        p = poisson_2d(; n=(4, 3), order)
        Γ = BoundaryTriangulation(p.model); dΓ = Measure(Γ, 2order)
        n = get_normal_vector(Γ)
        res(u, v) = ∫(∇(v) ⋅ ∇(u))p.dΩ - ∫((n ⋅ ∇(u)) * v)dΓ      # Nitsche-like consistency term
        jac(u, du, v) = ∫(∇(v) ⋅ ∇(du))p.dΩ - ∫((n ⋅ ∇(du)) * v)dΓ
        op = KernelOperator(res, jac, p.U, p.V)
        @test nofallback(op)
        c = check(res, jac, p.U, p.V, randx(p.U); op)
        @test c.res < TOL && c.jac < TOL
    end
end

@testset "3D hexahedra and tetrahedra: pressure follower load with the normal" begin
    for simplex in (false, true), order in (1, 2)
        model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (2, 2, 2))
        simplex && (model = simplexify(model))
        Ω = Triangulation(model); dΩ = Measure(Ω, 2order)
        V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, order), dirichlet_tags=[1, 3, 5, 7])
        U = TrialFESpace(V, x -> VectorValue(0.01, 0.0, 0.0))
        Γ = BoundaryTriangulation(model; tags=[26]); dΓ = Measure(Γ, 2order)
        n = get_normal_vector(Γ)
        pr = 0.3
        # pressure on the deformed configuration: p J F⁻ᵀ n (nonlinear in u on Γ)
        res(u, v) = ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))))dΩ - ∫(pr * (v ⋅ ((det ∘ (Fk ∘ (∇(u)'))) * ((inv ∘ (Fk ∘ (∇(u)'))) ⋅ n))))dΓ
        jac(u, du, v) = ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')))dΩ -
                        ∫(pr * (v ⋅ (((det ∘ (Fk ∘ (∇(u)'))) * (tr ∘ ((inv ∘ (Fk ∘ (∇(u)'))) ⋅ ∇(du)'))) * ((inv ∘ (Fk ∘ (∇(u)'))) ⋅ n)
                                   - (det ∘ (Fk ∘ (∇(u)'))) * ((inv ∘ (Fk ∘ (∇(u)'))) ⋅ ∇(du)' ⋅ (inv ∘ (Fk ∘ (∇(u)'))) ⋅ n))))dΓ
        op = KernelOperator(res, jac, U, V)
        @test nofallback(op)
        c = check(res, jac, U, V, randx(U); op)
        @test c.res < TOL && c.jac < TOL
    end
end

@testset "multi-field: surface load on one field, Robin on the other" begin
    c3 = coupled_3d()
    Γ = BoundaryTriangulation(c3.model; tags=[25, 26]); dΓ = Measure(Γ, 2)
    t = VectorValue(0.1, -0.2, 0.05)
    res((u, φ), (v, vφ)) = c3.res((u, φ), (v, vφ)) - ∫(v ⋅ t)dΓ + ∫(2.0 * vφ * φ)dΓ
    jac((u, φ), (du, dφ), (v, vφ)) = c3.jac((u, φ), (du, dφ), (v, vφ)) + ∫(2.0 * vφ * dφ)dΓ
    op = KernelOperator(res, jac, c3.U, c3.V)
    @test nofallback(op)
    c = check(res, jac, c3.U, c3.V, randx(c3.U); op)
    @test c.res < TOL && c.jac < TOL
end

@testset "volume field evaluated on the boundary (previous state) as auxiliary data" begin
    p = poisson_2d(; order=2)
    Γ = BoundaryTriangulation(p.model); dΓ = Measure(Γ, 4)
    uprev = FEFunction(p.U, randx(p.U; scale=1.0))
    res(u, v) = ∫(∇(v) ⋅ ∇(u))p.dΩ + ∫(v * (u - uprev) * uprev)dΓ
    jac(u, du, v) = ∫(∇(v) ⋅ ∇(du))p.dΩ + ∫(v * du * uprev)dΓ
    op = KernelOperator(res, jac, p.U, p.V)
    @test nofallback(op)
    c = check(res, jac, p.U, p.V, randx(p.U); op)
    @test c.res < TOL && c.jac < TOL
end

@testset "matrix-free Jacobian with boundary groups; update; allocations" begin
    p = poisson_2d(; n=(6, 5), order=2)
    Γ = BoundaryTriangulation(p.model); dΓ = Measure(Γ, 4)
    res(u, v) = ∫(∇(v) ⋅ ∇(u))p.dΩ + ∫(5.0 * v * u)dΓ
    jac(u, du, v) = ∫(∇(v) ⋅ ∇(du))p.dΩ + ∫(5.0 * v * du)dΓ
    op = KernelOperator(res, jac, p.U, p.V)
    x = randx(p.U); dx = randn(num_free_dofs(p.U))
    A = Algebra.jacobian(op, x)
    mf = MatrixFreeJacobian(op); prepare!(mf, x)
    @test reldiff(mf * dx, A * dx) < 1e-11
    res2(u, v) = ∫(∇(v) ⋅ ∇(u))p.dΩ + ∫(7.0 * v * u)dΓ
    jac2(u, du, v) = ∫(∇(v) ⋅ ∇(du))p.dΩ + ∫(7.0 * v * du)dΓ
    op2 = update(op, res2, jac2)
    @test op2.A0 === op.A0 && (op2.jgroups[end].geo === op.jgroups[end].geo || op2.jgroups[end].geo === op.rgroups[end].geo)
    c = check(res2, jac2, p.U, p.V, x; op=op2)
    @test c.res < TOL && c.jac < TOL
    @test all(a -> a.bytes == 0, allocations(op, x))
end

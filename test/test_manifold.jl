# FE spaces on a manifold triangulation: cells of lower dimension than the physical space (a curve in
# 2D or 3D, a surface in 3D). The case that motivated it is the L2 projection of the Maxwell traction
# onto the solid–air interface of the magneto-mechanical gripper (SoftMagnetoSensing): spaces on
# `BoundaryTriangulation(Ωsolid, tags="Interface")` (a `CompositeTriangulation`) and the mass matrix
# `∫(u ⋅ v)dΓ` on them. Values, tangential gradients (Gridap's `pinvJt ⋅ ∇̂`), FE functions on the
# manifold in the integrand and the matrix-free product, against pure Gridap, without hybrid fallback.

nofallback_mf(op) = isempty(op.fallback.rtrians) && isempty(op.fallback.jtrians)

"Model (0,2)×(0,1) split at x = 1; Γ = interface of the left part (x = 1), a CompositeTriangulation."
function interface_2d(; n=(8, 4), simplex=false)
    model = CartesianDiscreteModel((0, 2, 0, 1), n)
    simplex && (model = simplexify(model))
    cc = [sum(c) / length(c) for c in get_cell_coordinates(model)]
    Ωa = Triangulation(model, [x[1] < 1 for x in cc])
    Γ = BoundaryTriangulation(Ωa, tags="interior")
    (; model, Ωa, Γ)
end

# nonlinear pointwise laws (so that the Jacobian is not the residual's matrix)
ψv(u) = (1 + 0.5 * (u ⋅ u)) * u
dψv(u, du) = (1 + 0.5 * (u ⋅ u)) * du + (u ⋅ du) * u
ψs(g) = (1 + g ⋅ g) * g
dψs(g, dg) = (1 + g ⋅ g) * dg + 2 * (g ⋅ dg) * g

@testset "interface of a subdomain (CompositeTriangulation): mass and L2 projection" begin
    for simplex in (false, true), order in (1, 2)
        s = interface_2d(; simplex)
        @test s.Γ isa Gridap.Geometry.CompositeTriangulation
        @test num_cell_dims(s.Γ) == 1 && num_point_dims(s.Γ) == 2
        dΓ = Measure(s.Γ, 2order + 2)
        reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
        V = TestFESpace(s.Γ, reffe); U = TrialFESpace(V)
        # the traction field to project, an FE function on the interface (as tmaxw)
        t = interpolate_everywhere(x -> VectorValue(sin(3x[2]), x[1] * x[2]), FESpace(s.Γ, reffe))
        res(u, v) = ∫(u ⋅ v - v ⋅ t)dΓ
        jac(u, du, v) = ∫(du ⋅ v)dΓ
        x = randx(U; scale=1.0)
        for nthreads in (1, Threads.nthreads())
            op = KernelOperator(res, jac, U, V; nthreads)
            @test nofallback_mf(op)
            c = check(res, jac, U, V, x; op)
            @test c.res < TOL && c.jac < TOL
            @test all(a -> a.bytes == 0, allocations(op, x))
        end
        # the projection reproduces the FE function (same space)
        op = KernelOperator(res, jac, U, V)
        A = Algebra.jacobian(op, x); b = Algebra.residual(op, zeros(num_free_dofs(U)))
        @test norm(A \ (-b) - get_free_dof_values(t)) < 1e-10 * norm(get_free_dof_values(t))
    end
end

@testset "curve in 2D: tangential gradients, nonlinear laws, Dirichlet values" begin
    s = interface_2d()
    dΓ = Measure(s.Γ, 4)
    for (T, ψ, dψ) in ((Float64, ψs, dψs), (VectorValue{2,Float64}, nothing, nothing))
        V = TestFESpace(s.Γ, ReferenceFE(lagrangian, T, 2); dirichlet_tags="boundary")
        g = T <: Real ? (x -> x[2]^2) : (x -> VectorValue(x[2], -x[2]^2))
        U = TrialFESpace(V, g)
        f = T <: Real ? (x -> cos(x[2])) : (x -> VectorValue(cos(x[2]), 1.0))
        res, jac = if T <: Real
            ((u, v) -> ∫(∇(v) ⋅ (ψ ∘ ∇(u)) + u * u * v - f * v)dΓ,
             (u, du, v) -> ∫(∇(v) ⋅ (dψ ∘ (∇(u), ∇(du))) + 2 * u * du * v)dΓ)
        else
            ((u, v) -> ∫(∇(v) ⊙ ∇(u) + v ⋅ (ψv ∘ u) - f ⋅ v)dΓ,
             (u, du, v) -> ∫(∇(v) ⊙ ∇(du) + v ⋅ (dψv ∘ (u, du)))dΓ)
        end
        x = randx(U; scale=0.5)
        op = KernelOperator(res, jac, U, V)
        @test nofallback_mf(op)
        c = check(res, jac, U, V, x; op)
        @test c.res < TOL && c.jac < TOL
        @test all(a -> a.bytes == 0, allocations(op, x))
        mf = MatrixFreeJacobian(op); prepare!(mf, x)
        dx = randn(num_free_dofs(U))
        @test reldiff(mf * dx, Algebra.jacobian(op, x) * dx) < 1e-11
    end
end

@testset "surface in 3D: Laplace–Beltrami on the boundary of a box and of a subdomain" begin
    for simplex in (false, true), order in (1, 2)
        model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (3, 2, 2))
        simplex && (model = simplexify(model))
        cc = [sum(c) / length(c) for c in get_cell_coordinates(model)]
        Ωa = Triangulation(model, [x[1] < 0.5 for x in cc])
        for Γ in (BoundaryTriangulation(model), BoundaryTriangulation(Ωa, tags="interior"))
            @test num_cell_dims(Γ) == 2 && num_point_dims(Γ) == 3
            dΓ = Measure(Γ, 2order + 2)
            V = TestFESpace(Γ, ReferenceFE(lagrangian, Float64, order)); U = TrialFESpace(V)
            w = interpolate_everywhere(x -> 1 + x[1] * x[3], FESpace(Γ, ReferenceFE(lagrangian, Float64, order)))
            f(x) = x[1] + 2x[2] - x[3]
            res(u, v) = ∫(w * (∇(v) ⋅ (ψs ∘ ∇(u))) + u * v - f * v)dΓ
            jac(u, du, v) = ∫(w * (∇(v) ⋅ (dψs ∘ (∇(u), ∇(du)))) + du * v)dΓ
            x = randx(U; scale=0.5)
            op = KernelOperator(res, jac, U, V)
            @test nofallback_mf(op)
            c = check(res, jac, U, V, x; op)
            @test c.res < TOL && c.jac < TOL
            @test all(a -> a.bytes == 0, allocations(op, x))
            mf = MatrixFreeJacobian(op); prepare!(mf, x)
            dx = randn(num_free_dofs(U))
            @test reldiff(mf * dx, Algebra.jacobian(op, x) * dx) < 1e-11
        end
    end
end

@testset "curve in 3D (edges of a box): values and tangential gradients" begin
    model = CartesianDiscreteModel((0, 1, 0, 2, 0, 1), (2, 2, 2))
    Λ = Triangulation(ReferenceFE{1}, model)          # all the edges of the mesh
    @test num_cell_dims(Λ) == 1 && num_point_dims(Λ) == 3
    dΛ = Measure(Λ, 4)
    V = TestFESpace(Λ, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1)); U = TrialFESpace(V)
    res(u, v) = ∫(∇(v) ⊙ ∇(u) + v ⋅ (ψv ∘ u))dΛ
    jac(u, du, v) = ∫(∇(v) ⊙ ∇(du) + v ⋅ (dψv ∘ (u, du)))dΛ
    x = randx(U; scale=0.5)
    op = KernelOperator(res, jac, U, V)
    @test nofallback_mf(op)
    c = check(res, jac, U, V, x; op)
    @test c.res < TOL && c.jac < TOL
end

@testset "frame completion: |det| is the cell measure, inv is pinvJt on the tangent directions" begin
    for Jr in (TensorValue{1,2}(0.3, -1.2), TensorValue{2,3}(0.3, 1.1, -0.4, 0.2, 0.9, 0.5), TensorValue{1,3}(0.2, -0.7, 1.4))
        d, D = size(Jr)
        Jt = TensorValue{D,D}(ntuple(i -> ((r, c) = ((i - 1) % D + 1, (i - 1) ÷ D + 1); r <= d ? Jr[r, c] : 0.0), D * D))
        Jc = HK.complete_frame(Jt, d)
        @test abs(det(Jc)) ≈ meas(Jr)
        P = Gridap.Fields.pinvJt(Jr)                                # D × d
        iJ = inv(Jc)
        @test maximum(abs(iJ[i, K] - P[i, K]) for i in 1:D, K in 1:d) < 1e-14 * maximum(abs, P.data)
    end
end

@testset "manifold spaces: unsupported face integrals raise a clear error (→ hybrid fallback)" begin
    # boundary/skeleton groups of a space on a manifold would need the face map of a manifold cell:
    # the geometry table refuses them, and `compile_groups` sends such terms to the Gridap fallback
    s = interface_2d()
    dΓ = Measure(s.Γ, 2)
    xq = quad_points(dΓ); w = quad_weights(dΓ)
    @test_throws ErrorException HK.GeoTable(s.Γ, xq, w; Jf=TensorValue{1,1}(1.0))
    geo = HK.GeoTable(s.Γ, xq, w)
    @test geo.dc == 1 && eltype(geo.dM) == VectorValue{2,Float64} && all(g -> g[2] == 0.0, geo.dM)
end

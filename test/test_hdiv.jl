# H(div)-conforming (Raviart–Thomas) fields in the engine: reference tables of the vector shape
# functions, contravariant Piola map per point and per-cell dof signs applied at gather/scatter.
# Every case is checked against pure Gridap (tolerance TOL) and must not use the hybrid fallback.

nofallback(op) = isempty(op.fallback.rtrians) && isempty(op.fallback.jtrians)

"""
Perturbed (non-affine) Cartesian models in 2D and 3D, n-cubes or simplices. The model is converted to
an `UnstructuredDiscreteModel`: Gridap builds the face grids of a `CartesianDiscreteModel` with affine
face maps even when the model has a nonlinear `map`, so its boundary integrals on the curved faces of
hexahedra would not match the (exact, trilinear) geometry of the volume; the unstructured model uses
the bilinear face maps and agrees with the kernels.
"""
function hdiv_model(D::Int; n=D == 2 ? (3, 2) : (2, 2, 2), simplex=false, affine=false)
    model = if D == 2
        affine ? CartesianDiscreteModel((0, 1, 0, 1), n) :
        CartesianDiscreteModel((0, 1, 0, 1), n; map=x -> VectorValue(x[1] + 0.1 * x[2]^2, x[2] + 0.05 * sin(3x[1])))
    else
        affine ? CartesianDiscreteModel((0, 1, 0, 1, 0, 1), n) :
        CartesianDiscreteModel((0, 1, 0, 1, 0, 1), n; map=x -> VectorValue(x[1] + 0.1 * x[2] * x[3], x[2] - 0.05 * x[1]^2, x[3] + 0.05 * x[1] * x[2]))
    end
    simplex ? simplexify(model) : Gridap.Geometry.UnstructuredDiscreteModel(model)
end

"Mixed Darcy problem: flux q in RT_k (Dirichlet q⋅n on tags), pressure p in discontinuous P_k/Q_k."
function darcy(model, order; tags=num_cell_dims(model) == 2 ? [5, 6] : [21, 22])
    D = num_cell_dims(model)
    Ω = Triangulation(model); dΩ = Measure(Ω, 2order + 2)
    Vq = TestFESpace(model, ReferenceFE(raviart_thomas, Float64, order), conformity=:Hdiv, dirichlet_tags=tags)
    Vp = TestFESpace(model, ReferenceFE(lagrangian, Float64, order), conformity=:L2)
    V = MultiFieldFESpace([Vq, Vp])
    U = MultiFieldFESpace([TrialFESpace(Vq, x -> VectorValue(ntuple(i -> 0.1 * i * x[1], D))), TrialFESpace(Vp)])
    f(x) = sin(x[1]) + x[2]
    κ = 2.0
    res((q, p), (v, w)) = ∫(κ * (v ⋅ q) - (∇ ⋅ v) * p + w * (∇ ⋅ q) - w * f)dΩ
    jac((q, p), (dq, dp), (v, w)) = ∫(κ * (v ⋅ dq) - (∇ ⋅ v) * dp + w * (∇ ⋅ dq))dΩ
    (; model, Ω, dΩ, U, V, res, jac)
end

"""
Biot u–W (solid displacement in Lagrangian P1/Q1, relative fluid displacement in RT_k): drained
elasticity, Biot coupling through the divergences, mass-like couplings, a deformation-dependent
permeability (nonlinear in u) and a gradient–gradient term in W (exercises the full ∇W).
"""
function biot_uW(model, order; nonlinear=true)
    D = num_cell_dims(model)
    Ω = Triangulation(model); dΩ = Measure(Ω, 2order + 2)
    Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{D,Float64}, 1), dirichlet_tags=[1, 2])
    VW = TestFESpace(model, ReferenceFE(raviart_thomas, Float64, order), conformity=:Hdiv, dirichlet_tags=[D == 2 ? 7 : 25])
    V = MultiFieldFESpace([Vu, VW])
    U = MultiFieldFESpace([TrialFESpace(Vu, x -> VectorValue(ntuple(i -> 0.01 * x[i], D))),
                           TrialFESpace(VW, x -> VectorValue(ntuple(i -> i == 1 ? 0.02 : 0.0, D)))])
    λ, μ, α, M, ρ, ρf, m, ηκ, γ = 1.5, 1.0, 0.8, 3.0, 2.0, 1.0, 4.0, 0.7, 0.05
    Id = one(TensorValue{D,D,Float64,D * D})
    ε(u) = 0.5 * (∇(u) + ∇(u)')
    σ(u) = λ * tr(ε(u)) * Id + 2μ * ε(u)
    ζ(u, W) = α * (∇ ⋅ u) + (∇ ⋅ W)
    κinv(t) = ηκ / (1 + t)^3
    dκinv(t) = -3ηκ / (1 + t)^4
    invκ(u) = nonlinear ? κinv ∘ tr(∇(u)) : ηκ
    dκ(u, W, du, dW, w) = nonlinear ? (dκinv ∘ tr(∇(u))) * tr(∇(du)) * (w ⋅ W) : 0.0 * (w ⋅ dW)   # bilinear in both cases
    b(x) = VectorValue(ntuple(i -> 0.1 * x[i]^2, D))
    res((u, W), (v, w)) = ∫(ε(v) ⊙ σ(u) + α * M * (∇ ⋅ v) * ζ(u, W) + M * (∇ ⋅ w) * ζ(u, W) +
                            ρ * (v ⋅ u) + ρf * (v ⋅ W) + ρf * (w ⋅ u) + m * (w ⋅ W) +
                            invκ(u) * (w ⋅ W) + γ * (∇(w) ⊙ ∇(W)) - v ⋅ b)dΩ
    jac((u, W), (du, dW), (v, w)) = ∫(ε(v) ⊙ σ(du) + α * M * (∇ ⋅ v) * ζ(du, dW) + M * (∇ ⋅ w) * ζ(du, dW) +
                                      ρ * (v ⋅ du) + ρf * (v ⋅ dW) + ρf * (w ⋅ du) + m * (w ⋅ dW) +
                                      invκ(u) * (w ⋅ dW) + dκ(u, W, du, dW, w) + γ * (∇(w) ⊙ ∇(dW)))dΩ
    (; model, Ω, dΩ, U, V, res, jac)
end

@testset "Piola table against Gridap's basis (values, gradients, signs)" begin
    for D in (2, 3), simplex in (false, true), order in (0, 1)
        model = hdiv_model(D; simplex)
        Ω = Triangulation(model); dΩ = Measure(Ω, 2order + 2)
        V = TestFESpace(model, ReferenceFE(raviart_thomas, Float64, order), conformity=:Hdiv)
        xq = quad_points(dΩ)
        t = HK.field_table(V, xq, D)
        @test t isa HK.PiolaTable
        @test HK.is_piola(V)
        @test !HK.is_piola(TestFESpace(model, ReferenceFE(lagrangian, Float64, 1)))
        nd = length(testitem(get_cell_dof_ids(V)))
        @test t.nn == nd && t.nc == 1 && HK.ndofs(t) == nd
        signs = HK.cell_signs(V, nd)
        @test size(signs) == (nd, num_cells(Ω))
        @test all(s -> s == 1 || s == -1, signs)
        @test any(==(-1), signs)                       # shared facets flip the slave cell
        # physical values and gradients from the tables, Piola per point and the signs
        geo = HK.GeoTable(Ω, xq, quad_weights(dΩ))
        bv = get_fe_basis(V)(get_cell_points(dΩ)); bg = ∇(get_fe_basis(V))(get_cell_points(dΩ))
        W = zeros(nd, D + D * D)
        for cell in (1, num_cells(Ω)), q in 1:length(xq)
            dV, iJt, Jt = HK.point_geometry(geo, q, cell)
            HK.point_weights!(W, t, q, iJt, Jt)
            for j in 1:nd
                s = signs[j, cell]
                v = VectorValue(ntuple(c -> s * W[j, c], D))
                g = TensorValue{D,D}(ntuple(i -> s * W[j, D+i], D * D))
                @test v ≈ bv[cell][q, j] atol = 1e-13
                @test g ≈ bg[cell][q, j] atol = 1e-12
            end
        end
    end
end

@testset "Darcy mixed RT_k–P_k/Q_k: 2D/3D, n-cubes/simplices, non-affine" begin
    for D in (2, 3), simplex in (false, true), order in (D == 2 ? (0, 1, 2) : (0, 1))
        (simplex && D == 3 && order == 1) && continue     # RT1 on tets: the same code, kept out for time
        p = darcy(hdiv_model(D; simplex), order)
        op = @test_logs KernelOperator(p.res, p.jac, p.U, p.V)
        @test nofallback(op)
        @test op.signed
        c = check(p.res, p.jac, p.U, p.V, randx(p.U; scale=1.0); op)
        @test c.res < TOL && c.jac < TOL
    end
end

@testset "Biot u–W: linear and nonlinear, both paths, threads, update, zero allocations" begin
    # (D, simplex, order, nonlinear): the nonlinear permeability (dual-number tangent) on quads RT1 and hexahedra RT0;
    # every new weak form costs about a minute of Gridap compilation in `check`, hence the short list
    for (D, simplex, order, nonlinear) in ((2, false, 1, true), (2, true, 0, false), (3, false, 0, true), (3, true, 0, false))
        p = biot_uW(hdiv_model(D; simplex), order; nonlinear)
        x = randx(p.U; scale=0.1)
        for (linear_maps, nthreads) in ((true, Threads.nthreads()), (false, 1))   # linear maps + threads; dual numbers, serial
            op = KernelOperator(p.res, p.jac, p.U, p.V; linear_maps, nthreads)
            @test nofallback(op)
            c = check(p.res, p.jac, p.U, p.V, x; op)
            @test c.res < TOL && c.jac < TOL
            linear_maps && @test all(a -> a.bytes == 0, allocations(op, x))
        end
    end
    # update: other numbers in the form reuse tables, pattern and positions
    p = biot_uW(hdiv_model(2), 1)
    op = KernelOperator(p.res, p.jac, p.U, p.V)
    res2((u, W), (v, w)) = p.res((u, W), (v, w)) + ∫(0.3 * (w ⋅ W))p.dΩ
    jac2((u, W), (du, dW), (v, w)) = p.jac((u, W), (du, dW), (v, w)) + ∫(0.3 * (w ⋅ dW))p.dΩ
    op2 = HK.update(op, res2, jac2)
    @test op2.A0 === op.A0 && op2.nzpos === op.nzpos
    @test op2.jgroups[1].tables === op.jgroups[1].tables
    c = check(res2, jac2, p.U, p.V, randx(p.U); op=op2)
    @test c.res < TOL && c.jac < TOL
end

@testset "matrix-product and loop paths agree with a Piola field" begin
    p = biot_uW(hdiv_model(3), 1)
    x = randx(p.U)
    op = KernelOperator(p.res, p.jac, p.U, p.V; nthreads=1)
    g = op.jgroups[1]
    @test HK.contiguous(g)
    dvals = [get_dirichlet_dof_values(s) for s in HK.field_spaces(op.U)]
    auxd = HK.materialize_aux(g, 1)
    xl = zeros(op.ndl); HK.gather_cell!(xl, op.fields, x, dvals, 1)
    K1 = zeros(op.ndl, op.ndl); K2 = zeros(op.ndl, op.ndl)
    HK.element_jacobian!(K1, g, xl, auxd, 1, map(t -> zeros(t.nn, HK.wcols(t)), g.tables))
    HK.element_jacobian_mm!(K2, g, xl, auxd, 1, HK.mm_buffers(g)...)
    @test K1 ≈ K2
    gr = op.rgroups[1]
    r1 = zeros(op.ndl); r2 = zeros(op.ndl)
    HK.element_residual!(r1, gr, xl, HK.materialize_aux(gr, 1), 1)
    HK.element_residual_mm!(r2, gr, xl, HK.materialize_aux(gr, 1), 1, HK.rm_buffers(gr)...)
    @test r1 ≈ r2
end

@testset "boundary integrals with an RT field (w ⋅ n, Piola at the face points)" begin
    for D in (2, 3), simplex in (false, true)
        model = hdiv_model(D; simplex)
        order = 1
        Ω = Triangulation(model); dΩ = Measure(Ω, 2order + 2)
        Γ = BoundaryTriangulation(model; tags=D == 2 ? [6, 8] : [22, 26]); dΓ = Measure(Γ, 2order + 2)
        n = get_normal_vector(Γ)
        VW = TestFESpace(model, ReferenceFE(raviart_thomas, Float64, order), conformity=:Hdiv, dirichlet_tags=[D == 2 ? 7 : 25])
        UW = TrialFESpace(VW, x -> VectorValue(ntuple(i -> 0.0, D)))
        pb(x) = 1.0 + x[1]
        res(W, w) = ∫(w ⋅ W + (∇ ⋅ w) * (∇ ⋅ W))dΩ + ∫((w ⋅ n) * pb + 0.5 * (w ⋅ n) * (W ⋅ n))dΓ
        jac(W, dW, w) = ∫(w ⋅ dW + (∇ ⋅ w) * (∇ ⋅ dW))dΩ + ∫(0.5 * (w ⋅ n) * (dW ⋅ n))dΓ
        op = @test_logs KernelOperator(res, jac, UW, VW)
        @test nofallback(op)
        c = check(res, jac, UW, VW, randx(UW; scale=1.0); op)
        @test c.res < TOL && c.jac < TOL
    end
end

@testset "matrix-free Jacobian with RT fields (dense weights, signs)" begin
    for D in (2, 3), simplex in (false, true)
        p = biot_uW(hdiv_model(D; simplex), D == 2 ? 1 : 0)
        x = randx(p.U; scale=0.1)
        for nthreads in (1, Threads.nthreads())
            op = KernelOperator(p.res, p.jac, p.U, p.V; nthreads)
            A = Algebra.jacobian(op, x)
            mf = MatrixFreeJacobian(op)
            prepare!(mf, x)
            dx = randn(num_free_dofs(p.U))
            @test norm(mf * dx - A * dx) / norm(A * dx) < 1e-11
        end
    end
end

@testset "RT on a skeleton: fallback with a clear reason (not supported yet)" begin
    p = darcy(hdiv_model(2), 0)
    Λ = SkeletonTriangulation(p.model); dΛ = Measure(Λ, 2)
    res((q, pp), (v, w)) = p.res((q, pp), (v, w)) + ∫(jump(w) * jump(pp))dΛ + ∫(mean(v ⋅ q) * 1.0)dΛ   # mean(v⋅q): RT on the skeleton
    jac((q, pp), (dq, dp), (v, w)) = p.jac((q, pp), (dq, dp), (v, w)) + ∫(jump(w) * jump(dp))dΛ + ∫(mean(v ⋅ dq) * 1.0)dΛ
    op2 = @test_logs (:warn, r"skeleton") (:warn, r"skeleton") match_mode = :any KernelOperator(res, jac, p.U, p.V)
    @test !isempty(op2.fallback.rtrians)       # the skeleton terms go to Gridap, the volume ones to the kernels
    c = check(res, jac, p.U, p.V, randx(p.U; scale=1.0); op=op2)
    @test c.res < TOL && c.jac < TOL
end

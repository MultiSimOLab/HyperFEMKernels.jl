# FE functions in the weak form (not the unknown): a previous state `uh⁻`, interpolated fields
# (`Nh`, `Bah`), their gradients and nonlinear functions of them. Read as `FEArray` leaves and
# evaluated by the engine from the current nodal values (`fefunctions.jl`); the operations above
# them are compiled with the integrand. Checked against Gridap, including after the nodal values
# change in place, on affine and non-affine meshes and through a sub-triangulation.

fe_aux(op) = [a for g in vcat(op.rgroups, op.jgroups) for a in g.aux if a isa HK.FEArray]
gridap_aux(op) = [a for g in vcat(op.rgroups, op.jgroups) for a in g.aux if !(a isa HK.FEArray)]

"hMRE-like residual: elastic term plus a magnetic load built from ∇(uh⁻)', Nh and Bah through a nonlinear map."
function fe_problem(; n=(4, 3), simplex=false, order=1, perturb=0.0)
    model = CartesianDiscreteModel((0, 1, 0, 1), n)
    simplex && (model = simplexify(model))
    if perturb > 0          # non-affine quadrilaterals: Jt varies inside the cell (interior nodes moved)
        rng = MersenneTwister(3)
        umodel = Gridap.Geometry.UnstructuredDiscreteModel(model)            # Table ids, Vector coords
        grid = get_grid(umodel)
        coords = Gridap.Geometry.get_node_coordinates(grid)
        inner(c) = all(i -> 1e-9 < c[i] < 1 - 1e-9, 1:2)
        new = [inner(c) ? c + perturb * VectorValue(rand(rng) - 0.5, rand(rng) - 0.5) / n[1] : c for c in coords]
        pgrid = Gridap.Geometry.UnstructuredGrid(new, Gridap.Geometry.get_cell_node_ids(grid),
            Gridap.Geometry.get_reffes(grid), Gridap.Geometry.get_cell_type(grid), Gridap.Geometry.Oriented())
        model = Gridap.Geometry.UnstructuredDiscreteModel(pgrid, Gridap.Geometry.get_grid_topology(umodel),
            Gridap.Geometry.get_face_labeling(umodel))                        # same topology and tags
    end
    Ω = Triangulation(model); dΩ = Measure(Ω, 2order)
    V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, order), dirichlet_tags="boundary")
    U = TrialFESpace(V, x -> VectorValue(0.01, -0.02))
    uold = FEFunction(U, 0.05 * randn(num_free_dofs(U)))
    Vn = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, order))
    Nh = interpolate_everywhere(x -> VectorValue(1.0, 0.0), Vn)
    Bah = interpolate_everywhere(x -> VectorValue(0.0, -0.3 * x[1]), Vn)
    Vs = TestFESpace(model, ReferenceFE(lagrangian, Float64, order))
    ρh = interpolate_everywhere(x -> 1.0 + x[1] * x[2], Vs)
    F2(∇u) = one(∇u) + ∇u
    # (no keyword arguments in local functions of the form: a local kwarg closure is called
    # dynamically by Julia and allocates in the element kernel; `allocations(op, x)` reports it)
    invU(F) = (C = transpose(F) ⋅ F + 1e-12 * one(F); s = sqrt(det(C)); inv((C + s * one(F)) / sqrt(tr(C) + 2s)))
    mag(Fn, N, Ba) = -0.7 * (Ba ⊗ (invU(Fn) ⋅ N))
    res(u, v) = ∫(∇(v)' ⊙ (P_svk ∘ (F2 ∘ (∇(u)'))) + ρh * (v ⋅ ∇(ρh)))dΩ +
                0.5 * ∫((mag ∘ (F2 ∘ (∇(uold)'), Nh, Bah)) ⊙ (∇(v)'))dΩ
    jac(u, du, v) = ∫(∇(v)' ⊙ (dP_svk ∘ (F2 ∘ (∇(u)'), ∇(du)')))dΩ
    (; model, Ω, dΩ, U, V, res, jac, uold, Nh, Bah, ρh)
end

@testset "FE functions of the form are engine leaves; values and gradients, scalar and vector" begin
    for simplex in (false, true), order in (1, 2)
        p = fe_problem(; simplex, order)
        op = KernelOperator(p.res, p.jac, p.U, p.V)
        fes = fe_aux(op)
        # ∇(uold)', Nh, Bah, ρh, ∇(ρh): five FE leaves, no Gridap-evaluated auxiliary data left
        @test length(fes) == 5
        @test isempty(gridap_aux(op))
        @test count(a -> HK.fe_deriv(a) == 1, fes) == 2 && count(a -> HK.fe_deriv(a) == 0, fes) == 3
        x = randx(p.U)
        c = check(p.res, p.jac, p.U, p.V, x; op)
        @test c.res < TOL && c.jac < TOL
    end
end

@testset "the nodal values are read at every assembly (updated in place: previous state, other physics)" begin
    p = fe_problem()
    op = KernelOperator(p.res, p.jac, p.U, p.V)
    x = randx(p.U)
    b = Algebra.residual(op, x)
    get_free_dof_values(p.uold) .*= 3.0                       # the previous state of the next load step
    get_free_dof_values(p.Bah) .+= 0.1                        # a field computed by another physics
    b2 = Algebra.residual(op, x)
    @test !(b2 ≈ b)
    @test reldiff(b2, Algebra.residual(gridap_op(p), x)) < TOL
    # dense buffers kept, refilled
    g = op.rgroups[1]
    buf = g.auxcache[]
    Algebra.residual!(b, op, x)
    @test g.auxcache[][1] === buf[1]
end

@testset "non-affine cells (perturbed quadrilaterals): inv(Jt) per point" begin
    p = fe_problem(; n=(5, 4), perturb=0.3)
    op = KernelOperator(p.res, p.jac, p.U, p.V)
    @test length(fe_aux(op)) == 5
    c = check(p.res, p.jac, p.U, p.V, randx(p.U); op)
    @test c.res < TOL && c.jac < TOL
end

@testset "FE function of the parent model integrated on a sub-triangulation (Reindex)" begin
    model = CartesianDiscreteModel((0, 2, 0, 1), (8, 4))
    left = Bool[sum(c)[1] / length(c) < 1.0 for c in vec(collect(get_cell_coordinates(model)))]
    Ωs = Triangulation(model, left); dΩs = Measure(Ωs, 2)
    Vp = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, 1))      # on the whole model
    wh = interpolate_everywhere(x -> VectorValue(x[2], -x[1]), Vp)
    V = TestFESpace(Ωs, ReferenceFE(lagrangian, VectorValue{2,Float64}, 1), dirichlet_tags=[1])
    U = TrialFESpace(V, x -> VectorValue(0.0, 0.0))
    F2(∇u) = one(∇u) + ∇u
    res(u, v) = ∫(∇(v)' ⊙ (P_svk ∘ (F2 ∘ (∇(u)'))) + v ⋅ wh + (∇(v)' ⊙ ∇(wh)'))dΩs
    jac(u, du, v) = ∫(∇(v)' ⊙ (dP_svk ∘ (F2 ∘ (∇(u)'), ∇(du)')))dΩs
    op = KernelOperator(res, jac, U, V)
    fes = fe_aux(op)
    @test length(fes) == 2 && all(a -> a.fe.cellmap !== nothing, fes)
    c = check(res, jac, U, V, randx(U); op)
    @test c.res < TOL && c.jac < TOL
end

@testset "allocations of the assembly do not grow with the mesh (no Gridap evaluation per cell)" begin
    function bytes_per_call(n)
        p = fe_problem(; n)
        op = KernelOperator(p.res, p.jac, p.U, p.V)
        x = randx(p.U); b = Algebra.residual(op, x)
        @test all(a -> a.bytes == 0, allocations(op, x))                 # element kernels do not allocate
        Algebra.residual!(b, op, x)
        @allocated Algebra.residual!(b, op, x)
    end
    small, large = bytes_per_call((6, 4)), bytes_per_call((24, 16))     # 16 × more cells
    @test large < 2 * small + 20_000
end

@testset "a subtree the reader cannot translate stays one Gridap array (no regression)" begin
    p = poisson_2d(; order=1)
    g(x) = sin(x[1])                                                    # a function of x: Gridap data
    res(u, v) = ∫(∇(v) ⋅ ∇(u) - (g * v))p.dΩ
    op = KernelOperator(res, p.jac, p.U, p.V)
    @test isempty(fe_aux(op)) && length(gridap_aux(op)) == 1
    c = check(res, p.jac, p.U, p.V, randx(p.U); op)
    @test c.res < TOL
end

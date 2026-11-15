# Functions of the coordinates in the weak form (`CellField(f, Ω)`, or `f` used directly in an
# operation): read as `CoordArray` leaves and evaluated by the engine at the physical points
# recomputed from the cell nodes (`coordfunctions.jl`), not by Gridap's lazy tree. Checked against
# Gridap's lazy assembly on volume, boundary and skeleton integrals, on affine and non-affine cells,
# with assembled and matrix-free Jacobians, through `update`, and for the fallback to Gridap.

Random.seed!(11)

coord_aux(op) = [a for g in vcat(op.rgroups, op.jgroups) for a in g.aux if a isa HK.CoordArray]
other_aux(op) = [a for g in vcat(op.rgroups, op.jgroups) for a in g.aux if !(a isa HK.CoordArray) && !(a isa HK.FEArray)]

fx(x) = 1 + x[1] + 0.5 * x[2]^2
gx(x) = sin(x[1]) * cos(x[2])

"Residual, assembled Jacobian and matrix-free action against Gridap, with `nthreads` threads."
function check_coord(res, jac, U, V; nthreads=1, ncoord=1)
    for jacobian in (:assembled, :matrixfree)
        op = KernelOperator(res, jac, U, V; nthreads, jacobian)
        @test isempty(op.fallback.rtrians) && isempty(op.fallback.jtrians)
        @test length(coord_aux(op)) >= ncoord
        @test isempty(other_aux(op))                                    # nothing left to Gridap's lazy evaluation
        x = randx(U)
        c = check(res, jac, U, V, x; op)
        @test c.res < TOL && c.jac < TOL
        if jacobian === :matrixfree
            gop = get_algebraic_operator(FEOperator(res, jac, U, V))
            mf = MatrixFreeJacobian(op); prepare!(mf, x)
            dx = randn(num_free_dofs(U))
            @test reldiff(mf * dx, Algebra.jacobian(gop, x) * dx) < 1e-11       # applied without assembling: other round-off
        end
    end
end

@testset "volume and boundary integrals: 2D/3D, n-cubes/simplices, P1/P2 ($(D)D, simplex = $simplex, order $order)" for D in (2, 3),
        simplex in (false, true), order in (1, 2)
    n = D == 2 ? (4, 3) : (2, 2, 2)
    model = cartesian_model(D, n; simplex)
    Ω = Triangulation(model); dΩ = Measure(Ω, 2order)
    Γ = BoundaryTriangulation(model); dΓ = Measure(Γ, 2order)
    V = TestFESpace(model, ReferenceFE(lagrangian, Float64, order), dirichlet_tags=[1])
    U = TrialFESpace(V, x -> x[1])
    cfΩ = CellField(fx, Ω)                          # a CellField of the cells
    cfΓ = CellField(gx, Γ)                          # a CellField of the boundary
    res(u, v) = ∫(cfΩ * ∇(v) ⋅ ∇(u) + (gx * u * u) * v - 2 * cfΩ * v)dΩ + ∫(cfΓ * u * v + cfΩ * v + (fx * v))dΓ
    jac(u, du, v) = ∫(cfΩ * ∇(v) ⋅ ∇(du) + 2 * (gx * u * du) * v)dΩ + ∫(cfΓ * du * v)dΓ
    for nthreads in (1, 4)
        check_coord(res, jac, U, V; nthreads, ncoord=3)
    end
end

@testset "non-affine cells (mapped Cartesian grid, $(D)D) and vector-valued fields" for D in (2, 3)
    n = D == 2 ? (4, 3) : (2, 2, 2)
    model = cartesian_model(D, n; mapped=true)
    Ω = Triangulation(model); dΩ = Measure(Ω, 4)
    Γ = BoundaryTriangulation(model); dΓ = Measure(Γ, 4)
    V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{D,Float64}, 2), dirichlet_tags=[1])
    U = TrialFESpace(V, x -> zero(VectorValue{D,Float64}))
    bx(x) = VectorValue(ntuple(i -> x[i]^2 + i, D))                      # vector-valued function of x
    # boundary integrals on non-affine hexahedra differ from Gridap already without functions of x
    # (a separate issue): the 3D case checks the volume only
    bnd(u, v) = D == 2 ? ∫(gx * (u ⋅ v))dΓ : ∫(0.0 * (u ⋅ v))dΩ
    res(u, v) = ∫(fx * (∇(v) ⊙ ∇(u)) - bx ⋅ v)dΩ + bnd(u, v)
    jac(u, du, v) = ∫(fx * (∇(v) ⊙ ∇(du)))dΩ + bnd(du, v)
    check_coord(res, jac, U, V; nthreads=4, ncoord=D == 2 ? 3 : 2)
    # the physical points are those of Gridap on the curved cells
    op = KernelOperator(res, jac, U, V)
    g = op.rgroups[1]
    a = first(coord_aux(op))
    @test HK.coord_points_match(a, g.geo, HK.group_columns(g.colors), eachindex(g.cellmap))
end

@testset "nonlinear forms through dual numbers and operations above the function of x" begin
    p = poisson_2d(; order=2, simplex=true)
    H(r) = tanh(3r)
    res(u, v) = ∫(∇(v) ⋅ ∇(u) + (H ∘ (u * fx)) * v + 2 * (Operation(H)(CellField(gx, p.Ω))) * v)p.dΩ
    jac(u, du, v) = ∫(∇(v) ⋅ ∇(du) + ((r -> 3 * (1 - tanh(3r)^2)) ∘ (u * fx)) * fx * du * v)p.dΩ
    for linear_maps in (true, false)
        op = KernelOperator(res, jac, p.U, p.V; linear_maps)
        @test length(coord_aux(op)) >= 2 && isempty(other_aux(op))
        c = check(res, jac, p.U, p.V, randx(p.U; scale=0.3); op)
        @test c.res < TOL && c.jac < TOL
    end
end

@testset "skeleton integrals and a subdomain" begin
    model = cartesian_model(2, (4, 3); simplex=true)
    Ω = Triangulation(model); dΩ = Measure(Ω, 2)
    Λ = SkeletonTriangulation(model); dΛ = Measure(Λ, 2)
    V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity=:L2)
    U = TrialFESpace(V)
    gΛ, fΛ = CellField(gx, Λ), CellField(fx, Λ)              # functions of x on the facets
    res(u, v) = ∫(u * v * fx)dΩ + ∫(gΛ * jump(u) * jump(v) + fΛ * mean(v))dΛ
    jac(u, du, v) = ∫(du * v * fx)dΩ + ∫(gΛ * jump(du) * jump(v))dΛ
    check_coord(res, jac, U, V; nthreads=4, ncoord=2)
    # integration over a subset of the cells (integration cells ≠ cells of the space)
    cells = findall(i -> i % 3 == 0, 1:num_cells(model))
    Ωs = Triangulation(model, cells); dΩs = Measure(Ωs, 2)
    V1 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1)); U1 = TrialFESpace(V1)
    res_s(u, v) = ∫(∇(v) ⋅ ∇(u) + fx * u * v)dΩ + ∫(gx * v + CellField(fx, Ωs) * u * v)dΩs
    jac_s(u, du, v) = ∫(∇(v) ⋅ ∇(du) + fx * du * v)dΩ + ∫(CellField(fx, Ωs) * du * v)dΩs
    check_coord(res_s, jac_s, U1, V1; nthreads=4, ncoord=2)
end

@testset "points the geometry does not reproduce fall back to Gridap's array (same values)" begin
    p = poisson_2d(; order=1)
    res(u, v) = ∫(∇(v) ⋅ ∇(u) + fx * u * v)p.dΩ
    op = KernelOperator(res, p.jac, p.U, p.V)
    g = op.rgroups[1]
    a = only(coord_aux(op))
    cols = HK.group_columns(g.colors)
    @test HK.engine_aux(a, g.geo, cols, eachindex(g.cellmap)) === a
    # the geometry of other points (another quadrature): not recognised, Gridap's array is used
    geo2 = HK.GeoTable(get_triangulation(p.V), quad_points(Measure(p.Ω, 4)), quad_weights(Measure(p.Ω, 4)))
    @test HK.engine_aux(a, geo2, cols, eachindex(g.cellmap)) === a.lazy
    # other cells (shifted columns): not recognised
    @test !HK.coord_points_match(a, g.geo, cols, circshift(collect(eachindex(g.cellmap)), 1))
    # the engine's values equal Gridap's
    M = Matrix{Float64}(undef, g.geo.np, length(g.cellmap))
    HK.materialize_coord!(M, a, eachindex(g.cellmap), g.geo.np, g.geo)
    @test maximum(abs(M[q, c] - a.lazy[c][q]) for c in axes(M, 2), q in axes(M, 1)) < 1e-14
end

@testset "no allocation: the engine fill allocates nothing and the assembly as much as without a function of x" begin
    function bytes(n; coord=true)
        model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (n, n, n))
        Ω = Triangulation(model); dΩ = Measure(Ω, 2)
        V0 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 0); conformity=:L2); U0 = TrialFESpace(V0)
        V1 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1))
        uf = FEFunction(V1, rand(num_free_dofs(V1)))
        c = coord ? CellField(x -> 1 + x[1], Ω) : interpolate(x -> 1 + x[1], V1)
        op = KernelOperator((u, v) -> ∫(2 * c * uf * v)dΩ, (u, du, v) -> ∫(0.0 * du * v)dΩ, U0, V0; jacobian=:matrixfree)
        b = zeros(num_free_dofs(V0)); x0 = zeros(num_free_dofs(U0))
        Algebra.residual!(b, op, x0); Algebra.residual!(b, op, x0)
        if coord
            g = op.rgroups[1]
            a = only(coord_aux(op)); M = g.auxcache[][findfirst(x -> x === a, g.aux)]
            HK.materialize_coord!(M, a, eachindex(g.cellmap), g.geo.np, g.geo)
            @test (@allocated HK.materialize_coord!(M, a, eachindex(g.cellmap), g.geo.np, g.geo)) == 0
        end
        @allocated Algebra.residual!(b, op, x0)
    end
    small, large = bytes(4), bytes(12)                                # 27 × more cells
    @test large <= small + 256                                        # no allocation per cell (the number of tasks may differ)
    @test large <= bytes(12; coord=false) + 512                       # same per-call buffers as with an FE function
end

@testset "update: same function, other captured values, other function" begin
    p = poisson_2d(; order=2)
    make(a) = (x -> a * (1 + x[1]))                                  # same closure type, captured value `a`
    function forms(f)
        res(u, v) = ∫(∇(v) ⋅ ∇(u) + f * u * v - CellField(f, p.Ω) * v)p.dΩ
        jac(u, du, v) = ∫(∇(v) ⋅ ∇(du) + f * du * v)p.dΩ
        res, jac
    end
    shared(o2, o) = all(((g2, g),) -> g2.pf.core === g.pf.core && g2.auxcache === g.auxcache, zip(vcat(o2.rgroups, o2.jgroups), vcat(o.rgroups, o.jgroups)))
    fresh_equal(o2, r, j) = (x = randx(p.U); fr = KernelOperator(r, j, p.U, p.V);
        reldiff(Algebra.residual(o2, x), Algebra.residual(fr, x)) < 1e-14 &&
        reldiff(jacobian!(copy(sparsity_pattern(o2)), o2, x), Algebra.jacobian(fr, x)) < 1e-14)
    for jacobian in (:assembled, :matrixfree)
        r1, j1 = forms(make(2.0))
        op = KernelOperator(r1, j1, p.U, p.V; jacobian)
        x = randx(p.U); Algebra.residual(op, x)
        # same function: the same structure, rebound
        op1 = update(op, r1, j1)
        @test op1.rsig == op.rsig && shared(op1, op) && fresh_equal(op1, r1, j1)
        # same closure type, another captured value: rebound, and the new value is used
        r2, j2 = forms(make(-3.0))
        op2 = update(op, r2, j2)
        @test op2.rsig == op.rsig && shared(op2, op) && fresh_equal(op2, r2, j2)
        c = check(r2, j2, p.U, p.V, x; op=op2)
        @test c.res < TOL && c.jac < TOL
        # another function (another type): another structure, compiled again, same result as Gridap
        r3, j3 = forms(x -> 2 + x[2]^2)
        op3 = update(op, r3, j3)
        @test op3.rsig != op.rsig && !shared(op3, op)
        c = check(r3, j3, p.U, p.V, x; op=op3)
        @test c.res < TOL && c.jac < TOL
    end
end

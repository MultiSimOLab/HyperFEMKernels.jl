# Internal variables updated by the engine (`update_states!`, `KernelState`) against Gridap's `update_state!`.
# Every case: the states written by the kernel equal those written by Gridap (same updater, copies of the states).

using Gridap.Geometry

dense(s::CellState) = reduce(hcat, s.values)         # np × ncells
dense(s::KernelState) = s.values
same(a, b) = maximum(norm, dense(a) - dense(b)) <= 1e-12 * max(1.0, maximum(norm, dense(b)))
no_aux(su) = su !== nothing && isempty(su.aux)

# Gridap's update_state! only writes CellStates: a copy of a state as a CellState on the same points
function CellState_from(k::KernelState)
    c = CellState(k.values[1, 1], k.points)
    for cell in axes(k.values, 2), q in axes(k.values, 1)
        c.values[cell][q] = k.values[q, cell]
    end
    c
end
function CellState_from(s::CellState)
    c = CellState(s.values[1][1], s.points)          # deepcopy would copy the points (and the model)
    for cell in eachindex(c.values)
        c.values[cell] .= s.values[cell]
    end
    c
end
# Gridap writes the reference states (CellState copies); the kernel writes the originals; then compare.
function check_update(updater, states, fields...; kw...)
    refs = map(CellState_from, states)
    update_state!(updater, refs..., fields...)
    su = update_states!(updater, states..., fields...; kw...)
    ok = all(((s, r),) -> same(s, r), zip(states, refs))
    su, ok
end

function scalar_setup(; n=(4, 3), order=1, mapped=false, simplex=false, D=2)
    dom = D == 2 ? (0, 1, 0, 1) : (0, 1, 0, 1, 0, 1)
    cells = D == 2 ? n : (n..., 2)
    model = if mapped
        f(x) = D == 2 ? VectorValue(x[1] + 0.1 * x[1] * x[2], x[2] + 0.05 * x[1]^2) :
                        VectorValue(x[1] + 0.1 * x[1] * x[2], x[2] + 0.05 * x[3]^2, x[3] - 0.07 * x[1] * x[2])
        CartesianDiscreteModel(dom, cells; map=f)
    else
        CartesianDiscreteModel(dom, cells)
    end
    simplex && (model = simplexify(model))
    Ω = Triangulation(model); dΩ = Measure(Ω, 2order)
    V = TestFESpace(model, ReferenceFE(lagrangian, Float64, order), dirichlet_tags="boundary")
    U = TrialFESpace(V, x -> 1 + x[1] + 2x[2])
    uh = FEFunction(U, randn(num_free_dofs(U)))
    (; model, Ω, dΩ, U, V, uh)
end

@testset "KernelState: construction, evaluation, in a weak form" begin
    s = scalar_setup()
    k = KernelState(2.5, s.dΩ)
    c = CellState(2.5, s.dΩ)
    np = length(quad_points(s.dΩ))
    @test size(k.values) == (np, num_cells(s.Ω)) && all(==(2.5), k.values)
    @test get_triangulation(k) === s.Ω
    @test same(k, c)
    x = get_cell_points(s.dΩ)
    cols = evaluate(k, x)
    @test length(cols) == num_cells(s.Ω) && cols[3] == fill(2.5, np)
    k.values[2, 3] = -1.0
    @test cols[3][2] == -1.0                          # a view of the storage
    @test_throws ErrorException evaluate(k, get_cell_points(Measure(s.Ω, 4)))
    @test_throws ErrorException Gridap.CellData.get_data(k)
    kc = KernelState(c)                               # copy of a CellState
    @test same(kc, c) && kc.points === c.points
    ku = KernelState{Float64}(undef, s.dΩ)
    @test size(ku.values) == size(k.values)
    # the same state in a weak form, both storages: same residual and Jacobian; no copy for the KernelState
    for cell in 1:num_cells(s.Ω), q in 1:np
        v = 0.3 * cell + 0.1 * q
        c.values[cell][q] = v; k.values[q, cell] = v
    end
    g(a, u) = (1 + a^2) * u
    dg(a, du) = (1 + a^2) * du
    resc(u, v) = ∫((g ∘ (c, u)) * v + ∇(u) ⋅ ∇(v))s.dΩ
    jacc(u, du, v) = ∫((dg ∘ (c, du)) * v + ∇(du) ⋅ ∇(v))s.dΩ
    resk(u, v) = ∫((g ∘ (k, u)) * v + ∇(u) ⋅ ∇(v))s.dΩ
    jack(u, du, v) = ∫((dg ∘ (k, du)) * v + ∇(du) ⋅ ∇(v))s.dΩ
    x0 = randx(s.U; scale=0.5)
    opc = KernelOperator(resc, jacc, s.U, s.V); opk = KernelOperator(resk, jack, s.U, s.V)
    @test reldiff(Algebra.residual(opk, x0), Algebra.residual(opc, x0)) < TOL
    @test reldiff(Algebra.jacobian(opk, x0), Algebra.jacobian(opc, x0)) < TOL
    ck = check(resk, jack, s.U, s.V, x0; op=opk)
    @test ck.res < TOL && ck.jac < TOL
    @test any(a -> a isa HK.StateColumns, opk.rgroups[1].aux)
    @test HK.materialize_aux(opk.rgroups[1], 1)[1] === k.values
end

@testset "scalar field: value and gradient, both storages, meshes and orders" begin
    upd(a, u, g) = (true, a + u * (g ⋅ g) - 0.5 * u^2)
    for (mapped, simplex, order, D) in ((false, false, 1, 2), (true, false, 2, 2), (false, true, 1, 2), (true, false, 1, 3), (false, true, 2, 3))
        s = scalar_setup(; order, mapped, simplex, D)
        for storage in (CellState, KernelState)
            a = storage(0.0, s.dΩ)
            su, ok = check_update(upd, (a,), s.uh, ∇(s.uh))
            @test ok
            @test no_aux(su) && length(su.fes) == 2        # value and gradient leaves, nothing left to Gridap
            @test su.geo !== nothing
            @test allocations(su) == 0
            # a second step reads the state it wrote; different FE function object, same dof ids
            uh2 = FEFunction(s.U, randn(num_free_dofs(s.U)))
            su2, ok2 = check_update(upd, (a,), uh2, ∇(uh2))
            @test ok2
            @test su2.expr == su.expr                      # same generated function (no recompilation)
        end
    end
end

@testset "value only, gradient only, constants, expressions, flag false" begin
    s = scalar_setup(; order=2, mapped=true)
    x = get_cell_points(s.dΩ)                         # Gridap requires the states of one call on the same CellPoint object
    a = KernelState(1.0, x); b = CellState(1.0, x)
    su, ok = check_update((a, u) -> (true, a * u), (a,), s.uh)
    @test ok && no_aux(su) && su.geo === nothing && length(su.fes) == 1
    su, ok = check_update((a, g) -> (true, a + g ⋅ g), (a,), ∇(s.uh))
    @test ok && no_aux(su) && su.geo !== nothing
    # numbers and compositions in the arguments (a number of the expression is a runtime parameter)
    h(g) = exp(-g ⋅ g)
    su, ok = check_update((a, w) -> (true, a + w), (a,), 2.0 * (h ∘ ∇(s.uh)) + 3.0)
    @test ok && no_aux(su)
    su2, ok2 = check_update((a, w) -> (true, a + w), (a,), 5.0 * (h ∘ ∇(s.uh)) + 3.0)
    @test ok2 && su2.expr == su.expr
    # the updater says no update: nothing changes
    before = copy(dense(a))
    su, ok = check_update((a, u) -> (false, a + u), (a,), s.uh)
    @test ok && dense(a) == before
    # a state used in an expression, and a state read but not written
    su, ok = check_update((a, w) -> (true, a + w), (a,), b * s.uh + a)
    @test ok && length(su.aux) == 1                   # b is not an argument: read through Gridap
    @test length(su.states) == 1 && HK.nwritten(su) == 1
    su, ok = check_update((a, bb, u) -> (true, a + bb * u), (a,), b, s.uh)
    @test ok && no_aux(su) && HK.nwritten(su) == 2 && length(su.states) == 2   # b: a state argument that is only read
end

@testset "several states written, mixed storages, order of the arguments" begin
    s = scalar_setup(; mapped=true)
    x = get_cell_points(s.dΩ)
    a = CellState(0.0, x); b = KernelState(1.0, x); c = CellState(-1.0, x)
    upd3(a, b, c, u, g) = (true, a + u, b * (1 + c), c + g ⋅ g)
    refs = (CellState_from(a), CellState_from(b), CellState_from(c))
    update_state!(upd3, refs..., s.uh, ∇(s.uh))
    su = update_states!(upd3, a, b, c, s.uh, ∇(s.uh))
    @test same(a, refs[1]) && same(b, refs[2]) && same(c, refs[3])
    @test HK.nwritten(su) == 3 && no_aux(su)
    @test allocations(su) == 0
    # states interleaved with fields: the state arguments are written in their order (Gridap's loop writes
    # the first arguments whatever they are, so it is not the reference here)
    a2 = CellState(0.0, x); b2 = KernelState(1.0, x); c2 = CellState(-1.0, x)
    upd3i(a, b, u, c, g) = (true, a + u, b * (1 + c), c + g ⋅ g)
    update_states!(upd3i, a2, b2, s.uh, c2, ∇(s.uh))
    @test same(a2, a) && same(b2, b) && same(c2, c)
    # fewer states returned than state arguments: the first ones are written, the others are inputs (as in Gridap)
    refs = (CellState_from(a), CellState_from(b), CellState_from(c))
    upd2(a, b, c, u, g) = (true, a - c * u, b + g ⋅ g)
    update_state!(upd2, refs..., s.uh, ∇(s.uh))
    update_states!(upd2, a, b, c, s.uh, ∇(s.uh))
    @test same(a, refs[1]) && same(b, refs[2]) && same(c, refs[3])
    # more states returned than state arguments
    @test_throws Exception update_states!((a, b, c, u, g) -> (true, a, b, c, a), a, b, c, s.uh, ∇(s.uh))
    @test_throws ErrorException update_states!((u,) -> (true, u), s.uh)
end

@testset "vector field 3D with transposes and user functions, nonzero Dirichlet values" begin
    model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (3, 2, 2); map=x -> VectorValue(x[1] + 0.1x[2] * x[3], x[2], x[3] - 0.05x[1]^2))
    Ω = Triangulation(model); dΩ = Measure(Ω, 2)
    V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), dirichlet_tags=[1, 3, 5, 7])
    U = TrialFESpace(V, x -> VectorValue(0.2 + x[2], 0.1, -0.3x[1]))
    uh = FEFunction(U, 0.1 .* randn(num_free_dofs(U))); uh⁻ = FEFunction(U, 0.1 .* randn(num_free_dofs(U)))
    x = get_cell_points(dΩ)
    for storage in (CellState, KernelState)
        A = storage(one(TensorValue{3,3,Float64,9}), x)
        P = storage(zero(TensorValue{3,3,Float64,9}), x)
        upd(A, F, Fn) = (true, 0.5 * (A + F ⋅ inv(Fn)))
        su, ok = check_update(upd, (A,), Fk ∘ ∇(uh)', Fk ∘ ∇(uh⁻)')
        @test ok && no_aux(su) && length(su.fes) == 2 && allocations(su) == 0
        # two states from the same inputs, as in HyperFEM's visco problems (P from the law, A from the branch)
        updP(P, F, Fn, A) = (true, P_svk(F) + A ⋅ Fn)
        su, ok = check_update(updP, (P,), Fk ∘ ∇(uh)', Fk ∘ ∇(uh⁻)', A)
        @test ok && no_aux(su) && HK.nwritten(su) == 2 && length(su.states) == 2
        # the value of the field as well
        su, ok = check_update((A, u, F) -> (true, A + u ⊗ u + F), (A,), uh, Fk ∘ ∇(uh)')
        @test ok && no_aux(su) && length(su.fes) == 2
    end
end

@testset "multi-field: FE functions of different spaces" begin
    p = coupled_3d()
    xh = FEFunction(p.U, randx(p.U; scale=0.1))
    uh, φh = xh
    for storage in (CellState, KernelState)
        A = storage(0.0, p.dΩ)
        upd(a, u, φ, F, E) = (true, a + (u ⋅ u) * φ + tr(F) + E ⋅ E)
        su, ok = check_update(upd, (A,), uh, φh, Fk ∘ ∇(uh)', -∇(φh))
        @test ok && no_aux(su) && length(su.fes) == 4
        @test allocations(su) == 0
    end
end

@testset "states on a sub-triangulation of the space (cell map), and spaces on a sub-triangulation" begin
    s = two_subdomains()
    dΩa = Measure(s.Ωa, 2)
    V = TestFESpace(s.model, ReferenceFE(lagrangian, Float64, 1), dirichlet_tags=[7]); U = TrialFESpace(V, x -> x[2])
    uh = FEFunction(U, randn(num_free_dofs(U)))
    for storage in (CellState, KernelState)
        a = storage(0.0, dΩa)
        su, ok = check_update((a, u, g) -> (true, a + u * (g ⋅ g)), (a,), uh, ∇(uh))
        @test ok && no_aux(su) && all(f -> f.cellmap !== nothing, su.fes)
    end
    Vd = TestFESpace(s.Ωd, ReferenceFE(lagrangian, Float64, 2), dirichlet_tags=[7]); Ud = TrialFESpace(Vd, x -> x[2])
    ud = FEFunction(Ud, randn(num_free_dofs(Ud)))
    for storage in (CellState, KernelState)
        a = storage(0.0, dΩa)
        su, ok = check_update((a, u, g) -> (true, a + u * (g ⋅ g)), (a,), ud, ∇(ud))
        @test ok && no_aux(su)
    end
end

@testset "data the engine does not read goes through Gridap (auxiliary), fallback to Gridap's loop" begin
    s = scalar_setup(; mapped=true)
    a = CellState(0.0, s.dΩ)
    f = CellField(x -> sin(x[1]) * x[2], s.Ω)                    # function of x: auxiliary data
    su, ok = check_update((a, u, w) -> (true, a + u * w), (a,), s.uh, f)
    @test ok && length(su.aux) == 1 && length(su.fes) == 1
    # an interpolated field of another space on the same cells is read from its dofs
    W = FESpace(s.model, ReferenceFE(lagrangian, VectorValue{2,Float64}, 2))
    wh = interpolate_everywhere(x -> VectorValue(x[2], -x[1]), W)
    su, ok = check_update((a, u, w) -> (true, a + u * (w ⋅ w)), (a,), s.uh, wh)
    @test ok && no_aux(su) && length(su.fes) == 2
    # states on a boundary are not compiled yet: Gridap's algorithm, with a warning, same result
    Γ = BoundaryTriangulation(s.model); dΓ = Measure(Γ, 2)
    b = KernelState(0.0, dΓ)
    rb = CellState_from(b)
    update_state!((b, u) -> (true, b + u), rb, s.uh)
    @test_logs (:warn, r"Gridap's algorithm") match_mode = :any update_states!((b, u) -> (true, b + u), b, s.uh)
    @test same(b, rb)
end

@testset "threads vs serial, KernelUpdater and dispatch of update_state!" begin
    s = scalar_setup(; n=(12, 9), order=2, mapped=true)
    upd(a, u, g) = (true, a + u * (g ⋅ g))
    a1 = KernelState(0.0, s.dΩ); a2 = KernelState(0.0, s.dΩ); a3 = CellState(0.0, s.dΩ); a4 = KernelState(0.0, s.dΩ)
    update_states!(upd, a1, s.uh, ∇(s.uh); nthreads=1)
    update_states!(upd, a2, s.uh, ∇(s.uh); nthreads=4)
    @test same(a1, a2)
    update_state!(KernelUpdater(upd), a3, s.uh, ∇(s.uh))          # Gridap's name, the engine's loop
    @test same(a1, a3)
    update_state!(upd, a4, s.uh, ∇(s.uh))                         # a KernelState first: the engine's loop
    @test same(a1, a4)
    @test KernelUpdater(upd)(1.0, 2.0, VectorValue(1.0, 1.0)) == upd(1.0, 2.0, VectorValue(1.0, 1.0))
end

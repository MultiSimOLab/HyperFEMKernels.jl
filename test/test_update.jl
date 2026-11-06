# Cheap `update`: a capture with the structure of the operator's forms (same measures, operations,
# constants up to their values, FE functions) rebinds the compiled groups instead of compiling them
# again. Every case: residual and Jacobian of the updated operator equal (≤ 1e-14 relative) to those of a
# KernelOperator built from scratch with the new forms; the fast path is recognised by the shared
# compiled functions and auxiliary buffers; a changed structure takes the full path with the same result.

Random.seed!(5)
const UTOL = 1e-14

"Same residual and Jacobian as an operator built from scratch; the fast path shares cores and buffers (`fast = true`)."
function same_as_fresh(op2, res, jac, U, V; fast=true, matrixfree=false)
    fresh = KernelOperator(res, jac, U, V; jacobian=op2.jacobian)
    x = randx(U)
    @test reldiff(Algebra.residual(op2, x), Algebra.residual(fresh, x)) < UTOL
    A2 = jacobian!(copy(sparsity_pattern(op2)), op2, x)
    Af = jacobian!(copy(sparsity_pattern(fresh)), fresh, x)
    @test reldiff(A2, Af) < UTOL
    if matrixfree
        mf = MatrixFreeJacobian(op2); prepare!(mf, x)
        dx = randn(num_free_dofs(U))
        @test reldiff(mf * dx, Af * dx) < 1e-12
    end
    fresh
end
shared(op2, op) = all(((g2, g),) -> g2.pf.core === g.pf.core && g2.auxcache === g.auxcache && g2.tables === g.tables,
                      zip(vcat(op2.rgroups, op2.jgroups), vcat(op.rgroups, op.jgroups)))

@testset "fast path: numbers by value, by Ref, auxiliary FE function changed (volume + boundary + skeleton, ∇∇, multi-field)" begin
    model = cartesian_model(2, (4, 3); simplex=true)
    Ω = Triangulation(model); dΩ = Measure(Ω, 4)
    Γ = BoundaryTriangulation(model); dΓ = Measure(Γ, 4); n = get_normal_vector(Γ)
    Λ = SkeletonTriangulation(model); dΛ = Measure(Λ, 4)
    Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, 2), dirichlet_tags="boundary")
    Vφ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 2), dirichlet_tags="boundary")
    V = MultiFieldFESpace([Vu, Vφ])
    U = MultiFieldFESpace([TrialFESpace(Vu, x -> VectorValue(0.01x[2], 0.0)), TrialFESpace(Vφ, x -> x[1])])
    Vw = TestFESpace(model, ReferenceFE(lagrangian, Float64, 2))
    wh = interpolate(x -> x[1] * x[2], Vw)                 # auxiliary FE function (changed in place below)
    αr = Ref(0.3)                                           # a parameter read through a Ref inside an operation (HyperFEM's Magnetic)
    load(x) = αr[] * x
    function forms(β, Λf)                                    # β by value (captured), Λf a load factor
        res((u, φ), (v, vφ)) = ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))) + β * (∇∇(u) ⊙ ∇∇(v)) + ∇(vφ) ⋅ ∇(φ) + (load ∘ φ) * vφ * wh + Δ(φ) * tr(∇(v)))dΩ +
                                 Λf * ∫((v ⋅ n) * (v ⋅ n) + vφ * (φ * φ))dΓ + ∫(jump(φ) * jump(vφ) * (1.0 + Λf))dΛ
        jac((u, φ), (du, dφ), (v, vφ)) = ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')) + β * (∇∇(du) ⊙ ∇∇(v)) + ∇(vφ) ⋅ ∇(dφ) + (load ∘ dφ) * vφ * wh + Δ(dφ) * tr(∇(v)))dΩ +
                                           Λf * ∫(2.0 * (v ⋅ n) * (v ⋅ n) + 2.0 * (vφ * (φ * dφ)))dΓ + ∫(jump(dφ) * jump(vφ) * (1.0 + Λf))dΛ
        res, jac
    end
    for jacobian in (:assembled, :matrixfree)
        r1, j1 = forms(5e-3, 0.5)
        op = KernelOperator(r1, j1, U, V; jacobian)
        @test isempty(op.fallback.rtrians) && length(op.rgroups) >= 3
        same_as_fresh(op, r1, j1, U, V; matrixfree=jacobian === :matrixfree)
        # numbers by value and the load factor (coefficient and parameter)
        r2, j2 = forms(0.2, 1.5)
        op2 = update(op, r2, j2)
        @test shared(op2, op) && op2.pattern === op.pattern && op2.fallback === op.fallback
        same_as_fresh(op2, r2, j2, U, V; matrixfree=jacobian === :matrixfree)
        # through a Ref: no capture needed, the operation reads it
        αr[] = -0.7
        same_as_fresh(op2, r2, j2, U, V)
        # the auxiliary FE function changed in place: read at the next assembly
        get_free_dof_values(wh) .*= 3.0
        same_as_fresh(op2, r2, j2, U, V)
        # a third update: still the fast path; the matrix-free Jacobian is rebound (buffers reused)
        r3, j3 = forms(1.0, 0.1)
        op3 = update(op2, r3, j3)
        @test shared(op3, op) && op3.rsig == op.rsig && op3.jsig == op.jsig
        same_as_fresh(op3, r3, j3, U, V; matrixfree=jacobian === :matrixfree)
        if jacobian === :matrixfree
            mf = MatrixFreeJacobian(op)
            x = randx(U)
            jacobian!(mf, op3, x)
            @test mf.op === op3
            dx = randn(num_free_dofs(U))
            @test reldiff(mf * dx, jacobian!(copy(sparsity_pattern(op3)), op3, x) * dx) < 1e-12
        end
        αr[] = 0.3; get_free_dof_values(wh) ./= 3.0
    end
end

@testset "structure changes take the full path, with the same result as a fresh operator" begin
    model = cartesian_model(2, (4, 3))
    Ω = Triangulation(model); dΩ = Measure(Ω, 4)
    Ω1 = Triangulation(model, 1:6); dΩ1 = Measure(Ω1, 4)
    V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, 2), dirichlet_tags="boundary")
    U = TrialFESpace(V, x -> VectorValue(0.0, 0.01x[1]))
    base(u, v) = ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))))dΩ
    basej(u, du, v) = ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')))dΩ
    op = KernelOperator(base, basej, U, V)
    cases = [
        # a term added
        ((u, v) -> base(u, v) + 5e-3 * ∫(∇∇(u) ⊙ ∇∇(v))dΩ, (u, du, v) -> basej(u, du, v) + 5e-3 * ∫(∇∇(du) ⊙ ∇∇(v))dΩ),
        # another triangulation
        ((u, v) -> base(u, v) + ∫(v ⋅ u)dΩ1, (u, du, v) -> basej(u, du, v) + ∫(v ⋅ du)dΩ1),
        # another constitutive law (another operation type)
        ((u, v) -> ∫(∇(v)' ⊙ (Fk ∘ (∇(u)')))dΩ, (u, du, v) -> ∫(∇(v)' ⊙ (∇(du)'))dΩ),
        # another type of constant (an Int instead of a Float64)
        ((u, v) -> base(u, v) + ∫(2 * (v ⋅ u))dΩ, (u, du, v) -> basej(u, du, v) + ∫(2 * (v ⋅ du))dΩ),
    ]
    for (r, j) in cases
        op2 = update(op, r, j)
        @test !(op2.rsig == op.rsig && op2.jsig == op.jsig)
        same_as_fresh(op2, r, j, U, V)
        # and back to the original structure: fast path from op2's point of view is not available either
        op3 = update(op2, base, basej)
        @test op3.rsig == op.rsig && op3.jsig == op.jsig
        same_as_fresh(op3, base, basej, U, V)
    end
    # the same structure with a different value of a Float64 constant stays on the fast path
    op4 = update(op, (u, v) -> base(u, v) + ∫(2.0 * (v ⋅ u))dΩ, (u, du, v) -> basej(u, du, v) + ∫(2.0 * (v ⋅ du))dΩ)
    op5 = update(op4, (u, v) -> base(u, v) + ∫(3.0 * (v ⋅ u))dΩ, (u, du, v) -> basej(u, du, v) + ∫(3.0 * (v ⋅ du))dΩ)
    @test shared(op5, op4)
    same_as_fresh(op5, (u, v) -> base(u, v) + ∫(3.0 * (v ⋅ u))dΩ, (u, du, v) -> basej(u, du, v) + ∫(3.0 * (v ⋅ du))dΩ, U, V)
end

@testset "fast path with linear maps: the map values follow the parameters" begin
    # the linear maps' values depend on the parameters (M is computed with them): a parameter change on
    # the fast path must re-evaluate them (here a coefficient that multiplies a state input nonlinearly in u)
    model = cartesian_model(3, (2, 2, 2))
    Ω = Triangulation(model); dΩ = Measure(Ω, 2)
    Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), dirichlet_tags=[1, 2])
    Vφ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), dirichlet_tags=[7, 8])
    V = MultiFieldFESpace([Vu, Vφ]); U = MultiFieldFESpace([TrialFESpace(Vu, x -> VectorValue(0.0, 0.0, 0.0)), TrialFESpace(Vφ, x -> 0.1)])
    mk(β) = (((u, φ), (v, vφ)) -> ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))) + β * φ * tr(∇(v)) + ∇(vφ) ⋅ ∇(φ) + β * vφ * (tr ∘ (Fk ∘ (∇(u)'))))dΩ,
             ((u, φ), (du, dφ), (v, vφ)) -> ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')) + β * dφ * tr(∇(v)) + ∇(vφ) ⋅ ∇(dφ) + β * vφ * tr(∇(du)))dΩ)
    r1, j1 = mk(0.3)
    op = KernelOperator(r1, j1, U, V)
    @test all(g -> g.lin !== nothing, op.rgroups)
    r2, j2 = mk(-1.7)
    op2 = update(op, r2, j2)
    @test shared(op2, op) && all(((g2, g),) -> g2.lin.apply === g.lin.apply && g2.lin.state === g.lin.state, zip(op2.rgroups, op.rgroups))
    @test any(((g2, g),) -> g2.lin.vals != g.lin.vals, zip(op2.rgroups, op.rgroups))
    same_as_fresh(op2, r2, j2, U, V)
    # with the check against a full compilation (debug option)
    withenv("HFK_CHECK_REBIND" => "1") do
        op3 = update(op2, r1, j1)
        same_as_fresh(op3, r1, j1, U, V)
    end
end

@testset "the fast path allocates independently of the mesh size (no tables, no auxiliary buffers)" begin
    bytes = Int[]
    for n in ((4, 3), (16, 12))
        model = cartesian_model(2, n)
        Ω = Triangulation(model); dΩ = Measure(Ω, 4)
        V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, 2), dirichlet_tags="boundary")
        U = TrialFESpace(V, x -> VectorValue(0.0, 0.0))
        Vw = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1))
        wh = interpolate(x -> x[1], Vw)
        mk(β) = ((u, v) -> ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))) + β * (∇∇(u) ⊙ ∇∇(v)) + wh * (v ⋅ u))dΩ,
                 (u, du, v) -> ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')) + β * (∇∇(du) ⊙ ∇∇(v)) + wh * (v ⋅ du))dΩ)
        r1, j1 = mk(1.0); r2, j2 = mk(2.0)
        op = KernelOperator(r1, j1, U, V)
        x = randx(U)
        Algebra.residual(op, x); Algebra.jacobian(op, x)        # auxiliary buffers allocated
        update(op, r2, j2)
        b = @allocated update(op, r2, j2)
        push!(bytes, b)
    end
    @test bytes[2] < 2 * bytes[1]            # 16× more cells: not 16× more bytes
end

@testset "KernelFunctional: fast update, same values as a fresh functional" begin
    model = cartesian_model(2, (4, 3); simplex=true)
    Ω = Triangulation(model); dΩ = Measure(Ω, 4)
    Γ = BoundaryTriangulation(model); dΓ = Measure(Γ, 4)
    V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, 2), dirichlet_tags="boundary")
    U = TrialFESpace(V, x -> VectorValue(0.0, 0.01x[1]))
    duh = FEFunction(U, randn(num_free_dofs(U)))
    Ψ(F) = 0.5 * (transpose(F) ⋅ F - one(F)) ⊙ (transpose(F) ⋅ F - one(F))
    mk(β, Λ) = u -> ∫(Ψ ∘ (Fk ∘ (∇(u)')) + β * (∇∇(u) ⊙ ∇∇(u)))dΩ + Λ * ∫(u ⋅ u)dΓ
    fun = KernelFunctional(mk(0.1, 0.5), U)
    fun2 = update(fun, mk(0.4, -2.0))
    @test all(((g2, g),) -> g2.pf.core === g.pf.core && g2.auxcache === g.auxcache, zip(fun2.groups, fun.groups))
    x = randx(U)
    @test abs(fun2(x) - KernelFunctional(mk(0.4, -2.0), U)(x)) <= 1e-14 * abs(fun2(x))
    # pointwise extremum with the increment in an FE function, changed in place
    α(∇u, ∇du) = 1.0 + 0.1 * (∇u ⊙ ∇du)
    fmin = KernelFunctional(u -> ∫(α ∘ (∇(u)', ∇(duh)'))dΩ, U; reduce=:min)
    fmin2 = update(fmin, u -> ∫(α ∘ (∇(u)', ∇(duh)'))dΩ)
    @test fmin2.groups[1].pf.core === fmin.groups[1].pf.core
    get_free_dof_values(duh) .*= 2.0
    @test fmin2(x) == KernelFunctional(u -> ∫(α ∘ (∇(u)', ∇(duh)'))dΩ, U; reduce=:min)(x)
    # another structure: full path
    fun3 = update(fun, u -> ∫(Ψ ∘ (Fk ∘ (∇(u)')))dΩ)
    @test fun3.sig != fun.sig && abs(fun3(x) - KernelFunctional(u -> ∫(Ψ ∘ (Fk ∘ (∇(u)')))dΩ, U)(x)) <= 1e-14 * abs(fun3(x))
end

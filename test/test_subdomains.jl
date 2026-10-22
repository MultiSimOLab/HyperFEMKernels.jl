# Configuration of the hMRE + vacuum example (examples/hmre_vacuum_2d): FE spaces on a sub-triangulation of
# the model, integrals on parts of it with different integrands, the boundary of a subdomain as an internal
# interface (faces interior to the model), and an auxiliary field that is an FE function of a space defined on
# that interface. Every case against pure Gridap (tolerance TOL), without hybrid fallback.

nofallback_sub(op) = isempty(op.fallback.rtrians) && isempty(op.fallback.jtrians)

"Model (0,2)×(0,1); active region Ωd without the top row of cells, split at x = 1 into Ωa (left) and Ωb (right)."
function two_subdomains(; n=(8, 4))
    model = CartesianDiscreteModel((0, 2, 0, 1), n)
    cc = [sum(c) / length(c) for c in get_cell_coordinates(model)]
    dom = [x[2] < 0.75 for x in cc]
    Ωd = Triangulation(model, dom)
    Ωa = Triangulation(model, [d && x[1] < 1 for (d, x) in zip(dom, cc)])
    Ωb = Triangulation(model, [d && x[1] >= 1 for (d, x) in zip(dom, cc)])
    (; model, Ωd, Ωa, Ωb)
end

κa(g) = (1 + g ⋅ g) * g                    # nonlinear "solid" law on Ωa
dκa(g, dg) = (1 + g ⋅ g) * dg + 2 * (g ⋅ dg) * g

@testset "space on a sub-triangulation, integrals on its parts with different laws" begin
    s = two_subdomains()
    for order in (1, 2)
        dΩa = Measure(s.Ωa, 2order); dΩb = Measure(s.Ωb, 2order)
        V = TestFESpace(s.Ωd, ReferenceFE(lagrangian, Float64, order); dirichlet_tags=[7])   # left edge
        U = TrialFESpace(V, x -> x[2])
        res(φ, v) = ∫(∇(v) ⋅ (κa ∘ ∇(φ)))dΩa + ∫(2.0 * (∇(v) ⋅ ∇(φ)))dΩb
        jac(φ, dφ, v) = ∫(∇(v) ⋅ (dκa ∘ (∇(φ), ∇(dφ))))dΩa + ∫(2.0 * (∇(v) ⋅ ∇(dφ)))dΩb
        op = KernelOperator(res, jac, U, V)
        @test nofallback_sub(op)
        @test length(op.jgroups) == 2 && sort([length(g.cellmap) for g in op.jgroups]) == sort([num_cells(s.Ωa), num_cells(s.Ωb)])
        c = check(res, jac, U, V, randx(U; scale=0.5); op)
        @test c.res < TOL && c.jac < TOL
    end
end

@testset "boundary of a subdomain as interface: traction from an FE function on the interface" begin
    s = two_subdomains()
    Γ = BoundaryTriangulation(s.Ωa, tags="interior")       # faces of Ωa interior to the model: x = 1 and y = 0.75
    @test num_cells(Γ) > 0
    dΩa = Measure(s.Ωa, 2); dΓ = Measure(Γ, 4)
    reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
    # auxiliary field living on an FE space over the interface itself (as the Maxwell traction tmaxw)
    t = interpolate_everywhere(x -> VectorValue(0.1 * x[2], -0.2 + x[1]), FESpace(Γ, reffe))
    n = get_normal_vector(Γ)
    V = TestFESpace(s.Ωa, reffe; dirichlet_tags=[7], dirichlet_masks=[(true, true)])
    U = TrialFESpace(V, x -> VectorValue(0.0, 0.0))
    α = 3.0
    res(u, v) = ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))))dΩa + ∫(v ⋅ t)dΓ + ∫(α * (u ⋅ n) * (v ⋅ n))dΓ
    jac(u, du, v) = ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')))dΩa + ∫(α * (du ⋅ n) * (v ⋅ n))dΓ
    x = randx(U; scale=0.05)
    for nthreads in (1, Threads.nthreads())
        op = KernelOperator(res, jac, U, V; nthreads)
        @test nofallback_sub(op)
        @test any(g -> g.trian === Γ, op.rgroups)
        c = check(res, jac, U, V, x; op)
        @test c.res < TOL && c.jac < TOL
        @test all(a -> a.bytes == 0, allocations(op, x))
    end
    # the interface field is read at every assembly (updated in place, as in the staggered loop)
    op = KernelOperator(res, jac, U, V)
    b1 = copy(Algebra.residual(op, x))
    get_free_dof_values(t) .*= 2.0
    @test reldiff(Algebra.residual(op, x), Algebra.residual(gridap_op((; res, jac, U, V)), x)) < TOL
    @test !(Algebra.residual(op, x) ≈ b1)
end

@testset "field on the other subdomain driven by Dirichlet values on the interface" begin
    s = two_subdomains()
    dΩb = Measure(s.Ωb, 2)
    reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
    V = TestFESpace(s.Ωb, reffe; dirichlet_tags=["interior", "tag_8"])    # interface (and top) + right edge
    U = TrialFESpace(V, [x -> VectorValue(0.01 * x[2], 0.0), x -> VectorValue(0.0, 0.0)])
    res(u, v) = ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))))dΩb
    jac(u, du, v) = ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')))dΩb
    op = KernelOperator(res, jac, U, V)
    x = randx(U; scale=0.01)
    c = check(res, jac, U, V, x; op)
    @test c.res < TOL && c.jac < TOL
    # new interface displacement (as the staggered coupling): Dirichlet values changed in place, same operator
    TrialFESpace!(U, [x -> VectorValue(0.03 * x[2], -0.01), x -> VectorValue(0.0, 0.0)])
    @test reldiff(Algebra.residual(op, x), Algebra.residual(gridap_op((; res, jac, U, V)), x)) < TOL
    @test reldiff(Algebra.jacobian(op, x), Algebra.jacobian(gridap_op((; res, jac, U, V)), x)) < TOL
end

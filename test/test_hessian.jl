# Second derivatives (∇∇) of the unknown, test and trial functions in the engine.
# Every case against Gridap on affine cells (simplices, Cartesian n-cubes: where Gridap's ∇∇ is
# defined) at 1e-13 relative, and against the push-forward of `hessian_nonaffine` (HyperFEM's
# `push_hessian`, reproduced in common.jl) on non-affine quadrilaterals and hexahedra; exact
# Hessians of an interpolated quadratic field; FE functions (not the unknown) with ∇∇; the
# matrix-free Jacobian; ∇∇ on skeleton integrals (C⁰ interior penalty, Hessian jumps); zero allocations.

Random.seed!(3)
const HTOL = 1e-13
tnorm(a) = sqrt(a ⊙ a)      # Frobenius norm of any tensor value (Gridap has no norm for third-order tensors)

# --- exact quadratic field: u = (x² + 3y² + xy, y² − 2xy), constant known Hessian ------------
uq(x) = VectorValue(x[1]^2 + 3x[2]^2 + x[1] * x[2], x[2]^2 - 2x[1] * x[2])
const H1q = TensorValue(2.0, 1.0, 1.0, 6.0)        # ∂²u₁ = [2 1; 1 6]
const H2q = TensorValue(0.0, -2.0, -2.0, 2.0)      # ∂²u₂ = [0 −2; −2 2]
const Hq = ThirdOrderTensorValue{2,2,2}(ntuple(i -> begin
        a, b, k = Tuple(CartesianIndices((2, 2, 2))[i])
        k == 1 ? H1q[a, b] : H2q[a, b]
    end, 8))

@testset "interpolated quadratic field: exact constant Hessian at every point (Q2 affine, Q2 non-affine, P2)" begin
    for (mapped, simplex) in ((false, false), (true, false), (false, true))
        model = cartesian_model(2, (4, 3); mapped, simplex)
        Ω = Triangulation(model); dΩ = Measure(Ω, 4)
        V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, 2))
        uh = interpolate(uq, V)
        xq, w = quad_points(dΩ), quad_weights(dΩ)
        geo = HK.GeoTable(Ω, xq, w)
        @test geo.affine == simplex                 # the Q1 reference Hessians of quadrilaterals do not vanish (Cartesian cells: Φ = 0 at run time)
        t = HK.FieldTable(V, xq, 2)
        W = zeros(t.nn, HK.wcols(t))
        vals = get_cell_dof_values(uh)
        use = HK.FieldUse{true,true,true}()
        for cell in 1:num_cells(Ω)
            xl = collect(vals[cell])
            for q in 1:length(xq)
                dV, iJt, Jt = HK.point_geometry(geo, q, cell)
                Φ = HK.point_curvature(geo, q, cell, Val(true))
                mapped || @test all(c -> maximum(abs, Φ[c]) < 1e-14, 1:2)
                u, g, H = HK.field_point_values(t, xl, 0, q, iJt, Jt, Φ, use)
                @test H isa ThirdOrderTensorValue{2,2,2,Float64,8}
                @test tnorm(H - Hq) < 1e-11
                @test u isa VectorValue{2,Float64} && g isa TensorValue{2,2,Float64}
                # the Hessian columns of the physical weights reproduce it: Σ_n x_{n,c} W[n, 1+D+p]
                HK.point_weights!(W, t, q, iJt, Jt, Φ, Val(true))
                for c in 1:2, (p, (a, b)) in enumerate(HK.hess_pairs(2))
                    hp = sum(xl[HK.ldofidx(t, n, c)] * W[n, 3+p] for n in 1:t.nn)
                    @test abs(hp - Hq[a, b, c]) < 1e-11
                end
            end
        end
        # the derivatives not asked for are not computed (returned as zero), the asked ones unchanged
        xl = collect(vals[1])
        dV, iJt, Jt = HK.point_geometry(geo, 1, 1)
        Φ = HK.point_curvature(geo, 1, 1, Val(true))
        u0, g0, H0 = HK.field_point_values(t, xl, 0, 1, iJt, Jt, Φ, HK.FieldUse{false,false,true}())
        @test iszero(u0) && tnorm(H0 - Hq) < 1e-11
        u1, g1, H1 = HK.field_point_values(t, xl, 0, 1, iJt, Jt, nothing, HK.FieldUse{true,true,false}())
        @test iszero(H1) && !iszero(g1)
    end
end

# --- forms with Hessians -------------------------------------------------------------------
"Scalar problem with ∇∇, Δ, ∇ and a nonlinear value term; `order` and the mesh from `model`."
function scalar_hessian_problem(model; order=2, degree=2order)
    Ω = Triangulation(model); dΩ = Measure(Ω, degree)
    V = TestFESpace(model, ReferenceFE(lagrangian, Float64, order), dirichlet_tags="boundary")
    U = TrialFESpace(V, x -> x[1]^2 - 0.5 * x[2])
    β = 0.7
    res(u, v) = ∫(β * (∇∇(v) ⊙ ∇∇(u)) + ∇(v) ⋅ ∇(u) + v * (u * u))dΩ + 0.3 * ∫(Δ(v) * Δ(u))dΩ
    jac(u, du, v) = ∫(β * (∇∇(v) ⊙ ∇∇(du)) + ∇(v) ⋅ ∇(du) + 2.0 * (v * (u * du)))dΩ + 0.3 * ∫(Δ(v) * Δ(du))dΩ
    (; model, Ω, dΩ, U, V, res, jac)
end
"Vector problem: St Venant–Kirchhoff plus the Hessian regularisation β ∫ ∇∇u ⊙ ∇∇v (as in NeoMag)."
function vector_hessian_problem(model, D; order=2, degree=2order, β=5e-3)
    Ω = Triangulation(model); dΩ = Measure(Ω, degree)
    V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{D,Float64}, order), dirichlet_tags="boundary")
    U = TrialFESpace(V, x -> VectorValue(ntuple(i -> 0.02 * x[i] * x[mod1(i + 1, D)], D)))
    res(u, v) = ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))))dΩ + β * ∫(∇∇(u) ⊙ ∇∇(v))dΩ
    jac(u, du, v) = ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')))dΩ + β * ∫(∇∇(du) ⊙ ∇∇(v))dΩ
    (; model, Ω, dΩ, U, V, res, jac)
end

"Kernel operator vs a Gridap operator built from (possibly other) forms `gres`/`gjac`: relative differences and matrix-free."
function compare(prob, gres, gjac; kw...)
    op = KernelOperator(prob.res, prob.jac, prob.U, prob.V; kw...)
    @test isempty(op.fallback.rtrians) && isempty(op.fallback.jtrians)
    x = randx(prob.U)
    gop = get_algebraic_operator(FEOperator(gres, gjac, prob.U, prob.V))
    bg, Ag = Algebra.residual(gop, x), Algebra.jacobian(gop, x)
    bk, Ak = Algebra.residual(op, x), Algebra.jacobian(op, x)
    mf = MatrixFreeJacobian(op); prepare!(mf, x)
    dx = randn(num_free_dofs(prob.U))
    (res=reldiff(bk, bg), jac=reldiff(Ak, Ag), mf=reldiff(mf * dx, Ag * dx), op)
end

@testset "scalar and vector forms with ∇∇ against Gridap (affine cells: simplices P1–P3, Cartesian Q1–Q2, 2D and 3D)" begin
    for D in (2, 3), simplex in (true, false), order in (simplex ? (1, 2, 3) : (1, 2))
        n = D == 2 ? (3, 2) : (2, 2, 2)
        (D == 3 && order == 3 && !simplex) && continue
        model = cartesian_model(D, n; simplex)
        for (prob, name) in ((scalar_hessian_problem(model; order), "scalar"), (vector_hessian_problem(model, D; order), "vector"))
            for linear_maps in (true, false)
                c = compare(prob, prob.res, prob.jac; linear_maps)
                @test c.res < HTOL && c.jac < HTOL && c.mf < 1e-12
                # the linear-map shortcut applies to the residuals (state inputs P(F), ∇∇u) and to the scalar
                # Jacobian; the SVK tangent dP(F, dF) is not linear in F (dual numbers, as in test_assembly)
                linear_maps && @test all(g -> g.lin !== nothing, c.op.rgroups)
                linear_maps && name == "scalar" && @test all(g -> g.lin !== nothing, c.op.jgroups)
            end
            op = KernelOperator(prob.res, prob.jac, prob.U, prob.V)
            @test all(a -> a.bytes == 0, allocations(op, randx(prob.U)))
            # one thread and the default: same result
            op1 = KernelOperator(prob.res, prob.jac, prob.U, prob.V; nthreads=1)
            x = randx(prob.U)
            @test reldiff(Algebra.residual(op1, x), Algebra.residual(op, x)) < HTOL
            @test reldiff(Algebra.jacobian(op1, x), Algebra.jacobian(op, x)) < HTOL
        end
    end
end

@testset "non-affine quadrilaterals and hexahedra against hessian_nonaffine (Q1, Q2)" begin
    for D in (2, 3), order in (1, 2)
        model = cartesian_model(D, D == 2 ? (3, 2) : (2, 2, 2); mapped=true)
        for (prob, name) in ((scalar_hessian_problem(model; order), "scalar"), (vector_hessian_problem(model, D; order), "vector"))
            # Gridap's ∇∇ is not defined on these cells: the reference form uses hessian_nonaffine
            Hn = hessian_nonaffine
            gres, gjac = if name == "scalar"
                ((u, v) -> ∫(0.7 * (Hn(v) ⊙ Hn(u)) + ∇(v) ⋅ ∇(u) + v * (u * u))prob.dΩ + 0.3 * ∫(tr(Hn(v)) * tr(Hn(u)))prob.dΩ,
                 (u, du, v) -> ∫(0.7 * (Hn(v) ⊙ Hn(du)) + ∇(v) ⋅ ∇(du) + 2.0 * (v * (u * du)))prob.dΩ + 0.3 * ∫(tr(Hn(v)) * tr(Hn(du)))prob.dΩ)
            else
                ((u, v) -> ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))))prob.dΩ + 5e-3 * ∫(Hn(u) ⊙ Hn(v))prob.dΩ,
                 (u, du, v) -> ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')))prob.dΩ + 5e-3 * ∫(Hn(du) ⊙ Hn(v))prob.dΩ)
            end
            for linear_maps in (true, false)
                c = compare(prob, gres, gjac; linear_maps)
                @test c.res < 1e-12 && c.jac < 1e-12 && c.mf < 1e-12
            end
            op = KernelOperator(prob.res, prob.jac, prob.U, prob.V)
            @test !op.rgroups[1].geo.affine
            @test all(a -> a.bytes == 0, allocations(op, randx(prob.U)))
            # sum factorisation is not used for fields with Hessian slots (dense path with the curvature)
            mf = MatrixFreeJacobian(op)
            @test all(tp -> tp === nothing, mf.tps[1])
            @test size(mf.data[1].Phi, 1) == D^3
        end
    end
end

@testset "mixed use: Hessian only in the Jacobian, value and Hessian of the same field, multi-field" begin
    model = cartesian_model(2, (3, 3); simplex=true)
    Ω = Triangulation(model); dΩ = Measure(Ω, 4)
    Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, 2), dirichlet_tags="boundary")
    Vφ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 2), dirichlet_tags="boundary")
    V = MultiFieldFESpace([Vu, Vφ])
    U = MultiFieldFESpace([TrialFESpace(Vu, x -> VectorValue(0.01x[2], 0.0)), TrialFESpace(Vφ, x -> x[1])])
    # u: gradient and Hessian; φ: value, gradient and Hessian; coupling through Δφ and tr(∇u)
    res((u, φ), (v, vφ)) = ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))) + 0.1 * (∇∇(u) ⊙ ∇∇(v)) + Δ(φ) * tr(∇(v)) +
                             ∇(vφ) ⋅ ∇(φ) + 0.2 * (∇∇(vφ) ⊙ ∇∇(φ)) + vφ * (φ * φ) + Δ(vφ) * tr(∇(u)))dΩ
    jac((u, φ), (du, dφ), (v, vφ)) = ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')) + 0.1 * (∇∇(du) ⊙ ∇∇(v)) + Δ(dφ) * tr(∇(v)) +
                                       ∇(vφ) ⋅ ∇(dφ) + 0.2 * (∇∇(vφ) ⊙ ∇∇(dφ)) + 2.0 * (vφ * (φ * dφ)) + Δ(vφ) * tr(∇(du)))dΩ
    prob = (; U, V, res, jac)
    for linear_maps in (true, false)
        c = compare(prob, res, jac; linear_maps)
        @test c.res < HTOL && c.jac < HTOL && c.mf < 1e-12
    end
    # Hessian only in the Jacobian (a residual without it): different slot layouts per operator part
    res2((u, φ), (v, vφ)) = ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))) + ∇(vφ) ⋅ ∇(φ))dΩ
    jac2((u, φ), (du, dφ), (v, vφ)) = ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')) + ∇(vφ) ⋅ ∇(dφ) + 0.2 * (∇∇(vφ) ⊙ ∇∇(dφ)))dΩ
    c2 = compare((; U, V, res=res2, jac=jac2), res2, jac2)
    @test c2.res < HTOL && c2.jac < HTOL && c2.mf < 1e-12
    @test !HK.has_hess(c2.op.rgroups[1].smv[2]) && HK.has_hess(c2.op.jgroups[1].smv[2]) && HK.has_hess(c2.op.jgroups[1].smu[2])
    @test HK.hess_use(c2.op.rgroups[1]) isa Val{false} && HK.hess_use(c2.op.jgroups[1]) isa Val{true}
end

@testset "update with another β reuses the compiled kernels (numbers are parameters)" begin
    model = cartesian_model(2, (3, 2); simplex=true)
    Ω = Triangulation(model); dΩ = Measure(Ω, 4)
    V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, 2), dirichlet_tags="boundary")
    U = TrialFESpace(V, x -> VectorValue(0.0, 0.01x[1]))
    mk(β) = ((u, v) -> ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))))dΩ + β * ∫(∇∇(u) ⊙ ∇∇(v))dΩ,
             (u, du, v) -> ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')))dΩ + β * ∫(∇∇(du) ⊙ ∇∇(v))dΩ)
    r1, j1 = mk(5e-3); r2, j2 = mk(0.2)
    op1 = KernelOperator(r1, j1, U, V)
    op2 = update(op1, r2, j2)
    for (g1, g2) in zip(vcat(op1.rgroups, op1.jgroups), vcat(op2.rgroups, op2.jgroups))
        @test typeof(g2) === typeof(g1) && typeof(g2.pf.core) === typeof(g1.pf.core) && g2.tables === g1.tables
    end
    c = check(r2, j2, U, V, randx(U); op=op2)
    @test c.res < HTOL && c.jac < HTOL
    # matrix-free Jacobian rebound to the updated operator
    mf = MatrixFreeJacobian(op1)
    x = randx(U); dx = randn(num_free_dofs(U))
    jacobian!(mf, op2, x)
    @test reldiff(mf * dx, Algebra.jacobian(op2, x) * dx) < 1e-12
end

@testset "∇∇ of an FE function that is not the unknown: read from its nodal values by the engine" begin
    for (D, simplex) in ((2, true), (2, false), (3, true))
        model = cartesian_model(D, D == 2 ? (3, 2) : (2, 2, 2); simplex)
        Ω = Triangulation(model); dΩ = Measure(Ω, 4)
        Vs = TestFESpace(model, ReferenceFE(lagrangian, Float64, 2), dirichlet_tags="boundary")
        Vw = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{D,Float64}, 2))
        U = TrialFESpace(Vs, x -> x[1])
        wh = interpolate(x -> VectorValue(ntuple(i -> x[i]^2 + 0.5 * x[mod1(i + 1, D)]^2 * x[i], D)), Vw)   # a vector field
        sh = interpolate(x -> x[1]^2 * x[2] + x[2]^2, Vs)                                                       # a scalar field
        res(u, v) = ∫(∇(v) ⋅ ∇(u) + v * (tr(∇∇(sh)) * u) + (∇∇(v) ⊙ ∇∇(u)) * (1.0 + Δ(sh) * Δ(sh)) + v * (∇∇(wh) ⊙ ∇∇(wh)))dΩ
        jac(u, du, v) = ∫(∇(v) ⋅ ∇(du) + v * (tr(∇∇(sh)) * du) + (∇∇(v) ⊙ ∇∇(du)) * (1.0 + Δ(sh) * Δ(sh)))dΩ
        op = KernelOperator(res, jac, U, Vs)
        @test isempty(op.fallback.rtrians) && isempty(op.fallback.jtrians)
        fes = filter(a -> a isa HK.FEArray, collect(op.rgroups[1].aux))
        @test length(fes) == 2 && all(a -> HK.fe_deriv(a) == 2, fes)     # ∇∇(sh) and ∇∇(wh), each read once
        c = check(res, jac, U, Vs, randx(U); op)
        @test c.res < HTOL && c.jac < HTOL
        # the FE function changed in place: the next assembly reads the new values
        x = randx(U)
        b1 = Algebra.residual(op, x)
        get_free_dof_values(sh) .*= 2.0
        b2 = Algebra.residual(op, x)
        @test reldiff(b2, Algebra.residual(gridap_op((; res, jac, U, V=Vs)), x)) < HTOL && reldiff(b1, b2) > 1e-3
    end
end

# --- ∇∇ on skeleton integrals --------------------------------------------------------------
# The Hessian of each side uses the Jt⁻¹ and the curvature of that side's cell map at its mapped points.
# Reference forms: Gridap's ∇∇/Δ on affine cells, `hessian_nonaffine` (and its trace) on mapped n-cubes.

sk_noalloc(op, x) = all(a -> a.bytes == 0, allocations(op, x))
Hn_tr(a) = tr(hessian_nonaffine(a))

"""
C⁰ interior penalty (biharmonic-like) scalar problem on the skeleton of the whole mesh: jump of the
normal derivative, consistency terms {Δ}⟦∂ₙ⟧, and a nonlinear Hessian-jump term
(1 + {Δu}²) ⟦∇∇u⟧:⟦∇∇v⟧ (dual numbers). `H`, `L` are the Hessian and Laplacian used in the forms
(∇∇ and Δ for the kernels; `hessian_nonaffine` and its trace for the reference on mapped cells).
"""
function c0ip_problem(model; order=2, degree=2order, H=∇∇, L=Δ)
    Ω = Triangulation(model); dΩ = Measure(Ω, degree)
    Λ = SkeletonTriangulation(model); dΛ = Measure(Λ, degree)
    n = get_normal_vector(Λ)
    V = TestFESpace(model, ReferenceFE(lagrangian, Float64, order), dirichlet_tags="boundary")
    U = TrialFESpace(V, x -> x[1]^2 - 0.5 * x[2])
    hF = CellField(get_array(∫(1)dΛ), Λ)
    γ = 3.0
    dn(a) = jump(∇(a) ⋅ n)
    res(u, v) = ∫(H(v) ⊙ H(u) + v * (u * u))dΩ +
                ∫((γ / hF) * (dn(v) * dn(u)) - mean(L(u)) * dn(v) - mean(L(v)) * dn(u) +
                  (0.2 * hF) * ((jump(H(u)) ⊙ jump(H(v))) * (1.0 + mean(L(u)) * mean(L(u)))))dΛ
    jac(u, du, v) = ∫(H(v) ⊙ H(du) + 2.0 * (v * (u * du)))dΩ +
                    ∫((γ / hF) * (dn(v) * dn(du)) - mean(L(du)) * dn(v) - mean(L(v)) * dn(du) +
                      (0.2 * hF) * ((jump(H(du)) ⊙ jump(H(v))) * (1.0 + mean(L(u)) * mean(L(u))) +
                                    (jump(H(u)) ⊙ jump(H(v))) * (2.0 * mean(L(u)) * mean(L(du)))))dΛ
    (; model, Ω, dΩ, Λ, dΛ, U, V, res, jac)
end

"""
Vector problem as in the third-medium regularisation of TMC_gears (G1): SVK on the whole mesh, β ∫∇∇u⊙∇∇v on
a subdomain Ωc and, on the skeleton of Ωc (a composite triangulation), σβ/h_F ⟦∇u⟧:⟦∇v⟧ + c₂βh_F ⟦∇∇u⟧:⟦∇∇v⟧.
"""
function g1_problem(model, D; order=2, H=∇∇)
    Ω = Triangulation(model); dΩ = Measure(Ω, 2order)
    cc = [sum(c) / length(c) for c in get_cell_coordinates(model)]
    Ωc = Triangulation(model, [x[1] > 0.4 for x in cc]); dΩc = Measure(Ωc, 2order)
    Λ = SkeletonTriangulation(Ωc); dΛ = Measure(Λ, 2order)
    hF = CellField(get_array(∫(1)dΛ), Λ)
    β = 5e-3
    cJ, cJ2 = β / hF, (2.0 * β) * hF
    V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{D,Float64}, order), dirichlet_tags="boundary")
    U = TrialFESpace(V, x -> VectorValue(ntuple(i -> 0.02 * x[i] * x[mod1(i + 1, D)], D)))
    qJ(a, b) = jump(∇(a)) ⊙ jump(∇(b))
    qJ2(a, b) = jump(H(a)) ⊙ jump(H(b))
    res(u, v) = ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))))dΩ + β * ∫(H(u) ⊙ H(v))dΩc + ∫(cJ * qJ(u, v) + cJ2 * qJ2(u, v))dΛ
    jac(u, du, v) = ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')))dΩ + β * ∫(H(du) ⊙ H(v))dΩc + ∫(cJ * qJ(du, v) + cJ2 * qJ2(du, v))dΛ
    (; model, Ω, dΩ, Ωc, Λ, dΛ, U, V, res, jac)
end

@testset "∇∇ on skeleton integrals: C⁰ interior penalty and G1 regularisation against Gridap (affine cells)" begin
    # 3D: the scalar problem on tetrahedra here; the vector one (54 slots per side pair) on the hexahedra of the next testset
    for (D, simplex, order) in ((2, true, 2), (2, true, 3), (2, false, 2), (3, true, 2))
        model = cartesian_model(D, D == 2 ? (4, 3) : (2, 2, 2); simplex)
        probs = D == 2 ? ((c0ip_problem(model; order), "C0-IP"), (g1_problem(model, D; order), "G1")) : ((c0ip_problem(model; order), "C0-IP"),)
        for (prob, name) in probs
            for linear_maps in (true, false)
                c = compare(prob, prob.res, prob.jac; linear_maps)
                @test c.res < HTOL && c.jac < HTOL && c.mf < 1e-11
                @test any(g -> g isa HK.SkeletonGroup && HK.hess_use(g) isa Val{true}, c.op.jgroups)
            end
            op = KernelOperator(prob.res, prob.jac, prob.U, prob.V)
            x = randx(prob.U)
            @test sk_noalloc(op, x)
            op1 = KernelOperator(prob.res, prob.jac, prob.U, prob.V; nthreads=1)
            @test reldiff(Algebra.residual(op1, x), Algebra.residual(op, x)) < HTOL
            @test reldiff(Algebra.jacobian(op1, x), Algebra.jacobian(op, x)) < HTOL
            # simplices: no curvature stored for the matrix-free skeleton groups (the Q1 reference Hessians of
            # quadrilaterals do not vanish, so Cartesian cells store a curvature that is zero at run time)
            mf = MatrixFreeJacobian(op)
            for (g, gd) in zip(op.jgroups, mf.data)
                g isa HK.SkeletonGroup && @test (size(gd[1].Phi, 1) == 0 && size(gd[2].Phi, 1) == 0) == simplex
            end
        end
    end
end

@testset "∇∇ on skeleton integrals: non-affine quadrilaterals and hexahedra against hessian_nonaffine" begin
    for D in (2, 3)
        model = cartesian_model(D, D == 2 ? (3, 3) : (2, 2, 2); mapped=true)
        # 3D: Gridap takes the faces of a mapped Cartesian model as flat (normals, see test_hdiv.jl); not those of an unstructured one
        D == 3 && (model = Gridap.Geometry.UnstructuredDiscreteModel(model))
        # 3D: the scalar problem only (the vector one, 54 × 54 slots per facet, takes minutes to compile; its
        # curvature path is the same, checked in 2D and by the scalar problem here)
        probs = ((c0ip_problem(model), c0ip_problem(model; H=hessian_nonaffine, L=Hn_tr)),)
        D == 2 && (probs = (probs..., (g1_problem(model, D), g1_problem(model, D; H=hessian_nonaffine))))
        for (kprob, gprob) in probs
            for linear_maps in (true, false)
                c = compare(kprob, gprob.res, gprob.jac; linear_maps)
                @test c.res < 1e-12 && c.jac < 1e-12 && c.mf < 1e-11
            end
            op = KernelOperator(kprob.res, kprob.jac, kprob.U, kprob.V)
            @test sk_noalloc(op, randx(kprob.U))
            sk = [g for g in op.jgroups if g isa HK.SkeletonGroup]
            @test !isempty(sk) && all(g -> !g.geo.affine && !g.geom.affine, sk)
            mf = MatrixFreeJacobian(op)
            for (g, gd) in zip(op.jgroups, mf.data)
                g isa HK.SkeletonGroup && @test size(gd[1].Phi, 1) == D^3 && size(gd[2].Phi, 1) == D^3
            end
        end
    end
end

@testset "∇∇ on skeleton integrals: a field with a constant Hessian has no Hessian jump (Q2 non-affine, P2)" begin
    # uq is quadratic: interpolated exactly, its Hessian is the same constant on both sides of every facet
    for (mapped, simplex) in ((false, false), (true, false), (false, true))
        model = cartesian_model(2, (4, 3); mapped, simplex)
        Λ = SkeletonTriangulation(model); dΛ = Measure(Λ, 4)
        V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, 2))
        res(u, v) = ∫(jump(∇∇(u)) ⊙ jump(∇∇(v)) + (mean(∇∇(u)) ⊙ mean(∇∇(u)) - Hq ⊙ Hq) * mean(tr(∇(v))))dΛ
        jac(u, du, v) = ∫(jump(∇∇(du)) ⊙ jump(∇∇(v)) + (2.0 * (mean(∇∇(u)) ⊙ mean(∇∇(du)))) * mean(tr(∇(v))))dΛ
        op = KernelOperator(res, jac, V, V)
        @test isempty(op.fallback.rtrians) && isempty(op.fallback.jtrians)
        x = get_free_dof_values(interpolate(uq, V))
        b = Algebra.residual(op, x)
        @test norm(b, Inf) < 1e-10
        @test norm(Algebra.residual(op, x .+ 1e-3 .* randn(length(x))), Inf) > 1e-6
    end
end

@testset "∇∇ on skeleton integrals: Hessian only in the Jacobian, multi-field, update, matrix-free rebind" begin
    model = cartesian_model(2, (3, 3); simplex=true)
    Ω = Triangulation(model); dΩ = Measure(Ω, 4)
    Λ = SkeletonTriangulation(model); dΛ = Measure(Λ, 4)
    n = get_normal_vector(Λ)
    Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, 2), dirichlet_tags="boundary")
    Vφ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 2), dirichlet_tags="boundary")
    V = MultiFieldFESpace([Vu, Vφ])
    U = MultiFieldFESpace([TrialFESpace(Vu, x -> VectorValue(0.01x[2], 0.0)), TrialFESpace(Vφ, x -> x[1])])
    # coupling on the skeleton: mean(Δφ) with jump(∇v), jump(∇∇u) with mean(φ)
    mk(γ) = (((u, φ), (v, vφ)) -> ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))) + ∇(vφ) ⋅ ∇(φ))dΩ +
                                  ∫(γ * (jump(∇∇(u)) ⊙ jump(∇∇(v))) * (1.0 + mean(φ) * mean(φ)) + mean(Δ(φ)) * (jump(∇(v)) ⊙ jump(∇(u))) +
                                    jump(∇(vφ) ⋅ n) * jump(∇(φ) ⋅ n) + mean(Δ(vφ)) * mean(Δ(φ)))dΛ,
             ((u, φ), (du, dφ), (v, vφ)) -> ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')) + ∇(vφ) ⋅ ∇(dφ))dΩ +
                                  ∫(γ * (jump(∇∇(du)) ⊙ jump(∇∇(v))) * (1.0 + mean(φ) * mean(φ)) +
                                    γ * (jump(∇∇(u)) ⊙ jump(∇∇(v))) * (2.0 * mean(φ) * mean(dφ)) +
                                    mean(Δ(dφ)) * (jump(∇(v)) ⊙ jump(∇(u))) + mean(Δ(φ)) * (jump(∇(v)) ⊙ jump(∇(du))) +
                                    jump(∇(vφ) ⋅ n) * jump(∇(dφ) ⋅ n) + mean(Δ(vφ)) * mean(Δ(dφ)))dΛ)
    r1, j1 = mk(0.3); r2, j2 = mk(1.7)
    for linear_maps in (true, false)
        c = compare((; U, V, res=r1, jac=j1), r1, j1; linear_maps)
        @test c.res < HTOL && c.jac < HTOL && c.mf < 1e-11
    end
    # Hessian only in the Jacobian (a residual without skeleton Hessians)
    r0((u, φ), (v, vφ)) = ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))) + ∇(vφ) ⋅ ∇(φ))dΩ + ∫(jump(∇(vφ) ⋅ n) * jump(∇(φ) ⋅ n))dΛ
    j0((u, φ), (du, dφ), (v, vφ)) = ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')) + ∇(vφ) ⋅ ∇(dφ))dΩ +
                                   ∫(jump(∇(vφ) ⋅ n) * jump(∇(dφ) ⋅ n) + 0.5 * (jump(∇∇(dφ)) ⊙ jump(∇∇(vφ))))dΛ
    c0 = compare((; U, V, res=r0, jac=j0), r0, j0)
    @test c0.res < HTOL && c0.jac < HTOL && c0.mf < 1e-11
    @test all(g -> !(g isa HK.SkeletonGroup) || HK.hess_use(g) isa Val{false}, c0.op.rgroups)
    @test any(g -> g isa HK.SkeletonGroup && HK.hess_use(g) isa Val{true}, c0.op.jgroups)
    # update with another γ: same compiled kernels and tables; matrix-free rebound
    op1 = KernelOperator(r1, j1, U, V)
    op2 = update(op1, r2, j2)
    for (g1, g2) in zip(vcat(op1.rgroups, op1.jgroups), vcat(op2.rgroups, op2.jgroups))
        @test typeof(g2) === typeof(g1) && typeof(g2.pf.core) === typeof(g1.pf.core) && g2.tables === g1.tables
    end
    x = randx(U); dx = randn(num_free_dofs(U))
    c = check(r2, j2, U, V, x; op=op2)
    @test c.res < HTOL && c.jac < HTOL
    mf = MatrixFreeJacobian(op1)
    jacobian!(mf, op2, x)
    @test reldiff(mf * dx, Algebra.jacobian(op2, x) * dx) < 1e-11
    @test sk_noalloc(op2, x)
end

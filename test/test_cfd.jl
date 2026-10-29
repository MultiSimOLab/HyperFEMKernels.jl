# Incompressible Navier–Stokes, one Oseen–BDF2 step with SUPG/PSPG/LSIC stabilization, backflow
# stabilization and Windkessel (Robin) outlet terms: the weak form of examples/aorta_cfd/aorta_flow.jl
# (Taylor–Hood P2–P1 on tetrahedra) on a small channel, against pure Gridap.

nofallback_cfd(op) = isempty(op.fallback.rtrians) && isempty(op.fallback.jtrians)

"Channel (0,2)×(0,1)×(0,1) on tetrahedra: inlet x = 0, outlet x = 2, wall elsewhere; weak form of one step."
function oseen_channel(; n=(3, 2, 2), seed=1)
    Random.seed!(seed)
    model = simplexify(CartesianDiscreteModel((0, 2, 0, 1, 0, 1), n))
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "inlet", [25])
    add_tag_from_tags!(labels, "outlet", [26])
    add_tag_from_tags!(labels, "wall", [collect(1:20); 21; 22; 23; 24])
    ρ, μ, Δt = 1.0, 0.05, 0.01
    Ω = Triangulation(model); dΩ = Measure(Ω, 4)
    he = CellField(map(v -> (6sqrt(2) * v)^(1 / 3), collect(get_cell_measure(Ω))), Ω)
    zero3 = VectorValue(0.0, 0.0, 0.0)
    Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 2), conformity=:H1, dirichlet_tags=["inlet", "wall"])
    Vp = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity=:H1)
    V = MultiFieldFESpace([Vu, Vp])
    gin(x) = VectorValue(16 * x[2] * (1 - x[2]) * x[3] * (1 - x[3]), 0.0, 0.0)
    trial() = MultiFieldFESpace([TrialFESpace(Vu, [gin, zero3]), TrialFESpace(Vp)])
    U, U⁻, U⁻⁻ = trial(), trial(), trial()
    nf = num_free_dofs(U)
    xh⁻ = FEFunction(U⁻, randn(nf)); xh⁻⁻ = FEFunction(U⁻⁻, randn(nf))
    un, unm1 = xh⁻[1], xh⁻⁻[1]
    bdf2 = Ref(false)
    c0() = bdf2[] ? 1.5 / Δt : 1.0 / Δt
    c1() = bdf2[] ? 2.0 / Δt : 1.0 / Δt
    c2() = bdf2[] ? 0.5 / Δt : 0.0
    hist() = c1() * un - c2() * unm1
    wadv() = bdf2[] ? 2.0 * un - 1.0 * unm1 : un
    # elementwise Laplacian of uⁿ through the discontinuous P1 space
    Wlap = FESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), conformity=:L2)
    eks = [VectorValue(ntuple(i -> i == k ? 1.0 : 0.0, 3)) for k in 1:3]
    gk = [interpolate(∇(un) ⋅ eks[k], Wlap) for k in 1:3]
    update_lap! = () -> foreach(k -> Gridap.FESpaces.interpolate!(∇(un) ⋅ eks[k], get_free_dof_values(gk[k]), Wlap), 1:3)
    lapn = Operation((a, b, c) -> VectorValue(a, b, c))(∇ ⋅ gk[1], ∇ ⋅ gk[2], ∇ ⋅ gk[3])
    # outlet: Windkessel pressure (a number of the integrand, as in the original script) and Robin term; backflow
    Γo = BoundaryTriangulation(model, tags="outlet"); dΓo = Measure(Γo, 4); dΓb = Measure(Γo, 5)
    no = get_normal_vector(Γo)
    Pc = Ref(0.7)
    ZA = 2.3
    negpart(x) = 0.5 * (x - abs(x))
    τsupg(u, h) = ((2 * norm(u) / h)^2 + 9 * (4μ / (ρ * h^2))^2)^(-1 / 2)
    τpspg(u, h) = τsupg(u, h) / ρ
    τlsic(u, h) = ρ * norm(u) * h * 0.5
    Rvel(c) = (u, ∇u, w) -> ρ * c * u + ρ * (∇u' ⋅ w)
    εs(∇u) = 0.5 * (∇u + ∇u')
    function a((u, p), (v, q))
        c = c0(); w = wadv()
        τs = τsupg ∘ (un, he); τp = τpspg ∘ (un, he); τl = τlsic ∘ (un, he)
        Ru = Rvel(c) ∘ (u, ∇(u), w)
        ∫(ρ * c * (u ⋅ v))dΩ + ∫(2μ * (εs ∘ (∇(u))) ⊙ (εs ∘ (∇(v))))dΩ + ∫(ρ * ((∇(u)' ⋅ w) ⋅ v))dΩ -
        ∫(p * (∇ ⋅ v))dΩ - ∫(q * (∇ ⋅ u))dΩ +
        ∫(τs * ((∇(v)' ⋅ w) ⋅ Ru))dΩ + ∫(τs * ((∇(v)' ⋅ w) ⋅ ∇(p)))dΩ -
        ∫(τp * (∇(q) ⋅ Ru))dΩ - ∫(τp * (∇(q) ⋅ ∇(p)))dΩ +
        ∫(τl * (∇ ⋅ v) * (∇ ⋅ u))dΩ -
        ∫(ρ * (negpart ∘ (w ⋅ no)) * (u ⋅ v))dΓb + ∫(ZA * (u ⋅ no) * (v ⋅ no))dΓo
    end
    function l((v, q))
        w = wadv(); f = hist()
        τs = τsupg ∘ (un, he); τp = τpspg ∘ (un, he)
        rk = ρ * f + μ * lapn
        ∫(ρ * (f ⋅ v))dΩ + ∫(τs * ((∇(v)' ⋅ w) ⋅ rk))dΩ - ∫(τp * (∇(q) ⋅ rk))dΩ - ∫(Pc[] * (v ⋅ no))dΓo
    end
    res(x, y) = a(x, y) - l(y)
    jac(x, dx, y) = a(dx, y)
    (; model, U, V, U⁻, xh⁻, xh⁻⁻, a, l, res, jac, bdf2, Pc, update_lap!)
end

"Gridap's linear system of the step (Dirichlet values lifted), as assembled in the example."
function gridap_system(p)
    data = Gridap.FESpaces.collect_cell_matrix_and_vector(p.U, p.V, p.a(get_trial_fe_basis(p.U), get_fe_basis(p.V)),
                                                          p.l(get_fe_basis(p.V)), zero(p.U))
    Gridap.FESpaces.assemble_matrix_and_vector(SparseMatrixAssembler(p.U, p.V), data)
end

@testset "Oseen step: backward Euler and BDF2, residual and Jacobian at random states" begin
    for bdf2 in (false, true)
        p = oseen_channel()
        p.bdf2[] = bdf2
        op = KernelOperator(p.res, p.jac, p.U, p.V)
        @test nofallback_cfd(op)
        @test any(g -> g.trian isa BoundaryTriangulation, op.jgroups)
        c = check(p.res, p.jac, p.U, p.V, randx(p.U; scale=1.0); op)
        @test c.res < TOL && c.jac < TOL
    end
end

@testset "Oseen step: linear system with lifted Dirichlet values (A = J, b = -r(0))" begin
    p = oseen_channel()
    p.bdf2[] = true
    Ag, bg = gridap_system(p)
    for nthreads in (1, Threads.nthreads())
        op = KernelOperator(p.res, p.jac, p.U, p.V; nthreads)
        x0 = zeros(num_free_dofs(p.U))
        @test reldiff(jacobian(op, x0), Ag) < TOL
        @test reldiff(-residual(op, x0), bg) < TOL
    end
end

"Same generated functions (no recompilation) in every group of two operators."
same_code(op, op2) = all(zip(vcat(op.rgroups, op.jgroups), vcat(op2.rgroups, op2.jgroups))) do (g, g2)
    typeof(g2) === typeof(g) && typeof(g2.pf.core) === typeof(g.pf.core) && (g.lin === nothing) == (g2.lin === nothing) &&
        (g.lin === nothing || (typeof(g2.lin.apply) === typeof(g.lin.apply) && typeof(g2.lin.state) === typeof(g.lin.state)))
end

@testset "Oseen step: time loop (data in place, numbers through update)" begin
    p = oseen_channel()
    op = KernelOperator(p.res, p.jac, p.U, p.V)
    x0 = zeros(num_free_dofs(p.U))
    r1 = -residual(op, x0)
    # new step: previous states, Dirichlet values and Laplacian of uⁿ overwritten in place (read at every
    # assembly); the Windkessel pressure is a number of the integrand (needs `update`, same code)
    get_free_dof_values(p.xh⁻⁻) .= get_free_dof_values(p.xh⁻)
    get_free_dof_values(p.xh⁻) .= randn(num_free_dofs(p.U))
    get_dirichlet_dof_values(p.U[1]) .*= 0.4
    get_dirichlet_dof_values(p.U⁻[1]) .*= 0.7
    p.update_lap!()
    p.Pc[] = -1.3
    op1 = HyperFEMKernels.update(op, p.res, p.jac)
    @test same_code(op, op1)
    @test op1.A0 === op.A0
    Ag, bg = gridap_system(p)
    @test reldiff(-residual(op1, x0), bg) < TOL
    @test reldiff(jacobian(op1, x0), Ag) < TOL
    @test reldiff(r1, bg) > 1e-3                 # the data did change
    # BDF2 after the backward Euler step: other numbers (c0, c1, c2) and another advecting-velocity expression
    # (2uⁿ - uⁿ⁻¹ instead of uⁿ). The FE functions are leaves of the compiled integrand, so the new
    # expression is new code: `update` recompiles the groups once (the switch happens once per simulation)
    p.bdf2[] = true
    op2 = HyperFEMKernels.update(op1, p.res, p.jac)
    @test !same_code(op, op2)
    @test same_code(op2, HyperFEMKernels.update(op2, p.res, p.jac))      # and not again
    Ag, bg = gridap_system(p)
    @test reldiff(-residual(op2, x0), bg) < TOL
    @test reldiff(jacobian(op2, x0), Ag) < TOL
end

Random.seed!(1)

@testset "sparsity pattern equals Gridap's and positions are consistent" begin
    for prob in (poisson_2d(), elasticity_3d(), coupled_3d())
        op = KernelOperator(prob.res, prob.jac, prob.U, prob.V)
        Ag = Algebra.jacobian(gridap_op(prob), randx(prob.U))
        @test size(HK.sparsity_pattern(op)) == size(Ag)
        @test HK.sparsity_pattern(op).colptr == Ag.colptr && HK.sparsity_pattern(op).rowval == Ag.rowval
        @test all(p -> p == 0 || 1 <= p <= nnz(HK.sparsity_pattern(op)), HK.nonzero_positions(op))
        ndl = op.ndl
        for cell in (1, size(HK.nonzero_positions(op), 2)), ii in 1:ndl, jj in 1:ndl
            pos = HK.nonzero_positions(op)[ii+(jj-1)*ndl, cell]
            pos == 0 && continue
            r = findfirst(fl -> fl.loc < ii <= fl.loc + fl.nd, op.fields)
            c = findfirst(fl -> fl.loc < jj <= fl.loc + fl.nd, op.fields)
            row = op.fields[r].test_off + op.fields[r].test_ids[ii-op.fields[r].loc, cell]
            col = op.fields[c].trial_off + op.fields[c].trial_ids[jj-op.fields[c].loc, cell]
            @test HK.sparsity_pattern(op).rowval[pos] == row && HK.sparsity_pattern(op).colptr[col] <= pos < HK.sparsity_pattern(op).colptr[col+1]
        end
    end
end

@testset "colouring: cells of one colour never share a free test dof" begin
    for prob in (poisson_2d(; n=(6, 5)), elasticity_3d(; order=2))
        op = KernelOperator(prob.res, prob.jac, prob.U, prob.V)
        @test sort(vcat(op.colors...)) == 1:num_cells(prob.Ω)
        for color in op.colors
            seen = Set{Int}()
            for cell in color, fl in op.fields, j in 1:fl.nd
                id = fl.test_ids[j, cell]
                id > 0 || continue
                gid = fl.test_off + id
                @test !(gid in seen)
                push!(seen, gid)
            end
        end
    end
end

@testset "threaded and serial assembly agree; nthreads kwarg" begin
    e = elasticity_3d(; n=(6, 4, 4))
    x = randx(e.U)
    op1 = KernelOperator(e.res, e.jac, e.U, e.V; nthreads=1)
    opN = KernelOperator(e.res, e.jac, e.U, e.V; nthreads=max(2, Threads.nthreads()))
    @test reldiff(Algebra.residual(opN, x), Algebra.residual(op1, x)) < TOL
    @test reldiff(Algebra.jacobian(opN, x), Algebra.jacobian(op1, x)) < TOL
end

@testset "residual!/jacobian! overwrite, allocate_* and zero_initial_guess" begin
    p = poisson_2d()
    op = KernelOperator(p.res, p.jac, p.U, p.V)
    x = randx(p.U)
    b = allocate_residual(op, x); A = allocate_jacobian(op, x)
    @test length(b) == num_free_dofs(p.V) && size(A) == (num_free_dofs(p.V), num_free_dofs(p.U))
    @test A !== HK.sparsity_pattern(op) && nnz(A) == nnz(HK.sparsity_pattern(op))
    fill!(b, 7.0); fill!(nonzeros(A), 7.0)
    residual!(b, op, x); jacobian!(A, op, x)
    @test reldiff(b, Algebra.residual(gridap_op(p), x)) < TOL
    @test reldiff(A, Algebra.jacobian(gridap_op(p), x)) < TOL
    @test iszero(zero_initial_guess(op)) && length(zero_initial_guess(op)) == num_free_dofs(p.U)
    b2, A2 = residual_and_jacobian!(b, A, op, x)
    @test b2 === b && A2 === A
end

@testset "Dirichlet values are read from the space at each assembly" begin
    p = poisson_2d()
    op = KernelOperator(p.res, p.jac, p.U, p.V)
    x = randx(p.U)
    b1 = Algebra.residual(op, x)
    TrialFESpace!(p.U, x -> 5.0)              # change Dirichlet data in place
    b2 = Algebra.residual(op, x)
    @test reldiff(b2, Algebra.residual(gridap_op(p), x)) < TOL
    @test !(b1 ≈ b2)
end

@testset "update: new weak form, same structure" begin
    p = poisson_2d()
    res(Λ) = (u, v) -> ∫(∇(v) ⋅ ∇(u) - Λ * v)p.dΩ
    jac(Λ) = (u, du, v) -> ∫(Λ * ∇(v) ⋅ ∇(du))p.dΩ
    op = KernelOperator(res(1.0), jac(1.0), p.U, p.V)
    op2 = HK.update(op, res(2.0), jac(3.0))
    @test op2.pattern === op.pattern && op2.colors === op.colors
    @test op2.rgroups[1].geo === op.rgroups[1].geo          # tables reused
    x = randx(p.U)
    @test reldiff(Algebra.residual(op2, x), Algebra.residual(get_algebraic_operator(FEOperator(res(2.0), jac(3.0), p.U, p.V)), x)) < TOL
    @test reldiff(Algebra.jacobian(op2, x), Algebra.jacobian(get_algebraic_operator(FEOperator(res(2.0), jac(3.0), p.U, p.V)), x)) < TOL
    # Λ is inside the integrand: a parameter, so the generated functions are reused
    for (g, g2) in zip(vcat(op.rgroups, op.jgroups), vcat(op2.rgroups, op2.jgroups))
        @test typeof(g2.pf.core) === typeof(g.pf.core)
        @test (g.lin === nothing) == (g2.lin === nothing)
        g.lin isa HK.LinearForm && @test typeof(g2.lin.apply) === typeof(g.lin.apply) && typeof(g2.lin.state) === typeof(g.lin.state)
        g.lin isa HK.FactorForm && @test typeof(g2.lin) === typeof(g.lin)
    end
end

@testset "update with other numbers in the integrand: same code, nonlinear forms" begin
    e = elasticity_3d()
    res(Λ, t) = (u, v) -> ∫(Λ * (∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)')))) - v ⋅ t)e.dΩ
    jac(Λ) = (u, du, v) -> ∫(Λ * (∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)'))))e.dΩ
    op = KernelOperator(res(1.0, VectorValue(0.0, 0.0, 0.0)), jac(1.0), e.U, e.V)
    x = randx(e.U)
    for (Λ, t) in ((0.5, VectorValue(0.1, -0.2, 0.3)), (2.0, VectorValue(1.0, 0.0, 0.0)))
        op2 = HK.update(op, res(Λ, t), jac(Λ))
        for (g, g2) in zip(vcat(op.rgroups, op.jgroups), vcat(op2.rgroups, op2.jgroups))
            @test typeof(g2.pf.core) === typeof(g.pf.core)
            g.lin isa HK.LinearForm && @test typeof(g2.lin.apply) === typeof(g.lin.apply)
            g.lin isa HK.FactorForm && @test typeof(g2.lin) === typeof(g.lin)      # same generated code
        end
        c = check(res(Λ, t), jac(Λ), e.U, e.V, x; op=op2)
        @test c.res < TOL && c.jac < TOL
        @test all(a -> a.bytes == 0, allocations(op2, x))
    end
end

@testset "auxiliary data: dense materialisation and cache_aux" begin
    e = elasticity_3d()
    uold = interpolate(x -> VectorValue(0.01x[1], 0.0, 0.02x[3]), e.U)
    res(u, v) = ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))) + v ⋅ uold)e.dΩ
    op = KernelOperator(res, e.jac, e.U, e.V)
    g = op.rgroups[1]
    auxd = HK.materialize_aux(g, 2)
    @test length(auxd) == 1 && size(auxd[1]) == (g.geo.np, num_cells(e.Ω))
    @test eltype(auxd[1]) === VectorValue{3,Float64}
    x = randx(e.U)
    opc = KernelOperator(res, e.jac, e.U, e.V; cache_aux=true)
    @test reldiff(Algebra.residual(opc, x), Algebra.residual(gridap_op((; res, jac=e.jac, U=e.U, V=e.V)), x)) < TOL
    @test opc.rgroups[1].auxcache[] !== nothing       # filled on first use
    # with cache_aux the cached values are kept even if the FE function changes
    get_free_dof_values(uold) .*= 2
    b_cached = Algebra.residual(opc, x)
    b_fresh = Algebra.residual(op, x)
    @test !(b_cached ≈ b_fresh)
    # without cache_aux the buffers are allocated once and refilled in place: same object across
    # assemblies, current values, and no allocation of np × ncells arrays per call
    b = similar(b_fresh)
    Algebra.residual!(b, op, x)
    buf = op.rgroups[1].auxcache[]
    @test buf !== nothing && buf[1] !== auxd[1]                        # a buffer owned by the group, not the one above
    get_free_dof_values(uold) .*= 2
    Algebra.residual!(b, op, x)
    @test op.rgroups[1].auxcache[][1] === buf[1]                         # same buffer
    @test reldiff(b, Algebra.residual(gridap_op((; res, jac=e.jac, U=e.U, V=e.V)), x)) < TOL   # refilled
end

@testset "integration_cells and group colours on a subdomain" begin
    model = CartesianDiscreteModel((0, 2, 0, 1), (8, 4))
    left = Bool[sum(c)[1] / length(c) < 1.0 for c in vec(collect(get_cell_coordinates(model)))]
    sub = Triangulation(model, left)
    Ω = Triangulation(model)
    cmap = HK.integration_cells(sub, Ω)
    @test length(cmap) == count(left) && cmap == Int32.(findall(left))
    @test HK.integration_cells(Ω, Ω) == Int32.(1:32)
    colors = [Int32[1, 3, 5], Int32[2, 4, 6]]
    gc = HK.group_colors(colors, Int32[2, 3, 6], 6)
    @test gc == [Int32[2], Int32[1, 3]]
end

@testset "matrix-product and loop Jacobian paths agree" begin
    e = elasticity_3d(; order=2)
    x = randx(e.U)
    op = KernelOperator(e.res, e.jac, e.U, e.V; nthreads=1)
    g = op.jgroups[1]
    @test HK.contiguous(g)
    dvals = [get_dirichlet_dof_values(s) for s in HK.field_spaces(op.U)]
    auxd = HK.materialize_aux(g, 1)
    xl = zeros(op.ndl); HK.gather_cell!(xl, op.fields, x, dvals, 1)
    K1 = zeros(op.ndl, op.ndl); K2 = zeros(op.ndl, op.ndl)
    HK.element_jacobian!(K1, g, xl, auxd, 1, map(t -> zeros(t.nn, 4), g.tables))
    HK.element_jacobian_mm!(K2, g, xl, auxd, 1, HK.mm_buffers(g)...)
    @test K1 ≈ K2
    # every point-block size (including one that does not divide the number of points, and all points)
    qb0 = HK.POINT_BLOCK[]
    for qb in (1, 5, 8, 27, 0)
        HK.POINT_BLOCK[] = qb
        fill!(K2, 0.0); HK.element_jacobian_mm!(K2, g, xl, auxd, 1, HK.mm_buffers(g)...)
        @test K1 ≈ K2
    end
    HK.POINT_BLOCK[] = qb0
end

@testset "element kernels do not allocate" begin
    for prob in (poisson_2d(), elasticity_3d(; order=2), coupled_3d())
        op = KernelOperator(prob.res, prob.jac, prob.U, prob.V)
        @test all(a -> a.bytes == 0, allocations(op, randx(prob.U)))
    end
end

@testset "repeated assembly with one thread allocates nothing (element buffers kept in the groups)" begin
    # measured through functions with explicit arguments (closures over the test's variables box them)
    rbytes(b, op, x) = (Algebra.residual!(b, op, x); Algebra.residual!(b, op, x); @allocated Algebra.residual!(b, op, x))
    jbytes(A, op, x) = (Algebra.jacobian!(A, op, x); Algebra.jacobian!(A, op, x); (@allocated Algebra.jacobian!(A, op, x)))
    # volume, Lagrangian P1/P2 2D and vector 3D, with an FE function and a function of x as data
    for (p, extra) in ((poisson_2d(; order=1), true), (poisson_2d(; order=2, simplex=true), true), (elasticity_3d(), false))
        Vw = TestFESpace(p.model, ReferenceFE(lagrangian, Float64, 1))
        wh = interpolate(x -> 1 + x[1], Vw)
        fx(x) = 2 + x[1] * x[2]
        res = extra ? ((u, v) -> p.res(u, v) + ∫(wh * fx * (u ⋅ v))p.dΩ) : p.res
        jac = extra ? ((u, du, v) -> p.jac(u, du, v) + ∫(wh * fx * (du ⋅ v))p.dΩ) : p.jac
        op = KernelOperator(res, jac, p.U, p.V; nthreads=1)
        x = randx(p.U); b = Algebra.residual(op, x); A = Algebra.jacobian(op, x)
        @test rbytes(b, op, x) == 0
        @test jbytes(A, op, x) == 0
        c = check(res, jac, p.U, p.V, x; op)
        @test c.res < TOL && c.jac < TOL
    end
    # boundary and skeleton groups
    model = CartesianDiscreteModel((0, 1, 0, 1), (4, 4))
    Ω = Triangulation(model); dΩ = Measure(Ω, 2)
    Γ = BoundaryTriangulation(model); dΓ = Measure(Γ, 2)
    Λ = SkeletonTriangulation(model); dΛ = Measure(Λ, 2)
    V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity=:L2); U = TrialFESpace(V)
    res(u, v) = ∫(u * v)dΩ + ∫(u * v)dΓ + ∫(jump(u) * jump(v))dΛ
    jac(u, du, v) = res(du, v)
    op = KernelOperator(res, jac, U, V; nthreads=1)
    x = randx(U); b = Algebra.residual(op, x); A = Algebra.jacobian(op, x)
    @test rbytes(b, op, x) == 0
    @test jbytes(A, op, x) == 0
    # the buffers are shared by the groups `update` rebinds: no new allocation after an update either
    op2 = update(op, res, jac)
    @test rbytes(b, op2, x) == 0
    # multi-field (Dirichlet values gathered per field)
    p = coupled_3d()
    op = KernelOperator(p.res, p.jac, p.U, p.V; nthreads=1)
    x = randx(p.U); b = Algebra.residual(op, x); A = Algebra.jacobian(op, x)
    @test rbytes(b, op, x) == 0 && jbytes(A, op, x) == 0
    # with threads: a fixed amount per call (the tasks), the same for a mesh 4 × larger
    w = HK.WORK_PER_TASK[]
    HK.WORK_PER_TASK[] = 1                                            # always as many tasks as threads
    try
        function threaded_bytes(n)
            p = poisson_2d(; n=(n, n), order=2)
            op = KernelOperator(p.res, p.jac, p.U, p.V; nthreads=4)
            x = randx(p.U); b = Algebra.residual(op, x)
            rbytes(b, op, x)
        end
        small, large = threaded_bytes(20), threaded_bytes(40)
        @test large <= small + 512
    finally
        HK.WORK_PER_TASK[] = w
    end
end

@testset "buffers shared by the operators update derives; block size fixed with the buffers" begin
    # alternate assemblies of an operator and of one derived by `update` (same buffers), growing the tasks
    p = elasticity_3d(; n=(3, 3, 2), order=2)
    forms(s) = ((u, v) -> s * p.res(u, v), (u, du, v) -> s * p.jac(u, du, v))
    op = KernelOperator(forms(1.0)..., p.U, p.V; nthreads=1)
    op2 = update(op, forms(2.0)...)
    @test op2.rgroups[1].work === op.rgroups[1].work
    x = randx(p.U)
    fresh(s) = KernelOperator(forms(s)..., p.U, p.V; nthreads=1)
    b1, b2 = Algebra.residual(fresh(1.0), x), Algebra.residual(fresh(2.0), x)
    A1, A2 = Algebra.jacobian(fresh(1.0), x), Algebra.jacobian(fresh(2.0), x)
    for _ in 1:2
        @test reldiff(Algebra.residual(op, x), b1) < 1e-14 && reldiff(Algebra.residual(op2, x), b2) < 1e-14
        @test reldiff(Algebra.jacobian(op, x), A1) < 1e-14 && reldiff(Algebra.jacobian(op2, x), A2) < 1e-14
    end
    op4 = update(op, forms(2.0)...)
    w = HK.WORK_PER_TASK[]
    HK.WORK_PER_TASK[] = 1
    try
        op4t = KernelOperator(forms(2.0)..., p.U, p.V; nthreads=4)              # more tasks: buffer sets copied
        @test reldiff(Algebra.jacobian(op4t, x), A2) < 1e-13
    finally
        HK.WORK_PER_TASK[] = w
    end
    # POINT_BLOCK changed after the operator was built: the buffers keep their block size (27 points per cell)
    pb = HK.POINT_BLOCK[]
    try
        for qb in (0, 27, 5)
            HK.POINT_BLOCK[] = qb
            @test reldiff(Algebra.jacobian(op, x), A1) < 1e-13
            @test reldiff(Algebra.jacobian(update(op, forms(1.0)...), x), A1) < 1e-13
        end
    finally
        HK.POINT_BLOCK[] = pb
    end
end

@testset "tuning switches read from the environment when the package is loaded" begin
    code = "using HyperFEMKernels; const H = HyperFEMKernels; print(H.MATRIX_PRODUCTS[], ' ', H.WORK_PER_TASK[], ' ', H.POINT_BLOCK[])"
    run_with(env...) = read(addenv(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`, env...), String)
    @test run_with("HFK_MATRIX_PRODUCTS" => "0", "HFK_WORK_PER_TASK" => "500", "HFK_POINT_BLOCK" => "4") == "false 500 4"
    @test run_with("HFK_MATRIX_PRODUCTS" => "off", "HFK_WORK_PER_TASK" => "x", "HFK_POINT_BLOCK" => "0") == "false 20000 0"
    @test run_with("HFK_MATRIX_PRODUCTS" => "", "HFK_WORK_PER_TASK" => "", "HFK_POINT_BLOCK" => "") == "true 20000 12"
    # in the same process
    old = get(ENV, "HFK_WORK_PER_TASK", nothing)
    try
        ENV["HFK_WORK_PER_TASK"] = "777"; HK.read_env_settings!()
        @test HK.WORK_PER_TASK[] == 777
    finally
        old === nothing ? delete!(ENV, "HFK_WORK_PER_TASK") : (ENV["HFK_WORK_PER_TASK"] = old)
        HK.read_env_settings!()
    end
end

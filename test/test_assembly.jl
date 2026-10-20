Random.seed!(1)

@testset "sparsity pattern equals Gridap's and positions are consistent" begin
    for prob in (poisson_2d(), elasticity_3d(), coupled_3d())
        op = KernelOperator(prob.res, prob.jac, prob.U, prob.V)
        Ag = Algebra.jacobian(gridap_op(prob), randx(prob.U))
        @test size(op.A0) == size(Ag)
        @test op.A0.colptr == Ag.colptr && op.A0.rowval == Ag.rowval
        @test all(p -> p == 0 || 1 <= p <= nnz(op.A0), op.nzpos)
        ndl = op.ndl
        for cell in (1, size(op.nzpos, 2)), ii in 1:ndl, jj in 1:ndl
            pos = op.nzpos[ii+(jj-1)*ndl, cell]
            pos == 0 && continue
            r = findfirst(fl -> fl.loc < ii <= fl.loc + fl.nd, op.fields)
            c = findfirst(fl -> fl.loc < jj <= fl.loc + fl.nd, op.fields)
            row = op.fields[r].test_off + op.fields[r].test_ids[ii-op.fields[r].loc, cell]
            col = op.fields[c].trial_off + op.fields[c].trial_ids[jj-op.fields[c].loc, cell]
            @test op.A0.rowval[pos] == row && op.A0.colptr[col] <= pos < op.A0.colptr[col+1]
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
    @test A !== op.A0 && nnz(A) == nnz(op.A0)
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
    @test op2.A0 === op.A0 && op2.nzpos === op.nzpos && op2.colors === op.colors
    @test op2.rgroups[1].geo === op.rgroups[1].geo          # tables reused
    x = randx(p.U)
    @test reldiff(Algebra.residual(op2, x), Algebra.residual(get_algebraic_operator(FEOperator(res(2.0), jac(3.0), p.U, p.V)), x)) < TOL
    @test reldiff(Algebra.jacobian(op2, x), Algebra.jacobian(get_algebraic_operator(FEOperator(res(2.0), jac(3.0), p.U, p.V)), x)) < TOL
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
end

@testset "element kernels do not allocate" begin
    for prob in (poisson_2d(), elasticity_3d(; order=2), coupled_3d())
        op = KernelOperator(prob.res, prob.jac, prob.U, prob.V)
        @test all(a -> a.bytes == 0, allocations(op, randx(prob.U)))
    end
end

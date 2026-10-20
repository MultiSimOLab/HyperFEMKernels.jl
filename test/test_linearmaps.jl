Random.seed!(1)

@testset "dependency and state inputs of nodes" begin
    p = poisson_2d()
    f = HK.capture(p.jac, HK.probes(p.U, :unknown), HK.probes(p.U, :trial), HK.probes(p.V, :test))[1][1]
    t = f.terms[1].expr
    @test HK.dependent(t)
    acc = HK.state_inputs!(HK.Node[], t)
    @test isempty(acc)                          # ∇v⋅∇du: no state input
    e = elasticity_3d()
    fe = HK.capture(e.res, HK.probes(e.U, :unknown), HK.probes(e.V, :test))[1][1]
    acc = HK.state_inputs!(HK.Node[], fe.terms[1].expr)
    @test length(acc) == 1 && acc[1] isa HK.Op   # P(F)
    @test !HK.dependent(acc[1])
end

@testset "split_sum separates additive slot-dependent parts" begin
    p = poisson_2d()
    f = HK.capture(p.res, HK.probes(p.U, :unknown), HK.probes(p.V, :test))[1][1]
    parts = HK.split_sum(f.terms[1].expr)
    @test length(parts) == 2
    @test Set(last.(parts)) == Set([1.0, -1.0])
end

@testset "linear map reproduces dual-number coefficients (residual and Jacobian)" begin
    for prob in (poisson_2d(), elasticity_3d(), coupled_3d())
        x = randx(prob.U)
        opL = KernelOperator(prob.res, prob.jac, prob.U, prob.V; linear_maps=true, nthreads=1)
        opD = KernelOperator(prob.res, prob.jac, prob.U, prob.V; linear_maps=false, nthreads=1)
        @test all(g -> g.lin === nothing, opD.rgroups) && all(g -> g.lin === nothing, opD.jgroups)
        bL = Algebra.residual(opL, x); bD = Algebra.residual(opD, x)
        AL = Algebra.jacobian(opL, x); AD = Algebra.jacobian(opD, x)
        @test reldiff(bL, bD) < TOL
        @test reldiff(AL, AD) < TOL
    end
end

@testset "where the shortcut applies" begin
    p = poisson_2d()
    op = KernelOperator(p.res, p.jac, p.U, p.V)
    @test all(g -> g.lin !== nothing, op.rgroups)      # (∇u, f) are state inputs; constants
    @test all(g -> g.lin !== nothing, op.jgroups)      # constant coefficients
    e = elasticity_3d()
    op = KernelOperator(e.res, e.jac, e.U, e.V)
    @test all(g -> g.lin !== nothing, op.rgroups)      # linear in P(F)
    @test all(g -> g.lin === nothing, op.jgroups)      # dP(F, dF): no single state input
    # analytic tangent as a 4th-order tensor: Jacobian qualifies
    Cof(F) = C_svk()                                     # state-dependent in general; constant here
    jacC(u, du, v) = ∫(∇(v)' ⊙ (dbl_contract ∘ (Cof ∘ (Fk ∘ (∇(u)')), ∇(du)')))e.dΩ
    opC = KernelOperator(e.res, jacC, e.U, e.V)
    @test all(g -> g.lin !== nothing, opC.jgroups)
    x = randx(e.U)
    AC = Algebra.jacobian(opC, x)
    Ag = Algebra.jacobian(get_algebraic_operator(FEOperator(e.res, jacC, e.U, e.V)), x)
    @test reldiff(AC, Ag) < TOL
end

@testset "a nonlinear dependence on the state input falls back to dual numbers" begin
    p = poisson_2d()
    K(u) = TensorValue(1 + u^2, 0.0, 0.0, 2.0)
    jac(u, du, v) = ∫(∇(v) ⋅ ((K ∘ u) ⋅ ∇(du)) + v * du * du)p.dΩ   # second term is not bilinear: skipped
    res(u, v) = ∫(∇(v) ⋅ ((K ∘ u) ⋅ ∇(u)))p.dΩ
    op = KernelOperator(res, jac, p.U, p.V)
    @test all(g -> g.lin !== nothing, op.rgroups)
end

@testset "apply_maps on hand-built maps" begin
    maps = [HK.SparseMap([(Int32(1), Int32(1), Int32(1), 2.0), (Int32(3), Int32(1), Int32(2), -1.0)]),
            HK.SparseMap([(Int32(2), Int32(2), Int32(1), 1.0), (Int32(2), Int32(0), Int32(0), 4.0)])]
    ST = Tuple{VectorValue{2,Float64},Float64}
    ap = HK.generate_apply(maps, ST, 3)
    vals = HK.map_values(maps, (1.0, 0.5))
    @test vals == (2.0, -1.0, 0.5, 2.0)
    lf = HK.LinearForm{Nothing,Tuple{},Tuple{Float64,Float64},ST,typeof(ap),typeof(vals)}(nothing, (), (1.0, 0.5), maps, ap, vals, :())
    out = HK.apply_maps(lf, (VectorValue(3.0, 4.0), 7.0), Val(3))
    @test out ≈ SVector(6.0, 3.5 + 2.0, -4.0)
end

@testset "numbers are parameters (not inputs) and several state inputs are allowed (dynamics-like term)" begin
    e = elasticity_3d()
    uold = interpolate(x -> VectorValue(0.01x[1], 0.0, 0.02x[3]), e.U)
    vold = interpolate(x -> VectorValue(0.0, 0.1x[2], 0.0), e.U)
    Δt = 0.1
    res(u, v) = ∫(960.0 * (2 / Δt^2 * (u - uold) - 2 / Δt * vold) ⋅ v + ∇(v)' ⊙ (0.5 * (P_svk ∘ (Fk ∘ (∇(u)')))))e.dΩ
    jac(u, du, v) = ∫(960.0 * (2 / Δt^2 * du) ⋅ v + ∇(v)' ⊙ (0.5 * (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)'))))e.dΩ
    f = HK.capture(res, HK.probes(e.U, :unknown), HK.probes(e.V, :test))[1][1]
    @test any(n -> n isa HK.Const, HK.state_inputs!(HK.Node[], f.terms[1].expr)) == false
    op = KernelOperator(res, jac, e.U, e.V)
    @test all(g -> g.lin !== nothing, op.rgroups)     # inputs: u, uold, vold, P(F); numbers are parameters
    x = randx(e.U)
    c = check(res, jac, e.U, e.V, x; op)
    @test c.res < TOL && c.jac < TOL
    @test all(a -> a.bytes == 0, allocations(op, x))
end

@testset "maps: positions and path do not depend on the parameter values" begin
    p = poisson_2d()
    # a parameter that is zero must not remove entries: same positions (same code) as for nonzero values
    res(Λ) = (u, v) -> ∫(∇(v) ⋅ ∇(u) + Λ * v * u)p.dΩ
    jac(Λ) = (u, du, v) -> ∫(∇(v) ⋅ ∇(du) + Λ * v * du)p.dΩ
    op0 = KernelOperator(res(0.0), jac(0.0), p.U, p.V)
    op1 = KernelOperator(res(2.5), jac(2.5), p.U, p.V)
    for (g0, g1) in zip(vcat(op0.rgroups, op0.jgroups), vcat(op1.rgroups, op1.jgroups))
        @test g0.lin !== nothing && g1.lin !== nothing
        @test [(i, g, k) for m in g0.lin.maps for (i, g, k, _) in m] == [(i, g, k) for m in g1.lin.maps for (i, g, k, _) in m]
        @test typeof(g0.lin.apply) === typeof(g1.lin.apply)
    end
    x = randx(p.U)
    c = check(res(0.0), jac(0.0), p.U, p.V, x; op=op0); @test c.res < TOL && c.jac < TOL
    # a zero parameter must not hide a term that is not linear in the state inputs (w ⊗ w)
    w = CellField(x -> VectorValue(1.0 + x[1], 0.5), p.Ω)
    jacw(Λ) = (u, du, v) -> ∫(∇(v) ⋅ ∇(du) + Λ * (w ⋅ ∇(du)) * (w ⋅ ∇(v)))p.dΩ
    resw(Λ) = (u, v) -> ∫(∇(v) ⋅ ∇(u) + Λ * (w ⋅ ∇(u)) * (w ⋅ ∇(v)))p.dΩ
    opw0 = KernelOperator(resw(0.0), jacw(0.0), p.U, p.V)
    opw1 = KernelOperator(resw(1.0), jacw(1.0), p.U, p.V)
    @test (opw0.jgroups[1].lin === nothing) == (opw1.jgroups[1].lin === nothing)
    c = check(resw(1.0), jacw(1.0), p.U, p.V, x; op=HK.update(opw0, resw(1.0), jacw(1.0))); @test c.res < TOL && c.jac < TOL
end

Random.seed!(1)

@testset "dependency and state inputs of nodes" begin
    p = poisson_2d()
    f = HK.capture(p.jac, HK.probes(p.U, :unknown), HK.probes(p.U, :trial), HK.probes(p.V, :test))[1]
    t = f.terms[1].expr
    @test HK.dependent(t)
    acc = HK.state_inputs!(HK.Node[], t)
    @test isempty(acc)                          # ∇v⋅∇du: no state input
    e = elasticity_3d()
    fe = HK.capture(e.res, HK.probes(e.U, :unknown), HK.probes(e.V, :test))[1]
    acc = HK.state_inputs!(HK.Node[], fe.terms[1].expr)
    @test length(acc) == 1 && acc[1] isa HK.Op   # P(F)
    @test !HK.dependent(acc[1])
end

@testset "split_sum separates additive slot-dependent parts" begin
    p = poisson_2d()
    f = HK.capture(p.res, HK.probes(p.U, :unknown), HK.probes(p.V, :test))[1]
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
    lf = HK.LinearForm{Nothing,Tuple{},Tuple{Float64,Float64},Tuple{VectorValue{2,Float64},Float64}}(
        nothing, (), (1.0, 0.5), maps, :())
    out = HK.apply_maps(lf, (VectorValue(3.0, 4.0), 7.0), Val(3))
    @test out ≈ SVector(6.0, 3.5 + 2.0, -4.0)
end

@testset "constants are folded and several state inputs are allowed (dynamics-like term)" begin
    e = elasticity_3d()
    uold = interpolate(x -> VectorValue(0.01x[1], 0.0, 0.02x[3]), e.U)
    vold = interpolate(x -> VectorValue(0.0, 0.1x[2], 0.0), e.U)
    Δt = 0.1
    res(u, v) = ∫(960.0 * (2 / Δt^2 * (u - uold) - 2 / Δt * vold) ⋅ v + ∇(v)' ⊙ (0.5 * (P_svk ∘ (Fk ∘ (∇(u)')))))e.dΩ
    jac(u, du, v) = ∫(960.0 * (2 / Δt^2 * du) ⋅ v + ∇(v)' ⊙ (0.5 * (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)'))))e.dΩ
    f = HK.capture(res, HK.probes(e.U, :unknown), HK.probes(e.V, :test))[1]
    @test any(n -> n isa HK.Const, HK.state_inputs!(HK.Node[], f.terms[1].expr)) == false
    op = KernelOperator(res, jac, e.U, e.V)
    @test all(g -> g.lin !== nothing, op.rgroups)     # inputs: u, uold, vold, P(F); constants folded
    x = randx(e.U)
    c = check(res, jac, e.U, e.V, x; op)
    @test c.res < TOL && c.jac < TOL
    @test all(a -> a.bytes == 0, allocations(op, x))
end

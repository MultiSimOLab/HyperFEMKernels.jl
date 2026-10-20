@testset "clear errors for unsupported inputs" begin
    p = poisson_2d()
    # weak form that does not return integrals
    @test_throws ErrorException KernelOperator((u, v) -> 1.0, p.jac, p.U, p.V)
    # mismatched number of fields
    c = coupled_3d()
    @test_throws ErrorException KernelOperator(c.res, c.jac, c.U, p.V)
end

@testset "check reports relative differences" begin
    p = poisson_2d()
    c = check(p.res, p.jac, p.U, p.V, randx(p.U))
    @test c.res < TOL && c.jac < TOL
    wrong(u, v) = ∫(2.0 * ∇(v) ⋅ ∇(u))p.dΩ
    c2 = check(wrong, p.jac, p.U, p.V, randx(p.U); op=KernelOperator(p.res, p.jac, p.U, p.V))
    @test c2.res > 0.1
end

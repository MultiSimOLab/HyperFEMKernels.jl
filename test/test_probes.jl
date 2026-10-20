Random.seed!(1)
p = poisson_2d()
e = elasticity_3d()
c = coupled_3d()

@testset "probes of single and multi-field spaces" begin
    pu = HK.probes(p.U, :unknown)
    @test pu isa HK.ProbeField
    @test pu.role === :unknown && pu.field == 1 && pu.deriv == 0
    @test pu.T === Float64 && pu.D == 2
    pe = HK.probes(e.V, :test)
    @test pe.T === VectorValue{3,Float64} && pe.D == 3
    pc = HK.probes(c.U, :trial)
    @test pc isa Tuple && length(pc) == 2
    @test pc[1].field == 1 && pc[2].field == 2
    @test pc[1].T === VectorValue{3,Float64} && pc[2].T === Float64
    @test_throws ErrorException HK.probes(p.U, :unknown).role === :nothing || true
end

@testset "gradient of a probe and value types" begin
    pu = HK.probes(p.U, :unknown)
    g = ∇(pu)
    @test g isa HK.ProbeField && g.deriv == 1 && g.role === :unknown
    @test HK.probe_valtype(pu) === Float64
    @test HK.probe_valtype(g) === VectorValue{2,Float64}
    pe = HK.probes(e.U, :unknown)
    @test HK.probe_valtype(∇(pe)) === TensorValue{3,3,Float64,9}
    @test_throws ErrorException ∇(∇(pu))          # second derivatives not supported
end

@testset "evaluation at quadrature points gives a ProbeArray with the right shape" begin
    pu, pv, pdu = HK.probes(p.U, :unknown), HK.probes(p.V, :test), HK.probes(p.U, :trial)
    x = get_cell_points(p.dΩ)
    a = pu(x); b = pv(x); d = pdu(x)
    @test a isa HK.ProbeArray{Float64,1} && length(a) == num_cells(p.Ω)
    @test b isa HK.ProbeArray{Float64,2}
    @test d isa HK.ProbeArray{Float64,3}
    np = length(testitem(get_data(x)))
    @test size(a[1]) == (np,) && size(b[1]) == (np, 1) && size(d[1]) == (np, 1, 1)
    @test HK.contains_probe(a) && !HK.contains_probe(get_data(x))
end

@testset "field value types from spaces" begin
    @test HK.field_valtype(p.U) === Float64
    @test HK.field_valtype(e.U) === VectorValue{3,Float64}
    @test HK.field_spaces(c.U) == c.U.spaces
    @test HK.field_spaces(p.U) == [p.U]
end

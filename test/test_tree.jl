Random.seed!(1)
p = poisson_2d()
c = coupled_3d()

capture_res(prob) = HK.capture(prob.res, HK.probes(prob.U, :unknown), HK.probes(prob.V, :test))
capture_jac(prob) = HK.capture(prob.jac, HK.probes(prob.U, :unknown), HK.probes(prob.U, :trial), HK.probes(prob.V, :test))

@testset "capture returns one Form per triangulation with integral terms" begin
    forms = capture_res(p)
    @test length(forms) == 1
    f = forms[1]
    @test f.trian === p.Ω
    @test length(f.terms) == 1           # ∫(∇v⋅∇u - f v)dΩ is a single integral
    t = f.terms[1]
    @test t.coef == 1.0
    @test t.expr isa HK.Op
    @test length(t.weights) == num_cells(p.Ω)
end

@testset "leaves and auxiliary nodes" begin
    f = capture_res(p)[1]
    tl = HK.leaves(f.terms, :test); ul = HK.leaves(f.terms, :unknown)
    @test length(tl) == 2 && Set((l.deriv for l in tl)) == Set([0, 1])   # v and ∇v
    @test length(ul) == 1 && ul[1].deriv == 1                              # only ∇u
    @test isempty(HK.leaves(f.terms, :trial))
    aux = HK.aux_nodes(f.terms)
    @test length(aux) == 1                                                 # f evaluated at points
    fj = capture_jac(p)[1]
    @test length(HK.leaves(fj.terms, :trial)) == 1
    @test isempty(HK.aux_nodes(fj.terms))
end

@testset "coefficients: sums, differences and scalings of integrals" begin
    dΩ = p.dΩ
    r1(u, v) = ∫(∇(v) ⋅ ∇(u))dΩ + 2.0 * ∫(v * u)dΩ - ∫(v)dΩ + (-3.0) * ∫(v)dΩ
    f = HK.capture(r1, HK.probes(p.U, :unknown), HK.probes(p.V, :test))[1]
    @test sort([t.coef for t in f.terms]) == [-3.0, -1.0, 1.0, 2.0]
    r2(u, v) = ∫(v * u)dΩ * 4.0
    f2 = HK.capture(r2, HK.probes(p.U, :unknown), HK.probes(p.V, :test))[1]
    @test [t.coef for t in f2.terms] == [4.0]
    r3(u, v) = ∫(v)dΩ - ∫(v * u)dΩ * 2.0
    f3 = HK.capture(r3, HK.probes(p.U, :unknown), HK.probes(p.V, :test))[1]
    @test sort([t.coef for t in f3.terms]) == [-2.0, 1.0]
end

@testset "multi-field capture keeps field indices" begin
    f = capture_jac(c)[1]
    ls = HK.leaves(f.terms, :test)
    @test Set(l.field for l in ls) == Set([1, 2])
    ls = HK.leaves(f.terms, :trial)
    @test Set(l.field for l in ls) == Set([1, 2])
end

@testset "integrals on two measures become separate terms of one form" begin
    dΩ1 = Measure(p.Ω, 2); dΩ2 = Measure(p.Ω, 4)
    r(u, v) = ∫(∇(v) ⋅ ∇(u))dΩ1 + ∫(v * u)dΩ2
    f = HK.capture(r, HK.probes(p.U, :unknown), HK.probes(p.V, :test))
    @test length(f) == 1 && length(f[1].terms) == 2
    @test f[1].terms[1].weights !== f[1].terms[2].weights
end

@testset "numbers in the integrand become Const nodes" begin
    dΩ = p.dΩ
    r(u, v) = ∫(3.0 * v * u)dΩ
    f = HK.capture(r, HK.probes(p.U, :unknown), HK.probes(p.V, :test))[1]
    consts = HK.Node[]
    walk(n::HK.Op) = foreach(walk, n.args)
    walk(n::HK.Const) = push!(consts, n)
    walk(n) = nothing
    walk(f.terms[1].expr)
    @test length(consts) == 1 && consts[1].value == 3.0
    @test isempty(HK.aux_nodes(f.terms))
end

@testset "show" begin
    f = capture_res(p)[1]
    s = sprint(show, MIME"text/plain"(), f)
    @test occursin("Form on", s) && occursin("test[1]", s) && occursin("unknown[1] ∇", s)
end

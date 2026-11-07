# p-multigrid transfers (S4-A step 1): the prolongation between nested Lagrangian spaces on the same mesh is
# exact, its transpose is the restriction, Dirichlet dofs are never written, the injection of a state reads
# the fine FE function at the coarse nodes, and both transfers are allocation-free.
Random.seed!(3)

"Nested spaces of orders `pc < pf` on one Cartesian mesh (vector field, Dirichlet by tags and masks)."
function nested_vector(; n=(3, 2, 2), pc=1, pf=2, masks=nothing, dirichlet=x -> VectorValue(0.01x[1], -0.02x[2], 0.03x[3]^2))
    model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), n)
    T = VectorValue{3,Float64}
    # the Dirichlet boundary must be the same closed set on both levels: vertex tags, or the whole boundary ("boundary").
    # A face tag alone (21) is the open face in Gridap's Cartesian labeling: its edges and vertices are not Dirichlet, so
    # Q1 would have no Dirichlet dof on it while Q2 has the face centre — different Dirichlet boundaries.
    kw = masks === nothing ? (; dirichlet_tags=[1, 3, 5, 7]) : (; dirichlet_tags=["boundary"], dirichlet_masks=masks)
    Vc = TestFESpace(model, ReferenceFE(lagrangian, T, pc); kw...)
    Vf = TestFESpace(model, ReferenceFE(lagrangian, T, pf); kw...)
    (; model, Uc=TrialFESpace(Vc, dirichlet), Uf=TrialFESpace(Vf, dirichlet), Vc, Vf)
end

"Nested multi-field (u, φ) spaces of orders `pc < pf`."
function nested_coupled(; n=(2, 2, 2), pc=1, pf=2)
    model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), n)
    T = VectorValue{3,Float64}
    mk(p) = (TestFESpace(model, ReferenceFE(lagrangian, T, p), dirichlet_tags=[1, 2]),
             TestFESpace(model, ReferenceFE(lagrangian, Float64, p), dirichlet_tags=[7, 8]))
    Vuc, Vφc = mk(pc); Vuf, Vφf = mk(pf)
    du = x -> VectorValue(0.1x[2], 0.0, -0.1x[1]); dφ = x -> 0.1 + x[3]
    Uc = MultiFieldFESpace([TrialFESpace(Vuc, du), TrialFESpace(Vφc, dφ)])
    Uf = MultiFieldFESpace([TrialFESpace(Vuf, du), TrialFESpace(Vφf, dφ)])
    (; model, Uc, Uf)
end

"The trial space with homogeneous Dirichlet values (the prolongation acts on free dofs only)."
homogeneous(U::TrialFESpace) = TrialFESpace(U.space)
homogeneous(U::MultiFieldFESpace) = MultiFieldFESpace([homogeneous(s) for s in U.spaces])

"Free values of Gridap's interpolation of the coarse FE function with free values `xc` and zero Dirichlet values into `Uf`."
gridap_prolongation(Uc, Uf, xc) = get_free_dof_values(interpolate(FEFunction(homogeneous(Uc), xc), homogeneous(Uf)))

@testset "prolongation is the exact embedding (Q1→Q2, Q2→Q3, Q1→Q3, P1→P2)" begin
    for (name, sp) in (("Q1→Q2", nested_vector()), ("Q2→Q3", nested_vector(; n=(2, 2, 1), pc=2, pf=3)),
                       ("Q1→Q3", nested_vector(; n=(2, 2, 1), pc=1, pf=3)), ("Q1→Q2 masks", nested_vector(; masks=[(true, false, true)])))
        P = Prolongation(sp.Uc, sp.Uf)
        @test size(P) == (num_free_dofs(sp.Uf), num_free_dofs(sp.Uc))
        xc = randn(num_free_dofs(sp.Uc))
        yf = P * xc
        @test reldiff(yf, gridap_prolongation(sp.Uc, sp.Uf, xc)) < 1e-9     # Q3: Gridap's basis round-off ~1e-10
        y2 = zeros(size(P, 1)); prolongate!(y2, P, xc); @test y2 == yf
        y3 = zeros(size(P, 1)); mul!(y3, P, xc); @test y3 == yf
    end
    # simplices: P1 ⊂ P2 (dense path, no tensor product needed by the transfers)
    model = simplexify(CartesianDiscreteModel((0, 1, 0, 1), (3, 2)))
    Vc = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), dirichlet_tags="boundary")
    Vf = TestFESpace(model, ReferenceFE(lagrangian, Float64, 2), dirichlet_tags="boundary")
    Uc = TrialFESpace(Vc, x -> x[1]); Uf = TrialFESpace(Vf, x -> x[1])
    P = Prolongation(Uc, Uf)
    xc = randn(num_free_dofs(Uc))
    @test reldiff(P * xc, gridap_prolongation(Uc, Uf, xc)) < 1e-12
end

@testset "restriction is the transpose; Dirichlet dofs never written" begin
    for sp in (nested_vector(), nested_vector(; n=(2, 2, 1), pc=2, pf=3), nested_vector(; masks=[(true, false, true)]))
        P = Prolongation(sp.Uc, sp.Uf)
        nf, nc = size(P)
        xc = randn(nc); yf = randn(nf)
        @test dot(transpose(P) * yf, xc) ≈ dot(yf, P * xc) atol = 1e-12 * nf
        # the dense matrix of P from unit vectors; Pᵀ from the restriction of unit vectors
        Pm = reduce(hcat, [P * Float64.(1:nc .== j) for j in 1:nc])
        Rm = reduce(hcat, [transpose(P) * Float64.(1:nf .== i) for i in 1:nf])
        @test Rm ≈ Pm' atol = 1e-13
        # coarse Dirichlet dofs are read as zero: a coarse FE function with Dirichlet values and zero free values prolongs to zero
        @test iszero(P * zeros(nc))
    end
end

@testset "partition of unity: without Dirichlet dofs the rows of P sum to one" begin
    model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (2, 2, 1))
    for (pc, pf) in ((1, 2), (2, 3), (1, 3))
        Uc = TrialFESpace(TestFESpace(model, ReferenceFE(lagrangian, Float64, pc)))
        Uf = TrialFESpace(TestFESpace(model, ReferenceFE(lagrangian, Float64, pf)))
        P = Prolongation(Uc, Uf)
        @test all(≈(1.0; atol=1e-9), P * ones(size(P, 2)))
    end
end

@testset "multi-field (u, φ) Q1→Q2" begin
    sp = nested_coupled()
    P = Prolongation(sp.Uc, sp.Uf)
    xc = randn(size(P, 2))
    @test reldiff(P * xc, gridap_prolongation(sp.Uc, sp.Uf, xc)) < 1e-12
    yf = randn(size(P, 1))
    @test dot(transpose(P) * yf, xc) ≈ dot(yf, P * xc) atol = 1e-12 * size(P, 1)
end

@testset "injection of the state: fine FE function read at the coarse nodes (free and Dirichlet values)" begin
    for sp in (nested_vector(), nested_vector(; n=(2, 2, 1), pc=1, pf=3), nested_vector(; n=(2, 2, 1), pc=2, pf=3),
               nested_vector(; masks=[(true, false, true)]))
        P = Prolongation(sp.Uc, sp.Uf)
        xf = randn(size(P, 1)); df = HK.dirichlet_values(sp.Uf)
        uh = FEFunction(sp.Uf, xf)
        xc = zeros(size(P, 2)); dc = zeros(num_dirichlet_dofs(sp.Uc))
        inject!(xc, dc, P, xf, df)
        ref = interpolate_everywhere(uh, sp.Uc)            # Gridap's nodal interpolation = injection for Lagrangian spaces
        @test reldiff(xc, get_free_dof_values(ref)) < 1e-9
        @test reldiff(dc, ref.dirichlet_values) < 1e-8          # Q3: Gridap's basis round-off
        # a coarse FE function prolonged and injected back is itself (interpolation of a coarse polynomial is exact)
        yc = randn(size(P, 2)); yf = P * yc
        xc2 = zeros(size(P, 2)); inject!(xc2, dc, P, yf, zeros(length(df)))
        @test reldiff(xc2, yc) < 1e-9
    end
    sp = nested_coupled()
    P = Prolongation(sp.Uc, sp.Uf)
    xf = randn(size(P, 1)); df = HK.dirichlet_values(sp.Uf)
    xc = zeros(size(P, 2)); dc = zeros(length(HK.dirichlet_values(sp.Uc)))
    inject!(xc, dc, P, xf, df)
    ref = interpolate_everywhere(FEFunction(sp.Uf, xf), sp.Uc)
    @test reldiff(xc, get_free_dof_values(ref)) < 1e-12
    @test reldiff(dc, reduce(vcat, (f.dirichlet_values for f in ref.single_fe_functions))) < 1e-12
end

@testset "transfers are allocation-free and threaded by colour" begin
    sp = nested_vector(; n=(4, 3, 3))
    P = Prolongation(sp.Uc, sp.Uf)
    @test sum(length, P.colors) == num_cells(sp.model)
    xc = randn(size(P, 2)); yf = zeros(size(P, 1)); yc = zeros(size(P, 2))
    prolongate!(yf, P, xc); restrict!(yc, P, yf)
    a1 = @allocated prolongate!(yf, P, xc)
    a2 = @allocated restrict!(yc, P, yf)
    if Threads.nthreads() == 1
        @test a1 == 0 && a2 == 0
    else
        @test a1 < 2000 * length(P.colors) * Threads.nthreads() && a2 < 2000 * length(P.colors) * Threads.nthreads()   # task overhead only
    end
end

@testset "errors: different meshes, non-nested or non-nodal spaces, different Dirichlet boundaries, wrong lengths" begin
    sp = nested_vector()
    # an open face tag: Dirichlet on Q2 (face centre) but not on Q1 → the levels have different Dirichlet boundaries
    V1 = TestFESpace(sp.model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), dirichlet_tags=[21])
    V2 = TestFESpace(sp.model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 2), dirichlet_tags=[21])
    @test_throws ErrorException Prolongation(TrialFESpace(V1), TrialFESpace(V2))
    other = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (2, 2, 2))
    Vo = TestFESpace(other, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), dirichlet_tags=[1])
    @test_throws ErrorException Prolongation(TrialFESpace(Vo, x -> zero(VectorValue{3,Float64})), sp.Uf)
    @test_throws ErrorException Prolongation(sp.Uf, sp.Uc)                 # coarse with more nodes than fine
    Vs = TestFESpace(sp.model, ReferenceFE(lagrangian, Float64, 1), dirichlet_tags=[1])
    @test_throws ErrorException Prolongation(TrialFESpace(Vs, x -> 0.0), sp.Uf)   # different number of components
    P = Prolongation(sp.Uc, sp.Uf)
    @test_throws DimensionMismatch prolongate!(zeros(3), P, zeros(size(P, 2)))
    @test_throws DimensionMismatch restrict!(zeros(size(P, 2)), P, zeros(3))
end

Random.seed!(1)

function tables_for(prob, order)
    xq = quad_points(prob.dΩ)
    w = quad_weights(prob.dΩ)
    D = num_cell_dims(prob.Ω)
    HK.GeoTable(prob.Ω, xq, w), HK.FieldTable(prob.U, xq, D), xq, w
end

@testset "geometry table: Jacobian and volume of a Cartesian mesh" begin
    p = poisson_2d(; n=(4, 2))
    geo, _, xq, w = tables_for(p, 1)
    @test geo.np == length(xq) && geo.ng == 4
    vol = 0.0
    for cell in 1:num_cells(p.Ω), q in 1:geo.np
        dV, iJt = HK.point_geometry(geo, q, cell)
        vol += dV
        @test iJt ≈ inv(TensorValue(0.25, 0.0, 0.0, 0.5))   # cell size 1/4 × 1/2
    end
    @test vol ≈ 1.0
end

@testset "field table: nodes, components, local dof layout, values and gradients" begin
    for (prob, order, nn, nc) in ((poisson_2d(), 1, 4, 1), (poisson_2d(; order=2), 2, 9, 1),
                                  (elasticity_3d(), 1, 8, 3), (elasticity_3d(; order=2), 2, 27, 3))
        geo, t, xq, w = tables_for(prob, order)
        @test t.nn == nn && t.nc == nc
        @test sort(vec(t.ldof)) == 1:nn*nc
        @test size(t.N) == (length(xq), nn) && size(t.dN) == (length(xq), nn)
        @test t.Nt == permutedims(t.N)
        @test all(sum(t.N[q, :]) ≈ 1.0 for q in 1:length(xq))          # partition of unity
        @test all(sum(t.dN[q, :]) ≈ zero(VectorValue{num_cell_dims(prob.Ω),Float64}) for q in 1:length(xq))
        # against Gridap's basis values and gradients
        fields = HK.uniform_value(get_data(get_fe_basis(prob.U)), "basis")
        vals = evaluate(fields, xq)
        for n in 1:nn, c in 1:nc, q in 1:length(xq)
            j = t.ldof[n, c]
            vg = nc == 1 ? vals[q, j] : vals[q, j][c]
            @test vg ≈ t.N[q, n]
        end
    end
end

@testset "contiguity of component dofs (Gridap Lagrangian spaces)" begin
    _, t = tables_for(elasticity_3d(), 1)
    @test t isa HK.FieldTable{3,VectorValue{3,Float64},8,3,true}
    _, t2 = tables_for(poisson_2d(), 1)
    @test t2 isa HK.FieldTable{2,Float64,4,1,true}
end

@testset "cell dof matrix with Dirichlet ids" begin
    p = poisson_2d(; n=(3, 3))
    m = HK.cell_dof_matrix(p.U, 4)
    @test size(m) == (4, 9)
    @test any(<(0), m) && any(>(0), m)
    @test maximum(m) == num_free_dofs(p.U)
    @test -minimum(m) == num_dirichlet_dofs(p.U)
end

@testset "uniform_value accepts Fill and single-value CompressedArray" begin
    @test HK.uniform_value(HK.Fill(3, 5), "x") == 3
    ca = CompressedArray([7.0], fill(1, 4))
    @test HK.uniform_value(ca, "x") == 7.0
end

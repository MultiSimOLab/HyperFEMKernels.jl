Random.seed!(1)

@testset "SlotSpec: value/gradient presence and columns" begin
    sm = [1 2 3 4]                         # scalar field, value + ∇ in 3D
    sp = HK.SlotSpec(sm, 3)
    @test sp isa HK.SlotSpec{1,4,true,true}
    @test [HK.wcol(sp, jj) for jj in 1:4] == [1, 2, 3, 4]
    sp2 = HK.SlotSpec([0 1 2 3], 3)        # gradient only
    @test sp2 isa HK.SlotSpec{1,3,false,true}
    @test [HK.wcol(sp2, jj) for jj in 1:3] == [2, 3, 4]
    sp3 = HK.SlotSpec([5 0 0 0], 3)        # value only
    @test sp3 isa HK.SlotSpec{1,1,true,false}
    @test HK.wcol(sp3, 1) == 1
    sp0 = HK.SlotSpec(zeros(Int, 1, 4), 3)
    @test sp0 isa HK.SlotSpec{1,0,false,false}
    @test_throws ErrorException HK.SlotSpec([1 0 3 4], 3)   # partial use
end

@testset "point weights and field values against a known field (Q1 3D)" begin
    e = elasticity_3d(; n=(1, 1, 1))
    xq = quad_points(e.dΩ)
    w = quad_weights(e.dΩ)
    Vfree = TestFESpace(e.model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1))   # no Dirichlet
    geo = HK.GeoTable(e.Ω, xq, w); t = HK.FieldTable(Vfree, xq, 3)
    # u(x) = A x + b  (affine) → gradient A' (Gridap convention) everywhere
    A = TensorValue(0.1, 0.2, 0.3, -0.1, 0.05, 0.0, 0.02, 0.01, 0.4)
    uh = interpolate(x -> A ⋅ x + VectorValue(1.0, 2.0, 3.0), Vfree)
    xl = collect(get_cell_dof_values(uh)[1])
    W = zeros(t.nn, 4)
    coords = get_cell_coordinates(e.Ω)[1]
    for q in 1:geo.np
        dV, iJt = HK.point_geometry(geo, q, 1)
        HK.point_weights!(W, t, q, iJt)
        u, g = HK.field_point_values(t, xl, 0, W)
        @test g ≈ transpose(A)                      # ∇u in Gridap's convention
        xqphys = sum(W[n, 1] * coords[n] for n in 1:t.nn)
        @test u ≈ A ⋅ xqphys + VectorValue(1.0, 2.0, 3.0)
    end
end

@testset "contract_residual! against explicit sums (scalar field, value+gradient)" begin
    p = poisson_2d(; n=(1, 1))
    xq = quad_points(p.dΩ)
    w = quad_weights(p.dΩ)
    geo = HK.GeoTable(p.Ω, xq, w); t = HK.FieldTable(p.U, xq, 2)
    sp = HK.SlotSpec([1 2 3], 2)
    R = SVector(0.5, -1.0, 2.0)
    re = zeros(4); W = zeros(4, 3)
    dV, iJt = HK.point_geometry(geo, 1, 1)
    HK.point_weights!(W, t, 1, iJt)
    HK.contract_residual!(re, t, 0, sp, R, W, dV)
    for n in 1:4
        @test re[t.ldof[n, 1]] ≈ dV * (W[n, 1] * R[1] + W[n, 2] * R[2] + W[n, 3] * R[3])
    end
end

@testset "contract_jacobian! and matrix-product path agree with explicit sums (vector field)" begin
    e = elasticity_3d(; n=(1, 1, 1))
    xq = quad_points(e.dΩ)
    w = quad_weights(e.dΩ)
    geo = HK.GeoTable(e.Ω, xq, w); t = HK.FieldTable(e.U, xq, 3)
    sm = zeros(Int, 3, 4); sm[:, 2:4] .= reshape(1:9, 3, 3)'   # gradient-only slots (∂_J of comp c)
    sp = HK.SlotSpec(sm, 3)
    T = SMatrix{9,9}(randn(9, 9))
    nd = 24
    K1 = zeros(nd, nd); K2 = zeros(nd, nd)
    W = zeros(8, 4); B = zeros(geo.np * 3, 8); G = zeros(8, geo.np * 3, 3, 3)
    for q in 1:geo.np
        dV, iJt = HK.point_geometry(geo, q, 1)
        HK.point_weights!(W, t, q, iJt)
        HK.contract_jacobian!(K1, t, 0, sp, W, t, 0, sp, W, T, dV)
        HK.store_test_weights!(B, sp, W, q, 8)
        HK.store_trial_block!(G, sp, sp, W, T, q, dV, 8)
        # explicit: K[(n,c),(m,k)] += dV Σ_J Σ_L W[n,1+J] T[sm[c,J+1], sm[k,L+1]] W[m,1+L]
        for n in 1:8, c in 1:3, m in 1:8, k in 1:3
            s = 0.0
            for J in 1:3, L in 1:3
                s += W[n, 1+J] * T[sm[c, J+1], sm[k, L+1]] * W[m, 1+L]
            end
            K2[t.ldof[m, k], t.ldof[n, c]] += dV * s      # transposed storage
        end
    end
    @test K1 ≈ K2
    K3 = zeros(nd, nd)
    HK.block_products!(K3, t, 0, sp, B, t, 0, sp, G, geo.np)
    @test K3 ≈ K2
end

@testset "gemm_add! tiles (full tiles and remainders)" begin
    for (M, N, R) in ((8, 4, 5), (27, 27, 81), (3, 2, 4), (9, 5, 7))
        A4 = randn(M, R, 2, 2); B = randn(R, N)
        C = zeros(M + 2, N + 3)
        HK.gemm_add!(C, 1, 2, A4, 2, 1, B, M, N, R)
        @test C[2:M+1, 3:N+2] ≈ A4[:, :, 2, 1] * B
        @test all(C[1, :] .== 0) && all(C[:, 1:2] .== 0)
    end
end

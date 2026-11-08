# p-multigrid levels (S4-A step 2): the coarse operators rediscretised from the same weak form at the injected state.
# For a linear form with constant coefficients and nested spaces integrated with the same quadrature the rediscretised
# coarse operator is the Galerkin product Pᵀ A P exactly — one identity that checks the transfers, the Dirichlet
# handling and the coarse operators together. Nonlinear forms differ from it by O(h) at a smooth state. Lifecycle:
# forms followed per load step, lagged coarse builds, foreign operators (Dirichlet projection), non-finite guard.
Random.seed!(5)

const λ_lin, μ_lin = 2.0, 1.0
lin_stress(G) = λ_lin * tr(G) * one(G) + μ_lin * (G + G')

"Spaces of orders `orders` (finest first) on one Cartesian cube mesh, Dirichlet `g` on the whole boundary."
function level_spaces_cube(n, orders; g=x -> VectorValue(0.01x[1], -0.02x[2] * x[3], 0.005), coarse_zero=true)
    model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), n)
    T = VectorValue{3,Float64}
    sp = map(enumerate(orders)) do (l, p)
        V = TestFESpace(model, ReferenceFE(lagrangian, T, p), dirichlet_tags="boundary")
        U = l > 1 && coarse_zero ? TrialFESpace(V, x -> zero(T)) : TrialFESpace(V, g)
        (U, V)
    end
    dΩ = Measure(Triangulation(model), 2first(orders))
    (; model, spaces=collect(sp), dΩ, g)
end

linear_forms(dΩ; s=1.0) = ((u, v) -> ∫(∇(v)' ⊙ (s * (lin_stress ∘ (∇(u)'))))dΩ,
                            (u, du, v) -> ∫(∇(v)' ⊙ (s * (lin_stress ∘ (∇(du)'))))dΩ)
svk_forms(dΩ) = ((u, v) -> ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))))dΩ,
                 (u, du, v) -> ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')))dΩ)

"The Newton's side: a matrix-free operator on the fine level, prepared at `x`."
function fine_tangent(res, jac, U, V, x)
    op = KernelOperator(res, jac, U, V; jacobian=:matrixfree)
    mf = MatrixFreeJacobian(op)
    jacobian!(mf, op, x)
    op, mf
end

dense(P) = reduce(hcat, [P * Float64.(1:size(P, 2) .== j) for j in 1:size(P, 2)])
dense_mf(mf) = reduce(hcat, [mf * Float64.(1:size(mf, 2) .== j) for j in 1:size(mf, 2)])

@testset "linear form: the rediscretised coarse operator is Pᵀ A P (Q2→Q1, Q3→Q2→Q1)" begin
    for (n, orders, tol) in (((3, 2, 2), (2, 1), 1e-12), ((3, 3, 2), (3, 2, 1), 1e-9))
        c = level_spaces_cube(n, orders)
        res, jac = linear_forms(c.dΩ)
        Uf, Vf = c.spaces[1]
        x = randx(Uf)
        op, mf = fine_tangent(res, jac, Uf, Vf, x)
        lv = PLevels(c.spaces, op.res, op.jac)
        HK.update_levels!(lv, mf)
        @test HK.levels_ok(lv) && HK.nlevels(lv) == length(orders)
        Af = Matrix(Algebra.jacobian(KernelOperator(res, jac, Uf, Vf), x))
        Ptot = Matrix{Float64}(I, size(Af, 1), size(Af, 1))
        for l in 1:length(orders)-1
            Ptot = Ptot * dense(lv.P[l])
            Gal = Ptot' * Af * Ptot
            if l < length(orders) - 1          # intermediate matrix-free level
                @test norm(dense_mf(lv.mlevels[l].mf) - Gal) / norm(Gal) < tol
            else                               # assembled coarse level
                @test norm(Matrix(lv.coarse.A) - Gal) / norm(Gal) < tol
            end
        end
        @test HK.preconditioner_builds(lv) == 1
    end
end

@testset "injected state: Dirichlet values written into the coarse spaces, free values interpolated" begin
    c = level_spaces_cube((3, 2, 2), (2, 1))
    res, jac = linear_forms(c.dΩ)
    (Uf, Vf), (Uc, Vc) = c.spaces
    xf = get_free_dof_values(interpolate(x -> VectorValue(0.1x[1]^2, 0.0, -0.2x[2]), Uf))
    op, mf = fine_tangent(res, jac, Uf, Vf, xf)
    lv = PLevels(c.spaces, op.res, op.jac)
    HK.update_levels!(lv, mf)
    ref = interpolate_everywhere(FEFunction(Uf, xf), Uc)
    @test lv.coarse.x ≈ get_free_dof_values(ref) atol = 1e-12
    @test get_dirichlet_dof_values(Uc) ≈ get_dirichlet_dof_values(TrialFESpace(Vc, c.g)) atol = 1e-12   # g is in Q1: exact
end

@testset "nonlinear form: coarse tangent at the injected state differs from Pᵀ A P by O(h)" begin
    g = x -> VectorValue(0.05sin(2x[1]), 0.03x[2]^2, 0.02x[1] * x[3])
    errs = map((2, 4)) do k
        c = level_spaces_cube((k, k, k), (2, 1); g, coarse_zero=true)
        res, jac = svk_forms(c.dΩ)
        Uf, Vf = c.spaces[1]
        x = get_free_dof_values(interpolate(g, Uf))
        op, mf = fine_tangent(res, jac, Uf, Vf, x)
        lv = PLevels(c.spaces, op.res, op.jac)
        HK.update_levels!(lv, mf)
        Af = Matrix(Algebra.jacobian(KernelOperator(res, jac, Uf, Vf), x))
        Pm = dense(lv.P[1])
        Gal = Pm' * Af * Pm
        norm(Matrix(lv.coarse.A) - Gal) / norm(Gal)
    end
    @test errs[1] < 0.05
    @test errs[1] / errs[2] > 1.5          # first order in h (the state's gradient is interpolated with Q1)
end

@testset "lifecycle: forms followed per load step, lagged coarse builds, foreign operator reused" begin
    c = level_spaces_cube((3, 2, 2), (2, 1))
    res, jac = linear_forms(c.dΩ)
    Uf, Vf = c.spaces[1]
    x = randx(Uf)
    op, mf = fine_tangent(res, jac, Uf, Vf, x)
    lv = PLevels(c.spaces, op.res, op.jac; coarse_every=3)
    for _ in 1:4
        jacobian!(mf, op, x); HK.update_levels!(lv, mf)
    end
    @test lv.nupdate == 4 && HK.preconditioner_builds(lv) == 2      # updates 1 and 4
    # a new load step: the operator is `update`d with other closures → levels follow, coarse rebuilt at once
    res2, jac2 = linear_forms(c.dΩ; s=3.0)
    op2 = update(op, res2, jac2)
    jacobian!(mf, op2, x); HK.update_levels!(lv, mf)
    @test lv.nform == 1 && HK.preconditioner_builds(lv) == 3
    Pm = dense(lv.P[1])
    Af2 = Matrix(Algebra.jacobian(KernelOperator(res2, jac2, Uf, Vf), x))
    @test norm(Matrix(lv.coarse.A) - Pm' * Af2 * Pm) / norm(Af2) < 1e-12
    # the Dirichlet projection: an operator on the increment space (same free dofs, homogeneous Dirichlet) — reused
    ΔU = TrialFESpace(Vf)
    opΔ = KernelOperator(res2, jac2, ΔU, Vf; jacobian=:matrixfree)
    A_before = copy(lv.coarse.A)
    jacobian!(mf, opΔ, x); HK.update_levels!(lv, mf)
    @test lv.nskip == 1 && HK.preconditioner_builds(lv) == 3 && lv.coarse.A == A_before
    # back on the working operator: updates resume
    jacobian!(mf, op2, x); HK.update_levels!(lv, mf)
    @test lv.nskip == 1 && lv.nupdate == 7
end

@testset "non-finite guard: a NaN state clears levels_ok, the next finite state restores it" begin
    c = level_spaces_cube((2, 2, 2), (2, 1))
    res, jac = svk_forms(c.dΩ)
    Uf, Vf = c.spaces[1]
    x = randx(Uf)
    op, mf = fine_tangent(res, jac, Uf, Vf, x)
    lv = PLevels(c.spaces, op.res, op.jac)
    HK.update_levels!(lv, mf)
    @test HK.levels_ok(lv)
    xb = copy(x); xb[1] = NaN
    jacobian!(mf, op, xb); HK.update_levels!(lv, mf)
    @test !HK.levels_ok(lv) && lv.coarse.ns === nothing
    jacobian!(mf, op, x); HK.update_levels!(lv, mf)
    @test HK.levels_ok(lv) && lv.coarse.ns !== nothing
end

@testset "errors" begin
    c = level_spaces_cube((2, 2, 1), (2, 1))
    res, jac = linear_forms(c.dΩ)
    @test_throws ArgumentError PLevels(c.spaces[1:1], res, jac)
    @test_throws ArgumentError PLevels(c.spaces, res, jac; coarse_every=0)
    (Uf, Vf), (Uc, Vc) = c.spaces
    @test_throws ErrorException PLevels([(Uf, Vf), (Vc, Vc)], res, jac)          # coarse trial space not a TrialFESpace
    op, mf = fine_tangent(res, jac, Uf, Vf, randx(Uf))
    lv = PLevels(c.spaces, op.res, op.jac)
    other = level_spaces_cube((3, 2, 1), (2, 1))
    reso, jaco = linear_forms(other.dΩ)
    opo, mfo = fine_tangent(reso, jaco, other.spaces[1]..., randx(other.spaces[1][1]))
    @test_throws ErrorException HK.update_levels!(lv, mfo)                         # another number of free dofs
end

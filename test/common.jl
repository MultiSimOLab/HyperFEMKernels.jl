# Shared material (St Venant–Kirchhoff, written pointwise with Gridap's convention F = I + ∇u').
const λ_svk, μ_svk = 2.0, 1.0
E_svk(F) = 0.5 * (transpose(F) ⋅ F - one(F))
S_svk(F) = λ_svk * tr(E_svk(F)) * one(F) + 2μ_svk * E_svk(F)
P_svk(F) = F ⋅ S_svk(F)
dS_svk(F, dF) = (dE = 0.5 * (transpose(dF) ⋅ F + transpose(F) ⋅ dF); λ_svk * tr(dE) * one(F) + 2μ_svk * dE)
dP_svk(F, dF) = dF ⋅ S_svk(F) + F ⋅ dS_svk(F, dF)
Fk(∇u) = one(∇u) + ∇u

# Linear-elastic tangent as a 4th-order tensor (flattened TensorValue{9,9}): C[(iJ),(kL)]
function C_svk()
    δ(i, j) = i == j ? 1.0 : 0.0
    TensorValue{9,9}(ntuple(idx -> begin
        a, b = (idx - 1) % 9 + 1, (idx - 1) ÷ 9 + 1
        i, J = (a - 1) % 3 + 1, (a - 1) ÷ 3 + 1
        k, L = (b - 1) % 3 + 1, (b - 1) ÷ 3 + 1
        λ_svk * δ(i, J) * δ(k, L) + μ_svk * (δ(i, k) * δ(J, L) + δ(i, L) * δ(J, k))
    end, 81))
end

function poisson_2d(; n=(5, 4), order=1, simplex=false)
    model = CartesianDiscreteModel((0, 1, 0, 1), n)
    simplex && (model = simplexify(model))
    Ω = Triangulation(model); dΩ = Measure(Ω, 2order)
    V = TestFESpace(model, ReferenceFE(lagrangian, Float64, order), dirichlet_tags="boundary")
    U = TrialFESpace(V, x -> x[1] + 2x[2])
    f(x) = sin(x[1]) * x[2]
    res(u, v) = ∫(∇(v) ⋅ ∇(u) - f * v)dΩ
    jac(u, du, v) = ∫(∇(v) ⋅ ∇(du))dΩ
    (; model, Ω, dΩ, U, V, res, jac)
end

function elasticity_3d(; n=(3, 2, 2), order=1)
    model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), n)
    Ω = Triangulation(model); dΩ = Measure(Ω, 2order)
    V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, order), dirichlet_tags=[1, 3, 5, 7])
    U = TrialFESpace(V, x -> VectorValue(0.01, 0.0, 0.0))
    res(u, v) = ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))))dΩ
    jac(u, du, v) = ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')))dΩ
    (; model, Ω, dΩ, U, V, res, jac)
end

"Multi-field u–φ problem (3D, Q1) with value and gradient couplings."
function coupled_3d(; n=(2, 2, 2))
    model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), n)
    Ω = Triangulation(model); dΩ = Measure(Ω, 2)
    Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), dirichlet_tags=[1, 2])
    Vφ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), dirichlet_tags=[7, 8])
    V = MultiFieldFESpace([Vu, Vφ])
    U = MultiFieldFESpace([TrialFESpace(Vu, x -> VectorValue(0.0, 0.0, 0.0)), TrialFESpace(Vφ, x -> 0.1)])
    β = 0.3
    res((u, φ), (v, vφ)) = ∫(∇(v)' ⊙ (P_svk ∘ (Fk ∘ (∇(u)'))) + β * φ * tr(∇(v)) + ∇(vφ) ⋅ ∇(φ) +
                             β * vφ * (tr ∘ (Fk ∘ (∇(u)'))))dΩ
    jac((u, φ), (du, dφ), (v, vφ)) = ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')) + β * dφ * tr(∇(v)) +
                                       ∇(vφ) ⋅ ∇(dφ) + β * vφ * tr(∇(du)))dΩ
    (; model, Ω, dΩ, U, V, res, jac)
end

gridap_op(p) = get_algebraic_operator(FEOperator(p.res, p.jac, p.U, p.V))
randx(U; scale=0.01) = scale .* randn(num_free_dofs(U))
reldiff(a, b) = norm(a - b) / max(norm(b), eps())

quad_points(dΩ) = HK.uniform_value(get_data(get_cell_points(dΩ)), "points")
quad_weights(dΩ) = HK.uniform_value(Gridap.CellData.get_cell_quadrature(dΩ).cell_weight, "weights")

"Contraction of a flattened 4th-order tensor C[(iJ),(kL)] with a 2nd-order tensor A[kL] (pointwise)."
dbl_contract(C::TensorValue{9,9}, A::TensorValue{3,3}) = TensorValue{3,3}(ntuple(i -> sum(C[i, j] * A[j] for j in 1:9), 9))

# ---------------------------------------------------------------------------------------------
# Hessians on non-affine cells: reference reproduced from HyperFEM.jl (src/TensorAlgebra/Hessian.jl,
# PR #274): `push_hessian` pushes a reference Hessian forward with the curvature of the cell map,
#   H_abk = Jinv_ai Jinv_bj (Ĥ_ijk − ∇̂∇̂φ_ijc ∇u_ck),  ∇u = Jinv ⋅ ∇̂u,  Jinv = inv(Jt),
# and `hessian_nonaffine(u)` builds it for an FE function or basis (Gridap's ∇∇ only supports affine maps).
using Gridap.Fields: Operation
using Gridap.CellData: similar_cell_field
using Gridap.Geometry: get_cell_map

function push_hessian(Ĥ::ThirdOrderTensorValue{D,D,N}, ∇̂u::TensorValue{D,N},
                      Jt::TensorValue{D,D}, ∇∇φ::ThirdOrderTensorValue{D,D,D}) where {D,N}
    Jinv = inv(Jt)
    ∇u = Jinv ⋅ ∇̂u
    T = promote_type(eltype(Ĥ), eltype(Jinv))
    ThirdOrderTensorValue{D,D,N}(ntuple(Val(D * D * N)) do l
        a, b, k = Tuple(CartesianIndices((D, D, N))[l])
        s = zero(T)
        for i in 1:D, j in 1:D
            t = Ĥ[i, j, k]
            for c in 1:D
                t -= ∇∇φ[i, j, c] * ∇u[c, k]
            end
            s += Jinv[a, i] * Jinv[b, j] * t
        end
        s
    end)
end
function push_hessian(Ĥ::TensorValue{D,D}, ∇̂u::VectorValue{D},
                      Jt::TensorValue{D,D}, ∇∇φ::ThirdOrderTensorValue{D,D,D}) where {D}
    Jinv = inv(Jt)
    ∇u = Jinv ⋅ ∇̂u
    G = TensorValue{D,D}(ntuple(Val(D * D)) do l
        i, j = Tuple(CartesianIndices((D, D))[l])
        Ĥ[i, j] - sum(∇∇φ[i, j, c] * ∇u[c] for c in 1:D)
    end)
    Jinv ⋅ G ⋅ transpose(Jinv)
end
function hessian_nonaffine(a::CellField)
    trian = get_triangulation(a)
    cell_a = get_data(a)
    cell_Ĥ = lazy_map(Broadcasting(∇∇), cell_a)
    cell_∇̂ = lazy_map(Broadcasting(∇), cell_a)
    cell_map = get_cell_map(trian)
    cell_Jt = lazy_map(∇, cell_map)
    cell_∇∇φ = lazy_map(∇∇, cell_map)
    h = lazy_map(Broadcasting(Operation(push_hessian)), cell_Ĥ, cell_∇̂, cell_Jt, cell_∇∇φ)
    similar_cell_field(a, h, trian, DomainStyle(a))
end

"Cartesian model with a non-affine (bilinear/trilinear, through the mapped nodes) geometry; identity map when `mapped = false`."
function cartesian_model(D::Int, n; mapped=false, simplex=false)
    dom = D == 2 ? (0, 1, 0, 1) : (0, 1, 0, 1, 0, 1)
    cells = D == 2 ? n[1:2] : n
    f(x) = D == 2 ? VectorValue(x[1] + 0.1 * x[1] * x[2], x[2] + 0.05 * x[1]^2) :
           VectorValue(x[1] + 0.1 * x[1] * x[2], x[2] + 0.05 * x[3]^2, x[3] - 0.07 * x[1] * x[2])
    model = mapped ? CartesianDiscreteModel(dom, cells; map=f) : CartesianDiscreteModel(dom, cells)
    simplex ? simplexify(model) : model
end

# Problems and helpers shared between test files of different CI groups (HFK_TEST_GROUP, see runtests.jl):
# each group runs on its own, so a file cannot rely on definitions made by another file of the suite.

"Builds an operator whose form is deliberately sent to Gridap, capturing the fallback warnings (they are expected here, not noise for the test log)."
fallback_op(f, args...) = @test_logs (:warn, r"assembled by Gridap") match_mode = :any f(args...)

function product_gradient_problem(; order=1, simplex=false, n=(5, 4))
    p = poisson_2d(; n, order, simplex)
    Λ = SkeletonTriangulation(p.model); dΛ = Measure(Λ, 2order)
    γ = 10.0
    # the skeleton term makes the Gridap-assembled pattern larger than the kernel's cell pattern
    res(u, v) = ∫(∇(v) ⋅ ∇(u) - 2.0 * v + ∇(v) ⋅ ∇(u * u))p.dΩ + ∫(γ * jump(v) * jump(u))dΛ
    jac(u, du, v) = ∫(∇(v) ⋅ ∇(du) + 2.0 * (∇(v) ⋅ ∇(du * u)))p.dΩ + ∫(γ * jump(v) * jump(du))dΛ
    (; p, Λ, dΛ, res, jac)
end

"Interior-penalty Poisson problem: volume + jump of the unknown on the skeleton."
function ip_poisson(; order=1, simplex=false, n=(5, 4), γ=10.0)
    p = poisson_2d(; n, order, simplex)
    Λ = SkeletonTriangulation(p.model); dΛ = Measure(Λ, 2order)
    res(u, v) = ∫(∇(v) ⋅ ∇(u) - 2.0 * v)p.dΩ + ∫(γ * jump(v) * jump(u))dΛ
    jac(u, du, v) = ∫(∇(v) ⋅ ∇(du))p.dΩ + ∫(γ * jump(v) * jump(du))dΛ
    (; p, Λ, dΛ, res, jac)
end

"Model (0,2)×(0,1); active region Ωd without the top row of cells, split at x = 1 into Ωa (left) and Ωb (right)."
function two_subdomains(; n=(8, 4))
    model = CartesianDiscreteModel((0, 2, 0, 1), n)
    cc = [sum(c) / length(c) for c in get_cell_coordinates(model)]
    dom = [x[2] < 0.75 for x in cc]
    Ωd = Triangulation(model, dom)
    Ωa = Triangulation(model, [d && x[1] < 1 for (d, x) in zip(dom, cc)])
    Ωb = Triangulation(model, [d && x[1] >= 1 for (d, x) in zip(dom, cc)])
    (; model, Ωd, Ωa, Ωb)
end

"Channel (0,2)×(0,1)×(0,1) on tetrahedra: inlet x = 0, outlet x = 2, wall elsewhere; weak form of one step."
function oseen_channel(; n=(3, 2, 2), seed=1)
    Random.seed!(seed)
    model = simplexify(CartesianDiscreteModel((0, 2, 0, 1, 0, 1), n))
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "inlet", [25])
    add_tag_from_tags!(labels, "outlet", [26])
    add_tag_from_tags!(labels, "wall", [collect(1:20); 21; 22; 23; 24])
    ρ, μ, Δt = 1.0, 0.05, 0.01
    Ω = Triangulation(model); dΩ = Measure(Ω, 4)
    he = CellField(map(v -> (6sqrt(2) * v)^(1 / 3), collect(get_cell_measure(Ω))), Ω)
    zero3 = VectorValue(0.0, 0.0, 0.0)
    Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 2), conformity=:H1, dirichlet_tags=["inlet", "wall"])
    Vp = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity=:H1)
    V = MultiFieldFESpace([Vu, Vp])
    gin(x) = VectorValue(16 * x[2] * (1 - x[2]) * x[3] * (1 - x[3]), 0.0, 0.0)
    trial() = MultiFieldFESpace([TrialFESpace(Vu, [gin, zero3]), TrialFESpace(Vp)])
    U, U⁻, U⁻⁻ = trial(), trial(), trial()
    nf = num_free_dofs(U)
    xh⁻ = FEFunction(U⁻, randn(nf)); xh⁻⁻ = FEFunction(U⁻⁻, randn(nf))
    un, unm1 = xh⁻[1], xh⁻⁻[1]
    bdf2 = Ref(false)
    c0() = bdf2[] ? 1.5 / Δt : 1.0 / Δt
    c1() = bdf2[] ? 2.0 / Δt : 1.0 / Δt
    c2() = bdf2[] ? 0.5 / Δt : 0.0
    hist() = c1() * un - c2() * unm1
    wadv() = bdf2[] ? 2.0 * un - 1.0 * unm1 : un
    # elementwise Laplacian of uⁿ through the discontinuous P1 space
    Wlap = FESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), conformity=:L2)
    eks = [VectorValue(ntuple(i -> i == k ? 1.0 : 0.0, 3)) for k in 1:3]
    gk = [interpolate(∇(un) ⋅ eks[k], Wlap) for k in 1:3]
    update_lap! = () -> foreach(k -> Gridap.FESpaces.interpolate!(∇(un) ⋅ eks[k], get_free_dof_values(gk[k]), Wlap), 1:3)
    lapn = Operation((a, b, c) -> VectorValue(a, b, c))(∇ ⋅ gk[1], ∇ ⋅ gk[2], ∇ ⋅ gk[3])
    # outlet: Windkessel pressure (a number of the integrand, as in the original script) and Robin term; backflow
    Γo = BoundaryTriangulation(model, tags="outlet"); dΓo = Measure(Γo, 4); dΓb = Measure(Γo, 5)
    no = get_normal_vector(Γo)
    Pc = Ref(0.7)
    ZA = 2.3
    negpart(x) = 0.5 * (x - abs(x))
    τsupg(u, h) = ((2 * norm(u) / h)^2 + 9 * (4μ / (ρ * h^2))^2)^(-1 / 2)
    τpspg(u, h) = τsupg(u, h) / ρ
    τlsic(u, h) = ρ * norm(u) * h * 0.5
    Rvel(c) = (u, ∇u, w) -> ρ * c * u + ρ * (∇u' ⋅ w)
    εs(∇u) = 0.5 * (∇u + ∇u')
    function a((u, p), (v, q))
        c = c0(); w = wadv()
        τs = τsupg ∘ (un, he); τp = τpspg ∘ (un, he); τl = τlsic ∘ (un, he)
        Ru = Rvel(c) ∘ (u, ∇(u), w)
        ∫(ρ * c * (u ⋅ v))dΩ + ∫(2μ * (εs ∘ (∇(u))) ⊙ (εs ∘ (∇(v))))dΩ + ∫(ρ * ((∇(u)' ⋅ w) ⋅ v))dΩ -
        ∫(p * (∇ ⋅ v))dΩ - ∫(q * (∇ ⋅ u))dΩ +
        ∫(τs * ((∇(v)' ⋅ w) ⋅ Ru))dΩ + ∫(τs * ((∇(v)' ⋅ w) ⋅ ∇(p)))dΩ -
        ∫(τp * (∇(q) ⋅ Ru))dΩ - ∫(τp * (∇(q) ⋅ ∇(p)))dΩ +
        ∫(τl * (∇ ⋅ v) * (∇ ⋅ u))dΩ -
        ∫(ρ * (negpart ∘ (w ⋅ no)) * (u ⋅ v))dΓb + ∫(ZA * (u ⋅ no) * (v ⋅ no))dΓo
    end
    function l((v, q))
        w = wadv(); f = hist()
        τs = τsupg ∘ (un, he); τp = τpspg ∘ (un, he)
        rk = ρ * f + μ * lapn
        ∫(ρ * (f ⋅ v))dΩ + ∫(τs * ((∇(v)' ⋅ w) ⋅ rk))dΩ - ∫(τp * (∇(q) ⋅ rk))dΩ - ∫(Pc[] * (v ⋅ no))dΓo
    end
    res(x, y) = a(x, y) - l(y)
    jac(x, dx, y) = a(dx, y)
    (; model, U, V, U⁻, xh⁻, xh⁻⁻, a, l, res, jac, bdf2, Pc, update_lap!)
end

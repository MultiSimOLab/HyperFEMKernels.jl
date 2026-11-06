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

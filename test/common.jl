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

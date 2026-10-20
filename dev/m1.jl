# Milestone 1 smoke test: kernel backend vs Gridap on small problems.
using HyperFEMKernels, HyperFEM, Gridap, GridapGmsh
using Gridap.FESpaces, Gridap.Algebra
using LinearAlgebra, Printf, Random

Random.seed!(1)
report(name, c) = (@printf("%-40s res %.2e   jac %.2e\n", name, c.res, c.jac); flush(stdout))

# 1. Poisson 2D Q1, inhomogeneous Dirichlet
let
    model = CartesianDiscreteModel((0, 1, 0, 1), (6, 5))
    Ω = Triangulation(model); dΩ = Measure(Ω, 2)
    V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), dirichlet_tags="boundary")
    U = TrialFESpace(V, x -> x[1] + 2x[2])
    f(x) = sin(x[1]) * x[2]
    res(u, v) = ∫(∇(v) ⋅ ∇(u) - f * v)dΩ
    jac(u, du, v) = ∫(∇(v) ⋅ ∇(du))dΩ
    x = rand(num_free_dofs(U))
    report("Poisson 2D Q1", check(res, jac, U, V, x))
end

# 2. Neo-Hooke 3D (HyperFEM model), Q1 and Q2
for order in (1, 2)
    model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (3, 2, 2))
    Ω = Triangulation(model); dΩ = Measure(Ω, 2order)
    V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, order), dirichlet_tags=[1, 3, 5, 7])
    U = TrialFESpace(V, x -> VectorValue(0.01, 0.0, 0.0))
    Ψ, ∂ΨF, ∂ΨFF = NeoHookean3D(λ=10.0, μ=1.0)()
    F, _, _ = get_Kinematics(Kinematics(Mechano, Solid))
    res(u, v) = ∫(∇(v)' ⊙ (∂ΨF ∘ (F ∘ (∇(u)'))))dΩ
    jac(u, du, v) = ∫(∇(v)' ⊙ ((∂ΨFF ∘ (F ∘ (∇(u)'))) ⊙ (∇(du)')))dΩ
    x = 0.01 .* randn(num_free_dofs(U))
    report("Neo-Hooke 3D Q$order", check(res, jac, U, V, x))
end

# 3. EM 3D monolithic (HyperFEM README), test_static_EM.msh
let
    model = GmshDiscreteModel(joinpath(@__DIR__, "..", "..", "HyperFEM.jl", "test", "models", "test_static_EM.msh"))
    physmodel = ElectroMechModel(IdealDielectric(ε=1.0), NeoHookean3D(λ=10.0, μ=1.0))
    k = (Kinematics(Mechano, Solid), Kinematics(Electro, Solid))
    Ω = Triangulation(model); dΩ = Measure(Ω, 2)
    Du = DirichletBC(["fixedup"], [[0.0, 0.0, 0.0]], [Λ -> 1.0])
    Dφ = DirichletBC(["midsuf", "topsuf"], [0.0, 0.1], [Λ -> Λ, Λ -> Λ])
    Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), Du, conformity=:H1)
    Vφ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), Dφ, conformity=:H1)
    V = MultiFieldFESpace([Vu, Vφ])
    U = MultiFieldFESpace([TrialFESpace(Vu, Du, 1.0), TrialFESpace(Vφ, Dφ, 1.0)])
    res((u, φ), (v, vφ)) = HyperFEM.residual(physmodel, k, (u, φ), (v, vφ), dΩ)
    jac((u, φ), (du, dφ), (v, vφ)) = HyperFEM.jacobian(physmodel, k, (u, φ), (du, dφ), (v, vφ), dΩ)
    x = 0.01 .* randn(num_free_dofs(U))
    report("EM 3D monolithic (README mesh)", check(res, jac, U, V, x))
end

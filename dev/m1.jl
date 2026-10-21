# Milestone 1 smoke test: kernel backend vs Gridap on small problems.
using HyperFEMKernels, HyperFEM, Gridap, GridapGmsh
using Gridap.FESpaces, Gridap.Algebra
using LinearAlgebra, Printf, Random

Random.seed!(1)
report(name, c) = (@printf("%-40s res %.2e   jac %.2e\n", name, c.res, c.jac); flush(stdout))
function report_alloc(name, res, jac, U, V, x)
    a = allocations(KernelOperator(res, jac, U, V), x)
    println("  allocations ", name, ": ", join(["$(t.kind)[$(t.group)] $(t.bytes) B" for t in a], ", ")); flush(stdout)
end

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
    report_alloc("Neo-Hooke Q$order", res, jac, U, V, x)
end

# 3. EM 3D monolithic (HyperFEM README), test_static_EM.msh
let
    model = GmshDiscreteModel(joinpath(@__DIR__, "..", "test", "models", "test_static_EM.msh"))
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
    report_alloc("EM", res, jac, U, V, x)
end

# 4. hMRE 2D (Adv_Mater_Ultra_Soft_Magnet hMRE_2D.jl): Dirichlet masks, auxiliary fields, Λ * ∫
let
    model = GmshDiscreteModel(joinpath(@__DIR__, "..", "test", "models", "Short_magnet.msh"))
    params = [10.0, 12480.64495232286, 1.0, 2.0, 5195.545287237134, 0.2602717127043121, 2.0]
    modelmech = NonlinearMooneyRivlin2D(λ=(params[1] + params[2]) * 1e2, μ1=params[1] * 1e3, μ2=params[2], α1=params[3], α2=params[4]) +
                TransverseIsotropy2D(μ=params[5], α1=params[6], α2=params[7])
    μ0 = 4π * 1e-7; Banorm = 80.0E-9; αr = 10e-3 / μ0
    reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
    Ωs = Triangulation(model, tags=["Beam"]); dΩs = Measure(Ωs, 2)
    Du = DirichletBC(["Fixed_Solid", "Fixed_Solid_x"], [[0.0, 0.0], [0.0, 0.0]])
    V = TestFESpace(Ωs, reffeu, Du, conformity=:H1, dirichlet_masks=[(true, true), (true, false)])
    U = TrialFESpace(V, Du); U⁻ = TrialFESpace(V, Du)
    uh⁻ = FEFunction(U⁻, 1e-4 .* randn(num_free_dofs(U⁻)))
    _, ∂Ψs∂F, ∂Ψs∂FF = modelmech()
    F, _, _ = get_Kinematics(Kinematics(Mechano, Solid))
    V_N = TestFESpace(Ωs, reffeu)
    Nh = interpolate_everywhere(x -> VectorValue(1.0, 0.0), V_N)
    Bah = interpolate_everywhere(VectorValue([0.0, -Banorm]), V_N)
    function invU(F; ε=1e-12)
        C = transpose(F) ⋅ F + ε * I2; s = sqrt(det(C))
        inv((C + s * I2) / sqrt(tr(C) + 2s))
    end
    ∂Ψmag∂F(Fn, N, Ba) = (-αr / μ0) * (Ba ⊗ ((invU(Fn)) * N))
    Λ = 0.35
    res(u, v) = ∫((∇(v)' ⊙ (∂Ψs∂F ∘ (F ∘ (∇(u)'), Nh))))dΩs + Λ * ∫((∂Ψmag∂F ∘ (F ∘ (∇(uh⁻)'), Nh, Bah)) ⊙ (∇(v)'))dΩs
    jac(u, du, v) = ∫(∇(v)' ⊙ ((∂Ψs∂FF ∘ (F ∘ (∇(u)'), Nh)) ⊙ (∇(du)')))dΩs
    x = 1e-4 .* randn(num_free_dofs(U))
    report("hMRE 2D (masks, aux fields, Λ*∫)", check(res, jac, U, V, x))
    report_alloc("hMRE (closed-form invU)", res, jac, U, V, x)
end

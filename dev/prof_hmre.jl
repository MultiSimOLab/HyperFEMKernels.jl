using HyperFEMKernels, HyperFEM, Gridap, GridapGmsh
using Gridap.FESpaces, Gridap.Algebra
using Printf, Random, Profile
Random.seed!(1)
adv = joinpath(@__DIR__, "..", "..", "Adv_Mater_Ultra_Soft_Magnet")
model = GmshDiscreteModel(joinpath(adv, "data", "models", "Short_magnet.msh"))
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
invU(F; ε=1e-12) = (F_ = get_array(F); TensorValue(inv(sqrt(F_' * F_ + ε * get_array(I2)))))
∂Ψmag∂F(Fn, N, Ba) = (-αr / μ0) * (Ba ⊗ ((invU(Fn)) * N))
res(Λ) = (u, v) -> ∫((∇(v)' ⊙ (∂Ψs∂F ∘ (F ∘ (∇(u)'), Nh))))dΩs + Λ * ∫((∂Ψmag∂F ∘ (F ∘ (∇(uh⁻)'), Nh, Bah)) ⊙ (∇(v)'))dΩs
jac(Λ) = (u, du, v) -> ∫(∇(v)' ⊙ ((∂Ψs∂FF ∘ (F ∘ (∇(u)'), Nh)) ⊙ (∇(du)')))dΩs
x = 1e-4 .* randn(num_free_dofs(U))
op = KernelOperator(res(0.1), jac(0.1), U, V)
op = update(op, res(0.2), jac(0.2))
t = minimum(@elapsed(update(op, res(0.3), jac(0.3))) for _ in 1:5); @printf("update: %.4f s\n", t)
gop = get_algebraic_operator(FEOperator(res(0.3), jac(0.3), U, V))
b = allocate_residual(op, x); A = allocate_jacobian(op, x); residual!(b, op, x); jacobian!(A, op, x)
bg = allocate_residual(gop, x); Ag = allocate_jacobian(gop, x); residual!(bg, gop, x); jacobian!(Ag, gop, x)
best(f) = minimum(@elapsed(f()) for _ in 1:10)
@printf("kernel residual %.5f s, jacobian %.5f s\n", best(() -> residual!(b, op, x)), best(() -> jacobian!(A, op, x)))
@printf("gridap residual %.5f s, jacobian %.5f s\n", best(() -> residual!(bg, gop, x)), best(() -> jacobian!(Ag, gop, x)))
@printf("FEOperator build %.5f s\n", best(() -> get_algebraic_operator(FEOperator(res(0.3), jac(0.3), U, V))))
@printf("colors: %d, cells %d\n", length(op.colors), size(op.nzpos, 2))
Profile.clear(); @profile for _ in 1:20; update(op, res(0.3), jac(0.3)); end
Profile.print(format=:flat, sortedby=:count, mincount=40, noisefloor=1.0)

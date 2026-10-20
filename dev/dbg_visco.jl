ENV["HFK_DEBUG_LINEAR"] = "1"
using HyperFEMKernels, HyperFEM, Gridap, Gridap.FESpaces, Gridap.Algebra, Gridap.CellData, LinearAlgebra, Random, Printf
const HK = HyperFEMKernels
Random.seed!(1)
width, thick, xdiv, order = 0.2, 0.001, 3, 2
geometry = CartesianDiscreteModel((-0.5width, 0.5width, -0.5width, 0.5width, 0.0, thick), (xdiv, xdiv, 1))
model = GeneralizedMaxwell(NeoHookean3D(μ=13446.1, λ=0.0) + VolumetricEnergy(λ=2.5e6),
                           ViscousPolyconvex(μ=11053.7, τ=151.68), ViscousPolyconvex(μ=30480.1, τ=7.0731))
F, H, J = get_Kinematics(Kinematics(Mechano, Solid))
Δt = 0.2 * thick / (order * 0.1)
Ω = Triangulation(geometry); dΩ = Measure(Ω, 2order)
reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
Vu = TestFESpace(geometry, reffe, conformity=:H1)
Uu = TrialFESpace(Vu); Uu⁻ = TrialFESpace(Vu); Uυ = TrialFESpace(Vu)
uh⁺ = FEFunction(Uu, 1e-5 .* randn(num_free_dofs(Uu)))
uh⁻ = FEFunction(Uu⁻, 1e-5 .* randn(num_free_dofs(Uu)))
υh = FEFunction(Uυ, 1e-3 .* randn(num_free_dofs(Uυ)))
Fh = F ∘ ∇(uh⁺)'; Fh⁻ = F ∘ ∇(uh⁻)'
Ah = CellState(model, dΩ)
println("Ah: ", typeof(Ah), " n=", length(Ah))
Ph⁻ = CellState(zero(TensorValue{3,3,Float64}), dΩ)
update_time_step!(model, Δt)
Ψ, ∂Ψ∂F, ∂∂Ψ∂FF = model()
P_updater(P, F, Fn, A...) = (true, ∂Ψ∂F(F, Fn, A...))
update_state!(P_updater, Ph⁻, Fh, Fh⁻, Ah...)
update_state!(model, Ah, Fh, Fh⁻)
ρ₀ = 960.0
x = 1e-5 .* randn(num_free_dofs(Uu))
for (name, res, jac) in (
    ("mass only", (u, v) -> ∫(ρ₀ * (2 / Δt^2 * (u - uh⁻) - 2 / Δt * υh) ⋅ v)dΩ, (u, du, v) -> ∫(ρ₀ * (2 / Δt^2 * du) ⋅ v)dΩ),
    ("stress only, no state", (u, v) -> ∫(∇(v)' ⊙ (∂Ψ∂F ∘ (F ∘ ∇(u)', Fh⁻, Ah...)))dΩ, (u, du, v) -> ∫(∇(v)' ⊙ ((∂∂Ψ∂FF ∘ (F ∘ ∇(u)', Fh⁻, Ah...)) ⊙ ∇(du)'))dΩ),
    ("Ph⁻ only", (u, v) -> ∫(∇(v)' ⊙ Ph⁻ + v ⋅ u)dΩ, (u, du, v) -> ∫(v ⋅ du)dΩ),
    ("full", (u, v) -> ∫(ρ₀ * (2 / Δt^2 * (u - uh⁻) - 2 / Δt * υh) ⋅ v + ∇(v)' ⊙ (1 / 2 * (∂Ψ∂F ∘ (F ∘ ∇(u)', Fh⁻, Ah...) + Ph⁻)))dΩ,
             (u, du, v) -> ∫(ρ₀ * (2 / Δt^2 * du) ⋅ v + ∇(v)' ⊙ (1 / 2 * (∂∂Ψ∂FF ∘ (F ∘ ∇(u)', Fh⁻, Ah...)) ⊙ ∇(du)'))dΩ))
    op = KernelOperator(res, jac, Uu, Vu)
    c = check(res, jac, Uu, Vu, x; op)
    a = allocations(op, x)
    @printf("%-24s res %.2e jac %.2e | alloc %s | lin res %s jac %s\n", name, c.res, c.jac, [t.bytes for t in a],
            all(g -> g.lin !== nothing, op.rgroups), all(g -> g.lin !== nothing, op.jgroups))
    for g in op.rgroups; println("  aux types: ", map(a -> eltype(eltype(a)), g.aux)); end
    flush(stdout)
end

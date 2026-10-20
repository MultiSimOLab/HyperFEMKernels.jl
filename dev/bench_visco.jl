# Assembly timing for the visco-hyperelastic dynamic weak form (Jesús's script: Q2 plate 7×7×1, 2 branches).
using HyperFEMKernels, HyperFEM, Gridap, Gridap.FESpaces, Gridap.Algebra, Gridap.CellData
using LinearAlgebra, Random, Printf
Random.seed!(1)
width, thick = 0.2, 0.001
xdiv = parse(Int, get(ARGS, 1, "7")); order = parse(Int, get(ARGS, 2, "2")); href = parse(Int, get(ARGS, 3, "1"))
geometry = CartesianDiscreteModel((-0.5width, 0.5width, -0.5width, 0.5width, 0.0, thick), href .* (xdiv, xdiv, 1))
model = GeneralizedMaxwell(NeoHookean3D(μ=13446.1, λ=0.0) + VolumetricEnergy(λ=2.5e6),
                           ViscousPolyconvex(μ=11053.7, τ=151.68), ViscousPolyconvex(μ=30480.1, τ=7.0731))
F, H, J = get_Kinematics(Kinematics(Mechano, Solid))
Δt = 0.2 * thick / (order * href * 0.1)
Ω = Triangulation(geometry); dΩ = Measure(Ω, 2order)
reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
Vu = TestFESpace(geometry, reffe, conformity=:H1)
Uu = TrialFESpace(Vu); Uu⁻ = TrialFESpace(Vu); Uυ = TrialFESpace(Vu)
uh⁺ = FEFunction(Uu, 1e-5 .* randn(num_free_dofs(Uu))); uh⁻ = FEFunction(Uu⁻, 1e-5 .* randn(num_free_dofs(Uu)))
υh = FEFunction(Uυ, 1e-3 .* randn(num_free_dofs(Uυ)))
Fh = F ∘ ∇(uh⁺)'; Fh⁻ = F ∘ ∇(uh⁻)'
Ah = CellState(model, dΩ); Ph⁻ = CellState(zero(TensorValue{3,3,Float64}), dΩ)
update_time_step!(model, Δt)
Ψ, ∂Ψ∂F, ∂∂Ψ∂FF = model()
P_updater(P, F, Fn, A...) = (true, ∂Ψ∂F(F, Fn, A...))
update_state!(P_updater, Ph⁻, Fh, Fh⁻, Ah...); update_state!(model, Ah, Fh, Fh⁻)
ρ₀ = 960.0
res(u, v) = ∫(ρ₀ * (2 / Δt^2 * (u - uh⁻) - 2 / Δt * υh) ⋅ v + ∇(v)' ⊙ (1 / 2 * (∂Ψ∂F ∘ (F ∘ ∇(u)', Fh⁻, Ah...) + Ph⁻)))dΩ
jac(u, du, v) = ∫(ρ₀ * (2 / Δt^2 * du) ⋅ v + ∇(v)' ⊙ (1 / 2 * (∂∂Ψ∂FF ∘ (F ∘ ∇(u)', Fh⁻, Ah...)) ⊙ ∇(du)'))dΩ
x = 1e-5 .* randn(num_free_dofs(Uu))
gop = get_algebraic_operator(FEOperator(res, jac, Uu, Vu))
kop = KernelOperator(res, jac, Uu, Vu); kop1 = KernelOperator(res, jac, Uu, Vu; nthreads=1)
bg = allocate_residual(gop, x); Ag = allocate_jacobian(gop, x); bk = allocate_residual(kop, x); Ak = allocate_jacobian(kop, x)
residual!(bg, gop, x); jacobian!(Ag, gop, x); residual!(bk, kop, x); jacobian!(Ak, kop, x); residual!(bk, kop1, x); jacobian!(Ak, kop1, x)
best(f) = minimum(@elapsed(f()) for _ in 1:3)
tjg = best(() -> jacobian!(Ag, gop, x)); trg = best(() -> residual!(bg, gop, x))
tjk = best(() -> jacobian!(Ak, kop, x)); trk = best(() -> residual!(bk, kop, x))
tjk1 = best(() -> jacobian!(Ak, kop1, x)); trk1 = best(() -> residual!(bk, kop1, x))
@printf("visco Q%d plate %dx%dx1 — %d DOFs, %d cells, %d threads\n", order, href * xdiv, href * xdiv, length(x), num_cells(Ω), Threads.nthreads())
@printf("  Jacobian : Gridap %.4f s | kernel 1 thr %.4f s (×%.1f) | %d thr %.4f s (×%.1f) | rel.diff %.1e\n", tjg, tjk1, tjg / tjk1, Threads.nthreads(), tjk, tjg / tjk, norm(Ak - Ag) / norm(Ag))
@printf("  Residual : Gridap %.4f s | kernel 1 thr %.4f s (×%.1f) | %d thr %.4f s (×%.1f) | rel.diff %.1e\n", trg, trk1, trg / trk1, Threads.nthreads(), trk, trg / trk, norm(bk - bg) / norm(bg))
# cost of the law alone (per point, 1 thread), to see what fraction of the kernel time it is
Fq = TensorValue(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) + 1e-3 * TensorValue(randn(9)...)
Aq = map(a -> a.values[1][1], Ah)
ncall = 20000
t = @elapsed for _ in 1:ncall; ∂∂Ψ∂FF(Fq, Fq, Aq...); end
np = num_cells(Ω) * (order + 1)^3
@printf("  law ∂∂Ψ∂FF alone: %.2f µs/call → %.4f s for all %d points (%.0f %% of kernel 1-thread Jacobian)\n", 1e6 * t / ncall, t / ncall * np, np, 100 * t / ncall * np / tjk1)

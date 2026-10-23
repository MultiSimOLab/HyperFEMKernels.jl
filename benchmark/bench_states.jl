# Timing of the update of the internal variables of the visco-hyperelastic problem (Jesús's script: Q2 plate, 2 branches):
# Gridap's update_state! vs update_states! (CellState and KernelState storages), 1 thread and N threads.
# Arguments: xdiv order href (as bench_visco.jl).
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
Uu = TrialFESpace(Vu); Uu⁻ = TrialFESpace(Vu)
uh⁺ = FEFunction(Uu, 1e-5 .* randn(num_free_dofs(Uu))); uh⁻ = FEFunction(Uu⁻, 1e-5 .* randn(num_free_dofs(Uu)))
Fh = F ∘ ∇(uh⁺)'; Fh⁻ = F ∘ ∇(uh⁻)'
update_time_step!(model, Δt)
const Ψ, ∂Ψ∂F, ∂∂Ψ∂FF = model()          # const: a non-const global in the updater makes every point allocate
T33 = TensorValue{3,3,Float64,9}
P_updater(P, F, Fn, A...) = (true, ∂Ψ∂F(F, Fn, A...))
const branches = model.branches.branches
branch_updater(b) = (A, F, Fn) -> (true, HyperFEM.return_mapping(b, F, Fn, A))
best(f) = minimum(@elapsed(f()) for _ in 1:3)

# Gridap: the two calls of the script (P from the law, then the branches)
Ahg = CellState(model, dΩ); Pg = CellState(zero(T33), dΩ)
gridap!() = (update_state!(P_updater, Pg, Fh, Fh⁻, Ahg...); update_state!(model, Ahg, Fh, Fh⁻))
gridap!()
tg = best(gridap!)
dense(s::CellState) = reduce(hcat, s.values); dense(s::KernelState) = s.values

np = num_cells(Ω) * (order + 1)^3
@printf("visco Q%d plate %dx%dx1 — %d DOFs, %d cells, %d points, %d threads\n", order, href * xdiv, href * xdiv, num_free_dofs(Uu), num_cells(Ω), np, Threads.nthreads())
@printf("  Gridap update_state! (P + 2 branches)      : %.4f s\n", tg)
for storage in (CellState, KernelState)
    x = get_cell_points(dΩ)
    Ah = Tuple(storage(one(T33), x) for _ in branches); Ph⁻ = storage(zero(T33), x)
    # equality with Gridap after the same number of updates (both start from the initial states)
    Ahc = CellState(model, dΩ); Pc = CellState(zero(T33), dΩ)
    update_state!(P_updater, Pc, Fh, Fh⁻, Ahc...); update_state!(model, Ahc, Fh, Fh⁻)
    update_states!(P_updater, Ph⁻, Fh, Fh⁻, Ah...)
    for (b, A) in zip(branches, Ah); update_states!(branch_updater(b), A, Fh, Fh⁻); end
    @printf("  max |kernel - Gridap| after one update (%s): P %.1e, A %.1e\n", nameof(storage), maximum(norm, dense(Ph⁻) - dense(Pc)), maximum(maximum(norm, dense(A) - dense(Ac)) for (A, Ac) in zip(Ah, Ahc)))
    for nthreads in unique((1, Threads.nthreads()))
        kernel!() = begin
            update_states!(P_updater, Ph⁻, Fh, Fh⁻, Ah...; nthreads)
            for (b, A) in zip(branches, Ah)
                update_states!(branch_updater(b), A, Fh, Fh⁻; nthreads)
            end
        end
        kernel!()
        tk = best(kernel!)
        # capture + compile lookup only (no cells): the fixed cost per call
        su = update_states!(P_updater, Ph⁻, Fh, Fh⁻, Ah...; nthreads)
        tc = best(() -> HyperFEMKernels.StateUpdate(P_updater, (Ph⁻, Fh, Fh⁻, Ah...), x, [1, 4, 5], nthreads))
        @printf("  update_states! %-11s %2d thr            : %.4f s (×%.1f)   setup per call %.2f ms, allocations per cell %d B\n",
                string(nameof(storage)), nthreads, tk, tg / tk, 1e3 * tc, allocations(su))
    end
end
# the updaters alone (per point, 1 thread)
Fq = TensorValue(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) + 1e-3 * TensorValue(randn(9)...)
Aq = map(a -> a.values[1][1], Ahg)
ncall = 20000
t = @elapsed for _ in 1:ncall; ∂Ψ∂F(Fq, Fq, Aq...); for b in branches; HyperFEM.return_mapping(b, Fq, Fq, Aq[1]); end; end
@printf("  updaters alone: %.2f µs/point → %.4f s for all points (lower bound of the 1-thread kernel)\n", 1e6 * t / ncall, t / ncall * np)

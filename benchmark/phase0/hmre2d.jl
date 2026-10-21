# Phase 0 — profile of the hMRE 2D example (simplified magnetic field) from
# MultiSimOLab/Adv_Mater_Ultra_Soft_Magnet, src/hMRE_2D.jl, mesh Short_magnet.msh.
#
# Same model, mesh, boundary conditions and solver settings as the original script
# (20 load steps, Newton with LU, atol 1e-8, rtol 1e-2). The Newton loop is
# instrumented to split residual assembly, Jacobian assembly, LU and the rest.
#
# Usage (from the HyperFEMKernels.jl folder; mesh in test/models):
#   julia -t auto --project=benchmark benchmark/phase0/hmre2d.jl          # original invU (matrix sqrt)
#   julia -t auto --project=benchmark benchmark/phase0/hmre2d.jl closed   # closed-form 2D invU
#   julia -t auto --project=benchmark benchmark/phase0/hmre2d.jl sqrt 4   # refined mesh r = 4 (generate.jl hmre 4)
#
# Output: table on screen and CSV in benchmark/results/.

using HyperFEM
using Gridap, GridapGmsh
using Gridap.FESpaces: SparseMatrixAssembler, FEOperator, get_algebraic_operator, zero_free_values
using Gridap.Algebra: allocate_residual, allocate_jacobian, residual!, jacobian!
using GridapSolvers, GridapSolvers.NonlinearSolvers
using LinearAlgebra, SparseArrays, Printf, Dates

const MESH_ORIG = normpath(joinpath(@__DIR__, "..", "..", "test", "models", "Short_magnet.msh"))
meshfile(r) = r === nothing ? MESH_ORIG : joinpath(@__DIR__, "..", "meshes", "hmre_short_r$(r).msh")

# Original implementation (hMRE_2D.jl): inverse of the right stretch tensor via matrix sqrt.
function invU_sqrt(F; ε=1e-12)
    F_ = get_array(F)
    I2_ = get_array(I2)
    S = F_' * F_
    Sreg = S + ε * I2_
    Sroot = sqrt(Sreg)
    return TensorValue(inv(Sroot))
end

# Closed form in 2D: U = (C + √det C · I) / √(tr C + 2√det C), with C = FᵀF (+ εI as above).
function invU_closed(F; ε=1e-12)
    C = transpose(F) ⋅ F + ε * I2
    s = sqrt(det(C))
    U = (C + s * I2) / sqrt(tr(C) + 2s)
    return inv(U)
end

function check_invU()
    maxerr = 0.0
    for _ in 1:1000
        F = TensorValue(1.0 .+ 0.3 .* (2 .* rand(4) .- 1)...)
        det(F) > 0.1 || continue
        maxerr = max(maxerr, norm(get_array(invU_sqrt(F) - invU_closed(F))) / norm(get_array(invU_sqrt(F))))
    end
    return maxerr
end

function setup(invU, mesh)
    model = GmshDiscreteModel(mesh)
    params = [10.0, 12480.64495232286, 1.0, 2.0, 5195.545287237134, 0.2602717127043121, 2.0]
    modelmech = NonlinearMooneyRivlin2D(λ=(params[1] + params[2]) * 1e2, μ1=params[1] * 1e3, μ2=params[2],
                                        α1=params[3], α2=params[4]) +
                TransverseIsotropy2D(μ=params[5], α1=params[6], α2=params[7])
    μ0 = 4π * 1e-7
    Banorm = 80.0E-9
    αr = 10e-3 / μ0
    reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
    Ω = Triangulation(model, tags=["Beam"])
    dΩ = Measure(Ω, 2)
    Du = BoundaryConditions(BoundaryCondition("Fixed_Solid", [0.0, 0.0]), BoundaryCondition("Fixed_Solid_x", [0.0, 0.0]; mask=(true, false)))
    V = TestFESpace(Ω, reffeu, Du, conformity=:H1)
    U = TrialFESpace(V, Du)
    U⁻ = TrialFESpace(V, Du)
    x⁻ = zero_free_values(U⁻)
    uh⁻ = FEFunction(U⁻, x⁻)
    _, ∂Ψs∂F, ∂Ψs∂FF = modelmech()
    F, _, _ = get_Kinematics(Kinematics(Mechano, Solid))
    V_N = TestFESpace(Ω, reffeu)
    Nh = interpolate_everywhere((x) -> VectorValue(1.0, 0.0), V_N)
    Bah = interpolate_everywhere(VectorValue([0.0, -Banorm]), V_N)
    ∂Ψmag∂F(Fn, N, Ba) = (-αr / μ0) * (Ba ⊗ ((invU(Fn)) * N))
    res(Λ) = (u, v) -> ∫((∇(v)' ⊙ (∂Ψs∂F ∘ (F ∘ (∇(u)'), Nh))))dΩ +
                       Λ * ∫((∂Ψmag∂F ∘ (F ∘ (∇(uh⁻)'), Nh, Bah)) ⊙ (∇(v)'))dΩ
    jac(Λ) = (u, du, v) -> ∫(∇(v)' ⊙ ((∂Ψs∂FF ∘ (F ∘ (∇(u)'), Nh)) ⊙ (∇(du)')))dΩ
    return (; model, Ω, Du, U, V, U⁻, x⁻, uh⁻, res, jac)
end

"Newton loop timed by parts; same stepping and tolerances as hMRE_2D.jl (no bisection)."
function instrumented!(s; verbose=false, nsteps=20, maxiter=20, atol=1e-8, rtol=1e-2)
    U, V = s.U, s.V
    assem = SparseMatrixAssembler(U, V)
    x = zero_free_values(U)
    s.x⁻ .= 0.0
    t = Dict(:res => 0.0, :jac => 0.0, :solve => 0.0)
    n = Dict(:res => 0, :jac => 0, :solve => 0)
    total = @elapsed for step in 1:nsteps
        Λ = step / nsteps
        op = get_algebraic_operator(FEOperator(s.res(Λ), s.jac(Λ), U, V, assem))
        b = allocate_residual(op, x)
        A = allocate_jacobian(op, x)
        t[:res] += @elapsed residual!(b, op, x); n[:res] += 1
        r0 = norm(b)
        for _ in 1:maxiter
            t[:jac] += @elapsed jacobian!(A, op, x); n[:jac] += 1
            local dx
            t[:solve] += @elapsed (dx = lu(A) \ b); n[:solve] += 1
            x .-= dx
            t[:res] += @elapsed residual!(b, op, x); n[:res] += 1
            r = norm(b)
            verbose && (@printf("  step %d  |r| = %.3e\n", step, r); flush(stdout))
            (r < atol || r < rtol * r0) && break
        end
        s.x⁻ .= x
    end
    return (; total, t, n, ndofs = length(x), x = copy(x))
end

function hyperfem_solve(invU, mesh)
    s = setup(invU, mesh)
    uh = FEFunction(s.U, zero_free_values(s.U))
    nls = NewtonSolver(LUSolver(); maxiter=20, atol=1.e-8, rtol=1.e-2, verbose=false)
    cm = StaticNonlinearModel(s.res, s.jac, s.U, s.V, s.Du; nls=nls, xh=uh, xh⁻=s.uh⁻)
    t = @elapsed solve!(cm; stepping=(nsteps=20, maxbisec=1))
    return t, copy(get_free_dof_values(uh))
end

function main(args)
    variant = isempty(args) ? "sqrt" : args[1]
    invU = variant == "closed" ? invU_closed : invU_sqrt
    r = length(args) ≥ 2 ? parse(Float64, args[2]) : nothing
    mesh = meshfile(r)
    isfile(mesh) || error("missing $mesh: run benchmark/meshes/generate.jl hmre $r")
    tag = r === nothing ? variant : "$(variant)_r$(r)"
    println("hMRE 2D phase-0 profile — $(basename(mesh)), invU = $variant")
    println("Julia $(VERSION), threads $(Threads.nthreads()), BLAS threads $(BLAS.get_num_threads())")
    @printf("closed-form vs sqrt invU, max rel. difference on 1000 random F: %.2e\n", check_invU())
    flush(stdout)

    cold = instrumented!(setup(invU, mesh); verbose=true)
    println("cold run done"); flush(stdout)
    warm = instrumented!(setup(invU, mesh))
    hf_t, hf_x = hyperfem_solve(invU, mesh)
    ncells = num_cells(setup(invU, mesh).Ω)

    rest = warm.total - warm.t[:res] - warm.t[:jac] - warm.t[:solve]
    asm = warm.t[:res] + warm.t[:jac]
    @printf("\ncells (Beam): %d   DOFs (free): %d\n", ncells, warm.ndofs)
    @printf("%-22s %10s %8s %6s %12s\n", "part", "time [s]", "share", "calls", "per call [s]")
    for (name, key) in (("residual assembly", :res), ("Jacobian assembly", :jac), ("linear solve (LU)", :solve))
        @printf("%-22s %10.3f %7.1f%% %6d %12.4f\n", name, warm.t[key], 100warm.t[key]/warm.total,
                warm.n[key], warm.t[key]/warm.n[key])
    end
    @printf("%-22s %10.3f %7.1f%%\n", "rest", rest, 100rest/warm.total)
    @printf("%-22s %10.3f\n", "total (instrumented)", warm.total)
    @printf("\nassembly share: %.1f %%  (phase-0 gate: > 50 %%)\n", 100asm/warm.total)
    @printf("first run incl. compilation: %.2f s\n", cold.total)
    @printf("HyperFEM solve!: %.3f s (after warm-up)\n", hf_t)
    @printf("final displacement, instrumented vs HyperFEM: rel. diff %.2e\n", norm(warm.x - hf_x) / max(norm(hf_x), eps()))

    resdir = joinpath(@__DIR__, "..", "results"); mkpath(resdir)
    file = joinpath(resdir, "phase0_hmre2d_$(tag).csv")
    open(file, "w") do io
        println(io, "date,variant,cells,dofs,threads,t_res,n_res,t_jac,n_jac,t_solve,n_solve,t_rest,t_total,t_cold,t_hyperfem")
        println(io, join((Dates.today(), tag, ncells, warm.ndofs, Threads.nthreads(),
            warm.t[:res], warm.n[:res], warm.t[:jac], warm.n[:jac], warm.t[:solve], warm.n[:solve],
            rest, warm.total, cold.total, hf_t), ","))
    end
    println("written $file")
end

main(ARGS)

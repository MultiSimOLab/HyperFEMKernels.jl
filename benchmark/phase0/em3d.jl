# Phase 0 — profile of HyperFEM's monolithic electromechanical 3D problem.
#
# Splits wall time into residual assembly, Jacobian assembly, linear solve and
# the rest, using an instrumented Newton loop that does the same work as
# HyperFEM's StaticNonlinearModel + GridapSolvers' NewtonSolver, and compares
# its total with HyperFEM's own `solve!`.
#
# Usage (from the HyperFEMKernels.jl folder):
#   julia -t auto --project=benchmark benchmark/phase0/em3d.jl            # test_static_EM.msh, P1
#   julia -t auto --project=benchmark benchmark/phase0/em3d.jl cart 40 8 8 2   # Cartesian box, Q2
#
# Output: table on screen and CSV in benchmark/results/.

using HyperFEM
using Gridap, GridapGmsh
using Gridap.FESpaces: SparseMatrixAssembler, FEOperator, get_algebraic_operator, zero_free_values
using Gridap.Algebra: allocate_residual, allocate_jacobian, residual!, jacobian!
using GridapSolvers, GridapSolvers.NonlinearSolvers
using LinearAlgebra, SparseArrays, Printf, Dates

const HYPERFEM_DIR = normpath(joinpath(@__DIR__, "..", "..", "..", "HyperFEM.jl"))
const MESH = joinpath(HYPERFEM_DIR, "test", "models", "test_static_EM.msh")

function build_model(args)
    if isempty(args) || args[1] == "msh"
        order = length(args) ≥ 2 ? parse(Int, args[2]) : 1
        return GmshDiscreteModel(MESH), order, "msh", 0.1
    end
    nx, ny, nz = parse.(Int, args[2:4])
    order = length(args) ≥ 5 ? parse(Int, args[5]) : 1
    model = CartesianDiscreteModel((0.0, 0.1, 0.0, 0.01, 0.0, 0.002), (nx, ny, nz))
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "fixedup", CartesianTags.face0YZ⁺)
    add_tag_from_tags!(labels, "topsuf", CartesianTags.faceXY1⁺)
    add_tag_from_vertex_filter!(labels, "midsuf", model, x -> abs(x[3] - 0.001) < 1e-9)
    # same geometry and load as HyperFEM test/SimulationsTests/StaggeredElectroMechanicalSimulation.jl
    return model, order, "cart$(nx)x$(ny)x$(nz)", 5e-5
end

function setup(model, order; φtop = 0.1)
    physmodel = ElectroMechModel(IdealDielectric(ε=1.0), NeoHookean3D(λ=10.0, μ=1.0))
    k = (Kinematics(Mechano, Solid), Kinematics(Electro, Solid))
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2order)
    Du = DirichletBC(["fixedup"], [[0.0, 0.0, 0.0]], [Λ -> 1.0])
    Dφ = DirichletBC(["midsuf", "topsuf"], [0.0, φtop], [Λ -> Λ, Λ -> Λ])
    D = MultiFieldBC([Du, Dφ])
    reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
    reffeφ = ReferenceFE(lagrangian, Float64, order)
    Vu = TestFESpace(model, reffeu, Du, conformity=:H1)
    Vφ = TestFESpace(model, reffeφ, Dφ, conformity=:H1)
    Uu = TrialFESpace(Vu, Du, 1.0)
    Uφ = TrialFESpace(Vφ, Dφ, 1.0)
    V = MultiFieldFESpace([Vu, Vφ])
    U = MultiFieldFESpace([Uu, Uφ])
    res(Λ) = ((u, φ), (v, vφ)) -> HyperFEM.residual(physmodel, k, (u, φ), (v, vφ), dΩ)
    jac(Λ) = ((u, φ), (du, dφ), (v, vφ)) -> HyperFEM.jacobian(physmodel, k, (u, φ), (du, dφ), (v, vφ), dΩ)
    return (; Ω, D, U, V, res, jac)
end

"Newton loop timed by parts. Same steps, tolerances and LU solve as HyperFEM's driver."
function instrumented!(s; verbose=false, nsteps=5, maxiter=20, atol=1e-10, rtol=1e-8)
    U, V = s.U, s.V
    assem = SparseMatrixAssembler(U, V)
    x = zero_free_values(U)
    t = Dict(:res => 0.0, :jac => 0.0, :solve => 0.0)
    n = Dict(:res => 0, :jac => 0, :solve => 0)
    total = @elapsed for step in 1:nsteps
        Λ = step / nsteps
        TrialFESpace!(U, s.D, Λ)
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
    end
    return (; total, t, n, ndofs = length(x))
end

function hyperfem_solve(model, order, φtop)
    s = setup(model, order; φtop)
    nls = NewtonSolver(LUSolver(); maxiter=20, atol=1e-10, rtol=1e-8, verbose=false)
    cm = StaticNonlinearModel(s.res, s.jac, s.U, s.V, s.D; nls=nls)
    return @elapsed solve!(cm; stepping=(nsteps=5, maxbisec=5))
end

function main(args)
    model, order, tag, φtop = build_model(args)
    ncells = num_cells(model)
    println("HyperFEM EM 3D phase-0 profile — case $tag, order $order, $ncells cells")
    println("Julia $(VERSION), threads $(Threads.nthreads()), BLAS threads $(BLAS.get_num_threads())")

    flush(stdout)
    cold = instrumented!(setup(model, order; φtop); verbose=true)  # includes compilation
    println("cold run done"); flush(stdout)
    warm = instrumented!(setup(model, order; φtop))      # steady state
    hf_cold = hyperfem_solve(model, order, φtop)
    hf_warm = hyperfem_solve(model, order, φtop)

    rest = warm.total - warm.t[:res] - warm.t[:jac] - warm.t[:solve]
    asm = warm.t[:res] + warm.t[:jac]
    @printf("\nDOFs (free): %d\n", warm.ndofs)
    @printf("%-22s %10s %8s %6s %12s\n", "part", "time [s]", "share", "calls", "per call [s]")
    for (name, key) in (("residual assembly", :res), ("Jacobian assembly", :jac), ("linear solve (LU)", :solve))
        @printf("%-22s %10.3f %7.1f%% %6d %12.4f\n", name, warm.t[key], 100warm.t[key]/warm.total,
                warm.n[key], warm.t[key]/warm.n[key])
    end
    @printf("%-22s %10.3f %7.1f%%\n", "rest", rest, 100rest/warm.total)
    @printf("%-22s %10.3f\n", "total (instrumented)", warm.total)
    @printf("\nassembly share: %.1f %%  (phase-0 gate: > 50 %%)\n", 100asm/warm.total)
    @printf("first run incl. compilation: %.2f s\n", cold.total)
    @printf("HyperFEM solve!: %.3f s warm, %.2f s cold\n", hf_warm, hf_cold)

    resdir = joinpath(@__DIR__, "..", "results"); mkpath(resdir)
    file = joinpath(resdir, "phase0_em3d_$(tag)_p$(order).csv")
    open(file, "w") do io
        println(io, "date,case,order,cells,dofs,threads,t_res,n_res,t_jac,n_jac,t_solve,n_solve,t_rest,t_total,t_cold,t_hyperfem_warm,t_hyperfem_cold")
        println(io, join((Dates.today(), tag, order, ncells, warm.ndofs, Threads.nthreads(),
            warm.t[:res], warm.n[:res], warm.t[:jac], warm.n[:jac], warm.t[:solve], warm.n[:solve],
            rest, warm.total, cold.total, hf_warm, hf_cold), ","))
    end
    println("written $file")
end

main(ARGS)

# Phase 1 — full HyperFEM solve with backend = :gridap or :kernel (same Newton, same steps).
#
#   julia -t auto --project=benchmark benchmark/phase1/solve.jl em plate 2 1 kernel   # EM plate n=2, P1
#   julia -t auto --project=benchmark benchmark/phase1/solve.jl em box 20 4 4 2 gridap
#   julia -t auto --project=benchmark benchmark/phase1/solve.jl hmre 2 kernel          # hMRE r=2
#
# Each run: a cold solve (includes compilation) and a warm solve on a fresh model; the final
# free values are written to benchmark/results/ so both backends can be compared.

using HyperFEM, HyperFEMKernels
using Gridap, GridapGmsh, GridapSolvers, GridapSolvers.NonlinearSolvers
using Gridap.FESpaces
using LinearAlgebra, Printf, Serialization

const MESHDIR = joinpath(@__DIR__, "..", "meshes")
const MODELS = normpath(joinpath(@__DIR__, "..", "..", "test", "models"))

function em_model(args)
    if args[1] == "plate"
        n, order = parse(Int, args[2]), parse(Int, args[3])
        return GmshDiscreteModel(joinpath(MESHDIR, "em_plate_n$(n).msh")), order, 0.1, "plate_n$(n)_p$(order)", args[4:end]
    end
    nx, ny, nz, order = parse.(Int, args[2:5])
    m = CartesianDiscreteModel((0.0, 0.1, 0.0, 0.01, 0.0, 0.002), (nx, ny, nz))
    labels = get_face_labeling(m)
    add_tag_from_tags!(labels, "fixedup", CartesianTags.face0YZ⁺)
    add_tag_from_tags!(labels, "topsuf", CartesianTags.faceXY1⁺)
    add_tag_from_vertex_filter!(labels, "midsuf", m, x -> abs(x[3] - 0.001) < 1e-9)
    m, order, 5e-5, "box$(nx)x$(ny)x$(nz)_p$(order)", args[6:end]
end

function em_solve(model, order, φtop, backend)
    physmodel = ElectroMechModel(IdealDielectric(ε=1.0), NeoHookean3D(λ=10.0, μ=1.0))
    k = (Kinematics(Mechano, Solid), Kinematics(Electro, Solid))
    Ω = Triangulation(model); dΩ = Measure(Ω, 2order)
    Du = BoundaryConditions(BoundaryCondition("fixedup", [0.0, 0.0, 0.0]))
    Dφ = BoundaryConditions(BoundaryCondition("midsuf", 0.0, Λ -> Λ), BoundaryCondition("topsuf", φtop, Λ -> Λ))
    D = MultiFieldBC([Du, Dφ])
    Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, order), Du, conformity=:H1)
    Vφ = TestFESpace(model, ReferenceFE(lagrangian, Float64, order), Dφ, conformity=:H1)
    V = MultiFieldFESpace([Vu, Vφ])
    U = MultiFieldFESpace([TrialFESpace(Vu, Du, 1.0), TrialFESpace(Vφ, Dφ, 1.0)])
    res(Λ) = ((u, φ), (v, vφ)) -> HyperFEM.residual(physmodel, k, (u, φ), (v, vφ), dΩ)
    jac(Λ) = ((u, φ), (du, dφ), (v, vφ)) -> HyperFEM.jacobian(physmodel, k, (u, φ), (du, dφ), (v, vφ), dΩ)
    nls = NewtonSolver(LUSolver(); maxiter=20, atol=1e-10, rtol=1e-8, verbose=false)
    t_build = @elapsed (cm = StaticNonlinearModel(res, jac, U, V, D; nls=nls, backend=backend))
    t_solve = @elapsed solve!(cm; stepping=(nsteps=5, maxbisec=5))
    t_build, t_solve, copy(cm.caches[3]), num_free_dofs(U)
end

function hmre_solve(r, backend)
    mesh = r == "orig" ? joinpath(MODELS, "Short_magnet.msh") : joinpath(MESHDIR, "hmre_short_r$(parse(Float64, r)).msh")
    model = GmshDiscreteModel(mesh)
    params = [10.0, 12480.64495232286, 1.0, 2.0, 5195.545287237134, 0.2602717127043121, 2.0]
    modelmech = NonlinearMooneyRivlin2D(λ=(params[1] + params[2]) * 1e2, μ1=params[1] * 1e3, μ2=params[2], α1=params[3], α2=params[4]) +
                TransverseIsotropy2D(μ=params[5], α1=params[6], α2=params[7])
    μ0 = 4π * 1e-7; Banorm = 80.0E-9; αr = 10e-3 / μ0
    reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
    Ωs = Triangulation(model, tags=["Beam"]); dΩs = Measure(Ωs, 2)
    Du = BoundaryConditions(BoundaryCondition("Fixed_Solid", [0.0, 0.0]), BoundaryCondition("Fixed_Solid_x", [0.0, 0.0]; mask=(true, false)))
    V = TestFESpace(Ωs, reffeu, Du, conformity=:H1)
    U = TrialFESpace(V, Du); U⁻ = TrialFESpace(V, Du)
    uh = FEFunction(U, zero_free_values(U)); uh⁻ = FEFunction(U⁻, zero_free_values(U⁻))
    _, ∂Ψs∂F, ∂Ψs∂FF = modelmech()
    F, _, _ = get_Kinematics(Kinematics(Mechano, Solid))
    V_N = TestFESpace(Ωs, reffeu)
    Nh = interpolate_everywhere(x -> VectorValue(1.0, 0.0), V_N)
    Bah = interpolate_everywhere(VectorValue([0.0, -Banorm]), V_N)
    function invU(F; ε=1e-12)   # original implementation (matrix square root)
        F_ = get_array(F); S = F_' * F_ + ε * get_array(I2)
        TensorValue(inv(sqrt(S)))
    end
    ∂Ψmag∂F(Fn, N, Ba) = (-αr / μ0) * (Ba ⊗ ((invU(Fn)) * N))
    res(Λ) = (u, v) -> ∫((∇(v)' ⊙ (∂Ψs∂F ∘ (F ∘ (∇(u)'), Nh))))dΩs + Λ * ∫((∂Ψmag∂F ∘ (F ∘ (∇(uh⁻)'), Nh, Bah)) ⊙ (∇(v)'))dΩs
    jac(Λ) = (u, du, v) -> ∫(∇(v)' ⊙ ((∂Ψs∂FF ∘ (F ∘ (∇(u)'), Nh)) ⊙ (∇(du)')))dΩs
    nls = NewtonSolver(LUSolver(); maxiter=20, atol=1.e-8, rtol=1.e-2, verbose=false)
    t_build = @elapsed (cm = StaticNonlinearModel(res, jac, U, V, Du; nls=nls, xh=uh, xh⁻=uh⁻, backend=backend))
    t_solve = @elapsed solve!(cm; stepping=(nsteps=20, maxbisec=1))
    t_build, t_solve, copy(get_free_dof_values(uh)), num_free_dofs(U)
end

function main(args)
    backend = Symbol(args[end])
    if args[1] == "em"
        model, order, φtop, tag, _ = em_model(args[2:end-1])
        runf = () -> em_solve(model, order, φtop, backend)
        tag = "em_" * tag
    else
        runf = () -> hmre_solve(args[2], backend)
        tag = "hmre_r" * args[2]
    end
    println("Phase-1 solve — $tag, backend = $backend, threads $(Threads.nthreads())"); flush(stdout)
    tb0, ts0, _, _ = runf()
    tb1, ts1, x, n = runf()
    @printf("DOFs %d\ncold: build %.2f s + solve %.2f s\nwarm: build %.2f s + solve %.2f s = %.2f s\n", n, tb0, ts0, tb1, ts1, tb1 + ts1)
    resdir = joinpath(@__DIR__, "..", "results"); mkpath(resdir)
    serialize(joinpath(resdir, "phase1_$(tag)_$(backend).jls"), (x=x, tb=tb1, ts=ts1, cold=tb0 + ts0))
    other = joinpath(resdir, "phase1_$(tag)_$(backend == :kernel ? :gridap : :kernel).jls")
    if isfile(other)
        o = deserialize(other)
        @printf("vs %s: final state rel. diff %.2e, warm total %.2f s vs %.2f s (×%.2f)\n",
                backend == :kernel ? "gridap" : "kernel", norm(x - o.x) / norm(o.x), tb1 + ts1, o.tb + o.ts,
                backend == :kernel ? (o.tb + o.ts) / (tb1 + ts1) : (tb1 + ts1) / (o.tb + o.ts))
    end
end

main(ARGS)

# Milestone 1 timing: Jacobian and residual assembly, Gridap vs kernel (same matrix).
using HyperFEMKernels, HyperFEM, Gridap, GridapGmsh
using Gridap.FESpaces, Gridap.Algebra
using LinearAlgebra, Printf, Random
Random.seed!(1)

function em_case(model, order)
    physmodel = ElectroMechModel(IdealDielectric(ε=1.0), NeoHookean3D(λ=10.0, μ=1.0))
    k = (Kinematics(Mechano, Solid), Kinematics(Electro, Solid))
    Ω = Triangulation(model); dΩ = Measure(Ω, 2order)
    Du = BoundaryConditions(BoundaryCondition("fixedup", [0.0, 0.0, 0.0]))
    Dφ = BoundaryConditions(BoundaryCondition("midsuf", 0.0, Λ -> Λ), BoundaryCondition("topsuf", 0.1, Λ -> Λ))
    Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, order), Du, conformity=:H1)
    Vφ = TestFESpace(model, ReferenceFE(lagrangian, Float64, order), Dφ, conformity=:H1)
    V = MultiFieldFESpace([Vu, Vφ])
    U = MultiFieldFESpace([TrialFESpace(Vu, Du, 1.0), TrialFESpace(Vφ, Dφ, 1.0)])
    res((u, φ), (v, vφ)) = HyperFEM.residual(physmodel, k, (u, φ), (v, vφ), dΩ)
    jac((u, φ), (du, dφ), (v, vφ)) = HyperFEM.jacobian(physmodel, k, (u, φ), (du, dφ), (v, vφ), dΩ)
    res, jac, U, V
end

function bench(name, res, jac, U, V)
    x = 0.01 .* randn(num_free_dofs(U))
    gop = get_algebraic_operator(FEOperator(res, jac, U, V))
    tbuild = @elapsed (kop = KernelOperator(res, jac, U, V))
    bg = allocate_residual(gop, x); Ag = allocate_jacobian(gop, x)
    bk = allocate_residual(kop, x); Ak = allocate_jacobian(kop, x)
    # warm-up (compilation)
    tc_g = @elapsed (residual!(bg, gop, x); jacobian!(Ag, gop, x))
    tc_k = @elapsed (residual!(bk, kop, x); jacobian!(Ak, kop, x))
    best(f) = minimum(@elapsed(f()) for _ in 1:3)
    tjg = best(() -> jacobian!(Ag, gop, x)); tjk = best(() -> jacobian!(Ak, kop, x))
    trg = best(() -> residual!(bg, gop, x)); trk = best(() -> residual!(bk, kop, x))
    kopd = KernelOperator(res, jac, U, V; nthreads=1, linear_maps=false)
    jacobian!(Ak, kopd, x); residual!(bk, kopd, x)
    tjd1 = best(() -> jacobian!(Ak, kopd, x)); trd1 = best(() -> residual!(bk, kopd, x))
    kop1 = KernelOperator(res, jac, U, V; nthreads=1)
    jacobian!(Ak, kop1, x); residual!(bk, kop1, x)
    tjk1 = best(() -> jacobian!(Ak, kop1, x)); trk1 = best(() -> residual!(bk, kop1, x))
    ej = norm(Ak - Ag) / norm(Ag); er = norm(bk - bg) / norm(bg)
    @printf("\n%s — %d DOFs, %d threads\n", name, length(x), Threads.nthreads())
    @printf("  Jacobian : Gridap %.4f s | kernel 1 thr %.4f s (×%.1f) | %d thr %.4f s (×%.1f) | rel.diff %.1e\n",
            tjg, tjk1, tjg / tjk1, Threads.nthreads(), tjk, tjg / tjk, ej)
    @printf("  Residual : Gridap %.4f s | kernel 1 thr %.4f s (×%.1f) | %d thr %.4f s (×%.1f) | rel.diff %.1e\n",
            trg, trk1, trg / trk1, Threads.nthreads(), trk, trg / trk, er)
    @printf("  1 thread, dual-number path: Jacobian %.4f s, residual %.4f s | linear maps used: res %s, jac %s\n",
            tjd1, trd1, all(g -> g.lin !== nothing, kop.rgroups), all(g -> g.lin !== nothing, kop.jgroups))
    @printf("  first call (compilation): Gridap %.1f s, kernel %.1f s; kernel operator build %.2f s\n", tc_g, tc_k, tbuild)
    flush(stdout)
end

meshdir = joinpath(@__DIR__, "..", "benchmark", "meshes")
bench("EM plate n=2, P1", em_case(GmshDiscreteModel(joinpath(meshdir, "em_plate_n2.msh")), 1)...)
let m = CartesianDiscreteModel((0.0, 0.1, 0.0, 0.01, 0.0, 0.002), (20, 4, 4))
    labels = get_face_labeling(m)
    add_tag_from_tags!(labels, "fixedup", CartesianTags.face0YZ⁺)
    add_tag_from_tags!(labels, "topsuf", CartesianTags.faceXY1⁺)
    add_tag_from_vertex_filter!(labels, "midsuf", m, x -> abs(x[3] - 0.001) < 1e-9)
    bench("EM box 20x4x4, Q2", em_case(m, 2)...)
end

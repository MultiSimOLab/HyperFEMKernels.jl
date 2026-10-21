# =============================================================================
# hMRE beam with explicit surrounding air (vacuum), 2D, staggered magneto-mechanical solution:
#   problem 1: magnetic potential on solid + vacuum,
#   problem 2: mechanics of the solid (with the Maxwell traction of the vacuum on the interface),
#   problem 3: fictitious mechanics of the vacuum (mesh motion driven by the solid on the interface).
#
# Local copy of MultiSimOLab/Adv_Mater_Ultra_Soft_Magnet src/hMRE_Vacuum_2D.jl (commit 71a610e;
# Perez-Garcia, Ortigosa, Martinez-Frutos, Garcia-Gonzalez, Advanced Materials), with the same weak forms,
# models, boundary conditions and stepping. Changes: no DrWatson (mesh from test/models, output to
# results/), the setup wrapped in a function so that it can be built for each backend, and the drivers
# built with `backend` (HyperFEM branch kernel-backend). Needs HyperFEM with the fix of the legacy
# DirichletCoupling (branch fix-legacy-coupling, see docs/regresion_DirichletCoupling_HyperFEM.pdf).
#
# Usage (from the repository root, benchmark environment):
#   julia -t auto --project=benchmark examples/hmre_vacuum_2d/hmre_vacuum_2d.jl [mode=compare|gridap|kernel]
#        [nsteps=20] [nsubsteps=2] [vtk=off]
#   mode=compare : after a one-step warm-up of both backends, runs the whole staggered simulation with Gridap and with HyperFEMKernels, compares the final
#                  solutions of the three problems, and checks residual and Jacobian of each problem against
#                  Gridap at the final state (HyperFEMKernels.check).
# =============================================================================
using HyperFEM, HyperFEMKernels
using Gridap, GridapGmsh, GridapSolvers
using GridapSolvers.NonlinearSolvers
using Gridap.FESpaces
using Gridap.CellData
using LinearAlgebra, Printf

kw = Dict(split(a, "=")[1] => split(a, "=")[2] for a in ARGS if occursin("=", a))
mode = get(kw, "mode", "compare")
nsteps = parse(Int, get(kw, "nsteps", "20"))
nsubsteps = parse(Int, get(kw, "nsubsteps", "2"))
vtk = get(kw, "vtk", "off") == "on"
const MESH = normpath(joinpath(@__DIR__, "..", "..", "test", "models", "Short_magnet.msh"))
const OUT = joinpath(@__DIR__, "results")


"Everything of the original script up to the computational problems, with the drivers built for `backend`."
function build(backend::Symbol; verbose=false)
  geomodel = GmshDiscreteModel(MESH)

  # Constitutive models
  μ0 = 4π * 1e-7
  model_vacuum_mag = IdealMagnetic2D(μ0=μ0, χe=0.0)
  params = [10.0, 12480.64495232286, 1.0, 2.0, 5195.545287237134, 0.2602717127043121, 2.0]
  modelmech = NonlinearMooneyRivlin2D(λ=(params[1] + params[2]) * 1e2, μ1=params[1] * 1e3, μ2=params[2], α1=params[3], α2=params[4]) +
              TransverseIsotropy2D(μ=params[5], α1=params[6], α2=params[7])
  modelhard = HardMagnetic2D(μ0=μ0, αr=10e-3 / μ0, χe=0.0, χr=8.0)
  model_solid = MagnetoMechModel(magneto=modelhard, mechano=modelmech)
  model_vacuum_mech_ = NonlinearMooneyRivlin2D_CV(λ=1.0, μ1=1.0, μ2=0.0, α1=6.0, α2=1.0, γ=6.0)
  model_vacuum_mech = HessianRegularization(mechano=model_vacuum_mech_, δ=1e-6)

  # Finite elements
  order_solid = 1
  order_vacuum = 1
  reffeφ = ReferenceFE(lagrangian, Float64, order_solid)
  reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, order_solid)
  reffeu_vacuum = ReferenceFE(lagrangian, VectorValue{2,Float64}, order_vacuum)

  # Domains
  Ωdomain = Triangulation(geomodel, tags=["vacuumb", "Beam"])
  Ωvacuum = Triangulation(geomodel, tags=["vacuumb"])
  Ωsolid = Triangulation(geomodel, tags=["Beam"])
  dΩsolid = Measure(Ωsolid, 2 * order_solid)
  dΩvacuum_mag = Measure(Ωvacuum, 2 * order_solid)
  dΩvacuum_mec = Measure(Ωvacuum, 2 * order_vacuum)
  Γair_int = BoundaryTriangulation(Ωvacuum, tags="Interface")
  Γsolid_int = BoundaryTriangulation(Ωsolid, tags="Interface")
  dΓs = Measure(Γsolid_int, 4 * order_solid)

  # Dirichlet boundary conditions
  evolφ(Λ) = Λ
  Dφ = DirichletBC(["phibot", "phitop"], [0.2 * 5200, 0.0], [evolφ, evolφ])
  evolu(Λ) = 1.0
  Du_solid = DirichletBC(["Fixed_Solid", "Fixed_Solid_x"], [[0.0, 0.0], [0.0, 0.0]], [evolu, evolu])
  Du_vacuum = DirichletBC(["Fixed_Air", "Fixed_Air_x", "Interface"], [[0.0, 0.0], [0.0, 0.0], DirichletCoupling], [evolu, evolu, evolu])

  # FE spaces and state variables ((⁺) current stage, (⁻) previous stage)
  Vφ = TestFESpace(Ωdomain, reffeφ, Dφ, conformity=:H1)
  Uφ⁺ = TrialFESpace(Vφ, Dφ); Uφ⁻ = TrialFESpace(Vφ, Dφ)
  φh⁺ = FEFunction(Uφ⁺, zero_free_values(Uφ⁺)); φh⁻ = FEFunction(Uφ⁻, zero_free_values(Uφ⁻))
  Vu_solid = TestFESpace(Ωsolid, reffeu, Du_solid, conformity=:H1, dirichlet_masks=[(true, true), (true, false)])
  Uu_solid⁺ = TrialFESpace(Vu_solid, Du_solid); Uu_solid⁻ = TrialFESpace(Vu_solid, Du_solid)
  uh_solid⁺ = FEFunction(Uu_solid⁺, zero_free_values(Uu_solid⁺)); uh_solid⁻ = FEFunction(Uu_solid⁻, zero_free_values(Uu_solid⁻))
  Vu_vacuum = TestFESpace(Ωvacuum, reffeu_vacuum, Du_vacuum, conformity=:H1, dirichlet_masks=[(true, true), (true, false), (true, true)])
  Uu_vacuum⁺ = TrialFESpace(Vu_vacuum, Du_vacuum); Uu_vacuum⁻ = TrialFESpace(Vu_vacuum, Du_vacuum)
  uh_vacuum⁺ = FEFunction(Uu_vacuum⁺, zero_free_values(Uu_vacuum⁺)); uh_vacuum⁻ = FEFunction(Uu_vacuum⁻, zero_free_values(Uu_vacuum⁻))

  # Solid–vacuum coupling: staggered solid displacement interpolated to the vacuum interface
  Uair_int = FESpace(Γair_int, reffeu_vacuum)
  Usolid_int = FESpace(Γsolid_int, reffeu)
  uhsolid(Λ) = uh_solid⁻ + (uh_solid⁺ - uh_solid⁻) * Λ
  uhsolid_int = interpolate_everywhere(uh_solid⁺, Usolid_int)
  uhair_int = interpolate_everywhere(Interpolable(uhsolid_int), Uair_int)
  uhsolid_int_(Λ) = Interpolable(interpolate_everywhere!(uhsolid(Λ), get_free_dof_values(uhsolid_int), uhsolid_int.dirichlet_values, Usolid_int))
  uhair_int_(Λ) = interpolate_everywhere!(uhsolid_int_(Λ), get_free_dof_values(uhair_int), uhair_int.dirichlet_values, Uair_int)
  InterpolableBC!(Uu_vacuum⁺, Du_vacuum, "Interface", uhair_int_)

  # Maxwell stress in the vacuum and traction on the interface
  V_maxwell = TestFESpace(Ωvacuum, reffeu_vacuum)
  Pmaxw = FEFunction(V_maxwell, zero_free_values(V_maxwell))
  Pmaxw_ = get_free_dof_values(Pmaxw)
  V_traction = TestFESpace(Γsolid_int, reffeu)
  U_traction = TrialFESpace(V_traction)
  tmaxw = interpolate_everywhere(Pmaxw, Usolid_int)
  tmaxw_ = get_free_dof_values(tmaxw)
  Massoperator = StaticLinearModel((u, v) -> ∫(u ⋅ v)dΓs, U_traction, V_traction, NothingBC())

  # Weak forms
  Ψs, ∂Ψs∂F, ∂Ψs∂H0, ∂Ψs∂FF, ∂Ψs∂H0F, ∂Ψs∂H0H0 = model_solid()
  Ψv, ∂Ψv∂F, ∂Ψv∂H0, ∂Ψv∂FF, ∂Ψv∂H0F, ∂Ψv∂H0H0 = model_vacuum_mag()
  Ψvm, ∂Ψvm∂F, ∂Ψvm∂FF = model_vacuum_mech()
  F, H, J = get_Kinematics(Kinematics(Mechano, Solid))
  ℋ₀ = get_Kinematics(Kinematics(Magneto, Solid))
  uhvacuum(Λ) = uh_vacuum⁻ + (uh_vacuum⁺ - uh_vacuum⁻) * Λ
  φh(Λ) = φh⁻ + (φh⁺ - φh⁻) * Λ
  V_N = TestFESpace(Ωsolid, reffeu)
  Nh = interpolate_everywhere((x) -> VectorValue(1.0, 0.0), V_N)

  res_mag(Λ) = (φ, vφ) -> -1.0 * ∫((∇(vφ) ⋅ (∂Ψs∂H0 ∘ (F ∘ (∇(uhsolid(Λ))'), ℋ₀ ∘ (∇(φ)), Nh))))dΩsolid -
                          ∫((∇(vφ) ⋅ (∂Ψv∂H0 ∘ (F ∘ (∇(uhvacuum(Λ))'), ℋ₀ ∘ (∇(φ))))))dΩvacuum_mag
  jac_mag(Λ) = (φ, dφ, vφ) -> ∫(∇(vφ)' ⋅ ((∂Ψs∂H0H0 ∘ (F ∘ (∇(uhsolid(Λ))'), ℋ₀ ∘ (∇(φ)), Nh)) ⋅ ∇(dφ)))dΩsolid +
                              ∫(∇(vφ)' ⋅ ((∂Ψv∂H0H0 ∘ (F ∘ (∇(uhvacuum(Λ))'), ℋ₀ ∘ (∇(φ)))) ⋅ ∇(dφ)))dΩvacuum_mag
  res_mech(Λ) = begin
    assemble_vector!((v) -> ∫(∇(v)' ⊙ (∂Ψv∂F ∘ (F ∘ (∇(uhvacuum(Λ))'), ℋ₀ ∘ (∇(φh(Λ))))))dΩvacuum_mec, Pmaxw_, V_maxwell)
    interpolate_everywhere!(Interpolable(Pmaxw), tmaxw_, tmaxw.dirichlet_values, Usolid_int)
    Massoperator(tmaxw_)
    (u, v) -> ∫((∇(v)' ⊙ (∂Ψs∂F ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φh(Λ))), Nh))))dΩsolid + ∫((v ⋅ tmaxw))dΓs
  end
  jac_mech(Λ) = (u, du, v) -> ∫(∇(v)' ⊙ ((∂Ψs∂FF ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φh(Λ))), Nh)) ⊙ (∇(du)')))dΩsolid
  res_vacmech(Λ) = (u, v) -> ∫((∇(v)' ⊙ (∂Ψvm∂F ∘ (F ∘ (∇(u)')))))dΩvacuum_mec
  jac_vacmech(Λ) = (u, du, v) -> ∫(∇(v)' ⊙ ((∂Ψvm∂FF ∘ (F ∘ (∇(u)'))) ⊙ (∇(du)')))dΩvacuum_mec

  # Computational problems
  nls_mag = NewtonSolver(LUSolver(); maxiter=20, atol=1.e-9, rtol=1.e-8, verbose=verbose)
  cm_mag = StaticNonlinearModel(res_mag, jac_mag, Uφ⁺, Vφ, Dφ; nls=nls_mag, xh=φh⁺, backend=backend)
  nls_mech = NewtonSolver(LUSolver(); maxiter=20, atol=1.e-6, rtol=1.e-2, verbose=verbose)
  cm_mech = StaticNonlinearModel(res_mech, jac_mech, Uu_solid⁺, Vu_solid, Du_solid; nls=nls_mech, xh=uh_solid⁺, backend=backend)
  nls_vacmech = NewtonSolver(LUSolver(); maxiter=10, rtol=2, verbose=verbose)
  cm_vacmech = StaticNonlinearModel(res_vacmech, jac_vacmech, Uu_vacuum⁺, Vu_vacuum, Du_vacuum; nls=nls_vacmech, xh=uh_vacuum⁺, backend=backend)
  comp_model = StaggeredModel((cm_mag, cm_mech, cm_vacmech), (φh⁺, uh_solid⁺, uh_vacuum⁺), (φh⁻, uh_solid⁻, uh_vacuum⁻))

  problems = ((name="magnetic", res=res_mag, jac=jac_mag, U=Uφ⁺, V=Vφ, xh=φh⁺),
              (name="solid mechanics", res=res_mech, jac=jac_mech, U=Uu_solid⁺, V=Vu_solid, xh=uh_solid⁺),
              (name="vacuum mechanics", res=res_vacmech, jac=jac_vacmech, U=Uu_vacuum⁺, V=Vu_vacuum, xh=uh_vacuum⁺))
  (; comp_model, problems, Ωsolid, Ωvacuum, φh⁺, uh_solid⁺, uh_vacuum⁺)
end

function simulate(backend::Symbol; nsteps=nsteps, nsubsteps=nsubsteps, quiet=false)
  tb = @elapsed (s = build(backend))
  args = (Dict(:stepping => (nsteps=1, maxbisec=5), :ProjectDirichlet => true),
          Dict(:stepping => (nsteps=1, maxbisec=5)),
          Dict(:stepping => (nsteps=1, maxbisec=5), :ProjectDirichlet => true))
  ts = @elapsed solve!(s.comp_model; stepping=(nsteps=nsteps, nsubsteps=nsubsteps, maxbisec=1), kargsolve=args, verbose=false)
  quiet || @printf("%-6s: %d DOFs (φ %d, u solid %d, u vacuum %d) | build %.1f s | staggered solve %.1f s\n", backend,
          sum(p -> num_free_dofs(p.U), s.problems), (num_free_dofs(p.U) for p in s.problems)..., tb, ts)
  flush(stdout)
  if vtk && !quiet
    mkpath(OUT)
    writevtk(s.Ωsolid, joinpath(OUT, "hMRE_$(backend)"), cellfields=["uh" => s.uh_solid⁺])
    writevtk(s.Ωvacuum, joinpath(OUT, "vacuum_$(backend)"), cellfields=["uh" => s.uh_vacuum⁺, "-∇(φh)" => -∇(s.φh⁺)])
  end
  s, ts
end

println("hMRE + vacuum 2D (staggered), mode $mode, $nsteps steps × $nsubsteps substeps, $(Threads.nthreads()) threads")
if mode == "compare"
  # warm-up (compilation of HyperFEM, Gridap and the kernels) with one step, so that the timings compare runs
  tw = @elapsed foreach(b -> simulate(b; nsteps=1, nsubsteps=1, quiet=true), (:gridap, :kernel))
  @printf("warm-up (1 step each backend, compilation): %.1f s\n", tw)
  sg, tg = simulate(:gridap)
  sk, tk = simulate(:kernel)
  println("Final solutions, kernel vs Gridap (relative difference):")
  for (pg, pk) in zip(sg.problems, sk.problems)
    xg, xk = get_free_dof_values(pg.xh), get_free_dof_values(pk.xh)
    @printf("  %-17s ‖Δx‖/‖x‖ = %.2e   (‖x‖ = %.3e)\n", pg.name, norm(xk - xg) / norm(xg), norm(xg))
  end
  # at the converged state the residual is ~0 and its relative difference is meaningless: perturb it by 0.1 %
  println("Residual and Jacobian of each problem near the final state (HyperFEMKernels.check vs Gridap, Λ = 1):")
  for p in sk.problems
    x = get_free_dof_values(p.xh) .* (1 .+ 1e-3 .* randn(num_free_dofs(p.U)))
    op = KernelOperator(p.res(1.0), p.jac(1.0), p.U, p.V)
    c = check(p.res(1.0), p.jac(1.0), p.U, p.V, x; op)
    nd = count(g -> g.lin === nothing, op.jgroups)
    fb = isempty(op.fallback.rtrians) && isempty(op.fallback.jtrians) ? "none" : "residual $(length(op.fallback.rtrians)), Jacobian $(length(op.fallback.jtrians))"
    al = maximum(a -> a.bytes, allocations(op, x))
    @printf("  %-17s res %.2e  jac %.2e | groups: %d res, %d jac (%d dual) | fallback: %s | max bytes/element %d\n",
            p.name, c.res, c.jac, length(op.rgroups), length(op.jgroups), nd, fb, al)
  end
  @printf("Staggered solve: Gridap %.1f s, kernel %.1f s (x%.1f)\n", tg, tk, tg / tk)
elseif mode in ("gridap", "kernel")
  simulate(Symbol(mode))
end                                   # mode=none: only the definitions (used by dev scripts)
println("FIN HMRE VACUUM")

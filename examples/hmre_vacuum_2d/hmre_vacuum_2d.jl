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
# built with `backend` (HyperFEM branch kernel-backend). The Dirichlet conditions are written through
# BoundaryConditions, and the solid–vacuum coupling prescribes the state of the solid on the interface,
# BoundaryCondition("Interface", Coupling(uhair_int_)), a function of the stage parameter of the solver
# (HyperFEM PR #249 fixed the regression of the former DirichletCoupling; the Coupling prescription
# comes with the next PR).
#
# Usage (from the repository root, benchmark environment):
#   julia -t auto --project=benchmark examples/hmre_vacuum_2d/hmre_vacuum_2d.jl [mode=compare|gridap|kernel]
#        [nsteps=20] [nsubsteps=2] [vtk=off]
#   mode=compare : after a one-step warm-up of both backends, runs the whole staggered simulation with Gridap and with HyperFEMKernels, compares the final
#                  solutions of the three problems, and checks residual and Jacobian of each problem against
#                  Gridap at the final state (HyperFEMKernels.check).
# =============================================================================

kw = Dict(split(a, "=")[1] => split(a, "=")[2] for a in ARGS if occursin("=", a))
mode = get(kw, "mode", "compare")
nsteps = parse(Int, get(kw, "nsteps", "20"))
nsubsteps = parse(Int, get(kw, "nsubsteps", "2"))
vtk = get(kw, "vtk", "off") == "on"
const OUT = joinpath(@__DIR__, "results")


include(joinpath(@__DIR__, "setup.jl"))

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
  tw = @elapsed foreach(b -> simulate(b; nsteps=1, nsubsteps=1, quiet=true), (:lazy, :compiled))
  @printf("warm-up (1 step each backend, compilation): %.1f s\n", tw)
  sg, tg = simulate(:lazy)
  sk, tk = simulate(:compiled)
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

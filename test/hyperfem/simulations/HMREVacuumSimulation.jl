# hMRE beam with explicit surrounding vacuum, 2D, staggered magneto-mechanics (the three problems of
# examples/hmre_vacuum_2d: magnetic potential on solid + vacuum, solid mechanics with the Maxwell traction of
# the vacuum on the interface, fictitious mechanics of the vacuum driven by the solid through a Coupling
# prescription). Setup shared with the example (examples/hmre_vacuum_2d/setup.jl, Short_magnet.msh,
# 25 375 dofs). Reduced for a test: half of the applied potential (the original is 0.2 · 5200 V) and four
# staggered steps, with the same solvers, stepping options and projections as the example.
include(normpath(joinpath(@__DIR__, "..", "..", "..", "examples", "hmre_vacuum_2d", "setup.jl")))

function hmre_vacuum_simulation(; backend=:gridap, φbot=0.2 * 5200 / 2, nsteps=4, nsubsteps=1, verbose=false)
  s = build(backend; φbot, verbose)
  args = (Dict(:stepping => (nsteps=1, maxbisec=5), :ProjectDirichlet => true),
          Dict(:stepping => (nsteps=1, maxbisec=5)),
          Dict(:stepping => (nsteps=1, maxbisec=5), :ProjectDirichlet => true))
  solve!(s.comp_model; stepping=(nsteps=nsteps, nsubsteps=nsubsteps, maxbisec=1), kargsolve=args, verbose=verbose)
  (; φ=copy(get_free_dof_values(s.φh⁺)), u_solid=copy(get_free_dof_values(s.uh_solid⁺)),
     u_vacuum=copy(get_free_dof_values(s.uh_vacuum⁺)))
end

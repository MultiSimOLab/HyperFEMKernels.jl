# NeoMag (repo ../NeoMag, mesh data/models/NeoMag2D_id2.msh) with the compiled backend: cost of the
# operator updates per load step. Reports, after a 2-step warm-up in the same session, for 20 and 100
# staggered steps: t_solve, the time outside the Newton solves (t_solve − Newton of both problems), the
# assembly and line-search times of the magnetic and mechanical Newtons, the accumulated time and number
# of calls of HyperFEMKernelsExt.build_operator (wrapped here with a counter) and the peak memory; the
# 100-step result is saved for tools/compare_results.jl.
#
#   julia -t auto --project=<env with HyperFEM and the HyperFEMKernels to measure> benchmark/neomag/update_bench.jl <label> [profile]
#
# (the environment is NeoMag's Project.toml with the [sources] paths of the HyperFEMKernels version to
# measure: main 0.2.0, the ∇∇ commits, ∇∇ + cheap update). With `profile`, a flat @profile of 10 more
# steps is written to <NeoMag>/data/sims/bench_<label>_profile.txt.
using Printf, Serialization, Profile
label = ARGS[1]
const NEOMAG = abspath(joinpath(@__DIR__, "..", "..", "..", "NeoMag"))
include(joinpath(NEOMAG, "run_simulation_2d.jl"))
const EXT = Base.get_extension(HyperFEM, :HyperFEMKernelsExt)
const BUILD = Ref((0.0, 0))
@eval EXT function HyperFEM.ComputationalModels.build_operator(b::CompiledBackend, res, jac, U, V, assem, previous)
    t = time()
    op = previous isa KernelOperator ? update(previous, res, jac) : KernelOperator(res, jac, U, V; jacobian=b.jacobian)
    Main.BUILD[] = (Main.BUILD[][1] + (time() - t), Main.BUILD[][2] + 1)
    op
end
mesh = joinpath(NEOMAG, "data", "models", "NeoMag2D_id2.msh")
out(s) = joinpath(NEOMAG, "data", "sims", "bench_$(label)_$(s)")
println("== $label: HyperFEMKernels $(pkgversion(HyperFEMKernels)) at $(pkgdir(HyperFEMKernels)); $(Threads.nthreads()) threads")
neomag_simulation(mesh, out("warmup"); steps=2, vtk=false)                       # warm-up (compilation)
for steps in (20, 100)
    BUILD[] = (0.0, 0)
    r = neomag_simulation(mesh, out(steps); steps, vtk=false)
    steps == 100 && serialize(joinpath(out(steps), "result.jls"), r)
    outside = r.t_solve - r.stats_mag.t_solve - r.stats_mech.t_solve
    @printf("RESULT %s steps=%d t_solve=%.1f outside_newton=%.2f (%.3f/step) build_operator=%.2f s in %d calls | magnetic assembly %.2f linesearch %.2f | mechanics assembly %.2f linesearch %.2f | maxrss %.0f MB\n",
            label, steps, r.t_solve, outside, outside / steps, BUILD[][1], BUILD[][2],
            r.stats_mag.t_assembly, r.stats_mag.t_linesearch, r.stats_mech.t_assembly, r.stats_mech.t_linesearch, r.maxrss_MB)
end
if length(ARGS) > 1 && ARGS[2] == "profile"
    Profile.clear()
    @profile neomag_simulation(mesh, out("profile"); steps=10, vtk=false)
    open(joinpath(NEOMAG, "data", "sims", "bench_$(label)_profile.txt"), "w") do io
        Profile.print(io; format=:flat, sortedby=:count, mincount=50, noisefloor=2)
    end
    println("profile written")
end

# Time breakdown of the staggered solve of the hMRE + vacuum example (sampling profiler, inclusive samples).
#   julia -t auto --project=benchmark dev/prof_hmre_vacuum_solve.jl [backend=kernel|gridap]
using Profile, Printf
const B = Symbol(get(Dict(split(a, "=")[1] => split(a, "=")[2] for a in ARGS if occursin("=", a)), "backend", "kernel"))
empty!(ARGS); append!(ARGS, ["mode=none"])
include(joinpath(@__DIR__, "..", "examples", "hmre_vacuum_2d", "hmre_vacuum_2d.jl"))
args = (Dict(:stepping => (nsteps=1, maxbisec=5), :ProjectDirichlet => true), Dict(:stepping => (nsteps=1, maxbisec=5)),
        Dict(:stepping => (nsteps=1, maxbisec=5), :ProjectDirichlet => true))
s = build(B); solve!(s.comp_model; stepping=(nsteps=1, nsubsteps=1, maxbisec=1), kargsolve=args, verbose=false)   # compile
s = build(B)
Profile.init(n=10^8, delay=0.002)
t = @elapsed @profile solve!(s.comp_model; stepping=(nsteps=4, nsubsteps=2, maxbisec=1), kargsolve=args, verbose=false)
data = Profile.fetch(include_meta=false); lidict = Profile.getdict(data)
# split samples (each sample ends with a 0 separator), inclusive count of frames matching each pattern
cats = ["dirichlet_preconditioning!" => r"^dirichlet_preconditioning!$", "AffineFEOperator (Gridap)" => r"^AffineFEOperator$",
        "TrialFESpace! (Dirichlet values)" => r"^TrialFESpace!$", "InterpolableBC / point search" => r"_point_to_cell!|^InterpolableBC$",
        "res builder (res_mech side effects)" => r"^res_mech$", "assemble_vector! (Maxwell stress)" => r"^assemble_vector!$",
        "build_operator / update (kernel)" => r"^build_operator$|^update$", "KernelOperator residual!/jacobian!" => r"^run_cells!$|^fallback_",
        "Gridap residual!/jacobian! (FEOperator)" => r"^numeric_loop_vector!$|^numeric_loop_matrix!$|^numeric_loop_matrix_and_vector!$",
        "LU (UMFPACK)" => r"umfpack|^lu!?$", "GC" => r"^GC|gc_collect|jl_gc",
        "StaggeredModel solve! (total)" => r"^solve!$", "Newton solve! (total)" => r"^_solve_nr!$|^_solve_newton!$|^_solve!$|^solve!$"]
counts = Dict(k => 0 for (k, _) in cats); total = 0
sample = UInt64[]
for ip in data
    if ip == 0
        isempty(sample) && continue
        global total += 1
        names = Set{String}()
        for i in sample, fr in get(lidict, i, Base.StackTraces.StackFrame[])
            push!(names, string(fr.func))
        end
        any(n -> n == "_include" || n == "include_string", names) || (global total -= 1; empty!(sample); continue)   # main task only (idle workers excluded)
        for (k, re) in cats
            any(n -> occursin(re, n), names) && (counts[k] += 1)
        end
        empty!(sample)
    else
        push!(sample, ip)
    end
end
@printf("\nbackend %s: staggered solve of 4 steps x 2 substeps: %.1f s, %d samples (main task)\n", B, t, total)
for (k, _) in cats
    @printf("  %-42s %5.1f %%\n", k, 100 * counts[k] / max(total, 1))
end
println("FIN PROF SOLVE")

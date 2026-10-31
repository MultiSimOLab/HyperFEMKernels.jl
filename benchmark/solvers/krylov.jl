# Stage S1 (design/solvers.md): Krylov + AMG on the assembled tangent against LU, through HyperFEM's driver
# (NewtonRaphson), on the EM cases of the baseline (D4 refined 2026-10-30: the ×5 criterion is measured on EM).
# One case, one backend and one linear-solver configuration per process; appends a row to
# benchmark/results/solvers_krylov.csv.
#
#   julia -t 6 --project=benchmark/solvers benchmark/solvers/krylov.jl mech_box 52 11 11 compiled cg_gamg [rtol=1e-10] [nsteps=5] [bs=3] [opts=-pc_gamg_threshold,0.01]
#   julia -t 6 --project=benchmark/solvers benchmark/solvers/krylov.jl em_plate 6 compiled lu
#
# configuration = lu | mumps | <ksp>_<pc>[_nonull][_lagN][_neg], ksp ∈ cg gmres, pc ∈ gamg boomer, e.g. gmres_gamg_lag3
# (PETSc preconditioner reused for 3 Newton iterations), cg_gamg_nonull (without the rigid-body modes),
# gmres_boomer_neg (the φ equations of a multi-field case negated: positive-real form of the saddle point).
# rtol is the Krylov tolerance (relative to the right-hand side); nsteps overrides the case's load steps.
include(joinpath(@__DIR__, "cases.jl"))
using HyperFEMKernels, GridapPETSc
using Gridap.Algebra
using LinearAlgebra, SparseArrays, Printf, Dates

rss() = Sys.maxrss() / 2^30

function parse_args(args)
    i = findfirst(a -> a in ("lazy", "compiled"), args)
    words = args[1:i-1]; backend = Symbol(args[i]); config = args[i+1]
    kw = Dict{String,String}()
    for a in args[i+2:end]
        k, v = split(a, "="); kw[k] = v
    end
    rtol = parse(Float64, get(kw, "rtol", "1e-10"))
    nsteps = haskey(kw, "nsteps") ? parse(Int, kw["nsteps"]) : 0
    bs = parse(Int, get(kw, "bs", "1"))
    opts = replace(get(kw, "opts", ""), "," => " ")
    words, backend, config, rtol, nsteps, bs, opts
end

"Linear solver of a configuration word (needs the near-null space N and the trial space U)."
function make_solver(config::String, N, rtol, U, bs, opts)
    config == "lu" && return LUSolver()
    config == "mumps" && return PETScSolver(:preonly, :mumps)
    parts = split(config, "_")
    ksp = Symbol(parts[1]); pc = parts[2] == "boomer" ? :boomeramg : Symbol(parts[2])
    nullspace = "nonull" in parts ? nothing : N
    lag = findfirst(startswith("lag"), parts)
    reuse = lag === nothing ? 1 : parse(Int, parts[lag][4:end])
    negate = nothing
    if "neg" in parts
        U isa MultiFieldFESpace || error("_neg needs a multi-field case")
        nu = num_free_dofs(U[1]); negate = nu+1:num_free_dofs(U)
    end
    PETScSolver(ksp, pc; rtol, maxiter=300, reuse, nullspace, negate, blocksize=bs, options=opts)
end

function main(args)
    words, backend, config, rtol, nsteps_arg, bs, opts = parse_args(args)
    t0 = time()
    t_case = @elapsed ((case, tag) = build_case(words))
    nsteps = nsteps_arg > 0 ? nsteps_arg : case.nsteps
    t_null = @elapsed (N = near_nullspace(case.U))
    ls = make_solver(config, N, rtol, case.U, bs, opts)
    nls = NewtonRaphson(ls; case.nls_kw..., verbose=1)
    t_build = @elapsed (m = StaticNonlinearModel(case.res, case.jac, case.U, case.V, case.D; nls, backend, case.extra...))
    n = length(m.caches.x)
    @printf("%s  backend=%s  config=%s  rtol=%.0e  bs=%d  opts=\"%s\"  threads=%d  DOFs=%d  nullspace %d modes (%.2f s)  build %.1f s  rss %.2f GB\n",
            tag, backend, config, rtol, bs, opts, Threads.nthreads(), n, size(N, 2), t_null, t_build, rss())
    flush(stdout)
    st = @timed solve!(m; stepping=(nsteps=nsteps, maxbisec=2))
    x, flag = st.value
    log = nls.log
    nit = log.niters; nsolves = log.nsolves
    klin = log.nlinear; kmean = klin / max(nit, 1)
    klast = collect(linear_history(nls))
    @printf("  solve! %s: %d solves, %d Newton iterations, %.1f s (%.3f s/iteration), %.2f GB allocated, rss %.2f GB\n",
            flag == 1 ? "converged" : "FAILED", nsolves, nit, st.time, st.time / max(nit, 1), st.bytes / 2^30, rss())
    @printf("  per iteration: assembly %.3f s, setup %.3f s, linear solve %.3f s, line search %.3f s; Krylov %.1f its/iteration (last solve: %s)\n",
            log.t_assembly / nit, log.t_setup / nit, log.t_linear / nit, log.t_linesearch / nit, kmean, string(klast))
    nc = m.caches.nls_cache
    builds = nc.ns === nothing ? 0 : preconditioner_builds(nc.ns)
    resdir = normpath(joinpath(@__DIR__, "..", "results")); mkpath(resdir)
    file = joinpath(resdir, "solvers_krylov.csv")
    isfile(file) || open(file, "w") do f
        println(f, "date,case,backend,threads,dofs,config,rtol,bs,opts,nsteps,flag,nsolves,nit,klin,kmean,kmax_last,t_solve,t_assembly,t_setup,t_linear,t_linesearch,builds,alloc_gb,rss_gb,t_null,t_build,t_total")
    end
    open(file, "a") do f
        println(f, join((Dates.today(), tag, backend, Threads.nthreads(), n, config, rtol, bs, replace(opts, " " => ";"), nsteps, flag, nsolves, nit, klin, kmean,
                         isempty(klast) ? 0 : maximum(klast), st.time, log.t_assembly, log.t_setup, log.t_linear, log.t_linesearch,
                         builds, st.bytes / 2^30, rss(), t_null, t_build, time() - t0), ","))
    end
end

main(ARGS)

# Solver baseline (design/solvers.md, S0 + R0): per Newton iteration (residual, Jacobian, LU
# factorisation, triangular solve), per load step and per solve!, with backend :lazy or :compiled,
# on one case and one thread count per process. Appends a row to benchmark/results/solvers_baseline.csv.
#
#   julia -t 6 --project=benchmark/solvers benchmark/solvers/baseline.jl em_plate 4 compiled [lu_max] [driver]
#   julia -t 1 --project=benchmark/solvers benchmark/solvers/baseline.jl em_box 32 7 7 lazy
#   julia -t 18 --project=benchmark/solvers benchmark/solvers/baseline.jl hmre 7 compiled
#
# lu_max (default 300000): above this number of DOFs the LU is skipped (the factorisation does not
# fit or takes too long); only assembly is measured. With a third word `driver`, the full solve! of
# the case (its own number of load steps, as in benchmark/phase1) is also run through the driver and
# timed (stage R0): time and allocations per step and per iteration beyond assembly and LU.
# UMFPACK allocates through Julia's allocator, so the bytes of lu(A) are the memory of the factors.
include(joinpath(@__DIR__, "cases.jl"))
using HyperFEMKernels, GridapSolvers, GridapSolvers.NonlinearSolvers
using Gridap.Algebra
using LinearAlgebra, SparseArrays, Printf, Dates

function parse_args(args)
    i = findfirst(a -> a in ("lazy", "compiled"), args)
    words = args[1:i-1]; backend = Symbol(args[i])
    lu_max = length(args) ≥ i + 1 ? parse(Int, args[i+1]) : 300_000
    driver = length(args) ≥ i + 2 && args[i+2] == "driver"
    words, backend, lu_max, driver
end

best(f, n=3) = minimum(begin t = @elapsed f(); t end for _ in 1:n)
bytes(f) = @allocated f()
rss() = Sys.maxrss() / 2^30

function main(args)
    words, backend, lu_max, driver = parse_args(args)
    nsteps = case_nsteps = 0
    t0 = time()
    t_case = @elapsed ((case, tag) = build_case(words))
    rss_case = rss()
    nls = NewtonSolver(LUSolver(); case.nls_kw..., verbose=false)
    t_build = @elapsed (m = StaticNonlinearModel(case.res, case.jac, case.U, case.V, case.D; nls, backend, case.extra...))
    @printf("  mesh + spaces %.1f s (rss %.2f GB); StaticNonlinearModel %.1f s (rss %.2f GB)\n", t_case, rss_case, t_build, rss())
    op = m.caches.op
    x = m.caches.x
    n = length(x)
    @printf("%s  backend=%s  threads=%d  DOFs=%d  build %.1f s  (rss %.2f GB)\n", tag, backend, Threads.nthreads(), n, t_build, rss())

    # --- per Newton iteration, on the operator ---
    b = residual(op, x); A = jacobian(op, x)                      # warm-up: compilation, pattern
    residual!(b, op, x); jacobian!(A, op, x)
    t_res = best(() -> residual!(b, op, x));   b_res = bytes(() -> residual!(b, op, x))
    t_jac = best(() -> jacobian!(A, op, x));   b_jac = bytes(() -> jacobian!(A, op, x))
    nnzA = nnz(A); mat_gb = (nnzA * 12 + (n + 1) * 8) / 2^30
    rss_asm = rss()
    @printf("  residual! %.4f s (%d B)   jacobian! %.4f s (%d B)   nnz %d  matrix %.2f GB  rss %.2f GB\n",
            t_res, b_res, t_jac, b_jac, nnzA, mat_gb, rss_asm)

    t_lu = t_ldiv = NaN; rss_lu = NaN; lu_gb = NaN; b_lu = b_ldiv = 0
    if n ≤ lu_max
        dx = similar(b)
        try
            F = lu(A); ldiv!(dx, F, b)                                 # warm-up
            t_lu = @elapsed (F = lu(A))
            t_ldiv = best(() -> ldiv!(dx, F, b))
            b_lu = @allocated lu(A); b_ldiv = @allocated ldiv!(dx, F, b)
            rss_lu = rss()
            F = nothing; GC.gc()
            lu_gb = b_lu / 2^30
            @printf("  lu %.3f s  ldiv! %.4f s  LU factors %.2f GB  rss %.2f GB\n", t_lu, t_ldiv, lu_gb, rss_lu)
        catch e
            println("  LU failed: ", sprint(showerror, e)[1:min(end, 200)])
        end
    else
        println("  LU skipped (DOFs > lu_max = $lu_max)")
    end

    # --- per load step and per solve!, through the driver (R0) ---
    t_solve = NaN; b_solve = NaN; gc_solve = NaN; nit = 0; t_step = NaN; b_driver = NaN
    if driver && n ≤ lu_max
        nsteps = case.nsteps
        m2 = StaticNonlinearModel(case.res, case.jac, case.U, case.V, case.D;
                                  nls=NewtonSolver(LUSolver(); case.nls_kw..., verbose=true), backend, case.extra...)
        logfile = tempname()
        st = open(logfile, "w") do io
            redirect_stdout(io) do
                r = @timed solve!(m2; stepping=(nsteps=nsteps, maxbisec=1))
                flush(io); r
            end
        end
        log = read(logfile, String); rm(logfile)
        nit = count("> Iteration", log) - count("Iteration   0", log)
        nstp = count("STEP:", log)
        t_solve, b_solve, gc_solve = st.time, st.bytes, st.gctime
        t_step = t_solve / max(nstp, 1)
        # driver remainder: what the solve! allocated beyond assembly and LU of its iterations
        b_driver = b_solve - nit * (b_res + b_jac + b_lu + b_ldiv)   # what the driver itself allocates
        @printf("  solve! %d steps, %d Newton iterations: %.2f s (%.2f s/step, %.3f s/iteration), %.2f GB allocated (gc %.1f %%), rss %.2f GB\n",
                nstp, nit, t_solve, t_step, t_solve / max(nit, 1), b_solve / 2^30, 100gc_solve / t_solve, rss())
        @printf("  per iteration estimate: assembly %.3f s + lu %.3f s + ldiv %.4f s = %.3f s of %.3f s measured (rest %.1f %%)\n",
                t_res + t_jac, t_lu, t_ldiv, t_res + t_jac + t_lu + t_ldiv, t_solve / max(nit, 1),
                100 * (1 - (t_res + t_jac + t_lu + t_ldiv) * nit / t_solve))
        @printf("  driver allocations beyond assembly and LU: %.1f MB per iteration\n", b_driver / max(nit, 1) / 2^20)
    end

    resdir = normpath(joinpath(@__DIR__, "..", "results")); mkpath(resdir)
    file = joinpath(resdir, "solvers_baseline.csv")
    isfile(file) || open(file, "w") do f
        println(f, "date,case,backend,threads,dofs,nnz,t_build,t_res,b_res,t_jac,b_jac,matrix_gb,rss_asm_gb,t_lu,t_ldiv,lu_gb,rss_lu_gb,b_lu,b_ldiv,nsteps,nit,t_solve,b_solve,b_driver,gc_solve,rss_end_gb,t_total")
    end
    open(file, "a") do f
        println(f, join((Dates.today(), tag, backend, Threads.nthreads(), n, nnzA, t_build, t_res, b_res, t_jac, b_jac, mat_gb, rss_asm,
                         t_lu, t_ldiv, lu_gb, rss_lu, b_lu, b_ldiv, nsteps, nit, t_solve, b_solve, b_driver, gc_solve, rss(), time() - t0), ","))
    end
end

main(ARGS)

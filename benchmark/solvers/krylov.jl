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
# gmres_boomer_neg (the φ equations of a multi-field case negated: positive-real form of the saddle point);
# S2 block preconditioners: fs_schur_upper_a11, fs_schur_full_selfp, fs_additive, fs_multiplicative, with the
# suffixes _fgmres (FGMRES outside), _lagN, _lublocks (MUMPS on every block: the exact block solve), _boomer (BoomerAMG
# on every block), _phiboomer (GAMG on u, BoomerAMG on the potentials).
# The suffix _ew (R4a) adds the Eisenstat–Walker forcing term to the Newton (`forcing = EisenstatWalker()`): the Krylov
# tolerance of every iteration is chosen by the Newton, rtol is then the tolerance the solver was set up with; ewmax=
# sets its ηmax (the library default, 1e-2, if absent).
# S3 matrix-free configurations: mf_<krylov>_<prec>[_lagN][_ew], krylov ∈ gmres fgmres, prec ∈ gamg (PETSc GAMG with
# the rigid modes, one V-cycle), fs (PETSc fieldsplit additive, one V-cycle per block; multi-field cases), mumps,
# blockjacobi (the nodal block diagonal, no matrix): the Newton receives the MatrixFreeJacobian of the compiled backend
# (`CompiledBackend(jacobian = :matrixfree)`) and HyperFEMKernels' MatrixFreeSolver applies it; with gamg/fs/mumps the
# tangent is assembled inside `LaggedAssembled` every N Newton iterations (lagN; 1 if absent) for the preconditioner.
# rtol is the Krylov tolerance (relative to the right-hand side); nsteps overrides the case's load steps; restart= the GMRES
# restart of the mf_* solver (50; PETSc uses 30); probe=1 times the matrix-free product against the SpMV after the solve.
# S4-A p-multigrid: mf_<krylov>_pmg[lu|mumps][_lagN][_degK][_rngA][_q3] — one V-cycle (Chebyshev degree K, range A on the
# fine level; the case's order → Q1, or Q3 → Q2 → Q1 with _q3) per Krylov iteration, the Q1 level assembled every N Newton
# iterations with one GAMG V-cycle (rigid-body modes of the Q1 space; pmglu: LU, pmgmumps: MUMPS) as coarse solver.
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
    ewmax = haskey(kw, "ewmax") ? parse(Float64, kw["ewmax"]) : nothing   # ηmax of the forcing term (_ew); library default if absent
    probe = get(kw, "probe", "0") == "1"          # mf_*: time the matrix-free product vs the SpMV and the preconditioner application
    restart = parse(Int, get(kw, "restart", "50"))   # mf_*: GMRES restart (PETSc's default is 30)
    words, backend, config, rtol, nsteps, bs, opts, ewmax, probe, restart
end

"Matrix-free product vs SpMV of the assembled tangent, and one preconditioner application, at the final state (10 repetitions each)."
function probe_operator(m)
    nc = m.caches.nls_cache
    mf, ns = nc.A, nc.ns
    mf isa MatrixFreeJacobian || return
    op = mf.op
    A = copy(HyperFEMKernels.sparsity_pattern(op)); jacobian!(A, op, mf.x)
    v = randn(size(mf, 2)); y = similar(v); y2 = similar(v)
    mul!(y, mf, v); mul!(y2, A, v)
    tmf = minimum(@elapsed(mul!(y, mf, v)) for _ in 1:10)
    tsp = minimum(@elapsed(mul!(y2, A, v)) for _ in 1:10)
    @printf("  probe: matrix-free product %.4f s, SpMV %.4f s (nnz %d, %.0f nnz/row), |mf v - A v| / |A v| = %.1e", tmf, tsp, nnz(A), nnz(A) / size(A, 1), norm(y - y2) / norm(y2))
    if ns !== nothing && ns.prec !== nothing
        ldiv!(y, ns.prec, v)
        tp = minimum(@elapsed(ldiv!(y, ns.prec, v)) for _ in 1:10)
        @printf(", preconditioner application %.4f s", tp)
    end
    println()
end

"Polynomial order of a Lagrangian vector/scalar space on hexahedra/quadrilaterals ((p+1)^D nodes per cell)."
function hex_order(V)
    D = num_cell_dims(get_triangulation(V)); nc = num_components(HyperFEMKernels.field_valtype(V))
    nn = length(first(get_cell_dof_ids(V))) ÷ nc
    round(Int, nn^(1 / D)) - 1
end

"""
Coarse p-multigrid levels of a single-field case on the same model and boundary conditions (the benchmark's version of
the HyperFEM helper `level_spaces`, DS4A-1): the trial and test spaces of `orders`, finest first (the case's own pair).
"""
function pmg_level_spaces(case, orders)
    U, V = case.U, case.V
    U isa MultiFieldFESpace && error("pmg: multi-field cases are a follow-up of S4-A (DS4A-2)")
    model = get_background_model(get_triangulation(U))
    T = HyperFEMKernels.field_valtype(V)
    spaces = Any[(U, V)]
    for p in orders[2:end]
        Vl = TestFESpace(model, ReferenceFE(lagrangian, T, p), case.D, conformity=:H1)
        push!(spaces, (TrialFESpace(Vl, case.D, 0.0), Vl))
    end
    spaces
end

"Linear solver of a configuration word (needs the near-null space N and the trial space U; the case for pmg)."
function make_solver(config::String, N, rtol, U, bs, opts, restart=50; case=nothing)
    config == "lu" && return LUSolver()
    config == "mumps" && return PETScSolver(:preonly, :mumps)
    parts = split(config, "_")
    if parts[1] == "mf"                        # S3: mf_<krylov>_<prec>[_lagN]
        krylov = Symbol(parts[2])
        lag = findfirst(startswith("lag"), parts)
        every = lag === nothing ? 1 : parse(Int, parts[lag][4:end])
        pw = String(parts[findfirst(i -> i ≥ 3 && !startswith(parts[i], "lag"), eachindex(parts))])
        if startswith(pw, "pmg")                # S4-A: mf_<krylov>_pmg[lu|mumps][_lagN][_degK][_rngA][_q3] (coarse GAMG V-cycle by default)
            deg = findfirst(startswith("deg"), parts); rng = findfirst(startswith("rng"), parts)
            smoother = Chebyshev(; degree=deg === nothing ? 4 : parse(Int, parts[deg][4:end]),
                                 range=rng === nothing ? 20.0 : parse(Float64, parts[rng][4:end]))
            orders = "q3" in parts ? (3, 2, 1) : (hex_order(case.V), 1)
            spaces = pmg_level_spaces(case, orders)
            coarse = pw == "pmglu" ? LUSolver() : pw == "pmgmumps" ? PETScSolver(:preonly, :mumps) :
                     PETScSolver(:preonly, :gamg; nullspace=near_nullspace(spaces[end][1]), blocksize=bs, options=opts)
            return MatrixFreeSolver(; krylov, rtol, maxiter=300, restart, preconditioner=PMultigrid(spaces; smoother, coarse, coarse_every=every))
        end
        prec = pw == "blockjacobi" ? BlockJacobi() :
               pw == "mumps" ? LaggedAssembled(PETScSolver(:preonly, :mumps); every) :
               pw == "gamg" ? LaggedAssembled(PETScSolver(:preonly, :gamg; nullspace=N, blocksize=bs, options=opts); every) :
               pw == "fs" ? LaggedAssembled(PETScSolver(U; ksp=:preonly, split=:additive, options=opts); every) :
               error("unknown matrix-free preconditioner $pw (gamg, fs, mumps, blockjacobi)")
        pw == "fs" && !(U isa MultiFieldFESpace) && error("mf_*_fs needs a multi-field case")
        return MatrixFreeSolver(; krylov, rtol, maxiter=300, restart, preconditioner=prec)
    end
    if parts[1] == "fs"                        # S2: fs_<split>[_<fact>_<prec>][_<ksp>][_lagN][_lublocks]
        U isa MultiFieldFESpace || error("fs_* needs a multi-field case")
        split_ = Symbol(parts[2])
        fact = length(parts) ≥ 4 && parts[3] in ("upper", "lower", "full", "diag") ? Symbol(parts[3]) : :upper
        prec = length(parts) ≥ 4 && parts[4] in ("a11", "selfp") ? Symbol(parts[4]) : :a11
        ksp = "fgmres" in parts ? :fgmres : :gmres
        lag = findfirst(startswith("lag"), parts)
        reuse = lag === nothing ? 1 : parse(Int, parts[lag][4:end])
        blocks = "lublocks" in parts ? fill(:mumps, num_fields(U)) :
                 "boomer" in parts ? fill(:boomeramg, num_fields(U)) :          # BoomerAMG on every block
                 "phiboomer" in parts ? [:gamg; fill(:boomeramg, num_fields(U) - 1)] :   # GAMG on u, BoomerAMG on the potentials
                 fill(:gamg, num_fields(U))
        return PETScSolver(U; ksp, split=split_, schur=(fact=fact, precondition=prec), blocks, rtol, maxiter=300, reuse, options=opts)
    end
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
    words, backend, config, rtol, nsteps_arg, bs, opts, ewmax, probe, restart = parse_args(args)
    t0 = time()
    t_case = @elapsed ((case, tag) = build_case(words))
    nsteps = nsteps_arg > 0 ? nsteps_arg : case.nsteps
    t_null = @elapsed (N = near_nullspace(case.U))
    ew = endswith(config, "_ew")
    ls = make_solver(ew ? config[1:end-3] : config, N, rtol, case.U, bs, opts, restart; case)
    startswith(config, "mf_") && (backend = CompiledBackend(jacobian=:matrixfree))   # the Newton receives the operator
    nls = NewtonRaphson(ls; case.nls_kw..., verbose=1, forcing=!ew ? nothing : ewmax === nothing ? EisenstatWalker() : EisenstatWalker(ηmax=ewmax, η0=min(1e-2, ewmax)))
    t_build = @elapsed (m = StaticNonlinearModel(case.res, case.jac, case.U, case.V, case.D; nls, backend, case.extra...))
    n = length(m.caches.x)
    @printf("%s  backend=%s  config=%s  rtol=%.0e  bs=%d  opts=\"%s\"  threads=%d  DOFs=%d  nullspace %d modes (%.2f s)  build %.1f s  rss %.2f GB\n",
            tag, backend isa Symbol ? backend : "compiled/$(backend.jacobian)", config, rtol, bs, opts, Threads.nthreads(), n, size(N, 2), t_null, t_build, rss())
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
    probe && probe_operator(m)
    resdir = normpath(joinpath(@__DIR__, "..", "results")); mkpath(resdir)
    file = joinpath(resdir, "solvers_krylov.csv")
    isfile(file) || open(file, "w") do f
        println(f, "date,case,backend,threads,dofs,config,rtol,bs,opts,nsteps,flag,nsolves,nit,klin,kmean,kmax_last,t_solve,t_assembly,t_setup,t_linear,t_linesearch,builds,alloc_gb,rss_gb,t_null,t_build,t_total")
    end
    open(file, "a") do f
        println(f, join((Dates.today(), tag, backend isa Symbol ? backend : "compiled_mf", Threads.nthreads(), n, config, rtol, bs, replace(opts, " " => ";"), nsteps, flag, nsolves, nit, klin, kmean,
                         isempty(klast) ? 0 : maximum(klast), st.time, log.t_assembly, log.t_setup, log.t_linear, log.t_linesearch,
                         builds, st.bytes / 2^30, rss(), t_null, t_build, time() - t0), ","))
    end
end

main(ARGS)

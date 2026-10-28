# Compilation cost (time to first assembly): Gridap vs kernel backend.
# Every measurement runs in a fresh Julia process, so nothing is compiled beforehand
# except what the package images already contain.
#
#   julia -t auto --project=benchmark benchmark/bench_compile.jl
#
# Per case and backend it reports wall time and the part spent compiling
# (Base.cumulative_compile_time_ns) for: building the operator, the first residual!,
# the first jacobian!, and a second (warm) residual!+jacobian!. A last process per backend
# assembles the two cases in sequence, so the second one shows the marginal cost of a new
# weak form / element once the engine is compiled.
using Printf

const CASES = Dict(
    "em_p1" => "EM plate, P1 tets, u-φ",
    "em_q2" => "EM box 6x2x2, Q2 hex, u-φ",
)

if length(ARGS) >= 2 && ARGS[1] == "child"
    backend = ARGS[2]; cases = ARGS[3:end]
    Base.cumulative_compile_timing(true)
    ct() = Base.cumulative_compile_time_ns()[1] / 1e9
    function timed(f)
        c0 = ct(); t0 = time_ns(); r = f(); t = (time_ns() - t0) / 1e9
        r, t, ct() - c0
    end
    _, tu, cu = timed() do
        @eval using HyperFEM, Gridap, GridapGmsh, Gridap.FESpaces, Gridap.Algebra
        backend == "kernel" && @eval using HyperFEMKernels
        nothing
    end
    @eval include(joinpath(@__DIR__, "bench_cases.jl"))
    println("ROW,", backend, ",using,", tu, ",", cu)
    meshdir = joinpath(@__DIR__, "meshes")
    for (i, case) in enumerate(cases)
        (res, jac, U, V), ts, cs = timed() do
            case == "em_p1" ? Base.invokelatest(em_case, GmshDiscreteModel(joinpath(meshdir, "em_plate_n2.msh")), 1) :
                              Base.invokelatest(em_box, 6, 2, 2, 2)
        end
        x = 0.01 .* randn(Base.invokelatest(num_free_dofs, U))
        op, tb, cb = timed() do
            backend == "kernel" ? Base.invokelatest(KernelOperator, res, jac, U, V) :
                                  Base.invokelatest(get_algebraic_operator, FEOperator(res, jac, U, V))
        end
        b, A = Base.invokelatest() do
            allocate_residual(op, x), allocate_jacobian(op, x)
        end
        _, tr, cr = timed(() -> Base.invokelatest(residual!, b, op, x))
        _, tj, cj = timed(() -> Base.invokelatest(jacobian!, A, op, x))
        _, tw, cw = timed(() -> (Base.invokelatest(residual!, b, op, x); Base.invokelatest(jacobian!, A, op, x)))
        tag = string(case, i == 1 ? "" : "_after")
        println(join(Any["ROW", backend, tag, ts, cs, tb, cb, tr, cr, tj, cj, tw, cw, length(x)], ","))
        flush(stdout)
    end
    exit(0)
end

# ---------------- driver ----------------
println("Julia ", VERSION, ", ", Threads.nthreads(), " threads, ", Sys.cpu_info()[1].model)
cmdbase = `$(Base.julia_cmd()) -t $(Threads.nthreads()) --project=$(Base.active_project()) $(@__FILE__) child`
runs = [(b, c) for c in ["em_p1", "em_q2"] for b in ["gridap", "kernel"]]
append!(runs, [(b, "em_p1 em_q2") for b in ["gridap", "kernel"]])
rows = String[]
for (b, c) in runs
    @printf("\n-- fresh process: %s, %s\n", b, c); flush(stdout)
    out = read(`$cmdbase $b $(split(c))`, String)
    for l in split(out, '\n')
        startswith(l, "ROW,") ? push!(rows, l) : (isempty(l) || println("   ", l))
    end
end
f(x) = @sprintf("%6.1f", parse(Float64, x))
println("\nTimes in seconds: wall (compile part). warm = second residual!+jacobian!")
for r in rows
    p = split(r, ",")
    if p[3] == "using"
        @printf("%-7s using packages                %s (%s)\n", p[2], f(p[4]), f(p[5]))
    else
        @printf("%-7s %-12s DOFs %6s | spaces %s (%s) | operator %s (%s) | 1st res %s (%s) | 1st jac %s (%s) | warm %s\n",
                p[2], p[3], p[14], f(p[4]), f(p[5]), f(p[6]), f(p[7]), f(p[8]), f(p[9]), f(p[10]), f(p[11]), @sprintf("%.3f", parse(Float64, p[12])))
    end
end

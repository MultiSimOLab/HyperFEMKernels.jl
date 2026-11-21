# Measuring process of the performance gate (benchmark/gate/README.md), started by driver.jl:
#   julia -t N --project=<env of main or of the branch> worker.jl --groups a,b [--only id,id] [--inject f]
# Builds the groups of cases.jl (only the cases of --only, when given), reports cold times, warms every case up,
# then answers on stdin/stdout (protocol lines start with "@@ "; anything else on stdout is output of the cases):
#   sample <id>  -> "S <id> <t_gridap> <bytes_gridap> <t_kernel> <bytes_kernel>": minimum seconds and bytes per call
#                   over one batch of each backend (≥ 0.25 s and ≥ 3 calls, fewer for calls of seconds; NaN/-1 without a Gridap
#                   counterpart); the order of the two backends alternates between samples
#   calib        -> "C <seconds>" of a fixed computation (drift of the machine between rounds)
#   value <id>   -> "V <id> <value>"
#   quit
# --inject f (acceptance test of the gate only): every kernel call is followed by a busy wait of f times its warm
# duration (a regression of f, e.g. 0.10).
using Printf
const OPTS = Dict(ARGS[i] => ARGS[i+1] for i in 1:2:length(ARGS)-1)
const GROUPS_WANTED = split(get(OPTS, "--groups", ""), ',', keepempty=false)
const ONLY = Set(split(get(OPTS, "--only", ""), ',', keepempty=false))
const INJECT = parse(Float64, get(OPTS, "--inject", "0"))

# No watchdog thread here: a thread polling the parent slowed the construction of the operators ~20× (2026-11-21,
# aorta P2–P1: 876 s instead of 47 s). The driver kills its processes when it stops or when gate.sh disappears.
say(s) = (print(stdout, "\n@@ ", s, "\n"); flush(stdout))
progress(s) = (println(stderr, Libc.strftime("%H:%M:%S", time()), " ", s); flush(stderr))     # in logs/*.err
clean(s) = replace(string(s), r"[\s,;=]+" => "_")

t0 = time()
include(joinpath(@__DIR__, "cases.jl"))
say(@sprintf("COLD load - %.3f", time() - t0))
say("VERS julia=$(VERSION);HyperFEM=$(pkgversion(HyperFEM));Gridap=$(pkgversion(Gridap));HyperFEMKernels=$(pkgversion(HyperFEMKernels))")
for name in GROUPS_WANTED
    name in GROUP_NAMES || (say("ERROR $name unknown_group"); continue)
end
for (name, build) in GROUPS
    isempty(GROUPS_WANTED) || name in GROUPS_WANTED || continue
    n0 = length(ORDER)
    progress("building group $name")
    t = @elapsed try
        Base.invokelatest(build)
    catch err
        err isa InterruptException && rethrow()
        say("ERROR $name " * clean(first(sprint(showerror, err), 300)))
        foreach(id -> delete!(CASES, id), ORDER[n0+1:end])
        resize!(ORDER, n0)
        continue
    end
    if !isempty(ONLY)                  # stage 2: only the suspect cases are warmed up and measured
        foreach(id -> id in ONLY || delete!(CASES, id), ORDER[n0+1:end])
        filter!(id -> haskey(CASES, id), ORDER)
    end
    say(@sprintf("COLD %s build %.3f", name, t))
    for id in ORDER[n0+1:end]          # first calls (compilation of the call paths), one sample per process
        c = CASES[id]
        progress("first calls of $id")
        try
            c.gridap === nothing || say(@sprintf("COLD %s gridap %.3f", id, @elapsed Base.invokelatest(c.gridap)))
            say(@sprintf("COLD %s kernel %.3f", id, @elapsed Base.invokelatest(c.kernel)))
        catch err
            err isa InterruptException && rethrow()
            say("ERROR $id " * clean(first(sprint(showerror, err), 300)))
            delete!(CASES, id)
        end
    end
    filter!(id -> haskey(CASES, id), ORDER)
end
for id in ONLY
    haskey(CASES, id) || say("ERROR $id unknown_case")
end

# warm-up and batch sizes per backend: a batch lasts at least 0.25 s and has at least 3 calls (2 from 1 s per call,
# 1 from 5 s: whole simulations, whose time is dominated by the call itself)
batch_size(t) = clamp(ceil(Int, 0.25 / max(t, 1e-7)), t < 1 ? 3 : t < 5 ? 2 : 1, 10_000)
const BATCHK = Dict{String,Int}()
const BATCHG = Dict{String,Int}()
const TEST = Dict{String,Float64}()
for id in copy(ORDER)
    c = CASES[id]
    progress("warm-up of $id")
    try
        tk = minimum(@elapsed(c.kernel()) for _ in 1:(INJECT > 0 ? 5 : 2))
        tg = c.gridap === nothing ? 0.0 : minimum(@elapsed(c.gridap()) for _ in 1:2)
        TEST[id] = tk
        BATCHK[id] = batch_size(tk)
        BATCHG[id] = batch_size(tg)
    catch err
        err isa InterruptException && rethrow()
        say("ERROR $id " * clean(first(sprint(showerror, err), 300)))
        delete!(CASES, id); filter!(!=(id), ORDER)
        continue
    end
    inf = join(("$k=$(clean(v))" for (k, v) in sort(collect(c.info); by=first)), ";")
    say("INFO $id $(c.kind) $(c.exact_bytes ? 1 : 0) $(c.gridap === nothing ? 0 : 1) $(isempty(inf) ? "-" : inf)")
end

function spin(seconds)
    t = time_ns() + round(UInt64, 1e9 * seconds)
    while time_ns() < t
    end
end
"Minimum time and bytes per call over a batch of `k` calls (the minimum is steadier than the mean)."
function batch(f, k, inject)
    GC.gc(false)
    tmin, bmin = Inf, typemax(Int)
    for _ in 1:k
        s = @timed (f(); inject > 0 && spin(inject))
        tmin = min(tmin, s.time); bmin = min(bmin, s.bytes)
    end
    tmin, bmin
end
function calib1()
    x = 0.0
    t = @elapsed for i in 1:10_000_000
        x += sqrt(Float64(i))
    end
    x > 0 ? t : -t
end
calib() = minimum(calib1() for _ in 1:3)       # ~30 ms; the minimum of 3 is steadier than one run
calib()
say(@sprintf("READY %d %d %d %d", Threads.nthreads(), LinearAlgebra.BLAS.get_num_threads(), round(Int, Sys.maxrss() / 2^20), length(ORDER)))

const FLIP = Dict{String,Bool}()
for line in eachline(stdin)
    w = split(line)
    isempty(w) && continue
    if w[1] == "quit"
        break
    elseif w[1] == "calib"
        say(@sprintf("C %.6g", calib()))
    elseif w[1] == "value"
        c = CASES[w[2]]
        v = try
            Float64(c.value())
        catch
            NaN
        end
        say(@sprintf("V %s %.15g", w[2], v))
    elseif w[1] == "sample"
        id = w[2]; c = CASES[id]
        inj = INJECT * TEST[id]              # busy wait after each kernel call (acceptance test only)
        tg, bg, tk, bk = NaN, -1, NaN, -1
        kernel_first = get!(FLIP, id, false)
        FLIP[id] = !kernel_first
        if c.gridap === nothing
            tk, bk = batch(c.kernel, BATCHK[id], inj)
        elseif kernel_first
            tk, bk = batch(c.kernel, BATCHK[id], inj); tg, bg = batch(c.gridap, BATCHG[id], 0.0)
        else
            tg, bg = batch(c.gridap, BATCHG[id], 0.0); tk, bk = batch(c.kernel, BATCHK[id], inj)
        end
        say(@sprintf("S %s %.6g %d %.6g %d", id, tg, bg, tk, bk))
    end
end
say(@sprintf("BYE %d", round(Int, Sys.maxrss() / 2^20)))

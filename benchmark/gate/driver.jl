# Driver of the performance gate (benchmark/gate/README.md); started by gate.sh, plain Julia (no packages).
#
#   julia driver.jl --main-env E1 --branch-env E2 --out DIR [--threads 1,4] [--groups a,b | --cases id,id] [--quick]
#                   [--rounds 7] [--pairs 2] [--inject f] [--force] [--exceptions FILE] [--meta "k=v;k=v"]
#
# Stage 1: for each thread count, one pair of measuring processes (worker.jl: one of origin/main, then one of the
# branch; their warm-ups never overlap), all cases, `rounds` rounds; in each round every case is sampled by both
# processes, alternately (ABBA), each process timing Gridap and the kernel of the case alternately.
# Stage 2: the cases suspect after stage 1 (time against main, or parity with Gridap) are measured by `pairs` new
# pairs of processes; the verdict is taken on the interval over all the pairs (stats.jl).
# Output in DIR: samples.csv, report.md, summary.toml; exit code 0 PASS, 1 FAIL, 3 INCONCLUSIVE, 2 could not run.
using Printf, TOML
include(joinpath(@__DIR__, "stats.jl"))
include(joinpath(@__DIR__, "groups.jl"))

const OPT = let o = Dict{String,String}(), i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("--quick", "--force")
            o[a] = "1"
        else
            o[a] = ARGS[i+1]; i += 1
        end
        i += 1
    end
    o
end
const OUT = OPT["--out"]
const ENVS = Dict("main" => OPT["--main-env"], "branch" => OPT["--branch-env"])
const THREADS = parse.(Int, split(get(OPT, "--threads", "1,4"), ','))
const ROUNDS = parse(Int, get(OPT, "--rounds", "7"))
const PAIRS = parse(Int, get(OPT, "--pairs", "2"))
const INJECT = get(OPT, "--inject", "0")
const FORCE = haskey(OPT, "--force")
const QUICK = haskey(OPT, "--quick")
const ONLY = split(get(OPT, "--cases", ""), ',', keepempty=false)
const WANT_GROUPS = !isempty(ONLY) ? unique(filter(!isempty, group_of.(ONLY))) :
                    haskey(OPT, "--groups") ? split(OPT["--groups"], ',') : QUICK ? QUICK_GROUPS : GROUP_NAMES
const JULIA = Base.julia_cmd()
bad_groups = setdiff(WANT_GROUPS, GROUP_NAMES)
bad_ids = filter(id -> group_of(id) == "", ONLY)
if !isempty(bad_groups) || !isempty(bad_ids)
    println(stderr, "driver.jl: unknown ", isempty(bad_groups) ? "" : "groups $(join(bad_groups, ", ")) ",
            isempty(bad_ids) ? "" : "cases $(join(bad_ids, ", "))", " (groups: $(join(GROUP_NAMES, ", ")))")
    exit(2)
end
mkpath(joinpath(OUT, "logs"))
logmsg(s) = (println(stderr, Libc.strftime("%H:%M:%S", time()), " ", s); flush(stderr))

# --- exceptions (benchmark/gate/exceptions.toml): accepted limits, decided by Jesús ------------------------------
const EXC = let f = get(OPT, "--exceptions", joinpath(@__DIR__, "exceptions.toml"))
    isfile(f) ? get(TOML.parsefile(f), "exception", Any[]) : Any[]
end
function exception(id, nt, type)
    for e in EXC
        e["case"] == id && e["type"] == type && (get(e, "threads", 0) == 0 || e["threads"] == nt) && return e
    end
    nothing
end
parity_limit(id, nt) = (e = exception(id, nt, "slower_than_gridap"); e === nothing ? 1.02 : Float64(e["limit"]))
regress_limit(id, nt, t) = (e = exception(id, nt, "regression_vs_main"); e === nothing ? time_limit(t) : Float64(e["limit"]))

# --- other processes: CPU of everything except the gate (ps), to wait for a quiet machine -------------------------
const OURS = Set{Int}([getpid()])
function other_cpu()
    total = 0.0
    for l in eachline(`ps -A -o pid=,%cpu=`)
        f = split(l)
        length(f) == 2 || continue
        pid = tryparse(Int, f[1]); c = tryparse(Float64, f[2])
        (pid === nothing || c === nothing || pid in OURS) && continue
        total += c
    end
    total
end
"Wait until the other processes use less than one core (3 polls in a row); true if quiet."
function wait_quiet(maxwait)
    t0 = time(); calm = 0
    while time() - t0 < maxwait
        c = other_cpu()
        calm = c < 100 ? calm + 1 : 0
        calm >= 3 && return true
        sleep(5)
    end
    false
end

# --- measuring processes ----------------------------------------------------------------------------------------
mutable struct Worker
    label::String
    proc::Base.Process
    pid::Int
    info::Dict{String,Any}           # id => (kind, exact bytes, has Gridap, info)
    vers::String
    cold::Vector{Tuple{String,String,Float64}}
    errors::Vector{String}
    blas::Int
    maxrss::Int
end
"Next protocol line of a measuring process (\"@@ \" marks it; other output of the cases is skipped), or nothing at EOF."
function protocol_line(p)
    while !eof(p)
        l = readline(p)
        k = findfirst("@@ ", l)
        k === nothing || return l[last(k)+1:end]
    end
    nothing
end
function start_worker(label, nt, groups, tag, only=String[])
    args = ["--groups", join(groups, ',')]
    isempty(only) || append!(args, ["--only", join(only, ',')])
    label == "branch" && INJECT != "0" && append!(args, ["--inject", INJECT])
    cmd = `$JULIA --startup-file=no -t $nt --project=$(ENVS[label]) $(joinpath(@__DIR__, "worker.jl")) $args`
    p = open(pipeline(cmd; stderr=joinpath(OUT, "logs", "$(tag)_$(label).err")), "r+")
    pid = getpid(p); push!(OURS, pid)
    w = Worker(label, p, pid, Dict{String,Any}(), "", Tuple{String,String,Float64}[], String[], -1, -1)
    t0 = time()
    while true
        l = protocol_line(p)
        l === nothing && error("the $label process (threads $nt) died while building the cases: see $(OUT)/logs/$(tag)_$(label).err")
        f = split(l)
        isempty(f) && continue
        if f[1] == "COLD"
            push!(w.cold, (f[2], f[3], parse(Float64, f[4])))
        elseif f[1] == "INFO"
            w.info[f[2]] = (kind=Symbol(f[3]), exact=f[4] == "1", hasg=f[5] == "1", info=f[6])
        elseif f[1] == "VERS"
            w.vers = f[2]
        elseif f[1] == "ERROR"
            push!(w.errors, join(f[2:end], " ")); logmsg("$label: ERROR $(join(f[2:end], " "))")
        elseif f[1] == "READY"
            w.blas = parse(Int, f[3]); w.maxrss = parse(Int, f[4])
            logmsg(@sprintf("%s (%d threads) ready: %d cases, %.0f s, %d MiB", label, nt, parse(Int, f[5]), time() - t0, w.maxrss))
            return w
        end
    end
end
function ask(w::Worker, cmd, prefix)
    println(w.proc, cmd); flush(w.proc)
    while true
        l = protocol_line(w.proc)
        l === nothing && error("the $(w.label) process died (command: $cmd)")
        startswith(l, prefix) && return split(l)
    end
end
function stop!(w::Worker)
    try
        println(w.proc, "quit"); flush(w.proc)
        l = protocol_line(w.proc); l !== nothing && startswith(l, "BYE") && (w.maxrss = parse(Int, split(l)[2]))
    catch
    end
    try; wait(w.proc); catch; end
    delete!(OURS, w.pid)
end

# --- samples ----------------------------------------------------------------------------------------------------
const SAMPLES = NamedTuple[]
const VALUES = Dict{Tuple{Int,Int,String,String},Float64}()          # (threads, pair, label, id) => value
const WORKERS = Dict{Tuple{Int,Int,String},Worker}()                  # (threads, pair, label)
const ROUNDLOG = NamedTuple[]
const CSV = open(joinpath(OUT, "samples.csv"), "w")
println(CSV, "stage,pair,threads,round,label,id,t_gridap,bytes_gridap,t_kernel,bytes_kernel")

"One pair of processes measuring `ids` (all their cases when `ids` is empty)."
function measure_pair(stage, pair, nt, groups, ids)
    tag = "t$(nt)_p$(pair)"
    if !FORCE && !wait_quiet(600)
        error("the machine never got quiet (other processes ≥ 1 core for 10 min); use --force to measure anyway")
    end
    wm = start_worker("main", nt, groups, tag, ids)
    wb = start_worker("branch", nt, groups, tag, ids)
    WORKERS[(nt, pair, "main")] = wm; WORKERS[(nt, pair, "branch")] = wb
    for id in symdiff(collect(keys(wm.info)), collect(keys(wb.info)))      # built in one process only: an error, not a skip
        push!(wb.errors, "$id built by $(haskey(wm.info, id) ? "main" : "the branch") only")
    end
    sel = isempty(ids) ? [id for id in keys(wb.info) if haskey(wm.info, id)] : [id for id in ids if haskey(wb.info, id) && haskey(wm.info, id)]
    sort!(sel)
    try
        for r in 1:ROUNDS
            for attempt in 1:4
                load = other_cpu()
                if load >= 100 && !FORCE
                    wait_quiet(120); load = other_cpu()
                end
                c0 = Dict(w.label => parse(Float64, ask(w, "calib", "C ")[2]) for w in (wm, wb))
                order = isodd(r) ? (wm, wb) : (wb, wm)
                rows = NamedTuple[]
                for id in sel, w in order
                    f = ask(w, "sample $id", "S $id ")
                    push!(rows, (stage=stage, pair=pair, threads=nt, round=r, label=w.label, id=id, tg=parse(Float64, f[3]),
                                 bg=parse(Int, f[4]), tk=parse(Float64, f[5]), bk=parse(Int, f[6])))
                end
                c1 = Dict(w.label => parse(Float64, ask(w, "calib", "C ")[2]) for w in (wm, wb))
                drift = maximum(abs(c1[k] / c0[k] - 1) for k in keys(c0))
                ok = drift <= 0.10 && (FORCE || load < 100)
                if ok || attempt == 4
                    append!(SAMPLES, rows)
                    for x in rows
                        println(CSV, join((x.stage, x.pair, x.threads, x.round, x.label, x.id, x.tg, x.bg, x.tk, x.bk), ","))
                    end
                    flush(CSV)
                    push!(ROUNDLOG, (threads=nt, pair=pair, round=r, attempts=attempt, drift=drift, load=load, accepted_noisy=!ok))
                    logmsg(@sprintf("threads %d pair %d round %d/%d: %d cases, calibration drift %.1f %%, other CPU %.0f %%%s", nt, pair, r,
                                    ROUNDS, length(sel), 100drift, load, ok ? "" : " (kept after 4 attempts)"))
                    break
                end
                logmsg(@sprintf("threads %d pair %d round %d: discarded (calibration drift %.1f %%, other CPU %.0f %%), again", nt, pair, r, 100drift, load))
            end
        end
        for id in sel, w in (wm, wb)
            VALUES[(nt, pair, w.label, id)] = parse(Float64, ask(w, "value $id", "V $id ")[3])
        end
    finally
        stop!(wm); stop!(wb)
    end
    sel
end

# --- analysis ---------------------------------------------------------------------------------------------------
"Per-pair series of a case: (main kernel, branch kernel, main gridap, branch gridap) times over matched rounds."
function series(nt, id, pair)
    m = Dict(x.round => x for x in SAMPLES if x.threads == nt && x.pair == pair && x.id == id && x.label == "main")
    b = Dict(x.round => x for x in SAMPLES if x.threads == nt && x.pair == pair && x.id == id && x.label == "branch")
    rs = sort([r for r in keys(m) if haskey(b, r)])
    ([m[r].tk for r in rs], [b[r].tk for r in rs], [m[r].tg for r in rs], [b[r].tg for r in rs])
end
pairs_of(nt, id) = sort(unique(x.pair for x in SAMPLES if x.threads == nt && x.id == id))

function analyse(nt, id)
    info = WORKERS[(nt, 1, "branch")].info[id]
    kind, hasg, exact = info.kind, info.hasg, info.exact
    ps = pairs_of(nt, id)
    tm = Float64[]; tb = Float64[]; gm = Float64[]; gb = Float64[]
    rat = Float64[]; ctl = Float64[]; parb = Float64[]; parm = Float64[]
    for p in ps
        a, b, c, d = series(nt, id, p)
        isempty(a) && continue
        append!(tm, a); append!(tb, b); append!(gm, c); append!(gb, d)
        push!(rat, pair_ratio(a, b))
        if hasg
            push!(ctl, pair_ratio(c, d)); push!(parb, median(b ./ d)); push!(parm, median(a ./ c))
        end
    end
    bk_m = minimum(x.bk for x in SAMPLES if x.threads == nt && x.id == id && x.label == "main")
    bk_b = minimum(x.bk for x in SAMPLES if x.threads == nt && x.id == id && x.label == "branch")
    bg_m = hasg ? minimum(x.bg for x in SAMPLES if x.threads == nt && x.id == id && x.label == "main") : -1
    t_main = median(tm)
    est, lo, hi = interval(rat)
    lim = regress_limit(id, nt, t_main)
    abs_small = kind === :simulation && median(tb) - median(tm) <= 1e-3
    tv = length(rat) == 1 ? (rat[1] <= suspect_limit(t_main) || abs_small ? :pass : :inconclusive) :
         abs_small ? :pass : verdict(lo, hi, lim)
    plim = parity_limit(id, nt)
    pv = :none; pest = plo = phi = NaN
    if hasg
        pest, plo, phi = interval(parb)
        pv = length(parb) == 1 ? (parb[1] <= plim ? :pass : :inconclusive) : verdict(plo, phi, plim)
    end
    # memory: a FAIL for deterministic bytes (assemblies, products); a warning otherwise (builds, pattern builds and
    # simulations allocate amounts that change between processes running the same code, up to 25 % in the A/A runs)
    mem_over = bk_b > bytes_limit(kind, exact, bk_m)
    mv = mem_over && exact ? :fail : :pass
    vm = get(VALUES, (nt, 1, "main", id), NaN); vb = get(VALUES, (nt, 1, "branch", id), NaN)
    dv = !isfinite(vm) ? NaN : !isfinite(vb) ? Inf : abs(vb - vm) / max(abs(vm), 1e-300)   # NaN: not compared
    vv = isnan(dv) || dv <= value_tol(kind) ? :pass : :fail
    overall = any(==(:fail), (tv, pv, mv, vv)) ? :fail : any(==(:inconclusive), (tv, pv)) ? :inconclusive : :pass
    (id=id, threads=nt, kind=kind, hasg=hasg, npairs=length(rat), t_main=t_main, t_branch=median(tb), g_main=hasg ? median(gm) : NaN,
     g_branch=hasg ? median(gb) : NaN, est=est, lo=lo, hi=hi, limit=lim, tv=tv, ctl=isempty(ctl) ? NaN : gmean(ctl),
     par_main=isempty(parm) ? NaN : gmean(parm), pest=pest, plo=plo, phi=phi, plimit=plim, pv=pv, bk_m=bk_m, bk_b=bk_b, bg_m=bg_m, mv=mv,
     dv=dv, vv=vv, overall=overall, mem_warn=mem_over && !exact)
end
"Suspect after stage 1: worth the extra pairs of stage 2."
function suspect(nt, id)
    a = analyse(nt, id)
    a.tv === :inconclusive || a.pv === :inconclusive
end

# --- report -----------------------------------------------------------------------------------------------------
fmt_t(t) = isnan(t) ? "—" : t >= 1 ? @sprintf("%.3f s", t) : t >= 1e-3 ? @sprintf("%.2f ms", 1e3t) : @sprintf("%.1f µs", 1e6t)
fmt_b(b) = b < 0 ? "—" : b < 2^10 ? "$b B" : b < 2^20 ? @sprintf("%.1f KiB", b / 2^10) : @sprintf("%.1f MiB", b / 2^20)
fmt_r(e, lo, hi, n) = n <= 1 ? @sprintf("%.3f (1 pair)", e) : @sprintf("%.3f [%.3f–%.3f] (%d pairs)", e, lo, hi, n)
const WORD = Dict(:pass => "ok", :fail => "FAIL", :inconclusive => "INCONCLUSIVE", :none => "")
function why(a)
    w = String[]
    a.tv === :fail && push!(w, @sprintf("%+.1f %% vs main", 100(a.est - 1)))
    a.tv === :inconclusive && push!(w, "time vs main undecided")
    a.pv === :fail && push!(w, @sprintf("slower than Gridap ×%.3f (limit %.2f)", a.pest, a.plimit))
    a.pv === :inconclusive && push!(w, "parity with Gridap undecided")
    a.mv === :fail && push!(w, "memory $(fmt_b(a.bk_m)) → $(fmt_b(a.bk_b))")
    a.vv === :fail && push!(w, @sprintf("results differ (%.1e)", a.dv))
    a.mem_warn && push!(w, "memory $(fmt_b(a.bk_m)) → $(fmt_b(a.bk_b)) (warning: not deterministic)")
    isempty(w) || a.overall === :pass || return join(w, "; ")
    s = a.est < 0.97 ? @sprintf("ok (%+.0f %%)", 100(a.est - 1)) :
        a.est > 1.03 ? @sprintf("ok (%+.0f %%, under the %.0f %% threshold)", 100(a.est - 1), 100(a.limit - 1)) : "ok"
    isempty(w) ? s : s * "; " * join(w, "; ")
end
function report(results, errors, meta, minutes)
    nf = count(a -> a.overall === :fail, results) + length(errors)
    ni = count(a -> a.overall === :inconclusive, results)
    verdict_all = nf > 0 ? "FAIL" : ni > 0 ? "INCONCLUSIVE" : "PASS"
    io = IOBuffer()
    println(io, "## Performance gate: $verdict_all")
    println(io)
    println(io, "$(get(meta, "branch", "branch")) vs $(get(meta, "main", "origin/main")); $(get(meta, "machine", "")); Julia $(get(meta, "julia", VERSION)); ",
            "threads $(join(THREADS, ", ")); $(ROUNDS) rounds per pair, stage 2 with $(PAIRS) more pairs; $(round(Int, minutes)) min",
            INJECT != "0" ? "; INJECTED REGRESSION $(INJECT) (acceptance test)" : "")
    println(io)
    println(io, "Failures: $nf, inconclusive: $ni, cases: $(length(results)). Times per call (median of the samples); branch/main: ",
            "ratio of the compiled backend with its 95 % interval over the pairs of processes; comp/Gridap: compiled over Gridap in ",
            "the same process.", FORCE ? " MEASURED WITH --force (the machine was not required to be quiet)." : "")
    println(io)
    if !isempty(errors)
        println(io, "### Errors"); println(io)
        foreach(e -> println(io, "- ", e), errors); println(io)
    end
    bad = filter(a -> a.overall !== :pass, results)
    if !isempty(bad)
        println(io, "### Failures and inconclusive cases"); println(io)
        for a in bad
            println(io, "- $(a.id), $(a.threads) threads: $(WORD[a.overall]) — $(why(a))")
        end
        inc = filter(a -> a.overall === :inconclusive, results)
        if !isempty(inc)
            println(io)
            for nt in unique(a.threads for a in inc)
                ids = join(unique(a.id for a in inc if a.threads == nt), ",")
                println(io, "Repeat: `benchmark/gate/gate.sh --cases $ids --threads $nt --pairs 6`")
            end
        end
        println(io)
    end
    println(io, "### All cases"); println(io)
    println(io, "| case | thr | Gridap main | Gridap branch | compiled main | compiled branch | branch/main | comp/Gridap main | comp/Gridap branch | bytes main | bytes branch | verdict |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in results
        pm = a.hasg ? @sprintf("%.3f", a.par_main) : "—"
        pb = a.hasg ? (a.pv === :none ? "—" : fmt_r(a.pest, a.plo, a.phi, a.npairs)) : "—"
        println(io, "| $(a.id) | $(a.threads) | $(fmt_t(a.g_main)) | $(fmt_t(a.g_branch)) | $(fmt_t(a.t_main)) | $(fmt_t(a.t_branch)) | ",
                "$(fmt_r(a.est, a.lo, a.hi, a.npairs)) | $pm | $pb | $(fmt_b(a.bk_m)) | $(fmt_b(a.bk_b)) | $(a.overall === :pass ? why(a) : "**" * why(a) * "**") |")
    end
    println(io)
    if !isempty(EXC)
        println(io, "### Exceptions in force (benchmark/gate/exceptions.toml)"); println(io)
        for e in EXC
            println(io, "- $(e["case"]) ($(get(e, "threads", 0) == 0 ? "all thread counts" : "$(e["threads"]) threads")), $(e["type"]), limit $(e["limit"]): ",
                    "$(get(e, "reason", "")) [$(get(e, "measured", "")); $(get(e, "date", "")); $(get(e, "decision", ""))]")
        end
        println(io)
    end
    println(io, "### Cold latency (first pair of each thread count; one sample per process, informative)"); println(io)
    println(io, "| what | thr | main | branch | ratio |"); println(io, "|---|---|---|---|---|")
    for nt in THREADS
        haskey(WORKERS, (nt, 1, "main")) || continue
        cm = Dict((k, b) => t for (k, b, t) in WORKERS[(nt, 1, "main")].cold)
        for (k, b, t) in WORKERS[(nt, 1, "branch")].cold
            haskey(cm, (k, b)) || continue
            r = t / cm[(k, b)]
            flag = r > 1.25 && t - cm[(k, b)] > 1.0 ? " **slower**" : ""
            (b == "build" || b == "-" || flag != "") && println(io, "| $k $b | $nt | $(fmt_t(cm[(k, b)])) | $(fmt_t(t)) | $(@sprintf("%.2f", r))$flag |")
        end
    end
    println(io)
    println(io, "### Invariants (deterministic facts of the kernel operators: nnz, bytes, paths of the groups)"); println(io)
    nd = 0
    for nt in THREADS[1:1]
        haskey(WORKERS, (nt, 1, "main")) || continue
        im, ib = WORKERS[(nt, 1, "main")].info, WORKERS[(nt, 1, "branch")].info
        for id in sort(collect(keys(ib)))
            haskey(im, id) || continue
            ib[id].info == "-" && continue
            if im[id].info != ib[id].info
                println(io, "- $id: main `$(im[id].info)` → branch `$(ib[id].info)`"); nd += 1
            end
        end
    end
    nd == 0 && println(io, "No change.")
    println(io)
    println(io, "### Noise control"); println(io)
    println(io, "Gridap branch/main (the same code in both processes: the bias of the pairs): ",
            join((@sprintf("%s %d thr %.3f", a.id, a.threads, a.ctl) for a in results if a.hasg && (a.ctl > 1.03 || a.ctl < 0.97)), "; ") |> s -> isempty(s) ? "all within ±3 %." : s)
    nkept = count(r -> r.accepted_noisy, ROUNDLOG); nrep = sum(r -> r.attempts - 1, ROUNDLOG; init=0)
    println(io, "Rounds repeated for calibration drift > 10 % or other CPU ≥ 1 core: $nrep; kept noisy after 4 attempts: $nkept.")
    for ((nt, p, l), w) in sort(collect(WORKERS); by=first)
        println(io, "- threads $nt pair $p $l: BLAS threads $(w.blas), max RSS $(w.maxrss) MiB; $(w.vers)")
    end
    s = String(take!(io))
    write(joinpath(OUT, "report.md"), s)
    open(joinpath(OUT, "summary.toml"), "w") do f
        TOML.print(f, Dict("verdict" => verdict_all, "failures" => nf, "inconclusive" => ni, "cases" => length(results)))
    end
    print(s)
    verdict_all
end

# exit when gate.sh disappears (the processes of the driver are stopped with it)
const GATE = ccall(:getppid, Cint, ())
@async while true
    sleep(5)
    if ccall(:getppid, Cint, ()) != GATE
        for w in values(WORKERS); try; kill(w.proc); catch; end; end
        exit(2)
    end
end

function collect_errors!(errors, nt, pair)
    for l in ("main", "branch")
        haskey(WORKERS, (nt, pair, l)) || continue
        foreach(e -> push!(errors, "$l, $nt threads, pair $pair: $e"), WORKERS[(nt, pair, l)].errors)
    end
end

function main()
    t0 = time()
    meta = Dict(String(first(split(kv, '='))) => String(last(split(kv, '='; limit=2))) for kv in split(get(OPT, "--meta", ""), ';', keepempty=false))
    errors = String[]
    if !FORCE && !wait_quiet(600)
        println(stderr, "driver.jl: the machine never got quiet (other processes ≥ 1 core for 10 min); use --force"); exit(2)
    end
    # the two environments must resolve the same packages (only HyperFEMKernels may differ)
    mans = Dict(l => TOML.parsefile(joinpath(ENVS[l], "Manifest.toml"))["deps"] for l in ("main", "branch"))
    vers(m) = Dict(k => get(first(v), "version", get(first(v), "git-tree-sha1", "")) for (k, v) in m if k != "HyperFEMKernels")
    vm, vb = vers(mans["main"]), vers(mans["branch"])
    diffs = [k for k in union(keys(vm), keys(vb)) if get(vm, k, "-") != get(vb, k, "-")]
    if !isempty(diffs)
        println(stderr, "driver.jl: the environments of main and of the branch resolve different packages: ",
                join(("$k $(get(vm, k, "-")) / $(get(vb, k, "-"))" for k in sort(diffs)), ", "))
        exit(2)
    end
    ids1 = Dict{Int,Vector{String}}()
    for nt in THREADS
        logmsg("stage 1, $nt threads, groups $(join(WANT_GROUPS, ","))")
        ids1[nt] = measure_pair(1, 1, nt, WANT_GROUPS, ONLY)
        collect_errors!(errors, nt, 1)
        vm, vb = WORKERS[(nt, 1, "main")].vers, WORKERS[(nt, 1, "branch")].vers
        strip_k(v) = replace(v, r";HyperFEMKernels=[^;]*" => "")
        strip_k(vm) == strip_k(vb) || push!(errors, "the two environments differ (main: $vm; branch: $vb): HyperFEM, Gridap or Julia must be the same")
        missing_ids = setdiff(ONLY, ids1[nt])
        isempty(missing_ids) || push!(errors, "$nt threads: cases not measured: $(join(missing_ids, ", "))")
    end
    for nt in THREADS
        # with --cases (repeating undecided cases) every selected case gets the extra pairs
        sus = isempty(ONLY) ? filter(id -> suspect(nt, id), ids1[nt]) : ids1[nt]
        (isempty(sus) || PAIRS == 0) && continue
        logmsg("stage 2, $nt threads: $(length(sus)) cases: $(join(sus, ","))")
        groups = unique(group_of.(sus))
        for p in 2:1+PAIRS
            measure_pair(2, p, nt, groups, sus)
            collect_errors!(errors, nt, p)
        end
    end
    results = [analyse(nt, id) for nt in THREADS for id in ids1[nt]]
    v = report(results, unique(errors), meta, (time() - t0) / 60)
    close(CSV)
    exit(v == "PASS" ? 0 : v == "FAIL" ? 1 : 3)
end
try
    main()
catch err
    err isa InterruptException && exit(2)
    for w in values(WORKERS); try; kill(w.proc); catch; end; end
    println(stderr, "driver.jl: could not run: ", sprint(showerror, err))
    exit(2)
end

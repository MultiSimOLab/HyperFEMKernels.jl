# Statistics of the performance gate (benchmark/gate/README.md). No package dependencies.
#
# Unit of evidence: a PAIR of processes (one of origin/main, one of the branch, measured alternately). Two processes
# with the same code differ in a stable way (1–4 % on simulations, more on calls under 5 ms: A/A experiment of
# 2026-11-21, _work/perf-gate2/design/), so repetitions inside one pair cannot tell a regression from the bias of
# that pair: the interval is taken over the estimates of several pairs.

gmean(v) = exp(sum(log, v) / length(v))
mean(v) = sum(v) / length(v)
function sd(v)
    n = length(v)
    n < 2 && return NaN
    m = mean(v)
    sqrt(sum(x -> (x - m)^2, v) / (n - 1))
end
const T975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179)
"Student t quantile 0.975 with `df` degrees of freedom."
t975(df) = df <= length(T975) ? T975[df] : 1.96 + 2.4 / df

median(v) = (s = sort(v); n = length(s); isodd(n) ? s[(n+1)÷2] : (s[n÷2] + s[n÷2+1]) / 2)
"""
    pair_ratio(a, b) -> Float64

Estimate of one pair: median over its rounds of the per-round ratios b/a (b and a measured in the same round). The
median, not a mean: a single sample twice as long (seen in the A/A data, inside one process) must not move it.
"""
pair_ratio(a, b) = median(b ./ a)

"""
    interval(r) -> (est, lo, hi)

95 % interval of the ratio from the estimates `r` of several pairs (t interval on the logarithms). With one pair the
interval is empty (lo = hi = est): stage 1 only flags suspects.
"""
function interval(r::AbstractVector)
    l = log.(r)
    m = mean(l)
    length(l) < 2 && return (exp(m), exp(m), exp(m))
    h = t975(length(l) - 1) * sd(l) / sqrt(length(l))
    (exp(m), exp(m - h), exp(m + h))
end

"""
    verdict(lo, hi, limit) -> :fail, :pass or :inconclusive

`:fail` when the whole interval is above `limit`, `:pass` when it is below, `:inconclusive` otherwise.
"""
verdict(lo, hi, limit) = lo > limit ? :fail : hi <= limit ? :pass : :inconclusive

"Time threshold of a regression against main for a call of `t` seconds: 5 % (25 % under 5 ms, where the noise is higher)."
time_limit(t) = t < 0.005 ? 1.25 : 1.05
"Threshold of stage 1 (suspects measured again in stage 2): 3 % (25 % under 5 ms)."
suspect_limit(t) = t < 0.005 ? 1.25 : 1.03
"""
Allowed bytes per call of the branch from those of main: +256 B for calls whose bytes are deterministic (assemblies,
products); 5 % + 64 KiB for other calls (builds, updates: their bytes vary by ~2.5 % between processes with the same
code); 10 % + 1 MiB for simulations (~5 % between processes in the A/A data).
"""
bytes_limit(kind, exact, b) = exact ? b + 256 : kind === :simulation ? 1.10b + 2^20 : 1.05b + 65536
"Relative tolerance of the results of a case between main and the branch."
value_tol(kind) = kind === :simulation ? 1e-7 : 1e-10

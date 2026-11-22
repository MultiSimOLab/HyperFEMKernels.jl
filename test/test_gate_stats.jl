# Statistics of the performance gate (benchmark/gate/stats.jl): estimates, intervals, verdicts and limits.
module GateStats
include(joinpath(@__DIR__, "..", "benchmark", "gate", "stats.jl"))
end
@testset "performance gate: statistics" begin
    S = GateStats
    @test S.pair_ratio([1.0, 1.0, 1.0, 1.0, 1.0], [1.1, 1.1, 2.2, 1.1, 1.1]) ≈ 1.1     # a sample ×2 does not move the median
    @test S.pair_ratio([2.0, 2.0], [1.0, 3.0]) ≈ 1.0
    e, lo, hi = S.interval([1.1])
    @test e ≈ 1.1 && lo == hi == e                       # one pair: no interval
    e, lo, hi = S.interval([1.08, 1.10, 1.12, 1.09])
    @test lo < e < hi && 1.05 < lo                       # consistent pairs: the whole interval above 5 %
    @test S.verdict(lo, hi, 1.05) === :fail
    e, lo, hi = S.interval([0.99, 1.01, 1.00, 1.02])
    @test S.verdict(lo, hi, 1.05) === :pass
    e, lo, hi = S.interval([0.95, 1.12, 1.00, 1.09])
    @test S.verdict(lo, hi, 1.05) === :inconclusive
    @test S.time_limit(0.001) == 1.25 && S.time_limit(0.1) == 1.05
    @test S.suspect_limit(0.001) == 1.25 && S.suspect_limit(0.1) == 1.03
    @test S.bytes_limit(:call, true, 1000) == 1256
    @test S.bytes_limit(:call, false, 2^20) ≈ 1.05 * 2^20 + 65536
    @test S.bytes_limit(:simulation, false, 2^30) ≈ 1.10 * 2^30 + 2^20
    @test S.value_tol(:simulation) == 1e-7 && S.value_tol(:call) == 1e-10
    @test S.t975(3) ≈ 3.182 && S.t975(100) < 2.0
end

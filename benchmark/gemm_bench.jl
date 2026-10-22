# Micro-benchmark of the element matrix-product kernel: C[om+m, on+n] += Σ_r A[m, r, k, c] B[r, n]
# (A = G buffer, 4D; B = test weights). Current `gemm_add!` vs register-tiled variants with
# vectorised remainders, and OpenBLAS on the same shapes.
#
#   julia -t 1 --project=benchmark benchmark/gemm_bench.jl
using HyperFEMKernels, LinearAlgebra, Printf, Random
const HK = HyperFEMKernels
const SVector = HK.StaticArrays.SVector
BLAS.set_num_threads(1)
Random.seed!(1)

# ---- candidate: generic MB×NB tiles, A addressed linearly -----------------------------------
# (acc is passed, not captured: a closure over a reassigned variable would box it)
@inline accum(a, acc::NTuple{NB,Any}, B, r, n0) where {NB} =
    ntuple(j -> muladd.(a, @inbounds(B[r, n0+j-1]), acc[j]), Val(NB))
@inline function tile!(C, om, on, A, offA, lda, B, m0, n0, R, ::Val{MB}, ::Val{NB}) where {MB,NB}
    acc = ntuple(_ -> zero(SVector{MB,Float64}), Val(NB))
    @inbounds for r in 1:R
        base = offA + m0 - 1 + (r - 1) * lda
        a = SVector{MB,Float64}(ntuple(i -> A[base+i], Val(MB)))
        acc = accum(a, acc, B, r, n0)
    end
    @inbounds for j in 1:NB, i in 1:MB
        C[om+m0+i-1, on+n0+j-1] += acc[j][i]
    end
    nothing
end

@inline function colblock!(C, om, on, A, offA, lda, B, n0, M, R, vmb::Val{MB}, vnb) where {MB}
    m0 = 1
    while m0 + MB - 1 <= M
        tile!(C, om, on, A, offA, lda, B, m0, n0, R, vmb, vnb); m0 += MB
    end
    rem = M - m0 + 1
    if MB > 8 && rem >= 8
        tile!(C, om, on, A, offA, lda, B, m0, n0, R, Val(8), vnb); m0 += 8; rem -= 8
    end
    if MB > 4 && rem >= 4
        tile!(C, om, on, A, offA, lda, B, m0, n0, R, Val(4), vnb); m0 += 4; rem -= 4
    end
    if MB > 2 && rem >= 2
        tile!(C, om, on, A, offA, lda, B, m0, n0, R, Val(2), vnb); m0 += 2; rem -= 2
    end
    rem >= 1 && tile!(C, om, on, A, offA, lda, B, m0, n0, R, Val(1), vnb)
    nothing
end

function gemm_tiled!(C, om, on, A, k, c, B, M, N, R, vmb::Val{MB}, ::Val{NB}) where {MB,NB}
    lda = size(A, 1)
    offA = ((c - 1) * size(A, 3) + (k - 1)) * size(A, 1) * size(A, 2)
    n0 = 1
    while n0 + NB - 1 <= N
        colblock!(C, om, on, A, offA, lda, B, n0, M, R, vmb, Val(NB)); n0 += NB
    end
    while n0 <= N
        colblock!(C, om, on, A, offA, lda, B, n0, M, R, vmb, Val(1)); n0 += 1
    end
    C
end

# ---- benchmark --------------------------------------------------------------------------------
function timeit(f, n)
    f()
    best = Inf
    for _ in 1:7
        t = @elapsed for _ in 1:n; f(); end
        best = min(best, t / n)
    end
    best
end

function run(name, M, N, R; nk=3, nc=3)
    A = rand(M, R, nk, nc); B = rand(R, N)
    ndl = max(M, N) * max(nk, nc) + 8
    Cref = zeros(ndl, ndl)
    for c in 1:nc, k in 1:nk
        Cref[(k-1)*M+1:k*M, (c-1)*N+1:c*N] .+= A[:, :, k, c] * B
    end
    fl = 2.0 * M * N * R * nk * nc
    reps = max(1, round(Int, 2e8 / fl))
    @printf("%-22s M=%3d N=%3d R=%3d  (%d blocks)\n", name, M, N, R, nk * nc)
    C = zeros(ndl, ndl)
    allblocks(g) = () -> (for c in 1:nc, k in 1:nk; g(C, (k - 1) * M, (c - 1) * N, A, k, c, B, M, N, R); end)
    variants = Any[("current gemm_add!", HK.gemm_add!)]
    for (mb, nb) in ((8, 4), (8, 6), (8, 3), (12, 4), (16, 2), (4, 8))
        vm, vn = Val(mb), Val(nb)
        push!(variants, ("tile $(mb)x$(nb)", (C, om, on, A, k, c, B, M, N, R) -> gemm_tiled!(C, om, on, A, k, c, B, M, N, R, vm, vn)))
    end
    for (vn, g) in variants
        fill!(C, 0.0); allblocks(g)()
        err = maximum(abs, C - Cref) / maximum(abs, Cref)
        t = timeit(allblocks(g), reps)
        @printf("    %-20s %6.1f GFLOP/s   err %.0e\n", vn, fl / t / 1e9, err)
    end
    Am = [A[:, :, k, c] for k in 1:nk, c in 1:nc]; Cb = zeros(M, N)
    t = timeit(() -> (for c in 1:nc, k in 1:nk; mul!(Cb, Am[k, c], B, 1.0, 1.0); end), reps)
    @printf("    %-20s %6.1f GFLOP/s\n", "OpenBLAS", fl / t / 1e9)
    flush(stdout)
end

run("Q2 hex (EM u-u)", 27, 27, 81)
run("Q3 hex (EM u-u)", 64, 64, 192)
run("Q1 hex", 8, 8, 24)
run("P2 tet", 10, 10, 12)
run("Q2 quad 2D", 9, 9, 18; nk=2, nc=2)
run("P1 tet", 4, 4, 12)

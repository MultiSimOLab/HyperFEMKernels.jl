# Absolute efficiency of assembly and matrix-free apply: machine limits (STREAM bandwidth, FMA
# peak) vs measured kernels (GFLOP/s of the element matrix products, GB/s of scatter, matrix-free
# apply and SpMV), plus OpenBLAS on the same small products as a "compiled library" reference.
#
#   julia -t auto --project=benchmark benchmark/roofline.jl
using HyperFEMKernels, HyperFEM, Gridap, GridapGmsh, Gridap.FESpaces, Gridap.Algebra
using LinearAlgebra, SparseArrays, Random, Printf
const HK = HyperFEMKernels
const SVector = HK.StaticArrays.SVector
Random.seed!(1)
include(joinpath(@__DIR__, "bench_cases.jl"))
BLAS.set_num_threads(1)

best(f; n=5) = (f(); minimum(@elapsed(f()) for _ in 1:n))

# ---------------------------------------------------------------- machine ----------------------
sysinfo(k) = try strip(read(`sysctl -n $k`, String)) catch; "?" end
println("CPU: ", sysinfo("machdep.cpu.brand_string"), " | P-cores ", sysinfo("hw.perflevel0.physicalcpu"),
        " | E-cores ", sysinfo("hw.perflevel1.physicalcpu"), " | Julia threads ", Threads.nthreads())

function triad!(a, b, c, s, r)
    @inbounds @simd for i in r
        a[i] = b[i] + s * c[i]
    end
end
function stream(nt)
    n = 2^25
    a = zeros(n); b = rand(n); c = rand(n)
    chunks = collect(Iterators.partition(1:n, cld(n, nt)))
    par() = @sync for r in chunks
        Threads.@spawn triad!(a, b, c, 3.0, r)
    end
    t = best(par; n=10)
    3 * 8 * n / t / 1e9          # GB/s (STREAM convention: 2 reads + 1 write)
end

function fmaloop(N)
    x = SVector{8,Float64}(ntuple(i -> 1.0 + 1e-12i, 8)); y = SVector{8,Float64}(ntuple(i -> 1e-9i, 8))
    a1 = y; a2 = 2y; a3 = 3y; a4 = 4y; a5 = 5y; a6 = 6y
    for _ in 1:N
        a1 = map(muladd, a1, x, y); a2 = map(muladd, a2, x, y); a3 = map(muladd, a3, x, y)
        a4 = map(muladd, a4, x, y); a5 = map(muladd, a5, x, y); a6 = map(muladd, a6, x, y)
    end
    sum(a1 + a2 + a3 + a4 + a5 + a6)
end
function peak(nt)
    N = 20_000_000
    par() = @sync for _ in 1:nt
        Threads.@spawn fmaloop(N)
    end
    t = best(par; n=3)
    nt * N * 6 * 8 * 2 / t / 1e9   # GFLOP/s
end

bw1, bwN = stream(1), stream(Threads.nthreads())
pk1, pkN = peak(1), peak(Threads.nthreads())
@printf("STREAM triad: 1 thr %.1f GB/s | %d thr %.1f GB/s\n", bw1, Threads.nthreads(), bwN)
@printf("FMA peak:     1 thr %.1f GFLOP/s | %d thr %.1f GFLOP/s\n\n", pk1, Threads.nthreads(), pkN)
flush(stdout)

# ---------------------------------------------------------------- kernels ----------------------
lvof(::HK.SlotSpec{NC,LV}) where {NC,LV} = LV
ncof(::HK.SlotSpec{NC}) where {NC} = NC

function flops_products(g)
    np = g.geo.np; f1 = 0; f2 = 0
    for (tf, sf) in zip(g.tables, g.smv), (tg, sg) in zip(g.tables, g.smu)
        LVf, LVg = lvof(sf), lvof(sg)
        (LVf == 0 || LVg == 0) && continue
        f1 += ncof(sf) * ncof(sg) * 2 * tg.nn * tf.nn * np * LVf    # block products G·B
        f2 += ncof(sf) * ncof(sg) * LVf * np * tg.nn * LVg * 2      # building G at the points
    end
    f1, f2
end

loop_elem(g, op, x, dvals, auxd, xl, KeT, Wb, Bb, Gb) = for cell in 1:length(g.cellmap)
    HK.gather_cell!(xl, op.fields, x, dvals, g.cellmap[cell])
    HK.element_jacobian_mm!(KeT, g, xl, auxd, cell, Wb, Bb, Gb)
end
loop_products(g, KeT, Bb, Gb) = for cell in 1:length(g.cellmap)   # one block of points per cell, scaled below
    fill!(KeT, 0.0)
    map((tf, lf, smf, B, Gf) -> map((tg, lg, smg, G) -> HK.block_products!(KeT, tf, lf, smf, B, tg, lg, smg, G, HK.point_block(g.geo.np)),
                                    g.tables, g.locs, g.smu, Gf), g.tables, g.locs, g.smv, Bb, Gb)
end
loop_scatter(g, op, nzv, KeT) = for cell in 1:length(g.cellmap)
    HK.scatter_jacobian!(nzv, op.nzpos, KeT, op.ndl, g.cellmap[cell])
end
function loop_tangent(g::HK.Group{NSV,NSU}, op, x, dvals, auxd, xl, Wb) where {NSV,NSU}
    acc = 0.0
    for cell in 1:length(g.cellmap)
        HK.gather_cell!(xl, op.fields, x, dvals, g.cellmap[cell])
        for q in 1:g.geo.np
            dV, iJt = HK.point_geometry(g.geo, q, cell)
            map((t, W) -> HK.point_weights!(W, t, q, iJt), g.tables, Wb)
            FV = map((t, l, W) -> HK.field_point_values(t, xl, l, W), g.tables, g.locs, Wb)
            Aq = map(a -> @inbounds(a[q, cell]), auxd)
            T = HK.apply_maps(g.lin, g.lin.state(g.lin.ops, FV, Aq), Val(NSV * NSU))
            acc += T[1] * dV
        end
    end
    acc
end

# OpenBLAS on the same G·B shapes (one block, per cell), as a compiled-library reference
function blas_products(g, Bb, Gb, nc)
    t = 0.0; fl = 0
    for (i, (tf, sf)) in enumerate(zip(g.tables, g.smv)), (j, (tg, sg)) in enumerate(zip(g.tables, g.smu))
        (lvof(sf) == 0 || lvof(sg) == 0) && continue
        G = Gb[i][j]; B = Bb[i]
        M, R = size(G, 1), size(G, 2); N = size(B, 2)
        Gm = reshape(view(G, :, :, 1, 1), M, R) |> Matrix
        C = zeros(M, N)
        nb = ncof(sf) * ncof(sg)
        t += nb * nc * best(() -> mul!(C, Gm, B, 1.0, 1.0); n=200)
        fl += nb * nc * 2 * M * N * R
    end
    t, fl
end

function analyse(name, res, jac, U, V)
    x = 1e-5 .* randn(num_free_dofs(U)); dx = randn(num_free_dofs(U))
    op = KernelOperator(res, jac, U, V); op1 = KernelOperator(res, jac, U, V; nthreads=1)
    A = Algebra.jacobian(op, x); nzv = nonzeros(A)
    tN = best(() -> jacobian!(A, op, x)); t1 = best(() -> jacobian!(A, op1, x))
    dvals = [get_dirichlet_dof_values(s) for s in HK.field_spaces(op.U)]
    @printf("%s — %d DOFs, nnz %d, %d element dofs\n", name, length(x), nnz(A), op.ndl)
    @printf("  jacobian!: 1 thr %.2f ms | %d thr %.2f ms (speed-up %.1f)\n", 1e3t1, Threads.nthreads(), 1e3tN, t1 / tN)
    for g in op.jgroups
        nc = length(g.cellmap); np = g.geo.np
        auxd = HK.aux_data(op, g)
        xl = zeros(op.ndl); KeT = zeros(op.ndl, op.ndl)
        HK.contiguous(g) || (println("  (group without node-contiguous dofs: breakdown skipped)"); continue)
        Wb, Bb, Gb = HK.mm_buffers(g)
        te = best(() -> loop_elem(g, op, x, dvals, auxd, xl, KeT, Wb, Bb, Gb))
        tp = best(() -> loop_products(g, KeT, Bb, Gb)) * cld(np, HK.point_block(np))
        ts = best(() -> loop_scatter(g, op, nzv, KeT))
        tt = g.lin === nothing ? NaN : best(() -> loop_tangent(g, op, x, dvals, auxd, xl, Wb))
        f1, f2 = flops_products(g)
        nmap = g.lin === nothing ? 0 : sum(length, g.lin.maps)
        tb, fb = blas_products(g, Bb, Gb, nc)
        @printf("  per cell (1 thr, %d cells, %d points): total %.1f µs\n", nc, np, 1e6 * (te + ts) / nc)
        @printf("    point loop (geometry, weights, state, tangent, G): %.1f µs  [tangent+state alone %.1f µs = %.0f ns/point, %d map entries]\n",
                1e6 * (te - tp) / nc, 1e6 * tt / nc, 1e9 * tt / (nc * np), nmap)
        @printf("    G at points: %.2f MFLOP/cell\n", f2 / 1e6)
        @printf("    element matrix products: %.1f µs, %.2f MFLOP/cell → %.1f GFLOP/s (%.0f%% of 1-thr peak); OpenBLAS same shapes %.1f GFLOP/s\n",
                1e6 * tp / nc, f1 / 1e6, f1 * nc / tp / 1e9, 100 * f1 * nc / tp / 1e9 / pk1, fb / tb / 1e9)
        sb = nc * op.ndl^2 * (4 + 8 + 16)
        @printf("    scatter: %.1f µs (%.1f ns/entry, %.1f GB/s of nzpos+Ke+nzv traffic)\n",
                1e6 * ts / nc, 1e9 * ts / (nc * op.ndl^2), sb / ts / 1e9)
        @printf("    shares: point loop %.0f%%, products %.0f%%, scatter %.0f%%\n",
                100(te - tp) / (te + ts), 100tp / (te + ts), 100ts / (te + ts))
    end
    # matrix-free and SpMV, bandwidth view
    mf = MatrixFreeJacobian(op); mf1 = MatrixFreeJacobian(op1)
    prepare!(mf, x); prepare!(mf1, x); y = mf * dx
    ta = best(() -> mul!(y, mf, dx); n=20); ta1 = best(() -> mul!(y, mf1, dx); n=20)
    tsp = best(() -> mul!(y, A, dx); n=20)
    dbytes = sum(sizeof(d.T) + sizeof(d.iJt) + sizeof(d.dV) for d in mf.data)
    ncell = sum(length(g.cellmap) for g in op.jgroups)
    mfb = dbytes + ncell * op.ndl * (2 * 4 + 8 + 16)
    spb = nnz(A) * 16 + size(A, 2) * 8 * 3
    @printf("  matrix-free apply: 1 thr %.3f ms (%.1f GB/s, %.0f%% STREAM-1) | %d thr %.3f ms (%.1f GB/s, %.0f%% STREAM)\n",
            1e3ta1, mfb / ta1 / 1e9, 100 * mfb / ta1 / 1e9 / bw1, Threads.nthreads(), 1e3ta, mfb / ta / 1e9, 100 * mfb / ta / 1e9 / bwN)
    @printf("  SpMV (serial SparseArrays): %.3f ms (%.1f GB/s, %.0f%% STREAM-1)\n\n", 1e3tsp, spb / tsp / 1e9, 100 * spb / tsp / 1e9 / bw1)
    flush(stdout)
end

analyse("EM box 20x4x4 Q2", em_box(20, 4, 4, 2)...)
analyse("EM box 14x3x3 Q3", em_box(14, 3, 3, 3)...)
analyse("EM box 40x8x8 Q2", em_box(40, 8, 8, 2)...)
analyse("EM plate n=2 P1 (tets)", em_case(GmshDiscreteModel(joinpath(@__DIR__, "..", "benchmark", "meshes", "em_plate_n2.msh")), 1)...)

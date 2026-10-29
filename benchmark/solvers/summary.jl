# Markdown tables from benchmark/results/solvers_baseline.csv (last row per case/backend/threads wins).
#   julia --project=benchmark/solvers benchmark/solvers/summary.jl
using DelimitedFiles, Printf
file = normpath(joinpath(@__DIR__, "..", "results", "solvers_baseline.csv"))
raw, hdr = readdlm(file, ','; header=true)
col = Dict(string(h) => i for (i, h) in enumerate(vec(hdr)))
rows = Dict{Tuple{String,String,Int},Vector{Any}}()
for r in eachrow(raw)
    rows[(string(r[col["case"]]), string(r[col["backend"]]), Int(r[col["threads"]]))] = collect(r)
end
g(r, k) = r[col[k]]
f3(x) = x isa Number && isfinite(x) ? @sprintf("%.3f", x) : "—"
f2(x) = x isa Number && isfinite(x) ? @sprintf("%.2f", x) : "—"
cases = ["em_plate_n4", "em_plate_n6", "em_plate_n10", "em_box_32x7x7", "em_box_52x11x11", "em_box_90x18x18", "hmre_r7", "hmre_r14", "hmre_r30"]
println("### Per Newton iteration (seconds; best of 3; LU = UMFPACK through Gridap's LUSolver)\n")
println("| case | DOFs | nnz/row | threads | residual lazy | residual compiled | Jacobian lazy | Jacobian compiled | LU factor | triangular solve | matrix GB | LU factors GB |")
println("|---|---|---|---|---|---|---|---|---|---|---|---|")
for c in cases, t in (1, 6, 18)
    haskey(rows, (c, "compiled", t)) || continue
    k = rows[(c, "compiled", t)]; l = get(rows, (c, "lazy", t), nothing)
    n = Int(g(k, "dofs")); nnzr = g(k, "nnz") / n
    @printf("| %s | %d | %.0f | %d | %s | %s | %s | %s | %s | %s | %s | %s |\n", c, n, nnzr, t,
            l === nothing ? "—" : f3(g(l, "t_res")), f3(g(k, "t_res")), l === nothing ? "—" : f3(g(l, "t_jac")), f3(g(k, "t_jac")),
            f2(g(k, "t_lu")), f3(g(k, "t_ldiv")), f2(g(k, "matrix_gb")), f2(g(k, "lu_gb")))
end
println("\n### Full solve! through the driver (R0; case's own load steps; 6 threads)\n")
println("| case | backend | DOFs | steps | Newton it. | solve! s | s/iteration | assembly+LU estimate s/it | rest % | allocated GB | driver MB/it | peak RSS GB |")
println("|---|---|---|---|---|---|---|---|---|---|---|---|")
for c in cases, b in ("lazy", "compiled")
    r = get(rows, (c, b, 6), nothing)
    (r === nothing || !(g(r, "nit") isa Number) || g(r, "nit") == 0) && continue
    nit = g(r, "nit"); tit = g(r, "t_solve") / nit
    est = g(r, "t_res") + g(r, "t_jac") + g(r, "t_lu") + g(r, "t_ldiv")
    @printf("| %s | %s | %d | %d | %d | %.1f | %.3f | %.3f | %.1f | %.1f | %.1f | %.1f |\n", c, b, Int(g(r, "dofs")), Int(g(r, "nsteps")), nit,
            g(r, "t_solve"), tit, est, 100 * (1 - est / tit), g(r, "b_solve") / 2^30, g(r, "b_driver") / nit / 2^20, g(r, "rss_end_gb"))
end

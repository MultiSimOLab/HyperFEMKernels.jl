# Stage S0, first step: which preconditioners does the installed PETSc binary provide?
# Each configuration runs in its own Julia process on a 3D Poisson problem (Q1, n³ cells).
#
#   julia --project=benchmark/solvers benchmark/solvers/petsc_check.jl [n]
using Pkg, Printf, Dates

n = length(ARGS) ≥ 1 ? ARGS[1] : "40"
configs = [
    "CG + Jacobi"               => "-ksp_type cg -pc_type jacobi",
    "CG + GAMG"                 => "-ksp_type cg -pc_type gamg",
    "CG + BoomerAMG (hypre)"    => "-ksp_type cg -pc_type hypre -pc_hypre_type boomeramg",
    "GMRES + GAMG"              => "-ksp_type gmres -pc_type gamg",
    "LU (PETSc)"                => "-ksp_type preonly -pc_type lu",
    "LU (MUMPS)"                => "-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps",
    "LU (SuperLU_DIST)"         => "-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist",
    "Cholesky (MUMPS)"          => "-ksp_type preonly -pc_type cholesky -pc_factor_mat_solver_type mumps",
    "fieldsplit (available?)"   => "-ksp_type gmres -pc_type fieldsplit -pc_fieldsplit_type additive",
]
deps = Pkg.dependencies()
for (_, d) in deps
    d.name in ("PETSc_jll", "GridapPETSc", "HYPRE_jll", "MUMPS_jll", "SuperLU_DIST_jll", "Gridap") &&
        println(rpad(d.name, 18), d.version)
end
println("Poisson 3D, Q1, $(n)³ cells, rtol 1e-8, threads $(Threads.nthreads()), $(now())")
case = joinpath(@__DIR__, "petsc_case.jl")
rows = String[]
for (name, opts) in configs
    out = IOBuffer()
    ok = success(pipeline(`$(Base.julia_cmd()) --project=$(@__DIR__) $case $n $opts`; stdout=out, stderr=out))
    s = String(take!(out))
    m = match(r"RESULT dofs=(\d+) setup=([\d.]+) solve=([\d.]+) relres=(\S+)", s)
    it = match(r"converged due to (\w+) iterations (\d+)", s)
    if ok && m !== nothing
        line = @sprintf("%-26s yes   %-8s it=%-4s setup %.2f s  solve %.2f s  relres %s", name,
                        it === nothing ? "-" : it[1][1:min(end, 8)], it === nothing ? "-" : it[2], parse(Float64, m[2]), parse(Float64, m[3]), m[4])
    else
        err = match(r"(Unknown type[^\n]*|Could not locate[^\n]*|unknown[^\n]*|ERROR[^\n]*)", s)
        line = @sprintf("%-26s NO    %s", name, err === nothing ? last(split(strip(s), '\n')) : err[1])
    end
    println(line); flush(stdout); push!(rows, line)
end

# One PETSc configuration on a 3D Poisson problem (Q1, n³ cells), run by petsc_check.jl in its own
# process (PETSc is initialised once per process).
#
#   julia --project=benchmark/solvers benchmark/solvers/petsc_case.jl n "<PETSc options>"
using Gridap, GridapPETSc, SparseMatricesCSR, LinearAlgebra, Printf

n = parse(Int, ARGS[1])
opts = ARGS[2] * " -ksp_rtol 1e-8 -ksp_converged_reason"
model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (n, n, n))
V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); dirichlet_tags="boundary")
U = TrialFESpace(V, 0.0)
Ω = Triangulation(model); dΩ = Measure(Ω, 2)
a(u, v) = ∫(∇(u) ⋅ ∇(v))dΩ
l(v) = ∫(1.0 * v)dΩ
GridapPETSc.with(args=split(opts)) do
    assem = SparseMatrixAssembler(SparseMatrixCSR{0,PetscScalar,PetscInt}, Vector{PetscScalar}, U, V)
    op = AffineFEOperator(a, l, U, V, assem)
    A, b = get_matrix(op), get_vector(op)
    solver = PETScLinearSolver()
    x = similar(b); fill!(x, 0.0)
    ns = numerical_setup(symbolic_setup(solver, A), A)     # warm-up (compilation)
    solve!(x, ns, b)
    fill!(x, 0.0)
    t_setup = @elapsed (ns = numerical_setup(symbolic_setup(solver, A), A))
    t_solve = @elapsed solve!(x, ns, b)
    r = norm(A * x - b) / norm(b)
    @printf("RESULT dofs=%d setup=%.3f solve=%.3f relres=%.2e\n", length(b), t_setup, t_solve, r)
end

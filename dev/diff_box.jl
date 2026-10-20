using HyperFEMKernels, HyperFEM, Gridap
using Gridap.FESpaces, Gridap.Algebra
using LinearAlgebra, SparseArrays, Printf, Random
Random.seed!(1)
include(joinpath(@__DIR__, "bench_cases.jl"))
res, jac, U, V = em_box(20, 4, 4, 2)
x = parse(Float64, get(ENV, "XSCALE", "1e-5")) .* randn(num_free_dofs(U))
gop = get_algebraic_operator(FEOperator(res, jac, U, V))
kop = KernelOperator(res, jac, U, V)
Ag = jacobian(gop, x); Ak = jacobian(kop, x)
nu = num_free_dofs(U.spaces[1])
blk(A, r, c) = A[r, c]
ru, rφ = 1:nu, nu+1:size(Ag, 1)
for (name, r, c) in (("uu", ru, ru), ("uφ", ru, rφ), ("φu", rφ, ru), ("φφ", rφ, rφ))
    G = blk(Ag, r, c); K = blk(Ak, r, c)
    @printf("%s: ‖G‖ = %.3e  ‖K-G‖/‖G‖ = %.2e  max|K-G|/max|G| = %.2e\n", name, norm(G), norm(K - G) / norm(G),
            maximum(abs, K - G) / maximum(abs, G))
end
@printf("whole: %.2e ; Gridap vs Gridap(transpose order) n/a\n", norm(Ak - Ag) / norm(Ag))
# Gridap AD-free sanity: compare Gridap jacobian assembled twice
A2 = jacobian(gop, x); @printf("Gridap repeat diff %.2e\n", norm(A2 - Ag) / norm(Ag))

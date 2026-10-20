using HyperFEMKernels, Gridap, Gridap.FESpaces, Gridap.Algebra, LinearAlgebra, Random
const HK = HyperFEMKernels
include(joinpath(@__DIR__, "..", "test", "common.jl"))
model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (2, 2, 2))
Ω = Triangulation(model); dΩ = Measure(Ω, 2)
V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), dirichlet_tags=[1, 3, 5, 7])
U = TrialFESpace(V, x -> VectorValue(0.01, 0.0, 0.0))
Γ = BoundaryTriangulation(model; tags=[26]); dΓ = Measure(Γ, 2)
n = get_normal_vector(Γ); pr = 0.3
F(u) = Fk ∘ (∇(u)')
res1(u, v) = ∫(pr * (v ⋅ ((det ∘ F(u)) * ((inv ∘ F(u)) ⋅ n))))dΓ
res2(u, v) = ∫(v ⋅ ((inv ∘ F(u)) ⋅ n))dΓ
res3(u, v) = ∫(v ⋅ ((Fk ∘ (∇(u)')) ⋅ n))dΓ
res4(u, v) = ∫(v ⋅ (∇(u)' ⋅ n))dΓ
res5(u, v) = ∫(v ⋅ (∇(u) ⋅ n))dΓ
jac(u, du, v) = ∫(∇(v)' ⊙ (dP_svk ∘ (Fk ∘ (∇(u)'), ∇(du)')))dΩ
x = 0.01 .* randn(num_free_dofs(U))
for (name, r) in (("res5 ∇u⋅n", res5), ("res4 ∇u'⋅n", res4), ("res3 F⋅n", res3), ("res2 invF⋅n", res2), ("res1 full", res1))
    print(name, ": gridap "); flush(stdout)
    try
        Algebra.residual(get_algebraic_operator(FEOperator(r, jac, U, V)), x); print("ok; kernel ")
        op = KernelOperator(r, jac, U, V); c = check(r, jac, U, V, x; op)
        println("fallback=", op.fallback.rtrians, " diff ", c.res)
    catch e
        println("ERROR: ", sprint(showerror, e)[1:min(end, 300)])
    end
end

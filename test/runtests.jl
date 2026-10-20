using Test
using HyperFEMKernels
using Gridap
using Gridap.FESpaces, Gridap.Algebra
using LinearAlgebra, Random

const TOL = 1e-12

# St Venant–Kirchhoff, written pointwise (Gridap convention F = I + ∇u')
const λ, μ = 2.0, 1.0
E(F) = 0.5 * (transpose(F) ⋅ F - one(F))
S(F) = λ * tr(E(F)) * one(F) + 2μ * E(F)
P(F) = F ⋅ S(F)
dS(F, dF) = (dE = 0.5 * (transpose(dF) ⋅ F + transpose(F) ⋅ dF); λ * tr(dE) * one(F) + 2μ * dE)
dP(F, dF) = dF ⋅ S(F) + F ⋅ dS(F, dF)
Fk(∇u) = one(∇u) + ∇u

@testset "HyperFEMKernels" begin
    Random.seed!(1)

    @testset "Poisson 2D ($(name))" for (name, model) in (
            ("Q1", CartesianDiscreteModel((0, 1, 0, 1), (5, 4))),
            ("P1", simplexify(CartesianDiscreteModel((0, 1, 0, 1), (4, 3)))))
        Ω = Triangulation(model); dΩ = Measure(Ω, 2)
        V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), dirichlet_tags="boundary")
        U = TrialFESpace(V, x -> x[1] + 2x[2])
        f(x) = sin(x[1]) * x[2]
        res(u, v) = ∫(∇(v) ⋅ ∇(u) - f * v)dΩ
        jac(u, du, v) = ∫(∇(v) ⋅ ∇(du))dΩ
        c = check(res, jac, U, V, rand(num_free_dofs(U)))
        @test c.res < TOL
        @test c.jac < TOL
    end

    @testset "Hyperelasticity 3D Q$order" for order in (1, 2)
        model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (3, 2, 2))
        Ω = Triangulation(model); dΩ = Measure(Ω, 2order)
        V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, order), dirichlet_tags=[1, 3, 5, 7])
        U = TrialFESpace(V, x -> VectorValue(0.01, 0.0, 0.0))
        res(u, v) = ∫(∇(v)' ⊙ (P ∘ (Fk ∘ (∇(u)'))))dΩ
        jac(u, du, v) = ∫(∇(v)' ⊙ (dP ∘ (Fk ∘ (∇(u)'), ∇(du)')))dΩ
        x = 0.01 .* randn(num_free_dofs(U))
        c = check(res, jac, U, V, x)
        @test c.res < TOL
        @test c.jac < TOL
        @test all(a -> a.bytes == 0, allocations(KernelOperator(res, jac, U, V), x))
    end

    @testset "Hyperelasticity 3D P1 (tetrahedra)" begin
        model = simplexify(CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (2, 2, 2)))
        Ω = Triangulation(model); dΩ = Measure(Ω, 2)
        V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), dirichlet_tags=[1, 3, 5, 7])
        U = TrialFESpace(V, x -> VectorValue(0.0, 0.01, 0.0))
        res(u, v) = ∫(∇(v)' ⊙ (P ∘ (Fk ∘ (∇(u)'))))dΩ
        jac(u, du, v) = ∫(∇(v)' ⊙ (dP ∘ (Fk ∘ (∇(u)'), ∇(du)')))dΩ
        c = check(res, jac, U, V, 0.01 .* randn(num_free_dofs(U)))
        @test c.res < TOL
        @test c.jac < TOL
    end

    @testset "Multi-field u–φ with value and gradient couplings" begin
        model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (2, 2, 2))
        Ω = Triangulation(model); dΩ = Measure(Ω, 2)
        Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), dirichlet_tags=[1, 2])
        Vφ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), dirichlet_tags=[7, 8])
        V = MultiFieldFESpace([Vu, Vφ])
        U = MultiFieldFESpace([TrialFESpace(Vu, x -> VectorValue(0.0, 0.0, 0.0)), TrialFESpace(Vφ, x -> 0.1)])
        β = 0.3
        res((u, φ), (v, vφ)) = ∫(∇(v)' ⊙ (P ∘ (Fk ∘ (∇(u)'))) + β * φ * tr(∇(v)) + ∇(vφ) ⋅ ∇(φ) +
                                 β * vφ * (tr ∘ (Fk ∘ (∇(u)'))))dΩ
        jac((u, φ), (du, dφ), (v, vφ)) = ∫(∇(v)' ⊙ (dP ∘ (Fk ∘ (∇(u)'), ∇(du)')) + β * dφ * tr(∇(v)) +
                                           ∇(vφ) ⋅ ∇(dφ) + β * vφ * tr(∇(du)))dΩ
        x = 0.01 .* randn(num_free_dofs(U))
        c = check(res, jac, U, V, x)
        @test c.res < TOL
        @test c.jac < TOL
        @test all(a -> a.bytes == 0, allocations(KernelOperator(res, jac, U, V), x))
    end

    @testset "Dirichlet masks, auxiliary FE functions and scaled integrals (2D)" begin
        model = CartesianDiscreteModel((0, 2, 0, 1), (6, 3))
        Ω = Triangulation(model); dΩ = Measure(Ω, 2)
        V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, 1),
                        dirichlet_tags=["tag_7", "tag_8"], dirichlet_masks=[(true, true), (true, false)])
        U = TrialFESpace(V, [x -> VectorValue(0.0, 0.0), x -> VectorValue(0.02, 0.0)])
        Uold = TrialFESpace(V, [x -> VectorValue(0.0, 0.0), x -> VectorValue(0.0, 0.0)])
        uold = FEFunction(Uold, 0.01 .* randn(num_free_dofs(Uold)))
        N = interpolate_everywhere(x -> VectorValue(cos(x[2]), sin(x[2])), V)
        Λ = 0.4
        g(F, N) = (F ⋅ N) ⊗ N
        res(u, v) = ∫(∇(v)' ⊙ (P ∘ (Fk ∘ (∇(u)'))))dΩ - Λ * ∫((g ∘ (Fk ∘ (∇(uold)'), N)) ⊙ ∇(v)')dΩ +
                    ∫(v ⋅ uold)dΩ
        jac(u, du, v) = ∫(∇(v)' ⊙ (dP ∘ (Fk ∘ (∇(u)'), ∇(du)')))dΩ
        x = 0.01 .* randn(num_free_dofs(U))
        c = check(res, jac, U, V, x)
        @test c.res < TOL
        @test c.jac < TOL
        @test all(a -> a.bytes == 0, allocations(KernelOperator(res, jac, U, V), x))
        c2 = check(res, jac, U, V, x; op=KernelOperator(res, jac, U, V; cache_aux=true))
        @test c2.res < TOL
        @test c2.jac < TOL
    end

    @testset "Integration on a subdomain of the FE space (and two measures)" begin
        model = CartesianDiscreteModel((0, 2, 0, 1), (8, 4))
        left = Bool[sum(c)[1] / length(c) < 1.0 for c in vec(collect(get_cell_coordinates(model)))]
        sub = Triangulation(model, left)
        Ω = Triangulation(model); dΩ = Measure(Ω, 2); dΩs = Measure(sub, 2)
        V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), dirichlet_tags="boundary")
        U = TrialFESpace(V, x -> x[1])
        res(u, v) = ∫(∇(v) ⋅ ∇(u))dΩ + ∫(3.0 * v * u * u)dΩs
        jac(u, du, v) = ∫(∇(v) ⋅ ∇(du))dΩ + ∫(6.0 * v * u * du)dΩs
        c = check(res, jac, U, V, rand(num_free_dofs(U)))
        @test c.res < TOL
        @test c.jac < TOL
    end

    @testset "update keeps the structure and follows new parameters" begin
        model = CartesianDiscreteModel((0, 1, 0, 1), (4, 4))
        Ω = Triangulation(model); dΩ = Measure(Ω, 2)
        V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), dirichlet_tags="boundary")
        U = TrialFESpace(V, 0.0)
        res(Λ) = (u, v) -> ∫(∇(v) ⋅ ∇(u) - Λ * v)dΩ
        jac(Λ) = (u, du, v) -> ∫(∇(v) ⋅ ∇(du))dΩ
        op = KernelOperator(res(1.0), jac(1.0), U, V)
        op2 = update(op, res(2.0), jac(2.0))
        x = rand(num_free_dofs(U))
        c = check(res(2.0), jac(2.0), U, V, x; op=op2)
        @test c.res < TOL
        @test c.jac < TOL
        @test op2.A0 === op.A0
    end
end

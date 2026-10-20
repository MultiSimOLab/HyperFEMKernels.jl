using Test
using HyperFEMKernels

@testset "HyperFEMKernels" begin
    @testset "package loads" begin
        @test isdefined(HyperFEMKernels, :HyperFEMKernels)
    end
    # Phase 1: Poisson, linear elasticity and neo-Hookean (Q1/P1, 2D and 3D),
    # multifield u–φ and masked Dirichlet, each compared with Gridap at 1e-12.
end

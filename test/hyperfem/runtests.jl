# Tests that need HyperFEM (run in the benchmark environment: julia -t auto --project=benchmark test/hyperfem/runtests.jl).
# They check the kernel backend on HyperFEM's own weak forms: EM monolithic, hMRE, and a
# visco-hyperelastic dynamic problem with CellState internal variables (from Jesús's script).
using Test
using HyperFEMKernels, HyperFEM
using Gridap, GridapGmsh, Gridap.FESpaces, Gridap.Algebra, Gridap.CellData
using GridapSolvers, GridapSolvers.NonlinearSolvers
using LinearAlgebra, Random

const HK = HyperFEMKernels
const TOL = 1e-12
reldiff(a, b) = norm(a - b) / max(norm(b), eps())
gop(res, jac, U, V) = get_algebraic_operator(FEOperator(res, jac, U, V))

@testset "HyperFEM weak forms" begin
    Random.seed!(1)

    @testset "EM 3D monolithic (README), test_static_EM.msh" begin
        model = GmshDiscreteModel(joinpath(@__DIR__, "..", "models", "test_static_EM.msh"))
        physmodel = ElectroMechModel(IdealDielectric(ε=1.0), NeoHookean3D(λ=10.0, μ=1.0))
        k = (Kinematics(Mechano, Solid), Kinematics(Electro, Solid))
        Ω = Triangulation(model); dΩ = Measure(Ω, 2)
        Du = DirichletBC(["fixedup"], [[0.0, 0.0, 0.0]], [Λ -> 1.0])
        Dφ = DirichletBC(["midsuf", "topsuf"], [0.0, 0.1], [Λ -> Λ, Λ -> Λ])
        Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), Du, conformity=:H1)
        Vφ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), Dφ, conformity=:H1)
        V = MultiFieldFESpace([Vu, Vφ])
        U = MultiFieldFESpace([TrialFESpace(Vu, Du, 1.0), TrialFESpace(Vφ, Dφ, 1.0)])
        res((u, φ), (v, vφ)) = HyperFEM.residual(physmodel, k, (u, φ), (v, vφ), dΩ)
        jac((u, φ), (du, dφ), (v, vφ)) = HyperFEM.jacobian(physmodel, k, (u, φ), (du, dφ), (v, vφ), dΩ)
        x = 0.01 .* randn(num_free_dofs(U))
        op = KernelOperator(res, jac, U, V)
        c = check(res, jac, U, V, x; op)
        @test c.res < TOL && c.jac < TOL
        @test all(g -> g.lin !== nothing, op.rgroups) && all(g -> g.lin !== nothing, op.jgroups)
        @test all(a -> a.bytes == 0, allocations(op, x))
    end

    @testset "hMRE 2D, Short_magnet.msh" begin
        model = GmshDiscreteModel(joinpath(@__DIR__, "..", "models", "Short_magnet.msh"))
        params = [10.0, 12480.64495232286, 1.0, 2.0, 5195.545287237134, 0.2602717127043121, 2.0]
        modelmech = NonlinearMooneyRivlin2D(λ=(params[1] + params[2]) * 1e2, μ1=params[1] * 1e3, μ2=params[2], α1=params[3], α2=params[4]) +
                    TransverseIsotropy2D(μ=params[5], α1=params[6], α2=params[7])
        μ0 = 4π * 1e-7; Banorm = 80.0E-9; αr = 10e-3 / μ0
        reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
        Ωs = Triangulation(model, tags=["Beam"]); dΩs = Measure(Ωs, 2)
        Du = DirichletBC(["Fixed_Solid", "Fixed_Solid_x"], [[0.0, 0.0], [0.0, 0.0]])
        V = TestFESpace(Ωs, reffeu, Du, conformity=:H1, dirichlet_masks=[(true, true), (true, false)])
        U = TrialFESpace(V, Du); U⁻ = TrialFESpace(V, Du)
        uh⁻ = FEFunction(U⁻, 1e-4 .* randn(num_free_dofs(U⁻)))
        _, ∂Ψs∂F, ∂Ψs∂FF = modelmech()
        F, _, _ = get_Kinematics(Kinematics(Mechano, Solid))
        V_N = TestFESpace(Ωs, reffeu)
        Nh = interpolate_everywhere(x -> VectorValue(1.0, 0.0), V_N)
        Bah = interpolate_everywhere(VectorValue([0.0, -Banorm]), V_N)
        invU(F; ε=1e-12) = (F_ = get_array(F); TensorValue(inv(sqrt(F_' * F_ + ε * get_array(I2)))))   # original
        ∂Ψmag∂F(Fn, N, Ba) = (-αr / μ0) * (Ba ⊗ ((invU(Fn)) * N))
        Λ = 0.35
        res(u, v) = ∫((∇(v)' ⊙ (∂Ψs∂F ∘ (F ∘ (∇(u)'), Nh))))dΩs + Λ * ∫((∂Ψmag∂F ∘ (F ∘ (∇(uh⁻)'), Nh, Bah)) ⊙ (∇(v)'))dΩs
        jac(u, du, v) = ∫(∇(v)' ⊙ ((∂Ψs∂FF ∘ (F ∘ (∇(u)'), Nh)) ⊙ (∇(du)')))dΩs
        x = 1e-4 .* randn(num_free_dofs(U))
        op = KernelOperator(res, jac, U, V)
        c = check(res, jac, U, V, x; op)
        @test c.res < TOL && c.jac < TOL
        @test all(a -> a.bytes == 0, allocations(op, x))   # the magnetic term (uh⁻ only) is auxiliary
    end

    @testset "visco-hyperelastic dynamics with CellState internal variables (midpoint rule)" begin
        # Jesús's script (plate under a centre load), volume terms only; the Neumann term ∫(f⋅v)dΓ
        # is a boundary integral (phase 2) and is checked to raise a clear error below.
        width, thick, xdiv, order = 0.2, 0.001, 3, 2
        geometry = CartesianDiscreteModel((-0.5width, 0.5width, -0.5width, 0.5width, 0.0, thick), (xdiv, xdiv, 1))
        labels = get_face_labeling(geometry)
        add_tag_from_tags!(labels, "top", CartesianTags.faceXY1⁺)
        add_tag_from_vertex_filter!(labels, "center", geometry, p -> abs(p[1]) < 0.51width / xdiv && abs(p[2]) < 0.51width / xdiv)
        model = GeneralizedMaxwell(NeoHookean3D(μ=13446.1, λ=0.0) + VolumetricEnergy(λ=2.5e6),
                                   ViscousPolyconvex(μ=11053.7, τ=151.68), ViscousPolyconvex(μ=30480.1, τ=7.0731))
        F, H, J = get_Kinematics(Kinematics(Mechano, Solid))
        Δt = 0.2 * thick / (order * 0.1)
        Ω = Triangulation(geometry); dΩ = Measure(Ω, 2order)
        Γ = BoundaryTriangulation(geometry, tags="center"); dΓ = Measure(Γ, 2order)
        reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
        Vu = TestFESpace(geometry, reffe, conformity=:H1)
        Uu = TrialFESpace(Vu); Uu⁻ = TrialFESpace(Vu); Uυ = TrialFESpace(Vu)
        uh⁺ = FEFunction(Uu, 1e-5 .* randn(num_free_dofs(Uu)))
        uh⁻ = FEFunction(Uu⁻, 1e-5 .* randn(num_free_dofs(Uu)))
        υh = FEFunction(Uυ, 1e-3 .* randn(num_free_dofs(Uυ)))
        Fh = F ∘ ∇(uh⁺)'; Fh⁻ = F ∘ ∇(uh⁻)'
        Ah = CellState(model, dΩ)
        Ph⁻ = CellState(zero(TensorValue{3,3,Float64}), dΩ)
        update_time_step!(model, Δt)
        Ψ, ∂Ψ∂F, ∂∂Ψ∂FF = model()
        P_updater(P, F, Fn, A...) = (true, ∂Ψ∂F(F, Fn, A...))
        update_state!(P_updater, Ph⁻, Fh, Fh⁻, Ah...)
        update_state!(model, Ah, Fh, Fh⁻)
        ρ₀ = 960.0
        res(u, v) = begin
            a_mid = 2 / Δt^2 * (u - uh⁻) - 2 / Δt * υh
            P_mid = 1 / 2 * (∂Ψ∂F ∘ (F ∘ ∇(u)', Fh⁻, Ah...) + Ph⁻)
            ∫(ρ₀ * a_mid ⋅ v + ∇(v)' ⊙ P_mid)dΩ
        end
        jac(u, du, v) = begin
            Da = 2 / Δt^2 * du
            DP = 1 / 2 * (∂∂Ψ∂FF ∘ (F ∘ ∇(u)', Fh⁻, Ah...)) ⊙ ∇(du)'
            ∫(ρ₀ * Da ⋅ v + ∇(v)' ⊙ DP)dΩ
        end
        x = 1e-5 .* randn(num_free_dofs(Uu))
        op = KernelOperator(res, jac, Uu, Vu)
        c = check(res, jac, Uu, Vu, x; op)
        @test c.res < TOL && c.jac < TOL
        @test all(g -> g.lin !== nothing, op.rgroups) && all(g -> g.lin !== nothing, op.jgroups)
        # HyperFEM's ViscousPolyconvex law allocates (≈5.6 kB per call of ∂∂Ψ∂FF); the kernels themselves do not.
        # This is reported, not required, until the law is made allocation-free (GPU prerequisite).
        a = allocations(op, x)
        @info "visco-hyperelastic kernels: allocations per element call" a
        # internal variables change → the next assembly (no cache) must follow Gridap
        update_state!(model, Ah, F ∘ ∇(FEFunction(Uu, x))', Fh⁻)
        c2 = check(res, jac, Uu, Vu, x; op)
        @test c2.res < TOL && c2.jac < TOL
        # two time steps with Gridap's Newton on both operators give the same displacement
        f_mid(t) = VectorValue(0.0, 0.0, 20.0) * EvolutionFunctions.triangular(0.05)(t - Δt / 2)
        nls = NewtonSolver(LUSolver(); maxiter=20, atol=1e-10, rtol=1e-10, verbose=false)
        xg = copy(x); xk = copy(x)
        for step in 1:2
            # Dirichlet-free problem: add the load as a volume body force on the centre cells is not
            # the script's loading; we keep the script's homogeneous volume form (no Γ) here.
            opg = gop(res, jac, Uu, Vu); opk = HK.update(op, res, jac)
            Algebra.solve!(xg, nls, opg); Algebra.solve!(xk, nls, opk)
        end
        @test reldiff(xk, xg) < 1e-10
        # the script's surface load on Γ: a boundary group of the kernel (no fallback)
        resΓ(u, v) = res(u, v) - ∫(f_mid(Δt) ⋅ v)dΓ
        opΓ = KernelOperator(resΓ, jac, Uu, Vu)
        @test isempty(opΓ.fallback.rtrians) && isempty(opΓ.fallback.jtrians)
        @test any(g -> g.trian isa BoundaryTriangulation, opΓ.rgroups)
        cΓ = check(resΓ, jac, Uu, Vu, x; op=opΓ)
        @test cΓ.res < TOL && cΓ.jac < TOL
        xg = copy(x); xk = copy(x)
        Algebra.solve!(xg, nls, gop(resΓ, jac, Uu, Vu)); Algebra.solve!(xk, nls, opΓ)
        @test reldiff(xk, xg) < 1e-10
    end
end

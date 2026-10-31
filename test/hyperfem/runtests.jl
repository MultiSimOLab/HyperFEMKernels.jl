# Tests that need HyperFEM (run in the benchmark environment: julia -t auto --project=benchmark test/hyperfem/runtests.jl).
# They check the kernel backend on HyperFEM's own weak forms: EM monolithic, hMRE, and a
# visco-hyperelastic dynamic problem with CellState internal variables (from Jesús's script).
using Test
using HyperFEMKernels, HyperFEM
using Gridap, GridapGmsh, Gridap.FESpaces, Gridap.Algebra, Gridap.CellData
using LinearAlgebra, Random, SparseArrays

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
        Du = BoundaryConditions(BoundaryCondition("fixedup", [0.0, 0.0, 0.0]))
        Dφ = BoundaryConditions(BoundaryCondition("midsuf", 0.0, Λ -> Λ), BoundaryCondition("topsuf", 0.1, Λ -> Λ))
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
        Du = BoundaryConditions(BoundaryCondition("Fixed_Solid", [0.0, 0.0]), BoundaryCondition("Fixed_Solid_x", [0.0, 0.0]; mask=(true, false)))
        V = TestFESpace(Ωs, reffeu, Du, conformity=:H1)
        U = TrialFESpace(V, Du); U⁻ = TrialFESpace(V, Du)
        uh⁻ = FEFunction(U⁻, 1e-4 .* randn(num_free_dofs(U⁻)))
        _, ∂Ψs∂F, ∂Ψs∂FF = modelmech()
        F, _, _ = get_Kinematics(Kinematics(Mechano, Solid))
        V_N = TestFESpace(Ωs, reffeu)
        Nh = interpolate_everywhere(x -> VectorValue(1.0, 0.0), V_N)
        Bah = interpolate_everywhere(VectorValue([0.0, -Banorm]), V_N)
        # the magnetic term depends on uh⁻, Nh and Bah only: FE functions read as leaves of the compiled
        # integrand (re-read at every assembly), ∂Ψmag∂F compiled with it
        # (the original script's `invU(F; ε=1e-12)`: a local function with keyword arguments is called
        # dynamically by Julia and allocates in the element kernel, which `allocations` reports)
        invU0(F) = (F_ = get_array(F); TensorValue(inv(sqrt(F_' * F_ + 1e-12 * get_array(I2)))))   # original (matrix sqrt)
        invU(F) = (C = transpose(F) ⋅ F + 1e-12 * I2; s = sqrt(det(C)); inv((C + s * I2) / sqrt(tr(C) + 2s)))  # closed form, 2D
        Λ = 0.35
        jac(u, du, v) = ∫(∇(v)' ⊙ ((∂Ψs∂FF ∘ (F ∘ (∇(u)'), Nh)) ⊙ (∇(du)')))dΩs
        x = 1e-4 .* randn(num_free_dofs(U))
        for iU in (invU0, invU)
            ∂Ψmag∂F(Fn, N, Ba) = (-αr / μ0) * (Ba ⊗ ((iU(Fn)) * N))
            res(u, v) = ∫((∇(v)' ⊙ (∂Ψs∂F ∘ (F ∘ (∇(u)'), Nh))))dΩs + Λ * ∫((∂Ψmag∂F ∘ (F ∘ (∇(uh⁻)'), Nh, Bah)) ⊙ (∇(v)'))dΩs
            op = KernelOperator(res, jac, U, V)
            @test count(a -> a isa HyperFEMKernels.FEArray, op.rgroups[1].aux) == 3    # Nh (used twice), ∇uh⁻, Bah
            c = check(res, jac, U, V, x; op)
            @test c.res < TOL && c.jac < TOL
            @test all(a -> a.bytes == 0, allocations(op, x))
        end
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
        nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-10, rtol=1e-10, verbose=0)
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
    @testset "visco-hyperelastic internal variables updated by the engine (CellState and KernelState)" begin
        # Same plate as above. The states of the GeneralizedMaxwell branches (Cv per branch) and the stress of
        # the previous step are updated with update_states! from the same updaters HyperFEM uses, and compared
        # with Gridap's update_state!; then two time steps with kernel operator + kernel states against Gridap.
        width, thick, xdiv, order = 0.2, 0.001, 3, 2
        geometry = CartesianDiscreteModel((-0.5width, 0.5width, -0.5width, 0.5width, 0.0, thick), (xdiv, xdiv, 1))
        model = GeneralizedMaxwell(NeoHookean3D(μ=13446.1, λ=0.0) + VolumetricEnergy(λ=2.5e6),
                                   ViscousPolyconvex(μ=11053.7, τ=151.68), ViscousPolyconvex(μ=30480.1, τ=7.0731))
        F, H, J = get_Kinematics(Kinematics(Mechano, Solid))
        Δt = 0.2 * thick / (order * 0.1)
        Ω = Triangulation(geometry); dΩ = Measure(Ω, 2order)
        reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
        Vu = TestFESpace(geometry, reffe, conformity=:H1)
        Uu = TrialFESpace(Vu); Uu⁻ = TrialFESpace(Vu); Uυ = TrialFESpace(Vu)
        uh⁺ = FEFunction(Uu, 1e-5 .* randn(num_free_dofs(Uu)))
        uh⁻ = FEFunction(Uu⁻, 1e-5 .* randn(num_free_dofs(Uu)))
        υh = FEFunction(Uυ, 1e-3 .* randn(num_free_dofs(Uυ)))
        Fh = F ∘ ∇(uh⁺)'; Fh⁻ = F ∘ ∇(uh⁻)'
        update_time_step!(model, Δt)
        Ψ, ∂Ψ∂F, ∂∂Ψ∂FF = model()
        T33 = TensorValue{3,3,Float64,9}
        P_updater(P, F, Fn, A...) = (true, ∂Ψ∂F(F, Fn, A...))
        branch_updater(b) = (A, F, Fn) -> (true, HyperFEM.return_mapping(b, F, Fn, A))   # what update_state!(b, A, F, Fn) does
        branches = model.branches.branches                                                # NVisco holds the tuple
        dense(s::CellState) = reduce(hcat, s.values); dense(s::KernelState) = s.values
        same(a, b) = maximum(norm, dense(a) - dense(b)) <= 1e-12 * max(1.0, maximum(norm, dense(b)))
        ρ₀ = 960.0
        function forms(Ah, Ph⁻)
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
            res, jac
        end
        for storage in (CellState, KernelState)
            x = get_cell_points(dΩ)
            Ah = Tuple(storage(one(T33), x) for _ in branches); Ph⁻ = storage(zero(T33), x)
            Ahg = CellState(model, dΩ); Pg = CellState(zero(T33), dΩ)
            # one update at the random state
            update_state!(P_updater, Pg, Fh, Fh⁻, Ahg...); update_state!(model, Ahg, Fh, Fh⁻)
            suP = update_states!(P_updater, Ph⁻, Fh, Fh⁻, Ah...)
            sus = [update_states!(branch_updater(b), A, Fh, Fh⁻) for (b, A) in zip(branches, Ah)]
            @test same(Ph⁻, Pg) && all(same(A, Ag) for (A, Ag) in zip(Ah, Ahg))
            @test isempty(suP.aux) && length(suP.fes) == 2 && HK.nwritten(suP) == 3 && length(suP.states) == 3
            @test all(su -> isempty(su.aux) && length(su.fes) == 2 && HK.nwritten(su) == 1, sus)
            # ∂Ψ∂F and the return mapping of ViscousPolyconvex do not allocate (∂∂Ψ∂FF does, see above): the update must not
            @test allocations(suP) == 0 && all(su -> allocations(su) == 0, sus)
            # the weak form with the updated states, both storages, against Gridap
            resk, jack = forms(Ah, Ph⁻); resg, jacg = forms(Ahg, Pg)
            x0 = 1e-5 .* randn(num_free_dofs(Uu))
            opk = KernelOperator(resk, jack, Uu, Vu)
            @test reldiff(Algebra.residual(opk, x0), Algebra.residual(gop(resg, jacg, Uu, Vu), x0)) < TOL
            # two time steps: kernel operator + kernel states vs Gridap operator + Gridap states
            nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-10, rtol=1e-10, verbose=0)
            xg = copy(x0); xk = copy(x0)
            for step in 1:2
                Algebra.solve!(xg, nls, gop(resg, jacg, Uu, Vu))
                Algebra.solve!(xk, nls, HK.update(opk, resk, jack))
                Fg = F ∘ ∇(FEFunction(Uu, xg))'; Fkh = F ∘ ∇(FEFunction(Uu, xk))'
                update_state!(P_updater, Pg, Fg, Fh⁻, Ahg...); update_state!(model, Ahg, Fg, Fh⁻)
                suP2 = update_states!(P_updater, Ph⁻, Fkh, Fh⁻, Ah...)
                @test suP2.expr == suP.expr                       # same generated function every step
                for (b, A) in zip(branches, Ah)
                    update_states!(branch_updater(b), A, Fkh, Fh⁻)
                end
                @test reldiff(xk, xg) < 1e-10
                @test maximum(norm, dense(Ph⁻) - dense(Pg)) <= 1e-9 * maximum(norm, dense(Pg))
                @test all(maximum(norm, dense(A) - dense(Ag)) <= 1e-9 * maximum(norm, dense(Ag)) for (A, Ag) in zip(Ah, Ahg))
            end
        end
    end
    @testset "hMRE + vacuum 2D (staggered problems of examples/hmre_vacuum_2d), Short_magnet.msh" begin
        # The three weak forms of the example at random states, without HyperFEM's drivers: spaces on a
        # sub-triangulation (magnetic potential on solid + vacuum), integrals on its parts with different
        # constitutive models (HardMagnetic2D in MagnetoMechModel, IdealMagnetic2D, HessianRegularization of
        # NonlinearMooneyRivlin2D_CV), interface traction from an FE function on the interface.
        geomodel = GmshDiscreteModel(joinpath(@__DIR__, "..", "models", "Short_magnet.msh"))
        μ0 = 4π * 1e-7
        params = [10.0, 12480.64495232286, 1.0, 2.0, 5195.545287237134, 0.2602717127043121, 2.0]
        modelmech = NonlinearMooneyRivlin2D(λ=(params[1] + params[2]) * 1e2, μ1=params[1] * 1e3, μ2=params[2], α1=params[3], α2=params[4]) +
                    TransverseIsotropy2D(μ=params[5], α1=params[6], α2=params[7])
        model_solid = MagnetoMechModel(magneto=HardMagnetic2D(μ0=μ0, αr=10e-3 / μ0, χe=0.0, χr=8.0), mechano=modelmech)
        model_vacuum_mag = IdealMagnetic2D(μ0=μ0, χe=0.0)
        model_vacuum_mech = HessianRegularization(mechano=NonlinearMooneyRivlin2D_CV(λ=1.0, μ1=1.0, μ2=0.0, α1=6.0, α2=1.0, γ=6.0), δ=1e-6)
        reffeφ = ReferenceFE(lagrangian, Float64, 1); reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
        Ωdomain = Triangulation(geomodel, tags=["vacuumb", "Beam"])
        Ωvacuum = Triangulation(geomodel, tags=["vacuumb"]); Ωsolid = Triangulation(geomodel, tags=["Beam"])
        dΩsolid = Measure(Ωsolid, 2); dΩvacuum = Measure(Ωvacuum, 2)
        Γsolid_int = BoundaryTriangulation(Ωsolid, tags="Interface"); dΓs = Measure(Γsolid_int, 4)
        Vφ = TestFESpace(Ωdomain, reffeφ; dirichlet_tags=["phibot", "phitop"])
        Uφ = TrialFESpace(Vφ, [0.3 * 1040.0, 0.0])
        Vs = TestFESpace(Ωsolid, reffeu; dirichlet_tags=["Fixed_Solid", "Fixed_Solid_x"], dirichlet_masks=[(true, true), (true, false)])
        Us = TrialFESpace(Vs, [VectorValue(0.0, 0.0), VectorValue(0.0, 0.0)])
        Vv = TestFESpace(Ωvacuum, reffeu; dirichlet_tags=["Fixed_Air", "Fixed_Air_x", "Interface"], dirichlet_masks=[(true, true), (true, false), (true, true)])
        Uv = TrialFESpace(Vv, [VectorValue(0.0, 0.0), VectorValue(0.0, 0.0), x -> VectorValue(1e-4 * x[2], 0.0)])
        # states of the other problems (auxiliary data), random but small
        uh_solid = FEFunction(Us, 1e-4 .* randn(num_free_dofs(Us)))
        uh_vacuum = FEFunction(Uv, 1e-4 .* randn(num_free_dofs(Uv)))
        φh = FEFunction(Uφ, 50.0 .* randn(num_free_dofs(Uφ)))
        Nh = interpolate_everywhere(x -> VectorValue(1.0, 0.0), TestFESpace(Ωsolid, reffeu))
        tmaxw = interpolate_everywhere(x -> VectorValue(0.1, -0.3 * x[1]), FESpace(Γsolid_int, reffeu))
        _, ∂Ψs∂F, ∂Ψs∂H0, ∂Ψs∂FF, _, ∂Ψs∂H0H0 = model_solid()
        _, _, ∂Ψv∂H0, _, _, ∂Ψv∂H0H0 = model_vacuum_mag()
        _, ∂Ψvm∂F, ∂Ψvm∂FF = model_vacuum_mech()
        F, _, _ = get_Kinematics(Kinematics(Mechano, Solid)); ℋ₀ = get_Kinematics(Kinematics(Magneto, Solid))
        Λ = 0.4
        uhs = uh_solid * Λ; uhv = uh_vacuum * Λ               # staggered interpolation of the states (auxiliary expressions)
        res_mag(φ, vφ) = -1.0 * ∫((∇(vφ) ⋅ (∂Ψs∂H0 ∘ (F ∘ (∇(uhs)'), ℋ₀ ∘ (∇(φ)), Nh))))dΩsolid -
                         ∫((∇(vφ) ⋅ (∂Ψv∂H0 ∘ (F ∘ (∇(uhv)'), ℋ₀ ∘ (∇(φ))))))dΩvacuum
        jac_mag(φ, dφ, vφ) = ∫(∇(vφ)' ⋅ ((∂Ψs∂H0H0 ∘ (F ∘ (∇(uhs)'), ℋ₀ ∘ (∇(φ)), Nh)) ⋅ ∇(dφ)))dΩsolid +
                             ∫(∇(vφ)' ⋅ ((∂Ψv∂H0H0 ∘ (F ∘ (∇(uhv)'), ℋ₀ ∘ (∇(φ)))) ⋅ ∇(dφ)))dΩvacuum
        res_mech(u, v) = ∫((∇(v)' ⊙ (∂Ψs∂F ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φh)), Nh))))dΩsolid + ∫((v ⋅ tmaxw))dΓs
        jac_mech(u, du, v) = ∫(∇(v)' ⊙ ((∂Ψs∂FF ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φh)), Nh)) ⊙ (∇(du)')))dΩsolid
        res_vac(u, v) = ∫((∇(v)' ⊙ (∂Ψvm∂F ∘ (F ∘ (∇(u)')))))dΩvacuum
        jac_vac(u, du, v) = ∫(∇(v)' ⊙ ((∂Ψvm∂FF ∘ (F ∘ (∇(u)'))) ⊙ (∇(du)')))dΩvacuum
        for (name, res, jac, U, V, scale) in (("magnetic", res_mag, jac_mag, Uφ, Vφ, 50.0),
                                              ("solid", res_mech, jac_mech, Us, Vs, 1e-4),
                                              ("vacuum", res_vac, jac_vac, Uv, Vv, 1e-4))
            x = scale .* randn(num_free_dofs(U))
            op = KernelOperator(res, jac, U, V)
            @test isempty(op.fallback.rtrians) && isempty(op.fallback.jtrians)
            c = check(res, jac, U, V, x; op)
            @test c.res < TOL && c.jac < TOL
        end
    end

    @testset "drivers with backend=:compiled: Dirichlet projection (ProjectDirichlet=true)" begin
        CM = HyperFEM.ComputationalModels
        if !(isdefined(CM, :dirichlet_projection!) || isdefined(CM, :dirichlet_projection))
            @info "HyperFEM without the kernel backend of the drivers (needs HyperFEM ≥ 0.0.7): driver tests skipped"
        else
            # 1. The projection increment of a step, kernel vs Gridap, at a nonzero state; twice, so that the
            #    second call reuses the kernel operator of the first (`update`).
            geometry = CartesianDiscreteModel((0, 0.05, 0, 0.005, 0, 0.002), (5, 2, 2))
            labels = get_face_labeling(geometry)
            add_tag_from_tags!(labels, "fixed", CartesianTags.face0YZ⁺)
            add_tag_from_tags!(labels, "moving", CartesianTags.face1YZ⁺)
            physmodel = MooneyRivlin3D(λ=3.0, μ1=1.0, μ2=0.0, ρ=1.0)
            Ω = Triangulation(geometry); dΩ = Measure(Ω, 2)
            D_bc = BoundaryConditions(BoundaryCondition("fixed", [0.0, 0.0, 0.0]),
                                      BoundaryCondition("moving", [0.08, 0.0, 0.0], EvolutionFunctions.ramp()))
            reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, 1)
            k = Kinematics(Mechano, Solid)
            res(Λ) = (u, v) -> residual(physmodel, k, u, v, dΩ)
            jac(Λ) = (u, du, v) -> jacobian(physmodel, k, u, du, v, dΩ)
            models = map((:lazy, :compiled)) do backend
                V = TestFESpace(Ω, reffeu, D_bc, conformity=:H1); U = TrialFESpace(V, D_bc, 0.0)
                StaticNonlinearModel(res, jac, U, V, D_bc; nls=NewtonRaphson(LUSolver(); verbose=0), backend=backend)
            end
            Random.seed!(3)
            x = 1e-3 .* randn(num_free_dofs(models[1].spaces[1]))
            for (Λ, ∆Λ) in ((0.1, 0.1), (0.3, 0.2))
                duhs = map(models) do m
                    m.caches.x .= x
                    CM.get_dirichlet_preconditioner(m, Λ, ∆Λ)
                end
                dg, dk = map(get_free_dof_values, duhs)
                @test norm(dg) > 0
                @test reldiff(dk, dg) < 1e-9     # two LU solves of the same stiff system (rounding of the LU)
                # the linear systems themselves: Gridap's AffineFEOperator vs the kernel operator at du = 0
                mg, mk = models
                uh = FEFunction(mg.spaces[1], x)
                ∆Ug, ∆Uk = mg.spaces[3], mk.spaces[3]
                aop = AffineFEOperator((du, v) -> jac(Λ - ∆Λ)(uh, du, v), v -> -1.0 * res(Λ - ∆Λ)(uh, v), ∆Ug, mg.spaces[2])
                kop = mk.caches.projection
                z = zero_initial_guess(kop)
                @test reldiff(nonzeros(jacobian(kop, z)), nonzeros(get_matrix(aop))) < TOL
                @test reldiff(-residual(kop, z), get_vector(aop)) < TOL
                @test reldiff(get_dirichlet_dof_values(get_fe_space(duhs[2])), get_dirichlet_dof_values(get_fe_space(duhs[1]))) < TOL
            end
            @test models[2].caches.projection isa KernelOperator
            @test isempty(models[2].caches.projection.fallback.rtrians) && isempty(models[2].caches.projection.fallback.jtrians)

            # 2. HyperFEM's own simulations with the projection, with both backends: simulations/runtests.jl
        end
    end

    @testset "StaticLinearModel with backend=:compiled (matrix, load, Dirichlet lift, re-assembly, several loads, Stokes)" begin
        CM = HyperFEM.ComputationalModels
        if !isdefined(CM, :linear_system)
            @info "HyperFEM without the kernel backend of StaticLinearModel (needs HyperFEM ≥ 0.0.7): linear driver tests skipped"
        else
            # Linear elasticity, Dirichlet lift (nonzero displacement on "moving") and body load.
            geometry = CartesianDiscreteModel((0, 1.0, 0, 0.2, 0, 0.2), (5, 2, 2))
            labels = get_face_labeling(geometry)
            add_tag_from_tags!(labels, "fixed", CartesianTags.face0YZ⁺)
            add_tag_from_tags!(labels, "moving", CartesianTags.face1YZ⁺)
            Ω = Triangulation(geometry); dΩ = Measure(Ω, 2)
            D_bc = BoundaryConditions(BoundaryCondition("fixed", [0.0, 0.0, 0.0]),
                                      BoundaryCondition("moving", [0.01, 0.002, 0.0], EvolutionFunctions.ramp()))
            reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, 1)
            λ, μ = 3.0, 1.0
            ε(∇u) = 0.5 * (∇u + ∇u')
            σ(e) = λ * tr(e) * one(e) + 2μ * e
            a(u, v) = ∫((σ ∘ (ε ∘ ∇(u))) ⊙ (ε ∘ ∇(v)))dΩ
            f = VectorValue(0.0, 0.0, -1.0)
            l(v) = ∫(f ⋅ v)dΩ
            l2(v) = ∫(VectorValue(0.3, 0.0, 0.1) ⋅ v)dΩ
            spaces(Λ=1.0) = (V = TestFESpace(Ω, reffe, D_bc, conformity=:H1); (TrialFESpace(V, D_bc, Λ), V))
            system(m) = (m.caches.K, m.caches.b)
            @test_throws ErrorException StaticLinearModel(l, a, spaces()..., D_bc; backend=:other)

            # 1. lf + bf: K and b at construction, solution, re-assembly with other Dirichlet values
            mg, mk = map((:lazy, :compiled)) do backend
                StaticLinearModel(l, a, spaces()..., D_bc; backend)
            end
            Kg, bg = system(mg); Kk, bk = system(mk)
            @test norm(bg) > 0
            @test reldiff(nonzeros(Kk), nonzeros(Kg)) < TOL
            @test reldiff(bk, bg) < TOL
            @test mk.caches.op isa KernelOperator
            @test isempty(mk.caches.op.fallback.rtrians) && isempty(mk.caches.op.fallback.jtrians)
            xg = solve!(mg; Assembly=false); xk = solve!(mk; Assembly=false)
            @test norm(xg) > 0 && reldiff(xk, xg) < 1e-10
            for m in (mg, mk)
                TrialFESpace!(m.spaces[1], D_bc, 0.5)       # the lift of the new Dirichlet values is re-read
            end
            xg = copy(solve!(mg; Assembly=true)); xk = copy(solve!(mk; Assembly=true))
            @test reldiff(bk, bg) < TOL && reldiff(nonzeros(Kk), nonzeros(Kg)) < TOL
            @test reldiff(xk, xg) < 1e-10
            @test mk.caches.op isa KernelOperator

            # 2. bf only: the matrix, the right-hand side given to solve! and the functor (Assembly=true)
            mg, mk = map((:lazy, :compiled)) do backend
                StaticLinearModel(a, spaces()..., D_bc; backend)
            end
            Kg, bg = system(mg); Kk, bk = system(mk)
            @test reldiff(nonzeros(Kk), nonzeros(Kg)) < TOL
            @test iszero(bg) && iszero(bk)
            Random.seed!(5)
            r = randn(length(bg))
            xg = copy(solve!(mg, copy(r))); xk = copy(solve!(mk, copy(r)))
            @test reldiff(xk, xg) < 1e-10
            xg = copy(mg(copy(r); Assembly=true)); xk = copy(mk(copy(r); Assembly=true))
            @test reldiff(xk, xg) < 1e-10
            @test reldiff(nonzeros(Kk), nonzeros(Kg)) < TOL

            # 2b. the functor with an FEFunction: its right-hand side is ∫(xh * v)dΩ (scalar space), the matrix
            #     is re-assembled with the backend and the solution overwrites the free values of xh
            reffes = ReferenceFE(lagrangian, Float64, 1)
            Ds = BoundaryConditions(BoundaryCondition("fixed", 0.0), BoundaryCondition("moving", 1.0))
            κ(∇u) = 2.0 * ∇u
            as(u, v) = ∫((κ ∘ ∇(u)) ⋅ ∇(v))dΩ
            mg, mk = map((:lazy, :compiled)) do backend
                Vs = TestFESpace(Ω, reffes, Ds, conformity=:H1)
                StaticLinearModel(as, TrialFESpace(Vs, Ds, 1.0), Vs, Ds; backend)
            end
            Random.seed!(7)
            r = randn(num_free_dofs(mg.spaces[1]))
            xg = copy(mg(FEFunction(mg.spaces[1], copy(r)); Measure=dΩ, Assembly=true))
            xk = copy(mk(FEFunction(mk.spaces[1], copy(r)); Measure=dΩ, Assembly=true))
            @test norm(xg) > 0 && reldiff(xk, xg) < 1e-10
            @test reldiff(nonzeros(mk.caches.K), nonzeros(mg.caches.K)) < TOL
            @test reldiff(mk.caches.b, mg.caches.b) < TOL
            @test mk.caches.op isa KernelOperator

            # 3. several linear forms on repeated spaces (load vectors without Dirichlet lift)
            mg, mk = map((:lazy, :compiled)) do backend
                StaticLinearModel([l, l2], a, spaces()..., D_bc; backend)
            end
            xg = solve!(mg); xk = solve!(mk)
            @test all(norm.(CM.blocks(xg)) .> 0)
            @test reldiff(xk, xg) < 1e-10
            @test reldiff(mk.caches.b, mg.caches.b) < TOL     # last load vector (l2), no lift
            xg = solve!(mg; Assembly=false); xk = solve!(mk; Assembly=false)
            @test reldiff(xk, xg) < 1e-10

            # 4. HyperFEM's Stokes simulation (multi-field, linear form returning 0.0) with both backends:
            #    simulations/runtests.jl
        end
    end

    @testset "HyperFEM's state interface with backend=:compiled (CellState(model, dΩ; backend), update_state!(model, ...; backend))" begin
        PM = HyperFEM.PhysicalModels
        if !isdefined(PM, :new_state)
            @info "HyperFEM without the kernel backend of the states (needs HyperFEM ≥ 0.0.7): state interface tests skipped"
        else
            geometry = CartesianDiscreteModel((0, 0.05, 0, 0.005, 0, 0.002), (5, 2, 2))
            model = GeneralizedMaxwell(NeoHookean3D(μ=13446.1, λ=0.0) + VolumetricEnergy(λ=2.5e6),
                                       ViscousPolyconvex(μ=11053.7, τ=151.68), ViscousPolyconvex(μ=30480.1, τ=7.0731))
            em = ElectroMechModel(IdealDielectric(ε=1.0), model)          # the kwarg goes through the composed models
            update_time_step!(model, 0.01)
            Ω = Triangulation(geometry); dΩ = Measure(Ω, 2)
            F, _, _ = get_Kinematics(Kinematics(Mechano, Solid))
            V = TestFESpace(geometry, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), conformity=:H1)
            uh = FEFunction(V, 1e-3 .* randn(num_free_dofs(V))); uh⁻ = FEFunction(V, 1e-3 .* randn(num_free_dofs(V)))
            Fh = F ∘ ∇(uh)'; Fh⁻ = F ∘ ∇(uh⁻)'
            dense(s::CellState) = reduce(hcat, s.values); dense(s::KernelState) = s.values
            same(a, b) = maximum(norm, dense(a) - dense(b)) <= 1e-12 * max(1.0, maximum(norm, dense(b)))
            Ag = CellState(model, dΩ)                                    # Gridap's storage and loop (reference)
            Ak = CellState(model, dΩ; backend=:compiled)                   # KernelStates: the engine, with any backend
            Ac = CellState(model, dΩ)                                    # Gridap's storage, the engine's loop
            Ae = CellState(em, dΩ; backend=:compiled)
            @test all(s -> s isa KernelState, Ak) && all(s -> s isa KernelState, Ae) && all(s -> s isa CellState, Ac)
            @test_throws ErrorException CellState(model, dΩ; backend=:other)
            for _ in 1:2
                update_state!(model, Ag, Fh, Fh⁻)
                update_state!(model, Ak, Fh, Fh⁻)
                update_state!(model, Ac, Fh, Fh⁻; backend=:compiled)
                update_state!(em, Ae, Fh, nothing, Fh⁻; backend=:compiled)        # (state, F, E, Fn)
                @test all(same(a, g) for (a, g) in zip(Ak, Ag))
                @test all(same(a, g) for (a, g) in zip(Ac, Ag))
                @test all(same(a, g) for (a, g) in zip(Ae, Ag))
            end
            @test_throws ErrorException update_state!(model, Ac, Fh, Fh⁻; backend=:other)
        end
    end
end

# HyperFEM's simulation tests (test/SimulationsTests of HyperFEM.jl), with both backends:
# every simulation runs with :lazy and :compiled, the reference values are checked with each one and the kernel
# results are compared with Gridap's. They live in HyperFEM (their final place) and are included here, at top
# level (they load packages and define constants). Skipped when the HyperFEM next to this package has no kernel
# backend (e.g. main).
let CM = HyperFEM.ComputationalModels, PM = HyperFEM.PhysicalModels
    global hyperfem_sims = joinpath(pkgdir(HyperFEM), "test", "SimulationsTests", "runtests.jl")
    global hyperfem_has_backend = isdefined(CM, :dirichlet_projection) && isdefined(CM, :linear_system) &&
                                  isdefined(PM, :new_state) && isfile(hyperfem_sims) &&
                                  occursin("run_backends", read(hyperfem_sims, String))
end
if hyperfem_has_backend
    include(hyperfem_sims)
else
    @info "HyperFEM without the kernel backend (needs HyperFEM ≥ 0.0.7): HyperFEM's simulation tests skipped"
end

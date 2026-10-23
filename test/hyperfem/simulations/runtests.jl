# HyperFEM's seven simulation tests (test/SimulationsTests of HyperFEM.jl, copied here with a `backend`
# keyword) run with both backends: the Gridap run must reproduce HyperFEM's reference values, and the
# kernel run must reproduce the Gridap run. Needs HyperFEM on the branch kernel-backend (drivers, states and
# linear models with `backend`); otherwise the whole set is skipped with a message.
# Run alone with: julia -t auto --project=benchmark test/hyperfem/simulations/runtests.jl (after the setup
# of test/hyperfem/runtests.jl), or through test/hyperfem/runtests.jl.

if !@isdefined(reldiff)
    using Test, HyperFEMKernels, HyperFEM, Gridap, LinearAlgebra
    reldiff(a, b) = norm(a - b) / max(norm(b), eps())
end

@testset "HyperFEM simulations with backend=:gridap and backend=:kernel" begin
    CM, PM = HyperFEM.ComputationalModels, HyperFEM.PhysicalModels
    if !(isdefined(CM, :dirichlet_projection) && isdefined(CM, :linear_system) && isdefined(PM, :new_state))
        @info "HyperFEM without the kernel backend (branch kernel-backend): simulation tests skipped"
    else
        for f in ("ViscoElastic", "StaticMechanicalDirichlet", "StaticMechanicalNeumann", "StaggeredElectroMechanical",
                  "StaggeredViscoElectric", "Stokes", "MeshMovement", "TurekNavierStokes", "HMREVacuum")
            include(joinpath(@__DIR__, f * "Simulation.jl"))
        end
        TOLS = 1e-10          # kernel vs Gridap: same problem, two LU solves

        @testset "ViscoElastic (StaticNonlinearModel, CellStates, Dirichlet projection)" begin
            λg, σg = visco_elastic_simulation()
            @test σg[end] ≈ 22152.0463
            λk, σk = visco_elastic_simulation(backend=:kernel)
            @test reldiff(σk, σg) < TOLS && reldiff(λk, λg) < TOLS
        end

        @testset "StaticMechanicalDirichlet (StaticNonlinearModel, Dirichlet projection)" begin
            xg = static_mechanical_dirichlet_simulation()
            @test norm(xg) ≈ 0.27148722276
            xk = static_mechanical_dirichlet_simulation(backend=:kernel)
            @test reldiff(xk, xg) < TOLS
        end

        @testset "StaticMechanicalNeumann (StaticNonlinearModel, Neumann load through the model)" begin
            xg = static_mechanical_neumann_simulation()       # returns (x, convergence flag), as HyperFEM's test uses it
            @test norm(xg) ≈ 1.000148588846
            xk = static_mechanical_neumann_simulation(backend=:kernel)
            @test reldiff(xk[1], xg[1]) < TOLS && xk[2] == xg[2]
        end

        @testset "StaggeredElectroMechanical (StaggeredModel of two StaticNonlinearModels, Q2)" begin
            φg, ug = staggered_electro_mechanical_simulation()
            @test norm(ug[1]) ≈ 0.01492384116
            @test norm(φg[1]) ≈ 0.00023052926
            φk, uk = staggered_electro_mechanical_simulation(backend=:kernel)
            # same Newton histories (iterations and residuals to 3 digits in all 30 solves, checked by hand); the
            # thin Q2 plate (λ/μ = 10, aspect 50) is ill conditioned and the mechanical solves stop at a relative
            # residual of 6e-3 (atol 1e-10 on a residual of 1e-9), so the LU rounding shows up at 2e-10 in u
            @test reldiff(uk[1], ug[1]) < 1e-8 && reldiff(φk[1], φg[1]) < TOLS
        end

        @testset "StaggeredViscoElectric (operators per time step through build_operator, states through the backend)" begin
            tg, uzg = staggered_visco_electric_simulation()
            @test uzg[end] ≈ 0.002914376983
            @test norm(uzg) ≈ 0.004558785342
            tk, uzk = staggered_visco_electric_simulation(backend=:kernel)
            @test reldiff(uzk, uzg) < TOLS
        end

        @testset "Stokes (StaticLinearModel, multi-field)" begin
            # K and b agree to 1e-16, but the saddle-point LU differs at 1e-7: the kernel pattern stores the (empty)
            # p-p block, which changes UMFPACK's pivoting (residual of Gridap's solution 5e-8, of the kernel's 4e-11)
            xg = stokes_simulation()
            @test norm(xg) ≈ 4199.925167995936
            xk = stokes_simulation(backend=:kernel)
            @test reldiff(xk, xg) < 1e-6
        end

        @testset "MeshMovement (StaticNonlinearModel, injectivity line search, Dirichlet projection, interface stages)" begin
            xg = mesh_movement_simulation()
            @test isapprox(norm(xg), 2.8132015601158087, rtol=1e-14)
            xk = mesh_movement_simulation(backend=:kernel)
            @test reldiff(xk, xg) < TOLS
            xki = mesh_movement_simulation(backend=:kernel, interface_field=true)
            @test reldiff(xki, xg) < TOLS
        end

        # Simulation tests of this repository (not copies of HyperFEM's)
        @testset "Turek DFG 2D-1 Navier-Stokes (StaticLinearModel per time step, SUPG/PSPG/LSIC, BDF2 Oseen)" begin
            rg = turek_navier_stokes_simulation()
            @test isapprox(norm(rg.x), 13.67687986714, rtol=1e-8)       # measured with Gridap (12 steps of 0.5 s, h = 0.02)
            @test isapprox(rg.cD, 5.5795, rtol=0.03) && isapprox(rg.Δp, 0.1175, rtol=0.02)   # DFG reference values (fine meshes, P2-P1)
            rk = turek_navier_stokes_simulation(backend=:kernel)
            @test reldiff(rk.x, rg.x) < TOLS
            @test reldiff(rk.cD_history, rg.cD_history) < TOLS
        end

        @testset "hMRE + vacuum 2D (StaggeredModel of three problems, Coupling prescription, Maxwell traction)" begin
            rg = hmre_vacuum_simulation()
            @test isapprox(norm(rg.φ), 2.0940145799e4, rtol=1e-6)         # measured with Gridap (φbot = 520 V, 4 steps)
            @test isapprox(norm(rg.u_solid), 1.0426282896e-1, rtol=1e-6)
            @test isapprox(norm(rg.u_vacuum), 2.4014448097e-1, rtol=1e-6)
            rk = hmre_vacuum_simulation(backend=:kernel)
            @test reldiff(rk.φ, rg.φ) < TOLS && reldiff(rk.u_solid, rg.u_solid) < TOLS && reldiff(rk.u_vacuum, rg.u_vacuum) < TOLS
        end
    end
end

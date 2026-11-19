# Times HyperFEM's simulation tests with the registered HyperFEM (General registry, 0.3.x) and the registered
# HyperFEMKernels, with backend = :lazy and :compiled, as a baseline for benchmark/simulations/current.jl (same
# simulations, same columns). Environment: a copy of Project_official.toml instantiated by this script in
# benchmark/simulations/env_official (git-ignored); it is rebuilt when Project_official.toml changes. Each
# simulation runs twice; the second (warm) time is reported.
#   julia -t auto benchmark/simulations/official.jl
using Pkg
env = joinpath(@__DIR__, "env_official")
spec = joinpath(@__DIR__, "Project_official.toml")
envproject = joinpath(env, "Project.toml")
if !isfile(joinpath(env, "Manifest.toml")) || !isfile(envproject) || read(spec, String) != read(envproject, String)
    rm(env; recursive=true, force=true)
    mkpath(env); cp(spec, envproject; force=true)
    Pkg.activate(env); Pkg.instantiate()
else
    Pkg.activate(env)
end
using HyperFEMKernels, HyperFEM, Gridap, LinearAlgebra, Printf
deps = Pkg.dependencies()
ver(uuid) = deps[Base.UUID(uuid)].version
println("HyperFEM $(ver("ddbf423a-3df4-409e-9685-3dc416cae46c")), HyperFEMKernels $(ver("c52e38e0-a033-47df-8a03-a6c4ac881830")) (registered), Julia $VERSION, $(Threads.nthreads()) threads")
sd = joinpath(pkgdir(HyperFEM), "test", "SimulationsTests")
for f in ("ViscoElastic", "StaticMechanicalDirichlet", "StaticMechanicalNeumann", "StaggeredElectroMechanical",
          "StaggeredViscoElectric", "Stokes", "MeshMovement", "TurekNavierStokes", "HMREVacuum")
    include(joinpath(sd, f * "Simulation.jl"))
end
sims = [("ViscoElastic", b -> visco_elastic_simulation(backend=b), r -> r[2][end]),
        ("StaticMechanicalDirichlet", b -> static_mechanical_dirichlet_simulation(backend=b), r -> norm(r)),
        ("StaticMechanicalNeumann", b -> static_mechanical_neumann_simulation(backend=b), r -> norm(r[1])),
        ("StaggeredElectroMechanical", b -> staggered_electro_mechanical_simulation(backend=b), r -> norm(r[2][1])),
        ("StaggeredViscoElectric", b -> staggered_visco_electric_simulation(backend=b), r -> norm(r[2])),
        ("Stokes", b -> stokes_simulation(backend=b), r -> norm(r)),
        ("MeshMovement", b -> mesh_movement_simulation(backend=b), r -> norm(r)),
        ("TurekNavierStokes", b -> turek_navier_stokes_simulation(backend=b), r -> r.cD),
        ("HMREVacuum", b -> hmre_vacuum_simulation(backend=b), r -> norm(r.u_solid))]
println("simulation,gridap_cold_s,gridap_warm_s,kernel_cold_s,kernel_warm_s,value_gridap,value_kernel")
for (name, f, val) in sims
    tg1 = @elapsed rg = f(:lazy); tg2 = @elapsed rg = f(:lazy)
    tk1 = @elapsed rk = f(:compiled); tk2 = @elapsed rk = f(:compiled)
    @printf("%s,%.3f,%.3f,%.3f,%.3f,%.10g,%.10g\n", name, tg1, tg2, tk1, tk2, val(rg), val(rk))
end
println("FIN OFFICIAL")

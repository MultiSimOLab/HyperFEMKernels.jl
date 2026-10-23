# Times HyperFEM's simulation tests with the registered HyperFEM (General registry, v0.0.6), as a baseline for
# benchmark/simulations/current.jl. Environment: a copy of Project_official.toml instantiated by this script in
# benchmark/simulations/env_official (git-ignored). Each simulation runs twice; the second (warm) time is reported.
#   julia -t auto benchmark/simulations/official.jl
using Pkg
env = joinpath(@__DIR__, "env_official")
if !isfile(joinpath(env, "Manifest.toml"))
    mkpath(env); cp(joinpath(@__DIR__, "Project_official.toml"), joinpath(env, "Project.toml"); force=true)
    Pkg.activate(env); Pkg.instantiate()
else
    Pkg.activate(env)
end
using HyperFEM, Gridap, LinearAlgebra, Printf
using HyperFEM: projdir
ver = Pkg.dependencies()[Base.UUID("ddbf423a-3df4-409e-9685-3dc416cae46c")].version
println("HyperFEM $ver (registered), Julia $VERSION, $(Threads.nthreads()) threads")
data = projdir("test", "data")
include(joinpath(data, "ViscoElasticFastSimulation.jl"))        # the visco problem of the current test (reference 22152.0463)
include(joinpath(data, "StaticMechanicalDirichletSimulation.jl"))
include(joinpath(data, "StaticMechanicalNeumannSimulation.jl"))
include(joinpath(data, "StaggeredElectroMechanicalSimulation.jl"))
include(joinpath(data, "StaggeredViscoElectricSimulation.jl"))
include(joinpath(data, "Stokes.jl"))
sims = [("ViscoElastic", () -> visco_elastic_fast_simulation(t_end=2, writevtk=false, verbose=false), r -> r[2][end]),
        ("StaticMechanicalDirichlet", () -> static_mechanical_dirichlet_simulation(writevtk=false, verbose=false), r -> norm(r)),
        ("StaticMechanicalNeumann", () -> static_mechanical_neumann_simulation(writevtk=false, verbose=false), r -> norm(r)),
        ("StaggeredElectroMechanical", () -> staggered_electro_mechanical_simulation(is_vtk=false, verbose=false), r -> norm(r[2][1])),
        ("StaggeredViscoElectric", () -> staggered_visco_electric_simulation(t_end=0.5, writevtk=false, verbose=false), r -> norm(r[2])),
        ("Stokes", () -> Stokes(writevtk=false, verbose=false), r -> norm(r))]
println("simulation,official_cold_s,official_warm_s,value")
for (name, f, val) in sims
    t1 = @elapsed r = f()
    t2 = @elapsed r = f()
    @printf("%s,%.3f,%.3f,%.10g\n", name, t1, t2, val(r))
end
println("FIN OFFICIAL")

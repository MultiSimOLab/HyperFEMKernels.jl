# Times HyperFEM's simulation tests (test/SimulationsTests of the HyperFEM next to this package, branch
# kernel-backend) and backend = :lazy and :compiled. Each run twice; the second (warm) time is
# reported, with the convergence value, so that it can be compared with official.jl.
#   julia -t auto --project=benchmark benchmark/simulations/current.jl
using HyperFEMKernels, HyperFEM, Gridap, LinearAlgebra, Printf
println("HyperFEM $(pkgdir(HyperFEM)) (git: $(strip(read(`git -C $(pkgdir(HyperFEM)) rev-parse --short HEAD`, String)))), Julia $VERSION, $(Threads.nthreads()) threads")
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
println("FIN CURRENT")

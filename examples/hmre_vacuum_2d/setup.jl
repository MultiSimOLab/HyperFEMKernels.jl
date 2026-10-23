# Setup of the hMRE + vacuum 2D example (see hmre_vacuum_2d.jl for the description): models, spaces, coupling,
# weak forms and the staggered computational model, built for a backend. The builder `hmre_vacuum_model` lives
# in HyperFEM's simulation test test/SimulationsTests/HMREVacuumSimulation.jl (branch kernel-backend), which
# runs a reduced version of this example with both backends; the example builds the full problem with it.
using HyperFEM, HyperFEMKernels
using Gridap, GridapGmsh, GridapSolvers
using GridapSolvers.NonlinearSolvers
using Gridap.FESpaces
using Gridap.CellData
using LinearAlgebra, Printf

include(joinpath(pkgdir(HyperFEM), "test", "SimulationsTests", "HMREVacuumSimulation.jl"))

"The three staggered problems of the example built for `backend` (`φbot`: potential at the bottom of the box, 0.2 · 5200 in the original)."
build(backend::Symbol; kw...) = hmre_vacuum_model(backend; kw...)

ENV["HFK_DEBUG_LINEAR"] = "1"
using HyperFEMKernels, Gridap, Gridap.FESpaces, Gridap.Algebra, LinearAlgebra, Random
const HK = HyperFEMKernels; const TOL = 1e-12
using HyperFEMKernels.StaticArrays
include(joinpath(@__DIR__, "..", "test", "common.jl"))
e = elasticity_3d()
Cof(F) = C_svk()
jacC(u, du, v) = ∫(∇(v)' ⊙ (dbl_contract ∘ (Cof ∘ (Fk ∘ (∇(u)')), ∇(du)')))e.dΩ
opC = KernelOperator(e.res, jacC, e.U, e.V)
println(all(g -> g.lin !== nothing, opC.jgroups))

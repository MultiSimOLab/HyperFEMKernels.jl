"""
    HyperFEMKernels

Fused, threaded evaluation and assembly of Gridap weak forms.

Gridap keeps ownership of the mesh, `FESpace`s, Dirichlet data (including masks),
DOF numbering, `Measure`s, `FEFunction`s and post-processing. This package only
replaces the evaluation of `∫(...)dΩ` and the sparse assembly with:

1. static per-cell-type tables built from Gridap (`tables.jl`);
2. a fused element kernel that evaluates kinematics and the constitutive law once
   per quadrature point and contracts with all basis functions (`kernels.jl`);
3. a precomputed sparsity pattern and nonzero positions, Dirichlet lifting and
   allocation-free assembly (`assembly.jl`);
4. cell colouring for race-free threaded assembly (`coloring.jl`);
5. auxiliary fields evaluated at quadrature points (`fields.jl`);
6. built-in verification against Gridap (`verify.jl`).

Status: scaffolding only (phase 0). See `ESTADO.md`.
"""
module HyperFEMKernels

using LinearAlgebra
using SparseArrays
using StaticArrays
using ForwardDiff
using Gridap

include("tables.jl")
include("kernels.jl")
include("assembly.jl")
include("coloring.jl")
include("fields.jl")
include("verify.jl")

end # module

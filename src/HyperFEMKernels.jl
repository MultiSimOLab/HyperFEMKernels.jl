"""
    HyperFEMKernels

Fused, threaded evaluation and assembly of Gridap weak forms.

Gridap keeps ownership of the mesh, FE spaces, Dirichlet data, numbering, measures and
post-processing. The weak form is captured by calling the user's `res`/`jac` with probe leaves
and reading the lazy tree Gridap builds (`probes.jl`, `tree.jl`); it is translated into a
pointwise Julia function (`codegen.jl`) evaluated by fused kernels (`tables.jl`, `kernels.jl`)
and assembled in parallel with a precomputed sparsity pattern (`assembly.jl`, `coloring.jl`).
Integrals on a skeleton (interior facets, plus/minus/jump/mean) are assembled by facet kernels
that see both cells (`skeleton.jl`). Internal variables (`CellState`/`KernelState`) are updated by
the same kind of kernels (`states.jl`). `check` compares with Gridap (`verify.jl`).
"""
module HyperFEMKernels

using LinearAlgebra
using SparseArrays
using StaticArrays
using ForwardDiff
using RuntimeGeneratedFunctions
using Gridap
using Gridap.Arrays
using Gridap.Fields
using Gridap.CellData
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.TensorValues
using Gridap.Algebra

import Gridap.Arrays: CompressedArray, LazyArray, Broadcasting, testitem, array_cache, getindex!, evaluate
import Gridap.Fields: BroadcastingFieldOpMap, IntegrationMap
import Gridap.CellData: CellField, CellPoint, DomainContribution, ReferenceDomain
import Gridap.TensorValues: MultiValue, meas, outer
import Gridap.MultiField: MultiFieldStyle, ConsecutiveMultiFieldStyle
import Gridap.Algebra: NonlinearOperator
import Gridap.Helpers: first_and_tail

RuntimeGeneratedFunctions.init(@__MODULE__)

export KernelOperator, update, check, allocations, MatrixFreeJacobian, prepare!
export KernelState, KernelUpdater, update_states!

include("probes.jl")
include("tree.jl")
include("codegen.jl")
include("tables.jl")
include("linearmaps.jl")
include("kernels.jl")
include("assembly.jl")
include("coloring.jl")
include("skeleton.jl")
include("tensorproduct.jl")
include("matrixfree.jl")
include("states.jl")
include("verify.jl")

end # module

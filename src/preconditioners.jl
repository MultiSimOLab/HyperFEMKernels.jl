# Preconditioner slots for the matrix-free Jacobian.
#
# A slot is a configuration object (`LaggedAssembled`) and a state built by
# `preconditioner_setup(slot, mf)` at the first numerical setup of a `MatrixFreeSolver`, refreshed
# by `preconditioner_update!(state, mf)` at every later one (every Newton iteration) and applied by
# `LinearAlgebra.ldiv!(y, state, r)` inside the Krylov method. `mf` is the prepared operator at the
# current state (`mf.x`), so the slots never see a matrix unless they assemble one themselves.
#
# `numerical_setup_state(ss, A, x)` / `numerical_setup_state!(ns, A, x)` build the setup of the
# inner solver of `LaggedAssembled` with the state available (HyperFEM's PETSc solver takes the
# rigid-body modes from it); here they fall back to Gridap's `numerical_setup`/`numerical_setup!`,
# and HyperFEM's extension forwards them to its own.

"Setup of a linear solver on the matrix `A` at the state `x` (Gridap's `numerical_setup` by default)."
numerical_setup_state(ss::Algebra.SymbolicSetup, A, x) = numerical_setup(ss, A)
"Update of that setup (Gridap's `numerical_setup!` by default)."
numerical_setup_state!(ns::Algebra.NumericalSetup, A, x) = numerical_setup!(ns, A)

"Supertype of the preconditioner slots of `MatrixFreeSolver`."
abstract type OperatorPreconditioner end

function preconditioner_setup end
function preconditioner_update! end

# ---------------------------------------------------------------------------------------------
# LaggedAssembled: the tangent assembled every k updates, a Gridap solver on it

"""
    LaggedAssembled(solver; every = 3)

Preconditioner slot: the tangent is assembled (through the sparsity pattern of the operator) and
`solver` — any Gridap `LinearSolver`: `LUSolver()`, HyperFEM's `PETScSolver`/`AMGSolver` — is set
up on it at the first setup and then every `every` updates (Newton iterations); in between, the
Krylov method applies the matrix-free product with the preconditioner of a previous iterate. With
`every = 1` the preconditioner is exact at every iteration and the matrix-free product only saves
the matrix–vector product's memory traffic. The inner solver should not lag on its own (`reuse = 1`).
"""
struct LaggedAssembled{LS<:Algebra.LinearSolver} <: OperatorPreconditioner
    solver::LS
    every::Int
    function LaggedAssembled(solver::Algebra.LinearSolver; every::Int=3)
        every ≥ 1 || throw(ArgumentError("HyperFEMKernels: LaggedAssembled: every must be ≥ 1"))
        new{typeof(solver)}(solver, every)
    end
end

mutable struct LaggedAssembledState{P<:LaggedAssembled,NS}
    prec::P
    A::SparseMatrixCSC{Float64,Int}
    ns::NS
    nupdate::Int
    nbuild::Int
end

function preconditioner_setup(p::LaggedAssembled, mf::MatrixFreeJacobian)
    op = mf.op
    A = copy(sparsity_pattern(op))
    jacobian!(A, op, mf.x)
    ns = numerical_setup_state(symbolic_setup(p.solver, A), A, mf.x)
    LaggedAssembledState(p, A, ns, 1, 1)
end

function preconditioner_update!(st::LaggedAssembledState, mf::MatrixFreeJacobian)
    st.nupdate += 1
    if (st.nupdate - 1) % st.prec.every == 0
        jacobian!(st.A, mf.op, mf.x)
        numerical_setup_state!(st.ns, st.A, mf.x)
        st.nbuild += 1
    end
    st
end

LinearAlgebra.ldiv!(y::AbstractVector, st::LaggedAssembledState, r::AbstractVector) = solve!(y, st.ns, r)
preconditioner_builds(st::LaggedAssembledState) = st.nbuild

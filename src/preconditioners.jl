# Preconditioner slots for the matrix-free Jacobian.
#
# A slot is a configuration object (`LaggedAssembled`, `BlockJacobi`) and a state built by
# `preconditioner_setup(slot, mf)` at the first numerical setup of a `MatrixFreeSolver`, refreshed
# by `preconditioner_update!(state, mf)` at every later one (every Newton iteration) and applied by
# `LinearAlgebra.ldiv!(y, state, r)` inside the Krylov method. `mf` is the prepared operator at the
# current state (`mf.x`), so the slots never see a matrix unless they assemble one themselves.
#
# `numerical_setup_state(ss, A, x)` / `numerical_setup_state!(ns, A, x)` build the setup of the
# inner solver of `LaggedAssembled` with the state available (HyperFEM's PETSc solver takes the
# rigid-body modes from it); here they fall back to Gridap's `numerical_setup`/`numerical_setup!`,
# and HyperFEM's extension forwards them to its own.

"Setup of a linear solver on the matrix `A` at the state `x` (Gridap's `numerical_setup` by default; HyperFEM's extension adds its own)."
numerical_setup_state(ss, A, x) = numerical_setup(ss, A)
"Update of that setup (Gridap's `numerical_setup!` by default)."
numerical_setup_state!(ns, A, x) = numerical_setup!(ns, A)

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

# ---------------------------------------------------------------------------------------------
# BlockJacobi: nodal block diagonal extracted from the element tangents

"""
    BlockJacobi()

Preconditioner slot: the inverse of the nodal block diagonal of the tangent, per field (the
`nc × nc` block of the components of each node of a vector field, the diagonal entry of a scalar
one; couplings between fields are dropped). The blocks are gathered from the element tangents by a
kernel that scatters only the diagonal blocks (no sparsity pattern, `nc² × nodes` storage) at
every update and inverted with StaticArrays. Weak on its own; it is the Jacobi part of the
Chebyshev–Jacobi smoother of the p-multigrid (S4). Not available with H(div) fields or skeleton
integrals in the Jacobian.
"""
struct BlockJacobi <: OperatorPreconditioner end

"Blocks of one field: node → block, block → global dofs (0 = Dirichlet), accumulated and inverted blocks."
struct FieldBlocks{NC}
    loc::Int                    # offset of the field in the element vector
    ldof::Matrix{Int}           # nn × nc → local dof of the field
    block_of::Matrix{Int32}     # nn × ncells
    dofs::Matrix{Int32}         # nc × nblocks, global free dofs (0 = fixed)
    B::Array{Float64,3}         # nc × nc × nblocks, accumulated
    Binv::Array{Float64,3}      # nc × nc × nblocks, inverted
end

mutable struct BlockJacobiState
    fields::Vector{Any}         # FieldBlocks{NC} per field
    nthreads::Int
    nbuild::Int
end

function preconditioner_setup(::BlockJacobi, mf::MatrixFreeJacobian)
    op = mf.op
    op.signed && error("HyperFEMKernels: BlockJacobi is not available with H(div) fields yet")
    any(g -> g isa SkeletonGroup, op.jgroups) && error("HyperFEMKernels: BlockJacobi is not available with skeleton integrals in the Jacobian yet")
    isempty(op.fallback.jtrians) || error("HyperFEMKernels: BlockJacobi needs the whole Jacobian in the kernels (hybrid fallback present)")
    tables = op.jgroups[1].tables
    fields = Any[field_blocks(fl, t) for (fl, t) in zip(op.fields, tables)]
    st = BlockJacobiState(fields, op.nthreads, 0)
    block_diagonal!(st, mf)
    st
end

function field_blocks(fl::FieldLayout, t::FieldTable)
    nn, nc = t.nn, t.nc
    ncells = size(fl.test_ids, 2)
    block_of = zeros(Int32, nn, ncells)
    keys = Dict{NTuple{nc,Int32},Int32}()
    dofs = Int32[]
    for cell in 1:ncells, n in 1:nn
        key = ntuple(c -> fl.test_ids[t.ldof[n, c], cell], nc)
        blk = get!(keys, key) do
            for c in 1:nc
                id = key[c]
                push!(dofs, id > 0 ? Int32(fl.test_off + id) : Int32(0))
            end
            Int32(length(keys) + 1)
        end
        block_of[n, cell] = blk
    end
    nb = length(keys)
    FieldBlocks{nc}(fl.loc, t.ldof, block_of, reshape(dofs, nc, nb), zeros(nc, nc, nb), zeros(nc, nc, nb))
end
field_blocks(::FieldLayout, ::PiolaTable) = error("HyperFEMKernels: BlockJacobi is not available with H(div) fields yet")

"Accumulate the nodal blocks of the tangent at `mf.x` and invert them."
function block_diagonal!(st::BlockJacobiState, mf::MatrixFreeJacobian)
    op = mf.op
    for fb in st.fields
        fill!(fb.B, 0.0)
    end
    dvals = [get_dirichlet_dof_values(s) for s in field_spaces(op.U)]
    for g in op.jgroups
        run_cells!(Val(:blocks), nothing, st, op, g, mf.x, dvals)
    end
    for fb in st.fields
        invert_blocks!(fb, st.nthreads)
    end
    st.nbuild += 1
    st
end

"Scatter the diagonal node blocks of the element tangent `KeT` (stored transposed) of cell `sc`."
function scatter_blocks!(st::BlockJacobiState, KeT, sc::Integer)
    for fb in st.fields
        _scatter_blocks!(fb, KeT, sc)
    end
    st
end
function _scatter_blocks!(fb::FieldBlocks{NC}, KeT, sc::Integer) where {NC}
    B, ldof, loc = fb.B, fb.ldof, fb.loc
    @inbounds for n in axes(ldof, 1)
        blk = fb.block_of[n, sc]
        for cj in 1:NC, ci in 1:NC
            B[ci, cj, blk] += KeT[loc+ldof[n, cj], loc+ldof[n, ci]]
        end
    end
    fb
end

function invert_blocks!(fb::FieldBlocks{NC}, nthreads::Int) where {NC}
    nb = size(fb.B, 3)
    nchunks = clamp(nb ÷ 256, 1, nthreads)
    @sync for ch in Iterators.partition(1:nb, cld(nb, nchunks))
        Threads.@spawn _invert_blocks!(fb, ch)
    end
    fb
end
function _invert_blocks!(fb::FieldBlocks{NC}, blocks) where {NC}
    B, Binv, dofs = fb.B, fb.Binv, fb.dofs
    @inbounds for blk in blocks
        # rows/columns of fixed components → identity, so the block is invertible and they stay untouched
        M = SMatrix{NC,NC,Float64}(ntuple(k -> begin
                i, j = (k - 1) % NC + 1, (k - 1) ÷ NC + 1
                dofs[i, blk] == 0 || dofs[j, blk] == 0 ? (i == j ? 1.0 : 0.0) : B[i, j, blk]
            end, NC * NC))
        Mi = inv(M)
        for j in 1:NC, i in 1:NC
            Binv[i, j, blk] = Mi[i, j]
        end
    end
    fb
end

function preconditioner_update!(st::BlockJacobiState, mf::MatrixFreeJacobian)
    block_diagonal!(st, mf)
end

function LinearAlgebra.ldiv!(y::AbstractVector, st::BlockJacobiState, r::AbstractVector)
    for fb in st.fields
        apply_blocks!(y, fb, r, st.nthreads)
    end
    y
end
function apply_blocks!(y, fb::FieldBlocks{NC}, r, nthreads::Int) where {NC}
    nb = size(fb.Binv, 3)
    nchunks = clamp(nb ÷ 256, 1, nthreads)
    @sync for ch in Iterators.partition(1:nb, cld(nb, nchunks))
        Threads.@spawn _apply_blocks!(y, fb, r, ch)
    end
    y
end
function _apply_blocks!(y, fb::FieldBlocks{NC}, r, blocks) where {NC}
    Binv, dofs = fb.Binv, fb.dofs
    @inbounds for blk in blocks
        for i in 1:NC
            id = dofs[i, blk]
            id == 0 && continue
            s = 0.0
            for j in 1:NC
                jd = dofs[j, blk]
                jd == 0 || (s += Binv[i, j, blk] * r[jd])
            end
            y[id] = s
        end
    end
    y
end

preconditioner_builds(st::BlockJacobiState) = st.nbuild

# ---------------------------------------------------------------------------------------------
# Spectrum of the preconditioned operator (for the Chebyshev smoother of S4)

"Right-preconditioned operator `J P⁻¹` as a linear map for Krylov.jl's processes."
struct PreconditionedOperator{MF,P,V}
    mf::MF
    prec::P
    tmp::V
end
Base.size(A::PreconditionedOperator) = size(A.mf)
Base.size(A::PreconditionedOperator, i::Int) = size(A.mf, i)
Base.eltype(::PreconditionedOperator) = Float64
function LinearAlgebra.mul!(y::AbstractVector, A::PreconditionedOperator, v::AbstractVector)
    ldiv!(A.tmp, A.prec, v)
    mul!(y, A.mf, A.tmp)
end

"""
    eigenvalue_bounds(mf, prec; k = 10) -> (λmin, λmax)

Estimate of the extreme eigenvalues (real parts of the Ritz values) of the right-preconditioned
operator `J P⁻¹` by `k` Arnoldi iterations from a random vector (Krylov.jl's `arnoldi`); `prec` is a
preconditioner state (`ldiv!`) or `nothing`. The Chebyshev–Jacobi smoother of the p-multigrid
(S4) uses `λmax` and a fraction of it as its interval.
"""
function eigenvalue_bounds(mf::MatrixFreeJacobian, prec; k::Int=10)
    n = size(mf, 1)
    k = min(k, n)
    A = prec === nothing ? mf : PreconditionedOperator(mf, prec, zeros(n))
    b = randn(n)
    V, β, H = Krylov.arnoldi(A, b, k)
    λ = eigvals(Matrix(H[1:k, 1:k]))
    re = real.(λ)
    (minimum(re), maximum(re))
end

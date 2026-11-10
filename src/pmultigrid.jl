# p-multigrid levels (design/solvers-s4a-plan.md §3.1, S4-A step 2): the coarse levels of the working space on the
# same mesh, their operators rediscretised from the same weak form and linearised at the injected state.
#
# Level 1 is the working space: its operator and prepared tangent are the Newton's (`mf`), never owned here.
# Levels 2 … L−1 are matrix-free (`KernelOperator(…; jacobian = :matrixfree)` + `MatrixFreeJacobian`); level L
# (the lowest order, normally Q1) is assembled and set up with the coarse solver. At every `update_levels!(lv, mf)`
# (one per Newton iteration, from the preconditioner slot):
#   - the forms of the levels follow the Newton's operator: when `mf.op` carries other `res`/`jac` closures (a new
#     load step: HyperFEM's driver `update`s its operator), every level operator is `update`d with them;
#   - the state is injected level by level (free and Dirichlet values: the coarse nodal interpolation of the fine FE
#     function, `inject!`), the Dirichlet values written into the level's trial space — nothing comes from HyperFEM;
#   - matrix-free levels are re-prepared; the coarse matrix is re-assembled and its solver set up every
#     `coarse_every` updates, and always when the forms changed (a lagged coarse tangent of the previous load step's
#     form would be another operator);
#   - while `mf` is bound to another operator on the same free dofs (HyperFEM's Dirichlet projection, on the increment
#     space) the levels are not rebuilt: the last prepared ones are reused as they are (DS4A-10); the forms of a foreign
#     operator are followed only if it is the first one seen;
#   - a non-finite tangent on any level (a non-injective injected state: det F ≤ 0 at a coarse quadrature point where
#     the fine state is fine) clears `levels_ok`, which the slot reports as a failed linear solve.

"A matrix-free level: its operator (rebound per load step), prepared tangent, spaces and injected state."
mutable struct MFLevel{MF}
    op::Any                       # KernelOperator; its type changes with the closures of `update`
    mf::MF
    U::FESpace
    V::FESpace
    x::Vector{Float64}            # injected free values
    d::Vector{Float64}            # injected Dirichlet values (concatenated over the fields)
end

"The assembled coarse level: operator, matrix, coarse solver and its setup."
mutable struct CoarseLevel{LS}
    op::Any
    U::FESpace
    V::FESpace
    A::SparseMatrixCSC{Float64,Int}
    solver::LS
    ns::Any                       # numerical setup of `solver` on `A` (nothing until the first build)
    x::Vector{Float64}
    d::Vector{Float64}
end

"The levels of a p-multigrid hierarchy (see the constructor)."
mutable struct PLevels{ML<:Tuple,C<:CoarseLevel,PT<:Tuple}
    U::FESpace                    # working trial space (level 1)
    V::FESpace
    mlevels::ML                   # MFLevel per intermediate level (levels 2 … L−1), possibly empty
    coarse::C
    P::PT                         # P[l]: level l+1 → level l (level 1 = fine)
    res::Any                      # the forms the level operators follow (identity of the closures)
    jac::Any
    built::Bool
    coarse_every::Int
    since::Int                    # updates since the last coarse build
    nupdate::Int
    nbuild::Int                   # coarse builds (assembly + setup)
    nskip::Int                    # updates on a foreign operator (Dirichlet projection): levels reused
    nform::Int                    # form changes (load steps) followed
    nconsec::Int                  # consecutive skipped updates (a working space that is never seen again is a misconfiguration)
    ok::Bool
    dfine::Vector{Float64}        # Dirichlet values of the working space (buffer)
    nthreads::Int
end

"""
    PLevels(spaces, res, jac; coarse_solver = LUSolver(), coarse_every = 1, nthreads = Threads.nthreads())

The coarse levels of a p-multigrid hierarchy on one mesh. `spaces` lists `(U, V)` per level, **finest first**: the
first pair is the working space of the Newton (its trial space must be the one the operator is built on), the last the
assembled coarse level. Every level must have the same fields, the same closed Dirichlet boundary and nested Lagrangian
reference elements (see `Prolongation`); the trial spaces of the coarse levels must be `TrialFESpace`s (their Dirichlet
values are overwritten with the injected ones). The level operators are captured from `res`, `jac` (the forms of the
Newton's operator; the preconditioner slot passes `mf.op.res`, `mf.op.jac`) and follow the operator's forms afterwards.
The levels are a `Tuple` (no dynamic dispatch per level visit in the cycle); the operators inside are untyped fields,
one dispatch per level per call.
"""
function PLevels(spaces::AbstractVector, res, jac; coarse_solver::Algebra.LinearSolver=LUSolver(), coarse_every::Int=1,
                 nthreads::Int=Threads.nthreads())
    L = length(spaces)
    L ≥ 2 || throw(ArgumentError("HyperFEMKernels: PLevels needs at least two levels (fine and coarse)"))
    coarse_every ≥ 1 || throw(ArgumentError("HyperFEMKernels: PLevels: coarse_every must be ≥ 1"))
    for (U, V) in spaces[2:end], s in field_spaces(U)
        (s isa TrialFESpace || num_dirichlet_dofs(s) == 0) ||
            error("HyperFEMKernels: the trial spaces of the coarse p-multigrid levels must be TrialFESpaces (their Dirichlet values are set by injection), got $(nameof(typeof(s)))")
    end
    nfree(U) = sum(num_free_dofs, field_spaces(U))
    for l in 2:L
        nfree(spaces[l][2]) == nfree(spaces[l][1]) || error("HyperFEMKernels: level $l: test and trial spaces with different free dofs")
    end
    P = Tuple(Prolongation(spaces[l+1][1], spaces[l][1]; nthreads) for l in 1:L-1)
    mlev = Tuple(begin
                     U, V = spaces[l]
                     op = KernelOperator(res, jac, U, V; nthreads, jacobian=:matrixfree)
                     MFLevel(op, MatrixFreeJacobian(op), U, V, zeros(nfree(U)), copy(dirichlet_values(U)))
                 end for l in 2:L-1)
    Uc, Vc = spaces[L]
    opc = KernelOperator(res, jac, Uc, Vc; nthreads)
    coarse = CoarseLevel(opc, Uc, Vc, copy(sparsity_pattern(opc)), coarse_solver, nothing, zeros(nfree(Uc)), copy(dirichlet_values(Uc)))
    PLevels(spaces[1][1], spaces[1][2], mlev, coarse, P, res, jac, false, coarse_every, 0, 0, 0, 0, 0, 0, true,
            copy(dirichlet_values(spaces[1][1])), nthreads)
end

nlevels(lv::PLevels) = length(lv.mlevels) + 2
levels_ok(lv::PLevels) = lv.ok
preconditioner_builds(lv::PLevels) = lv.nbuild

"Write concatenated Dirichlet values into the fields of a trial space (in place)."
function set_dirichlet_values!(U::FESpace, d::AbstractVector)
    off = 0
    for s in field_spaces(U)
        dv = get_dirichlet_dof_values(s)
        n = length(dv)
        copyto!(dv, 1, d, off + 1, n)
        off += n
    end
    off == length(d) || error("HyperFEMKernels: Dirichlet values of length $(length(d)) for a space with $off Dirichlet dofs")
    U
end

"Concatenated Dirichlet values of a trial space into a buffer."
function get_dirichlet_values!(d::AbstractVector, U::FESpace)
    off = 0
    for s in field_spaces(U)
        dv = get_dirichlet_dof_values(s)
        copyto!(d, off + 1, dv, 1, length(dv))
        off += length(dv)
    end
    d
end

"All the stored tangents of a prepared matrix-free Jacobian are finite."
function tangents_finite(mf::MatrixFreeJacobian)
    for d in mf.data
        for gd in (d isa Tuple ? d : (d,))
            all(isfinite, gd.T) || return false
        end
    end
    mf.Afb === nothing || all(isfinite, nonzeros(mf.Afb)) || return false
    true
end

"`update` the level operators with new forms (a new load step of the Newton's operator)."
function update_level_operators!(lv::PLevels, res, jac)
    for ml in lv.mlevels
        ml.op = update(ml.op, res, jac)            # jacobian!(ml.mf, ml.op, x) rebinds the prepared tangent
    end
    c = lv.coarse
    newop = update(c.op, res, jac)
    newop.pattern === c.op.pattern || (c.A = copy(sparsity_pattern(newop)); c.ns = nothing)   # another structure: new matrix and setup
    c.op = newop
    lv.res, lv.jac = res, jac
    lv.nform += 1
    lv
end

"""
    update_levels!(lv::PLevels, mf::MatrixFreeJacobian) -> lv

Bring the coarse levels to the state of the prepared working tangent `mf` (see the header of this file):
follow its forms, inject the state, re-prepare the matrix-free levels, re-assemble and set up the coarse level every
`coarse_every` updates (and when the forms changed), check every tangent is finite (`levels_ok`). On a foreign operator
(another trial space on the same test space: the Dirichlet projection) the last prepared levels are reused if they are
usable, and prepared from it otherwise (first update, or levels that failed at a rejected iterate). The Dirichlet
values are read from the operator's trial space at this call and the free values from `mf.x` (the state of the last
`prepare!`): HyperFEM calls `jacobian!` and `numerical_setup!` back to back, with no change of the Dirichlet values in
between.
"""
function update_levels!(lv::PLevels, mf::MatrixFreeJacobian)
    _prepared(mf)
    op = mf.op
    lv.nupdate += 1
    nf = sum(num_free_dofs, field_spaces(lv.U))
    size(mf, 2) == nf || error("HyperFEMKernels: the operator has $(size(mf, 2)) free dofs, the p-multigrid working space $nf")
    if op.U !== lv.U
        op.V === lv.V || error("HyperFEMKernels: p-multigrid: the operator is not on the working test space of the levels (the first pair of `spaces` must be the spaces of the Newton's operator)")
        # another trial space on the same test space and free dofs: HyperFEM's Dirichlet projection (on ΔU). Reuse the
        # last prepared levels — unless they failed (a non-finite level after a rejected iterate: rebuild them at this
        # state, or the projection of the cut-back step would fail too) or were never built
        if lv.built && lv.ok
            lv.nskip += 1
            lv.nconsec += 1
            lv.nconsec == 5 && @warn "HyperFEMKernels: p-multigrid: 5 consecutive updates on an operator whose trial space is not the working space of the levels; the levels are not being updated (is the first pair of `spaces` the Newton's (U, V)?)" maxlog = 1
            return lv
        end
    else
        lv.nconsec = 0
    end
    newform = op.res !== lv.res || op.jac !== lv.jac
    newform && update_level_operators!(lv, op.res, op.jac)
    # state, level by level
    ok = true
    xprev = mf.x
    dprev = length(lv.dfine) == num_dirichlet_dofs_total(op.U) ? get_dirichlet_values!(lv.dfine, op.U) : dirichlet_values(op.U)
    for (i, ml) in enumerate(lv.mlevels)
        inject!(ml.x, ml.d, lv.P[i], xprev, dprev)
        set_dirichlet_values!(ml.U, ml.d)
        jacobian!(ml.mf, ml.op, ml.x)
        ok &= tangents_finite(ml.mf)
        xprev, dprev = ml.x, ml.d
    end
    c = lv.coarse
    inject!(c.x, c.d, lv.P[end], xprev, dprev)
    set_dirichlet_values!(c.U, c.d)
    lv.since += 1
    if newform || c.ns === nothing || lv.since ≥ lv.coarse_every
        jacobian!(c.A, c.op, c.x)
        cok = all(isfinite, nonzeros(c.A))
        if cok
            c.ns = c.ns === nothing ? numerical_setup_state(symbolic_setup(c.solver, c.A), c.A, c.x) :
                   numerical_setup_state!(c.ns, c.A, c.x)
            lv.nbuild += 1
            lv.since = 0
        else
            c.ns = nothing                    # a NaN matrix is not set up; the next update rebuilds
        end
        ok &= cok
    end
    lv.ok = ok
    lv.built = true
    lv
end

num_dirichlet_dofs_total(U::FESpace) = sum(num_dirichlet_dofs, field_spaces(U); init=0)

# ---------------------------------------------------------------------------------------------
# PMultigrid: the V-cycle as a preconditioner slot of MatrixFreeSolver (S4-A step 4)

"""
    PMultigrid(spaces; smoother = Chebyshev(), coarse = LUSolver(), coarse_every = 1)

Preconditioner slot of `MatrixFreeSolver`: one V-cycle of p-multigrid on the levels `spaces` (`(U, V)` per level,
finest first — the first is the working space of the Newton; see `PLevels`). On every level but the coarsest:
pre-smoothing from zero with `smoother` (Chebyshev on the block-preconditioned tangent), restriction of the residual,
the coarser level, prolongation of its correction, post-smoothing with the same polynomial; on the coarsest (assembled)
level one application of `coarse` — any Gridap `LinearSolver`: `LUSolver()` in tests; in production one AMG V-cycle
(HyperFEM's `PETScSolver(:preonly, :gamg)` with the rigid-body modes, which it receives through
`numerical_setup_state`) or MUMPS for thin structures. A coarse solver that is a Krylov solve to a tolerance makes the
cycle a variable operator: use `krylov = :fgmres` then; with a fixed coarse operator the cycle is linear (GMRES), and
symmetric for an SPD tangent with an SPD coarse solver (same pre- and post-smoother: CG applies).

Per Newton iteration (`preconditioner_update!`): the levels follow the operator (`update_levels!`), the smoothers
re-extract their blocks and interval. A non-finite tangent on any level or a non-usable smoother interval makes
`preconditioner_ok` false: `MatrixFreeSolver` then returns at once with `linear_converged == false` (and a direct
`ldiv!` returns NaN), so the Newton fails the iteration. A negative Ritz value of the block-preconditioned tangent
(an indefinite tangent, past a limit point or a buckling load) is reported the same way: Chebyshev would amplify the
negative modes.

Notes. The coarse operators are captured from the Newton's forms, so they integrate with the working space's quadrature
(e.g. 27 points per Q1 cell under a Q2 measure): this makes the rediscretised coarse operator of a linear form equal to
the Galerkin product exactly, at ≈ 3× the cost of a Q1 assembly with its own quadrature. The Dirichlet values of the
coarse trial spaces are overwritten in place at every update: spaces handed to `PMultigrid` belong to it. The smoother
needs the nodal blocks of the working tangent, so the working Jacobian must be entirely in the kernels (no hybrid
fallback, no skeleton terms) with invertible nodal blocks (no zero diagonal block, as the pressure of a mixed
formulation). Single-field spaces in practice (multi-field (u, φ) saddle points need block smoothers).
"""
struct PMultigrid{LS<:Algebra.LinearSolver} <: OperatorPreconditioner
    spaces::Vector{Any}
    smoother::Chebyshev
    coarse::LS
    coarse_every::Int
    function PMultigrid(spaces::AbstractVector; smoother::Chebyshev=Chebyshev(), coarse::Algebra.LinearSolver=LUSolver(),
                        coarse_every::Int=1)
        length(spaces) ≥ 2 || throw(ArgumentError("HyperFEMKernels: PMultigrid needs at least two levels"))
        coarse_every ≥ 1 || throw(ArgumentError("HyperFEMKernels: PMultigrid: coarse_every must be ≥ 1"))
        new{typeof(coarse)}(collect(Any, spaces), smoother, coarse, coarse_every)
    end
end

"Work vectors of one smoothed level: its residual and prolongated correction, the restricted rhs and the coarser solution."
struct CycleWork
    r::Vector{Float64}
    e::Vector{Float64}
    bc::Vector{Float64}
    xc::Vector{Float64}
end

mutable struct PMultigridState{LV<:PLevels,SM<:Tuple,W<:Tuple}
    prec::PMultigrid
    lv::LV
    smoothers::SM                 # ChebyshevState of levels 1 … L−1 (level 1 on the Newton's tangent)
    work::W                       # CycleWork of levels 1 … L−1
    ok::Bool
    ncycle::Int
end

function preconditioner_setup(p::PMultigrid, mf::MatrixFreeJacobian)
    _prepared(mf)
    lv = PLevels(p.spaces, mf.op.res, mf.op.jac; coarse_solver=p.coarse, coarse_every=p.coarse_every, nthreads=mf.nthreads)
    update_levels!(lv, mf)
    mfs = (mf, map(ml -> ml.mf, lv.mlevels)...)
    smoothers = map(m -> smoother_setup(p.smoother, m), mfs)
    sizes = (size(mf, 1), map(ml -> length(ml.x), lv.mlevels)..., length(lv.coarse.x))
    work = ntuple(l -> CycleWork(zeros(sizes[l]), zeros(sizes[l]), zeros(sizes[l+1]), zeros(sizes[l+1])), length(sizes) - 1)
    st = PMultigridState(p, lv, smoothers, work, true, 0)
    st.ok = levels_ok(lv) && all(s -> s.ok, smoothers)
    st
end

function preconditioner_update!(st::PMultigridState, mf::MatrixFreeJacobian)
    _prepared(mf)
    lv = st.lv
    nskip = lv.nskip
    update_levels!(lv, mf)
    skipped = lv.nskip > nskip
    smoother_update!(st.smoothers[1], mf)              # the fine smoother always follows the operator the Krylov method applies
    if !skipped && levels_ok(lv)                       # intermediate levels re-prepared: their smoothers follow
        for (i, ml) in enumerate(lv.mlevels)
            smoother_update!(st.smoothers[i+1], ml.mf)
        end
    end
    st.ok = levels_ok(lv) && all(s -> s.ok, st.smoothers)
    st
end

"One V-cycle on `A x = b` from the level of `smoothers[1]` down (tuples peeled at compile time: no dispatch per level)."
function vcycle!(x, b, smoothers::Tuple, Ps::Tuple, work::Tuple, coarse::CoarseLevel)
    S, P, w = first(smoothers), first(Ps), first(work)
    smooth!(x, S, b; zero_guess=true, residual=w.r)                  # pre-smoothing, residual for the restriction
    restrict!(w.bc, P, w.r)
    if length(smoothers) == 1
        solve!(w.xc, coarse.ns, w.bc)                                 # coarsest level
    else
        vcycle!(w.xc, w.bc, Base.tail(smoothers), Base.tail(Ps), Base.tail(work), coarse)
    end
    prolongate!(w.e, P, w.xc)
    x .+= w.e
    smooth!(x, S, b)                                                  # post-smoothing (same polynomial: symmetric cycle)
    x
end

function LinearAlgebra.ldiv!(y::AbstractVector, st::PMultigridState, r::AbstractVector)
    if !st.ok
        fill!(y, NaN)                                                 # reported by the Krylov solve as not converged
        return y
    end
    st.ncycle += 1
    vcycle!(y, r, st.smoothers, st.lv.P, st.work, st.lv.coarse)
end

preconditioner_builds(st::PMultigridState) = preconditioner_builds(st.lv)
preconditioner_ok(st::PMultigridState) = st.ok
"An indefinite tangent on some level (a non-positive Ritz value of a smoother with finite blocks), the S5 signal."
preconditioner_indefinite(st::PMultigridState) = !st.ok && levels_ok(st.lv) && any(smoother_indefinite, st.smoothers)

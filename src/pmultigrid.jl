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
        s isa TrialFESpace || error("HyperFEMKernels: the trial spaces of the coarse p-multigrid levels must be TrialFESpaces (their Dirichlet values are set by injection), got $(nameof(typeof(s)))")
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
    PLevels(spaces[1][1], spaces[1][2], mlev, coarse, P, res, jac, false, coarse_every, 0, 0, 0, 0, 0, true,
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
with the same free dofs (the Dirichlet projection) the last prepared levels are reused; if the first update comes on a
foreign operator, the levels are prepared from it once.
"""
function update_levels!(lv::PLevels, mf::MatrixFreeJacobian)
    _prepared(mf)
    op = mf.op
    lv.nupdate += 1
    nf = sum(num_free_dofs, field_spaces(lv.U))
    size(mf, 2) == nf || error("HyperFEMKernels: the operator has $(size(mf, 2)) free dofs, the p-multigrid working space $nf")
    if op.U !== lv.U && lv.built
        lv.nskip += 1
        return lv
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

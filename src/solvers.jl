# Krylov solvers on the matrix-free Jacobian (Gridap `LinearSolver` interface).
#
# `MatrixFreeSolver` solves `J dx = -R` with Krylov.jl applying `mul!(y, mf, v)` of a
# `MatrixFreeJacobian` (no direct indexing of the operator: the same code runs on GPU vectors and
# on distributed vectors once the kernels do). Right preconditioning, so the residual the method
# monitors and the tolerance refer to is the true one, ‖b − J x‖ ≤ atol + rtol ‖b‖ (the convention
# of HyperFEM's PETSc/AMG solvers). The preconditioner is one of the operator slots of
# `preconditioners.jl`, applied with `ldiv!`.
#
# Hooks read by a Newton solver (HyperFEM's `NewtonRaphson` through its HyperFEMKernels extension):
# `linear_converged`, `linear_iterations`, `set_tolerance!`, `tolerance`, `preconditioner_builds`.
# They are generic functions of this package, not exported.

"""
    MatrixFreeSolver(; krylov = :gmres, rtol = 1e-8, atol = 0.0, maxiter = 500, restart = 50,
                     preconditioner = nothing, vector = Vector{Float64}, verbose = 0)

Gridap `LinearSolver` for a `KernelOperator` with `jacobian = :matrixfree`: a Krylov method of
Krylov.jl (`:gmres`, `:fgmres`, `:cg`) applied to the `MatrixFreeJacobian` returned by
`allocate_jacobian`, right-preconditioned by `preconditioner` (an operator slot: `LaggedAssembled`,
`BlockJacobi`; `nothing` = none). The stopping test is on the true residual,
`‖b − J x‖ ≤ atol + rtol ‖b‖`; `restart` is the GMRES memory; `maxiter` counts all iterations.
`:cg` applies the preconditioner on the left (symmetric form; the monitored norm is then the
preconditioned one, as in Krylov.jl). `vector` is the vector type of the Krylov workspace.
"""
struct MatrixFreeSolver{P,S} <: Algebra.LinearSolver
    krylov::Symbol
    rtol::Float64
    atol::Float64
    maxiter::Int
    restart::Int
    preconditioner::P
    verbose::Int
end

function MatrixFreeSolver(; krylov::Symbol=:gmres, rtol::Real=1e-8, atol::Real=0.0, maxiter::Int=500, restart::Int=50,
                          preconditioner=nothing, vector::Type=Vector{Float64}, verbose::Int=0)
    krylov in (:gmres, :fgmres, :cg) || throw(ArgumentError("HyperFEMKernels: MatrixFreeSolver: krylov must be :gmres, :fgmres or :cg (got :$krylov)"))
    _check_rtol(rtol)
    atol ≥ 0 || throw(ArgumentError("HyperFEMKernels: MatrixFreeSolver: atol must be ≥ 0"))
    maxiter > 0 && restart > 0 || throw(ArgumentError("HyperFEMKernels: MatrixFreeSolver: maxiter and restart must be positive"))
    MatrixFreeSolver{typeof(preconditioner),vector}(krylov, Float64(rtol), Float64(atol), maxiter, restart, preconditioner, verbose)
end

_check_rtol(r) = (0 ≤ r < 1 || throw(ArgumentError("HyperFEMKernels: the relative tolerance must be in [0, 1) (got $r)")); Float64(r))

struct MatrixFreeSolverSS{S<:MatrixFreeSolver} <: Algebra.SymbolicSetup
    solver::S
end

Algebra.symbolic_setup(s::MatrixFreeSolver, ::MatrixFreeJacobian) = MatrixFreeSolverSS(s)
Algebra.symbolic_setup(::MatrixFreeSolver, ::AbstractMatrix) =
    error("HyperFEMKernels: MatrixFreeSolver applies a MatrixFreeJacobian; build the KernelOperator with jacobian = :matrixfree")

"""
Numerical setup of a `MatrixFreeSolver`: the operator at the current state, the Krylov.jl workspace
(allocated once, reused by every solve), the preconditioner state and the last solve's statistics.
"""
mutable struct MatrixFreeSolverNS{S<:MatrixFreeSolver,W,P,V} <: Algebra.NumericalSetup
    solver::S
    mf::MatrixFreeJacobian
    workspace::W
    prec::P                  # preconditioner state (`nothing` when there is none)
    r::V                     # scratch for the true residual when the method stops without its own test
    rtol::Float64            # tolerance in force for the next solves (`set_tolerance!`)
    iters::Int
    converged::Bool
    rnorm::Float64           # relative residual of the last solve when it had to be checked (NaN otherwise)
    nsetup::Int
end

function _workspace(s::MatrixFreeSolver{P,S}, n::Int) where {P,S}
    s.krylov === :gmres && return Krylov.GmresWorkspace(n, n, S; memory=min(s.restart, n))
    s.krylov === :fgmres && return Krylov.FgmresWorkspace(n, n, S; memory=min(s.restart, n))
    Krylov.CgWorkspace(n, n, S)
end

function Algebra.numerical_setup(ss::MatrixFreeSolverSS, mf::MatrixFreeJacobian)
    s = ss.solver
    n = size(mf, 1)
    n == size(mf, 2) || error("HyperFEMKernels: MatrixFreeSolver needs a square operator (got $(size(mf)))")
    prec = s.preconditioner === nothing ? nothing : preconditioner_setup(s.preconditioner, mf)
    ws = _workspace(s, n)
    MatrixFreeSolverNS(s, mf, ws, prec, similar(ws.x), s.rtol, 0, true, NaN, 1)
end

function Algebra.numerical_setup!(ns::MatrixFreeSolverNS, mf::MatrixFreeJacobian)
    ns.mf = mf
    ns.nsetup += 1
    ns.prec === nothing || preconditioner_update!(ns.prec, mf)
    ns
end

function Algebra.solve!(x::AbstractVector, ns::MatrixFreeSolverNS, b::AbstractVector)
    s, ws, P = ns.solver, ns.workspace, ns.prec
    ldiv = P !== nothing
    N = ldiv ? P : LinearAlgebra.I
    kw = (; atol=s.atol, rtol=ns.rtol, itmax=s.maxiter, verbose=s.verbose)
    if s.krylov === :gmres
        Krylov.gmres!(ws, ns.mf, b; N, ldiv, restart=true, kw...)
    elseif s.krylov === :fgmres
        Krylov.fgmres!(ws, ns.mf, b; N, ldiv, restart=true, kw...)
    else
        Krylov.cg!(ws, ns.mf, b; M=N, ldiv, kw...)
    end
    copyto!(x, ws.x)
    ns.iters = ws.stats.niter
    ns.converged = ws.stats.solved
    ns.rnorm = NaN
    if !ns.converged
        # Krylov.jl stops without its residual test on an Arnoldi breakdown (an exact preconditioner
        # solves in one iteration) or at the iteration limit: decide on the true residual
        r = ns.r
        mul!(r, ns.mf, x)
        r .= b .- r
        ns.rnorm = norm(r) / norm(b)
        ns.converged = norm(r) ≤ s.atol + ns.rtol * norm(b)
    end
    x
end

# --- hooks (see the header) --------------------------------------------------------------------

"Whether the last `solve!` of the setup reached its tolerance."
linear_converged(ns::MatrixFreeSolverNS) = ns.converged
"Krylov iterations of the last `solve!`."
linear_iterations(ns::MatrixFreeSolverNS) = ns.iters
"Relative tolerance of the next solves (`nothing` restores the solver's own)."
set_tolerance!(ns::MatrixFreeSolverNS, rtol) = (ns.rtol = rtol === nothing ? ns.solver.rtol : _check_rtol(rtol); ns)
"Relative tolerance in force."
tolerance(ns::MatrixFreeSolverNS) = ns.rtol
"Number of preconditioner builds (assemblies of the lagged tangent, block extractions) so far."
preconditioner_builds(ns::MatrixFreeSolverNS) = ns.prec === nothing ? 0 : preconditioner_builds(ns.prec)

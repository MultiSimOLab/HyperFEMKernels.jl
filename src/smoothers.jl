# Chebyshev smoother preconditioned by the nodal block diagonal (S4-A step 3, design/solvers-s4a-plan.md §3.3).
#
# For an SPD operator A and an SPD preconditioner M (here the nodal blocks of `BlockJacobi`) the Chebyshev iteration
# of degree k on the interval [a, b] multiplies the error by the polynomial p_k(M⁻¹A) with
#     p_k(λ) = T_k((θ − λ)/δ) / T_k(θ/δ),   θ = (b + a)/2, δ = (b − a)/2,
# so every eigencomponent of M⁻¹A in [a, b] is damped by at most 1/T_k(θ/δ). As a smoother the interval is the top
# of the spectrum: b = 1.1 λmax (the Ritz estimate is from below), a = b/α. Components below a are left to the coarse
# level. The iteration needs no inner products (no global reductions: GPU/MPI friendly) and costs one product and one
# block application per degree, the first product saved when the initial guess is zero.

"""
    Chebyshev(; degree = 2, range = 20.0, inflate = 1.1, eig_iters = 10)

Smoother configuration: Chebyshev iteration of `degree` on `[b/range, b]`, `b = inflate · λmax`, with `λmax` the Ritz
estimate of the block-preconditioned operator after `eig_iters` Arnoldi steps from a fixed start vector (reproducible
iteration counts). Defaults: degree 2, range 20 (DS4A-9, decided 2026-11-08 by measurement: with one AMG V-cycle as
coarse solver the p-multigrid cycle is coarse-limited, degrees 2–5 give 26.6–21.7 Krylov iterations on the Q2 cube at
197k dofs and degree 2 the lowest time; degree 1 doubles the iterations; the range changes them by ±2 %). deal.II's
step-37 uses degree 5 and range 15–20 with an exact coarse solve, PETSc's GAMG smoother `esteig` [0.1, 1.1]·λmax.
"""
struct Chebyshev
    degree::Int
    range::Float64
    inflate::Float64
    eig_iters::Int
    function Chebyshev(; degree::Int=2, range::Real=20.0, inflate::Real=1.1, eig_iters::Int=10)
        degree ≥ 1 || throw(ArgumentError("HyperFEMKernels: Chebyshev: degree must be ≥ 1"))
        range > 1 || throw(ArgumentError("HyperFEMKernels: Chebyshev: range must be > 1"))
        inflate ≥ 1 || throw(ArgumentError("HyperFEMKernels: Chebyshev: inflate must be ≥ 1"))
        eig_iters ≥ 2 || throw(ArgumentError("HyperFEMKernels: Chebyshev: eig_iters must be ≥ 2"))
        new(degree, Float64(range), Float64(inflate), eig_iters)
    end
end

"Damping factor of the Chebyshev smoother on its interval: max |p_k(λ)| = 1/T_k(θ/δ) for λ ∈ [b/range, b]."
function chebyshev_damping(c::Chebyshev)
    b = 1.0; a = b / c.range
    θ, δ = (b + a) / 2, (b - a) / 2
    1 / cosh(c.degree * acosh(θ / δ))
end

"Smoother state on one level: the operator, its nodal blocks, the interval and the work vectors."
mutable struct ChebyshevState{MF}
    cfg::Chebyshev
    mf::MF                        # the level's prepared tangent (the Newton's on the fine level)
    blocks::BlockJacobiState
    λest::Float64                 # Ritz estimate of λmax(M⁻¹A)
    λmin::Float64                 # smallest Ritz value (negative: indefinite tangent, the smoother is not usable)
    a::Float64
    b::Float64
    start::Vector{Float64}        # fixed start vector of the Arnoldi estimate
    r::Vector{Float64}
    z::Vector{Float64}
    d::Vector{Float64}
    nbuild::Int
    ok::Bool                      # the last estimate was finite and positive (otherwise the interval is not usable)
end

"Fixed start vector for the eigenvalue estimate (deterministic, not aligned with any special mode)."
function chebyshev_start(n::Int)
    v = [sin(1.0 + 0.7i) + 0.3cos(2.3i) for i in 1:n]
    v ./ norm(v)
end

"""
    smoother_setup(cfg::Chebyshev, mf) -> ChebyshevState

Blocks of the prepared tangent `mf` (from its stored tangents) and the interval of the smoother.
"""
function smoother_setup(cfg::Chebyshev, mf::MatrixFreeJacobian)
    blocks = preconditioner_setup(BlockJacobi(), mf; strict=false)    # a NaN tangent clears `ok` instead of throwing
    n = size(mf, 1)
    st = ChebyshevState(cfg, mf, blocks, 0.0, 0.0, 0.0, 0.0, chebyshev_start(n), zeros(n), zeros(n), zeros(n), 0, true)
    smoother_interval!(st)
end

"Refresh the blocks and the interval at the current state of `mf` (once per Newton iteration)."
function smoother_update!(st::ChebyshevState, mf::MatrixFreeJacobian)
    st.mf = mf
    preconditioner_update!(st.blocks, mf)
    smoother_interval!(st)
end

function smoother_interval!(st::ChebyshevState)
    if !st.blocks.ok                       # a non-finite tangent (its nodal blocks are not finite): nothing to estimate
        st.ok = false
        return st
    end
    λmin, λmax = eigenvalue_bounds(st.mf, st.blocks; k=st.cfg.eig_iters, b=st.start)
    st.λest, st.λmin = λmax, λmin
    # not usable: a non-finite or non-positive estimate, or a negative Ritz value (an indefinite tangent past a limit
    # point or a buckling load: Chebyshev would amplify the negative modes; S5 handles indefinite tangents)
    st.ok = isfinite(λmax) && λmax > 0 && λmin > 0
    st.ok || return st
    st.b = st.cfg.inflate * λmax
    st.a = st.b / st.cfg.range
    st.nbuild += 1
    st
end

"""
    smooth!(x, st, rhs; zero_guess = false, residual = nothing) -> x

`degree` Chebyshev steps on `A x = rhs` from `x` (from zero if `zero_guess`, which saves the first product). With
`residual` a vector, the residual `rhs − A x` after the last step is written into it (one more product, which a V-cycle
needs anyway for the restriction). Allocation-free.
"""
function smooth!(x::AbstractVector, st::ChebyshevState, rhs::AbstractVector; zero_guess::Bool=false, residual=nothing)
    st.ok || error("HyperFEMKernels: Chebyshev: the smoother has no usable interval (non-finite or indefinite tangent)")
    r, z, d, mf = st.r, st.z, st.d, st.mf
    a, b = st.a, st.b
    θ, δ = (b + a) / 2, (b - a) / 2
    σ = θ / δ
    if zero_guess
        fill!(x, 0.0)
        copyto!(r, rhs)
    else
        mul!(r, mf, x)
        @. r = rhs - r
    end
    ldiv!(z, st.blocks, r)
    ρ = 1 / σ
    @. d = z / θ
    k = st.cfg.degree
    for i in 1:k
        x .+= d
        (i == k && residual === nothing) && break
        mul!(z, mf, d)                   # z = A d (reused as scratch)
        r .-= z
        i == k && break
        ldiv!(z, st.blocks, r)
        ρn = 1 / (2σ - ρ)
        @. d = ρn * ρ * d + (2ρn / δ) * z
        ρ = ρn
    end
    residual === nothing || copyto!(residual, r)
    x
end

preconditioner_builds(st::ChebyshevState) = st.nbuild

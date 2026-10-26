# Tensor-product (sum-factorisation) structure of Lagrangian spaces on n-cubes.
#
# On quadrilaterals/hexahedra with a tensor-product Gauss quadrature, the nodal shape functions
# and the quadrature points are products of 1D ones: N_(i,j,k)(ξ) = n_i(ξ1) n_j(ξ2) n_k(ξ3).
# Interpolation to the points and the contraction back are then done direction by direction
# (sum factorisation), with O(Q^(D+1)) instead of O(Q^(2D)) operations per cell and component.
#
# The 1D matrices and the permutations between Gridap's node/point numbering and the tensor
# numbering are found numerically from the ND tables, so no assumption on Gridap's ordering is
# needed (and a mismatch is detected: `TPTable` returns `nothing`).

"1D basis values/derivatives at the 1D points and the permutations to Gridap's numbering."
struct TPTable{D,Q,P,L}
    N1::SMatrix{P,Q,Float64,L}      # N1[a, i]: 1D shape function i at 1D point a
    dN1::SMatrix{P,Q,Float64,L}
    nodeperm::Vector{Int}           # tensor node (i, j[, k]) linear index → table node n
    pointperm::Vector{Int}          # tensor point (a, b[, c]) linear index → quadrature point q
end

npoints1d(::TPTable{D,Q,P}) where {D,Q,P} = P
nnodes1d(::TPTable{D,Q,P}) where {D,Q,P} = Q

"1D Lagrange basis on [0,1] with Q equispaced nodes, values and derivatives at `ξ`."
function lagrange1d(Q::Int, ξ::Float64)
    xs = Q == 1 ? [0.5] : collect(range(0.0, 1.0; length=Q))
    N = zeros(Q); dN = zeros(Q)
    for i in 1:Q
        v = 1.0
        for j in 1:Q
            j == i && continue
            v *= (ξ - xs[j]) / (xs[i] - xs[j])
        end
        N[i] = v
        d = 0.0
        for m in 1:Q
            m == i && continue
            t = 1.0 / (xs[i] - xs[m])
            for j in 1:Q
                (j == i || j == m) && continue
                t *= (ξ - xs[j]) / (xs[i] - xs[j])
            end
            d += t
        end
        dN[i] = d
    end
    N, dN
end

"""
    TPTable(t::FieldTable, xq) -> TPTable or nothing

Tensor-product structure of a field table whose reference element is an n-cube with Q^D
equispaced Lagrange nodes and P^D tensor-product quadrature points, or `nothing`.
"""
TPTable(::PiolaTable, xq::AbstractVector) = nothing
function TPTable(t::FieldTable{D,T,NN}, xq::AbstractVector) where {D,T,NN}
    Q = round(Int, NN^(1 / D)); Q^D == NN || return nothing
    np = length(xq)
    P = round(Int, np^(1 / D)); P^D == np || return nothing
    # 1D points: distinct coordinates along direction 1
    xs1 = sort(unique(round.(getindex.(xq, 1); digits=12)))
    length(xs1) == P || return nothing
    for d in 2:D
        sort(unique(round.(getindex.(xq, d); digits=12))) ≈ xs1 || return nothing
    end
    N1 = zeros(P, Q); dN1 = zeros(P, Q)
    for a in 1:P
        N1[a, :], dN1[a, :] = lagrange1d(Q, xs1[a])
    end
    # point permutation: tensor (a1..aD) → q
    pointperm = zeros(Int, np)
    for (lin, idx) in enumerate(CartesianIndices(ntuple(_ -> P, D)))
        target = Tuple(idx)
        q = findfirst(p -> all(d -> isapprox(p[d], xs1[target[d]]; atol=1e-10), 1:D), xq)
        q === nothing && return nothing
        pointperm[lin] = q
    end
    # node permutation: tensor (i1..iD) → n, by matching the ND values at all points
    nodeperm = zeros(Int, NN)
    used = falses(NN)
    for (lin, idx) in enumerate(CartesianIndices(ntuple(_ -> Q, D)))
        ii = Tuple(idx)
        vals = [prod(d -> N1[Tuple(CartesianIndices(ntuple(_ -> P, D))[pl])[d], ii[d]], 1:D) for pl in 1:np]
        n = findfirst(n -> !used[n] && all(pl -> isapprox(t.N[pointperm[pl], n], vals[pl]; atol=1e-7), 1:np), 1:NN)   # Q3 values carry ~1e-9 round-off
        n === nothing && return nothing
        used[n] = true
        nodeperm[lin] = n
    end
    TPTable{D,Q,P,P * Q}(SMatrix{P,Q,Float64,P * Q}(N1), SMatrix{P,Q,Float64,P * Q}(dN1), nodeperm, pointperm)
end

# ---------------------------------------------------------------------------------------------
# Sum-factorised kernels (one scalar component). Tensor-ordered buffers, Q = nodes per
# direction, P = points per direction. All loops have static bounds.

"""
    tp_interpolate!(out, tp, u)

Values and reference gradients at the tensor-ordered points of a scalar nodal field `u`
(tensor-ordered, length Q^D): `out[pl, 1]` = value, `out[pl, 1+d]` = ∂/∂ξ_d.
"""
function tp_interpolate!(out::Matrix{Float64}, tp::TPTable{2,Q,P}, u::AbstractVector, tmp::Matrix{Float64}) where {Q,P}
    N1, dN1 = tp.N1, tp.dN1
    # sweep direction 1: tmp[a, j] = Σ_i N1[a,i] u[i,j]  (and with dN1 into tmp2)
    @inbounds for j in 1:Q, a in 1:P
        s = 0.0; ds = 0.0
        for i in 1:Q
            v = u[i+(j-1)*Q]
            s += N1[a, i] * v; ds += dN1[a, i] * v
        end
        tmp[a, j] = s; tmp[a+P, j] = ds
    end
    @inbounds for b in 1:P, a in 1:P
        v = 0.0; g1 = 0.0; g2 = 0.0
        for j in 1:Q
            v += N1[b, j] * tmp[a, j]; g1 += N1[b, j] * tmp[a+P, j]; g2 += dN1[b, j] * tmp[a, j]
        end
        pl = a + (b - 1) * P
        out[pl, 1] = v; out[pl, 2] = g1; out[pl, 3] = g2
    end
    out
end

function tp_interpolate!(out::Matrix{Float64}, tp::TPTable{3,Q,P}, u::AbstractVector, tmp::Matrix{Float64}) where {Q,P}
    N1, dN1 = tp.N1, tp.dN1
    # tmp1[a, (j,k)] : value and ∂1 after sweep 1  → stored in tmp[:, 1:2*Q*Q]
    @inbounds for k in 1:Q, j in 1:Q, a in 1:P
        s = 0.0; ds = 0.0
        for i in 1:Q
            v = u[i+(j-1)*Q+(k-1)*Q*Q]
            s += N1[a, i] * v; ds += dN1[a, i] * v
        end
        c = j + (k - 1) * Q
        tmp[a, c] = s; tmp[a+P, c] = ds
    end
    # sweep 2: tmp2[(a,b), k] for value, ∂1, ∂2 → stored in tmp[:, 2*Q*Q+1 : 2*Q*Q+3*Q]
    o2 = 2 * Q * Q
    @inbounds for k in 1:Q, b in 1:P, a in 1:P
        v = 0.0; g1 = 0.0; g2 = 0.0
        for j in 1:Q
            c = j + (k - 1) * Q
            v += N1[b, j] * tmp[a, c]; g1 += N1[b, j] * tmp[a+P, c]; g2 += dN1[b, j] * tmp[a, c]
        end
        r = a + (b - 1) * P
        tmp[r, o2+k] = v; tmp[r, o2+Q+k] = g1; tmp[r, o2+2Q+k] = g2
    end
    @inbounds for c in 1:P, b in 1:P, a in 1:P
        r = a + (b - 1) * P
        v = 0.0; g1 = 0.0; g2 = 0.0; g3 = 0.0
        for k in 1:Q
            v += N1[c, k] * tmp[r, o2+k]; g1 += N1[c, k] * tmp[r, o2+Q+k]
            g2 += N1[c, k] * tmp[r, o2+2Q+k]; g3 += dN1[c, k] * tmp[r, o2+k]
        end
        pl = r + (c - 1) * P * P
        out[pl, 1] = v; out[pl, 2] = g1; out[pl, 3] = g2; out[pl, 4] = g3
    end
    out
end

"""
    tp_contract!(r, tp, coef, tmp)

Nodal contraction of per-point coefficients: `r[n] = Σ_pl N_n(pl) coef[pl,1] + Σ_d ∂_d N_n(pl) coef[pl,1+d]`
(tensor-ordered nodes and points; coefficients already include the weights).
"""
function tp_contract!(r::AbstractVector, tp::TPTable{2,Q,P}, coef::Matrix{Float64}, tmp::Matrix{Float64}) where {Q,P}
    N1, dN1 = tp.N1, tp.dN1
    # sweep over direction 2 first: tmp[a, j] = Σ_b N1[b,j] c0[a,b] + dN1[b,j] c2[a,b] ; tmp[a+P, j] = Σ_b N1[b,j] c1[a,b]
    @inbounds for j in 1:Q, a in 1:P
        s = 0.0; s1 = 0.0
        for b in 1:P
            pl = a + (b - 1) * P
            s += N1[b, j] * coef[pl, 1] + dN1[b, j] * coef[pl, 3]
            s1 += N1[b, j] * coef[pl, 2]
        end
        tmp[a, j] = s; tmp[a+P, j] = s1
    end
    @inbounds for j in 1:Q, i in 1:Q
        s = 0.0
        for a in 1:P
            s += N1[a, i] * tmp[a, j] + dN1[a, i] * tmp[a+P, j]
        end
        r[i+(j-1)*Q] = s
    end
    r
end

function tp_contract!(r::AbstractVector, tp::TPTable{3,Q,P}, coef::Matrix{Float64}, tmp::Matrix{Float64}) where {Q,P}
    N1, dN1 = tp.N1, tp.dN1
    o2 = 2 * Q * Q
    # sweep 3: for each (a,b), k: t0 = Σ_c N1[c,k] c0 + dN1[c,k] c3 ; t1 = Σ_c N1 c1 ; t2 = Σ_c N1 c2
    @inbounds for k in 1:Q, b in 1:P, a in 1:P
        r2 = a + (b - 1) * P
        t0 = 0.0; t1 = 0.0; t2 = 0.0
        for c in 1:P
            pl = r2 + (c - 1) * P * P
            t0 += N1[c, k] * coef[pl, 1] + dN1[c, k] * coef[pl, 4]
            t1 += N1[c, k] * coef[pl, 2]
            t2 += N1[c, k] * coef[pl, 3]
        end
        tmp[r2, o2+k] = t0; tmp[r2, o2+Q+k] = t1; tmp[r2, o2+2Q+k] = t2
    end
    # sweep 2: for each a, (j,k): s0 = Σ_b N1[b,j] t0 + dN1[b,j] t2 ; s1 = Σ_b N1[b,j] t1
    @inbounds for k in 1:Q, j in 1:Q, a in 1:P
        s0 = 0.0; s1 = 0.0
        for b in 1:P
            r2 = a + (b - 1) * P
            s0 += N1[b, j] * tmp[r2, o2+k] + dN1[b, j] * tmp[r2, o2+2Q+k]
            s1 += N1[b, j] * tmp[r2, o2+Q+k]
        end
        c = j + (k - 1) * Q
        tmp[a, c] = s0; tmp[a+P, c] = s1
    end
    @inbounds for k in 1:Q, j in 1:Q, i in 1:Q
        c = j + (k - 1) * Q
        s = 0.0
        for a in 1:P
            s += N1[a, i] * tmp[a, c] + dN1[a, i] * tmp[a+P, c]
        end
        r[i+(j-1)*Q+(k-1)*Q*Q] = s
    end
    r
end

"Size of the scratch matrix for the sum-factorised kernels."
tp_scratch(::TPTable{2,Q,P}) where {Q,P} = zeros(2P, Q)
tp_scratch(::TPTable{3,Q,P}) where {Q,P} = zeros(2P * P, 2Q * Q + 3Q)

# Fused element kernels.
#
# Per cell and quadrature point: geometry, shape-function weights W (values and physical
# gradients, stored node-contiguous), values and gradients of the unknowns, auxiliary values;
# then the residual coefficients R (or the tangent T) in slot space, contracted with W:
#
#     r[(n,c)]          += dV Σ_jj W[n, jj] R[slot(c, jj)]
#     K[(n,c), (m,k)]   += dV Σ_jj Σ_ll W_f[n, jj] T[slot_f(c, jj), slot_g(k, ll)] W_g[m, ll]
#
# The innermost loops run over nodes with unit stride (when the dofs of each component are
# consecutive), so they vectorise on CPU and map to coalesced accesses on GPU. No BLAS calls.
#
# H(div) (Raviart–Thomas) fields use the same contractions with one pseudo-component and one
# derivative index per value component and per gradient entry; their W is the contravariant Piola
# map of the reference tables, applied per point (see "H(div) fields" below and `PiolaTable`).

"""
    SlotSpec{NC,LV,HV,HG,S,COLS}

Slots of the leaves of one field (test or trial) in the slot vector: `sm[c, jj]` is the slot of
component `c` and "derivative index" `jj`, where the value comes first if present (`HV`) and then
∂/∂x_1…∂/∂x_D if the gradient is present (`HG`). `LV` = number of derivative indices; `COLS[jj]`
is the column of the physical weight matrix W (`point_weights!`) of derivative index `jj`.

For an H(div) field (`PiolaTable`) there is one pseudo-component and the derivative indices run over
the value components (D) and then the gradient entries [J,c] column-major (D²), in the order of the
columns of W.
"""
struct SlotSpec{NC,LV,HV,HG,S,COLS}
    sm::S
end

SlotSpec(::FieldTable, smap::Matrix{Int}, D::Int) = SlotSpec(smap, D)
SlotSpec(::PiolaTable{D}, smap::Matrix{Int}, ::Int) where {D} = piola_slotspec(smap, D)

"Nodal field: `smap[c, 1 + J]` = slot of component `c`, value (`J = 0`) or ∂/∂x_J (`slot_map`)."
function SlotSpec(smap::Matrix{Int}, D::Int)
    nc = size(smap, 1)
    hv = any(>(0), smap[:, 1])
    hg = any(>(0), smap[:, 2:end])
    cols = vcat(hv ? [1] : Int[], hg ? collect(2:D+1) : Int[])
    lv = length(cols)
    sm = SMatrix{nc,lv,Int,nc * lv}(smap[:, cols])
    all(>(0), sm) || error("HyperFEMKernels: partial use of a field's components is not supported")
    SlotSpec{nc,lv,hv,hg,typeof(sm),Tuple(cols)}(sm)
end

"H(div) field: the slots of all D value components and/or all D² gradient entries, in one row."
function piola_slotspec(smap::Matrix{Int}, D::Int)
    size(smap, 1) == D || error("HyperFEMKernels: an H(div) field has $D components")
    hv = any(>(0), smap[:, 1])
    hg = any(>(0), smap[:, 2:end])
    slots = vcat(hv ? smap[:, 1] : Int[], hg ? vec(permutedims(smap[:, 2:end])) : Int[])   # [J,c] column-major: J fastest
    cols = vcat(hv ? collect(1:D) : Int[], hg ? collect(D+1:D+D*D) : Int[])
    lv = length(slots)
    sm = SMatrix{1,lv,Int,lv}(permutedims(slots))
    all(>(0), sm) || error("HyperFEMKernels: partial use of a field's components is not supported")
    SlotSpec{1,lv,hv,hg,typeof(sm),Tuple(cols)}(sm)
end

"Column of the weight matrix W for derivative index jj (nodal: W[:,1] = values, W[:,1+J] = ∂_J)."
@inline wcol(::SlotSpec{NC,LV,HV,HG,S,COLS}, jj) where {NC,LV,HV,HG,S,COLS} = COLS[jj]

"Shape-function weights of `t` at point `q`: W[n,1] = N_n, W[n,1+J] = ∂N_n/∂x_J."
@inline point_weights!(W::Matrix{Float64}, t::FieldTable, q::Int, iJt, Jt) = point_weights!(W, t, q, iJt)
@inline function point_weights!(W::Matrix{Float64}, t::FieldTable{D,T,NN}, q::Int, iJt) where {D,T,NN}
    @inbounds @simd for n in 1:NN
        W[n, 1] = t.Nt[n, q]
    end
    @inbounds for J in 1:D
        a = ntuple(K -> iJt[J, K], Val(D))
        @simd for n in 1:NN
            s = 0.0
            for K in 1:D
                s += a[K] * t.dNt[n, K, q]
            end
            W[n, 1+J] = s
        end
    end
    W
end

@inline ldofidx(t::AbstractFieldTable{D,T,NN,NC,true}, n, c) where {D,T,NN,NC} = t.ldof[1, c] + n - 1
@inline ldofidx(t::AbstractFieldTable{D,T,NN,NC,false}, n, c) where {D,T,NN,NC} = t.ldof[n, c]

# ---------------------------------------------------------------------------------------------
# Reference-gradient evaluation and contraction (no per-point transformation of the shape functions)
#
# With Jt the (transposed) Jacobian of the cell map and iJt = Jt⁻¹, a physical gradient is
# ∂N_n/∂x_J = Σ_K iJt[J,K] ∂N_n/∂ξ_K. The chain rule is applied to the field, not to the shape
# functions: the reference gradient of the field, Ĝ[c,K] = Σ_n x_{n,c} ∂N_n/∂ξ_K, is accumulated
# in one pass over the nodes (all components and directions at once) and transformed once,
# ∇u[J,c] = Σ_K iJt[J,K] Ĝ[c,K]; the residual coefficients R[c,J] of the gradient slots are
# transformed once per point, R̃[c,K] = Σ_J iJt[J,K] R[c,J], and contracted with the reference
# gradient tables. The tables are shared by all cells (GPU: constant memory); no per-cell buffer.

"""
    FieldUse{HV,HG}

Which derivatives of an unknown field the weak form evaluates at the points: its value (`HV`)
and/or its gradient (`HG`). What is not used is not computed.
"""
struct FieldUse{HV,HG} end

@inline _smat(iJt::TensorValue{D,D,Float64}) where {D} = SMatrix{D,D,Float64,D * D}(iJt.data)   # both column-major [J,K]

"""
    field_point_values(t, xl, off, q, iJt, use) -> (u, ∇u)

Value and gradient (Gridap convention: `∇u[J,c] = ∂u_c/∂x_J`) of a field at point `q` from its
local dof values (from position `off` of `xl`) and the reference tables, in a single pass over
the nodes. Derivatives not in `use` are returned as zero.
"""
@inline field_point_values(t::FieldTable, xl, off::Int, q::Int, iJt, Jt, use::FieldUse) = field_point_values(t, xl, off, q, iJt, use)
@inline function field_point_values(t::FieldTable{D,T,NN,NC}, xl, off::Int, q::Int, iJt,
                                    ::FieldUse{HV,HG}) where {D,T,NN,NC,HV,HG}
    v = zero(SVector{NC,Float64})
    G = zero(SMatrix{NC,D,Float64,NC * D})          # Ĝ[c,K]
    @inbounds for n in 1:NN
        x = SVector{NC,Float64}(ntuple(c -> xl[off+ldofidx(t, n, c)], Val(NC)))
        HV && (v = muladd.(x, t.Nt[n, q], v))
        if HG
            d = SVector{D,Float64}(ntuple(K -> t.dNt[n, K, q], Val(D)))
            G = muladd.(x * transpose(d), 1.0, G)
        end
    end
    g = HG ? _smat(iJt) * transpose(G) : zero(SMatrix{D,NC,Float64,D * NC})   # [J,c]
    _field_values(T, v, g)
end
@inline _field_values(::Type{Float64}, v::SVector{1}, g::SMatrix{D,1}) where {D} =
    (v[1], VectorValue{D,Float64}(Tuple(g)))
@inline _field_values(::Type{VectorValue{NC,Float64}}, v::SVector{NC}, g::SMatrix{D,NC}) where {D,NC} =
    (VectorValue{NC,Float64}(Tuple(v)), TensorValue{D,NC,Float64,D * NC}(Tuple(g)))

"""
    field_values(tables, locs, use, xl, q, iJt, Jt) -> FV

`field_point_values` of every field, as a tuple. Generated (no closures) and inlined, so that the
node loops of every field are compiled in the caller's loop.
"""
@inline @generated function field_values(tables::NTuple{NF,Any}, locs, use, xl, q, iJt, Jt) where {NF}
    ex = [:(field_point_values(tables[$i], xl, locs[$i], q, iJt, Jt, use[$i])) for i in 1:NF]
    :(($(ex...),))
end

"Same with one `iJt` per field (skeleton: the inverse Jacobian of each side)."
@inline @generated function field_values_sides(tables::NTuple{NF,Any}, locs, use, xl, q, iJs) where {NF}
    ex = [:(field_point_values(tables[$i], xl, locs[$i], q, iJs[$i], use[$i])) for i in 1:NF]
    :(($(ex...),))
end

"""
    residual_weights(t, sp, np) -> B (nn × np·LV)

Reference weights of the slots of one field at all points: `B[n, (q-1)LV + jj] = N_n(ξ_q)` for the
value slot and `∂N_n/∂ξ_K(ξ_q)` for the gradient slots. Shared by all cells; the element residual
of a cell is `re_f = B · C_f`, with `C_f[(q-1)LV + jj, c]` the transformed coefficients of the cell
(`residual_coefs!`).
"""
function residual_weights(t::FieldTable{D,T,NN,NC}, sp::SlotSpec{NC,LV,HV,HG}, np::Int) where {D,T,NN,NC,LV,HV,HG}
    B = zeros(NN, np * LV)
    for q in 1:np, n in 1:NN
        HV && (B[n, (q-1)*LV+1] = t.Nt[n, q])
        HG && for K in 1:D
            B[n, (q-1)*LV+(HV ? 1 : 0)+K] = t.dNt[n, K, q]
        end
    end
    B
end

"Transformed residual coefficients of one field at point `q` into `C[(q-1)LV + jj, c]` (see `residual_weights`)."
@inline function residual_coefs!(C, ::FieldTable{D}, sp::SlotSpec{NC,LV,HV,HG}, R, q::Int, iJt, Jt, dV) where {D,NC,LV,HV,HG}
    LV == 0 && return C
    @inbounds for c in 1:NC
        HV && (C[(q-1)*LV+1, c] = dV * R[sp.sm[c, 1]])
        if HG
            for K in 1:D
                s = 0.0
                for J in 1:D
                    s += iJt[J, K] * R[sp.sm[c, (HV ? 1 : 0)+J]]
                end
                C[(q-1)*LV+(HV ? 1 : 0)+K, c] = dV * s
            end
        end
    end
    C
end

"""
    contract_residual!(re, t, loc, sp, R, q, iJt, dV)

Residual contribution of one field at point `q` from the reference tables:
`re[(n,c)] += dV (R[c,0] N_n + Σ_K R̃[c,K] ∂N_n/∂ξ_K)` with `R̃[c,K] = Σ_J iJt[J,K] R[c,J]`.
"""
@inline contract_residual!(re, t::FieldTable, loc::Int, sp::SlotSpec, R, q::Int, iJt, Jt, dV) = contract_residual!(re, t, loc, sp, R, q, iJt, dV)
@inline function contract_residual!(re, t::FieldTable{D,T,NN,NC}, loc::Int, sp::SlotSpec{NC,LV,HV,HG},
                                    R, q::Int, iJt, dV) where {D,T,NN,NC,LV,HV,HG}
    LV == 0 && return nothing
    @inbounds for c in 1:NC
        r0 = HV ? dV * R[sp.sm[c, 1]] : 0.0
        rt = ntuple(K -> begin
                s = 0.0
                if HG
                    for J in 1:D
                        s += iJt[J, K] * R[sp.sm[c, (HV ? 1 : 0)+J]]
                    end
                end
                dV * s
            end, Val(D))
        @simd for n in 1:NN
            acc = HV ? r0 * t.Nt[n, q] : 0.0
            if HG
                for K in 1:D
                    acc += rt[K] * t.dNt[n, K, q]
                end
            end
            re[loc+ldofidx(t, n, c)] += acc
        end
    end
    nothing
end

# ---------------------------------------------------------------------------------------------
# H(div) fields: contravariant Piola map per point (see `PiolaTable`)
#
# With v̂ the reference vector shape function and Ĝ = ∇̂v̂ its reference gradient ([K,L] = ∂v̂_L/∂ξ_K),
# the physical value and gradient (without the dof sign, applied at gather/scatter) are
#     v[c]    = (1/|det Jt|) Σ_K v̂[K] Jt[K,c]
#     ∇v[J,c] = (1/|det Jt|) Σ_K Σ_L iJt[J,K] Ĝ[K,L] Jt[L,c]
# Both are linear in (v̂, Ĝ): they are applied once per point to the field (sum over the dofs of
# the reference tables first) and, transposed, to the residual coefficients.

@inline _smat_t(Jt::TensorValue{D,D,Float64}) where {D} = SMatrix{D,D,Float64,D * D}(Jt.data)

"Physical weights of an H(div) field at point `q`: W[n, c] = v_n[c], W[n, D + J + D(c-1)] = ∇v_n[J,c]."
@inline function point_weights!(W::Matrix{Float64}, t::PiolaTable{D,NN}, q::Int, iJt, Jt) where {D,NN}
    J = _smat_t(Jt); iJ = _smat_t(iJt)
    s = 1.0 / meas(Jt)
    @inbounds for n in 1:NN
        v = SVector{D,Float64}(ntuple(K -> t.Vt[n, K, q], Val(D)))
        G = SMatrix{D,D,Float64,D * D}(ntuple(i -> t.dVt[n, (i-1)%D+1, (i-1)÷D+1, q], Val(D * D)))   # [K,L]
        pv = s * (transpose(J) * v)                  # [c]
        pg = s * (iJ * G * J)                        # [J,c]
        for c in 1:D
            W[n, c] = pv[c]
        end
        for i in 1:D*D
            W[n, D+i] = pg[i]
        end
    end
    W
end

"Value and gradient of an H(div) field at point `q` from its local (signed) dof values and the reference tables."
@inline function field_point_values(t::PiolaTable{D,NN}, xl, off::Int, q::Int, iJt, Jt,
                                    ::FieldUse{HV,HG}) where {D,NN,HV,HG}
    v = zero(SVector{D,Float64})
    G = zero(SMatrix{D,D,Float64,D * D})
    @inbounds for n in 1:NN
        x = xl[off+n]
        HV && (v = muladd.(SVector{D,Float64}(ntuple(K -> t.Vt[n, K, q], Val(D))), x, v))
        HG && (G = muladd.(SMatrix{D,D,Float64,D * D}(ntuple(i -> t.dVt[n, (i-1)%D+1, (i-1)÷D+1, q], Val(D * D))), x, G))
    end
    s = 1.0 / meas(Jt)
    J = _smat_t(Jt)
    pv = HV ? s * (transpose(J) * v) : v
    pg = HG ? s * (_smat_t(iJt) * G * J) : G
    (VectorValue{D,Float64}(Tuple(pv)), TensorValue{D,D,Float64,D * D}(Tuple(pg)))
end

"""
Transformed residual coefficients of an H(div) field at a point: the coefficients `R` of the value
slots (`Rv[c]`) and gradient slots (`Rg[J,c]`) pulled back to the reference tables,
`R̃v[K] = (1/|det Jt|) Σ_c Jt[K,c] Rv[c]`, `R̃g[K,L] = (1/|det Jt|) Σ_J Σ_c iJt[J,K] Rg[J,c] Jt[L,c]`,
as one static vector in the order of the derivative indices (values, then gradient entries with K fastest).
"""
@inline function piola_coefs(sp::SlotSpec{1,LV,HV,HG}, R, iJt, Jt, dV, ::Val{D}) where {LV,HV,HG,D}
    s = dV / meas(Jt)
    J = _smat_t(Jt)
    rv = if HV
        Rv = SVector{D,Float64}(ntuple(c -> @inbounds(R[sp.sm[1, c]]), Val(D)))
        s * (J * Rv)
    else
        zero(SVector{0,Float64})
    end
    rg = if HG
        o = HV ? D : 0
        Rg = SMatrix{D,D,Float64,D * D}(ntuple(i -> @inbounds(R[sp.sm[1, o+i]]), Val(D * D)))   # [J,c]
        s * (transpose(_smat_t(iJt)) * Rg * transpose(J))                                      # [K,L]
    else
        zero(SMatrix{0,0,Float64,0})
    end
    SVector{LV,Float64}((Tuple(rv)..., Tuple(rg)...))
end

"Reference weights of the slots of an H(div) field at all points (see the nodal `residual_weights`)."
function residual_weights(t::PiolaTable{D,NN}, sp::SlotSpec{1,LV,HV,HG}, np::Int) where {D,NN,LV,HV,HG}
    B = zeros(NN, np * LV)
    for q in 1:np, n in 1:NN
        HV && for K in 1:D
            B[n, (q-1)*LV+K] = t.Vt[n, K, q]
        end
        HG && for L in 1:D, K in 1:D
            B[n, (q-1)*LV+(HV ? D : 0)+K+D*(L-1)] = t.dVt[n, K, L, q]
        end
    end
    B
end

@inline function residual_coefs!(C, ::PiolaTable{D}, sp::SlotSpec{1,LV}, R, q::Int, iJt, Jt, dV) where {D,LV}
    LV == 0 && return C
    rt = piola_coefs(sp, R, iJt, Jt, dV, Val(D))
    @inbounds for jj in 1:LV
        C[(q-1)*LV+jj, 1] = rt[jj]
    end
    C
end

"Residual contribution of an H(div) field at point `q` from the reference tables."
@inline function contract_residual!(re, t::PiolaTable{D,NN}, loc::Int, sp::SlotSpec{1,LV,HV,HG},
                                    R, q::Int, iJt, Jt, dV) where {D,NN,LV,HV,HG}
    LV == 0 && return nothing
    rt = piola_coefs(sp, R, iJt, Jt, dV, Val(D))
    @inbounds @simd for n in 1:NN
        acc = 0.0
        if HV
            for K in 1:D
                acc += rt[K] * t.Vt[n, K, q]
            end
        end
        if HG
            o = HV ? D : 0
            for L in 1:D, K in 1:D
                acc += rt[o+K+D*(L-1)] * t.dVt[n, K, L, q]
            end
        end
        re[loc+n] += acc
    end
    nothing
end

"Value and gradient (Gridap convention) of a field at the point from its local dof values and the physical weights `W`."
@inline function field_point_values(t::FieldTable{D,Float64,NN}, xl, off::Int, W) where {D,NN}
    u = 0.0
    @inbounds @simd for n in 1:NN
        u += xl[off+ldofidx(t, n, 1)] * W[n, 1]
    end
    g = ntuple(J -> begin
        s = 0.0
        @inbounds @simd for n in 1:NN
            s += xl[off+ldofidx(t, n, 1)] * W[n, 1+J]
        end
        s
    end, Val(D))
    (u, VectorValue{D,Float64}(g))
end

@inline function field_point_values(t::FieldTable{D,VectorValue{NC,Float64},NN,NC}, xl, off::Int, W) where {D,NN,NC}
    u = ntuple(c -> begin
        s = 0.0
        @inbounds @simd for n in 1:NN
            s += xl[off+ldofidx(t, n, c)] * W[n, 1]
        end
        s
    end, Val(NC))
    g = ntuple(i -> begin
        J = (i - 1) % D + 1
        c = (i - 1) ÷ D + 1
        s = 0.0
        @inbounds @simd for n in 1:NN
            s += xl[off+ldofidx(t, n, c)] * W[n, 1+J]
        end
        s
    end, Val(D * NC))
    (VectorValue{NC,Float64}(u), TensorValue{D,NC,Float64,D * NC}(g))
end

"Residual contribution of one field."
@inline function contract_residual!(re, t::AbstractFieldTable{D,T,NN,NC}, loc::Int, sp::SlotSpec{NC,LV},
                                    R, W, dV) where {D,T,NN,NC,LV}
    LV == 0 && return nothing
    @inbounds for c in 1:NC
        r = ntuple(jj -> dV * R[sp.sm[c, jj]], Val(LV))
        cols = ntuple(jj -> wcol(sp, jj), Val(LV))
        @simd for n in 1:NN
            acc = 0.0
            for jj in 1:LV
                acc += r[jj] * W[n, cols[jj]]
            end
            re[loc+ldofidx(t, n, c)] += acc
        end
    end
    nothing
end

"Jacobian block (test field f, trial field g), stored transposed: KeT[trial dof, test dof]."
@inline function contract_jacobian!(KeT, tf::AbstractFieldTable{D,Tf,NNf,NCf}, locf::Int, spf::SlotSpec{NCf,LVf}, Wf,
                                    tg::AbstractFieldTable{D,Tg,NNg,NCg}, locg::Int, spg::SlotSpec{NCg,LVg}, Wg,
                                    T, dV) where {D,Tf,NNf,NCf,LVf,Tg,NNg,NCg,LVg}
    (LVf == 0 || LVg == 0) && return nothing
    colsf = ntuple(jj -> wcol(spf, jj), Val(LVf))
    colsg = ntuple(ll -> wcol(spg, ll), Val(LVg))
    @inbounds for n in 1:NNf
        wn = ntuple(jj -> dV * Wf[n, colsf[jj]], Val(LVf))
        for c in 1:NCf
            i = locf + ldofidx(tf, n, c)
            for k in 1:NCg
                # z[ll] = Σ_jj W_f[n, jj] T[slot_f(c, jj), slot_g(k, ll)]
                z = ntuple(ll -> begin
                        su = spg.sm[k, ll]
                        acc = 0.0
                        for jj in 1:LVf
                            acc += wn[jj] * T[spf.sm[c, jj], su]
                        end
                        acc
                    end, Val(LVg))
                @simd for m in 1:NNg
                    acc = 0.0
                    for ll in 1:LVg
                        acc += z[ll] * Wg[m, colsg[ll]]
                    end
                    KeT[locg+ldofidx(tg, m, k), i] += acc
                end
            end
        end
    end
    nothing
end

# ---------------------------------------------------------------------------------------------
# Jacobian as a matrix product (per cell, written once)
#
# For test field f (component c) and trial field g (component k):
#
#     K_ck[m, n] = Σ_r G_ck[m, r] B_f[r, n],   r = (q, jj)
#     G_ck[m, (q,jj)] = dV_q Σ_ll T_q[slot_f(c,jj), slot_g(k,ll)] W_g,q[m, ll]
#     B_f[(q,jj), n]  = W_f,q[n, jj]
#
# G and B are filled during the point loop; the products are done once per cell with a
# register-tiled kernel written in plain Julia (no BLAS, see `gemm_add!`), so the same code can run per thread
# block on a GPU. Requires node-contiguous dofs (FieldTable CONTIG = true).

"B_f[(q-1)*LV + jj, n] = W[n, col(jj)] for point q."
@inline function store_test_weights!(B, sp::SlotSpec{NC,LV}, W, q::Int, NN::Int) where {NC,LV}
    @inbounds for n in 1:NN, jj in 1:LV
        B[(q-1)*LV+jj, n] = W[n, wcol(sp, jj)]
    end
    B
end

"G[m, (q-1)*LVf+jj, k, c] = dV Σ_ll T[slot_f(c,jj), slot_g(k,ll)] W_g[m, ll] for point q."
@inline function store_trial_block!(G, spf::SlotSpec{NCf,LVf}, spg::SlotSpec{NCg,LVg}, Wg, T, q::Int, dV,
                                    NNg::Int) where {NCf,LVf,NCg,LVg}
    (LVf == 0 || LVg == 0) && return G
    colsg = ntuple(ll -> wcol(spg, ll), Val(LVg))
    @inbounds for c in 1:NCf, k in 1:NCg, jj in 1:LVf
        z = ntuple(ll -> dV * T[spf.sm[c, jj], spg.sm[k, ll]], Val(LVg))
        r = (q - 1) * LVf + jj
        @simd for m in 1:NNg
            acc = 0.0
            for ll in 1:LVg
                acc += z[ll] * Wg[m, colsg[ll]]
            end
            G[m, r, k, c] = acc
        end
    end
    G
end

"""
    gemm_add!(C, om, on, A, k, c, B, M, N, R)

`C[om + m, on + n] += Σ_r A[m, r, k, c] B[r, n]` for m ≤ M, n ≤ N, r ≤ R. Register-tiled
(8×4 tiles of C, remainders in 4/2/1-row and 1-column tiles, all with static sizes), plain
Julia, no BLAS. On the M5 Max it runs at 30–45 GFLOP/s for the Q2/Q3 shapes (OpenBLAS: 40).
"""
@inline function gemm_add!(C, om::Int, on::Int, A, k::Int, c::Int, B, M::Int, N::Int, R::Int)
    lda = size(A, 1)
    offA = ((c - 1) * size(A, 3) + (k - 1)) * size(A, 1) * size(A, 2)
    n0 = 1
    while n0 + 3 <= N
        gemm_cols!(C, om, on, A, offA, lda, B, n0, M, R, Val(4)); n0 += 4
    end
    while n0 <= N
        gemm_cols!(C, om, on, A, offA, lda, B, n0, M, R, Val(1)); n0 += 1
    end
    C
end

"Columns n0 … n0+NB-1 of the product, rows in tiles of 8 then 4, 2, 1."
@inline function gemm_cols!(C, om, on, A, offA, lda, B, n0, M, R, vnb::Val{NB}) where {NB}
    m0 = 1
    while m0 + 7 <= M
        gemm_tile!(C, om, on, A, offA, lda, B, m0, n0, R, Val(8), vnb); m0 += 8
    end
    rem = M - m0 + 1
    if rem >= 4
        gemm_tile!(C, om, on, A, offA, lda, B, m0, n0, R, Val(4), vnb); m0 += 4; rem -= 4
    end
    if rem >= 2
        gemm_tile!(C, om, on, A, offA, lda, B, m0, n0, R, Val(2), vnb); m0 += 2; rem -= 2
    end
    rem >= 1 && gemm_tile!(C, om, on, A, offA, lda, B, m0, n0, R, Val(1), vnb)
    nothing
end

"One MB×NB tile of C kept in registers over the whole r loop."
@inline function gemm_tile!(C, om, on, A, offA, lda, B, m0, n0, R, ::Val{MB}, ::Val{NB}) where {MB,NB}
    acc = ntuple(_ -> zero(SVector{MB,Float64}), Val(NB))
    @inbounds for r in 1:R
        base = offA + m0 - 1 + (r - 1) * lda
        a = SVector{MB,Float64}(ntuple(i -> A[base+i], Val(MB)))
        acc = gemm_accum(a, acc, B, r, n0)
    end
    @inbounds for j in 1:NB, i in 1:MB
        C[om+m0+i-1, on+n0+j-1] += acc[j][i]
    end
    nothing
end
# acc is passed and returned, never captured: a closure over a reassigned variable would box it
@inline gemm_accum(a, acc::NTuple{NB,Any}, B, r, n0) where {NB} =
    ntuple(j -> muladd.(a, @inbounds(B[r, n0+j-1]), acc[j]), Val(NB))

"Add the products of one block (f, g) to KeT (transposed element matrix)."
@inline function block_products!(KeT, tf::AbstractFieldTable{D,Tf,NNf,NCf,true}, locf::Int, spf::SlotSpec{NCf,LVf}, B,
                                 tg::AbstractFieldTable{D,Tg,NNg,NCg,true}, locg::Int, spg::SlotSpec{NCg,LVg}, G,
                                 np::Int) where {D,Tf,NNf,NCf,LVf,Tg,NNg,NCg,LVg}
    (LVf == 0 || LVg == 0) && return KeT
    R = np * LVf
    for c in 1:NCf, k in 1:NCg
        gemm_add!(KeT, locg + tg.ldof[1, k] - 1, locf + tf.ldof[1, c] - 1, G, k, c, B, NNg, NNf, R)
    end
    KeT
end

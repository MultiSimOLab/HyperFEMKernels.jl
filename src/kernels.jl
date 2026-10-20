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

"""
    SlotSpec{NC,LV,HV,HG}

Slots of the leaves of one field (test or trial) in the slot vector: `sm[c, jj]` is the slot of
component `c` and "derivative index" `jj`, where the value comes first if present (`HV`) and then
∂/∂x_1…∂/∂x_D if the gradient is present (`HG`). `LV` = number of derivative indices.
"""
struct SlotSpec{NC,LV,HV,HG,S}
    sm::S
end

function SlotSpec(smap::Matrix{Int}, D::Int)
    nc = size(smap, 1)
    hv = any(>(0), smap[:, 1])
    hg = any(>(0), smap[:, 2:end])
    cols = vcat(hv ? [1] : Int[], hg ? collect(2:D+1) : Int[])
    lv = length(cols)
    sm = SMatrix{nc,lv,Int,nc * lv}(smap[:, cols])
    all(>(0), sm) || error("HyperFEMKernels: partial use of a field's components is not supported")
    SlotSpec{nc,lv,hv,hg,typeof(sm)}(sm)
end

"Column of the weight matrix W for derivative index jj (W[:,1] = values, W[:,1+J] = ∂_J)."
@inline wcol(::SlotSpec{NC,LV,HV}, jj) where {NC,LV,HV} = HV ? jj : jj + 1

"Shape-function weights of `t` at point `q`: W[n,1] = N_n, W[n,1+J] = ∂N_n/∂x_J."
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

@inline ldofidx(t::FieldTable{D,T,NN,NC,true}, n, c) where {D,T,NN,NC} = t.ldof[1, c] + n - 1
@inline ldofidx(t::FieldTable{D,T,NN,NC,false}, n, c) where {D,T,NN,NC} = t.ldof[n, c]

"Value and gradient (Gridap convention) of a field at the point from its local dof values."
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
@inline function contract_residual!(re, t::FieldTable{D,T,NN,NC}, loc::Int, sp::SlotSpec{NC,LV},
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
@inline function contract_jacobian!(KeT, tf::FieldTable{D,Tf,NNf,NCf}, locf::Int, spf::SlotSpec{NCf,LVf}, Wf,
                                    tg::FieldTable{D,Tg,NNg,NCg}, locg::Int, spg::SlotSpec{NCg,LVg}, Wg,
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
# register-tiled kernel written in plain Julia (no BLAS), so the same code can run per thread
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

"C[om + m, on + n] += Σ_r A[m, r, k, c] B[r, n] for m ≤ M, n ≤ N, r ≤ R (8×4 register tiles)."
@inline function gemm_add!(C, om::Int, on::Int, A, k::Int, c::Int, B, M::Int, N::Int, R::Int)
    @inbounds for n0 in 1:4:N
        nb = min(4, N - n0 + 1)
        for m0 in 1:8:M
            mb = min(8, M - m0 + 1)
            if nb == 4 && mb == 8
                c1 = c2 = c3 = c4 = zero(SVector{8,Float64})
                for r in 1:R
                    a = SVector{8,Float64}(ntuple(i -> A[m0+i-1, r, k, c], Val(8)))
                    c1 += a * B[r, n0]
                    c2 += a * B[r, n0+1]
                    c3 += a * B[r, n0+2]
                    c4 += a * B[r, n0+3]
                end
                for i in 1:8
                    C[om+m0+i-1, on+n0] += c1[i]
                    C[om+m0+i-1, on+n0+1] += c2[i]
                    C[om+m0+i-1, on+n0+2] += c3[i]
                    C[om+m0+i-1, on+n0+3] += c4[i]
                end
            else
                for j in n0:n0+nb-1, i in m0:m0+mb-1
                    s = 0.0
                    for r in 1:R
                        s += A[i, r, k, c] * B[r, j]
                    end
                    C[om+i, on+j] += s
                end
            end
        end
    end
    C
end

"Add the products of one block (f, g) to KeT (transposed element matrix)."
@inline function block_products!(KeT, tf::FieldTable{D,Tf,NNf,NCf,true}, locf::Int, spf::SlotSpec{NCf,LVf}, B,
                                 tg::FieldTable{D,Tg,NNg,NCg,true}, locg::Int, spg::SlotSpec{NCg,LVg}, G,
                                 np::Int) where {D,Tf,NNf,NCf,LVf,Tg,NNg,NCg,LVg}
    (LVf == 0 || LVg == 0) && return KeT
    R = np * LVf
    for c in 1:NCf, k in 1:NCg
        gemm_add!(KeT, locg + tg.ldof[1, k] - 1, locf + tf.ldof[1, c] - 1, G, k, c, B, NNg, NNf, R)
    end
    KeT
end

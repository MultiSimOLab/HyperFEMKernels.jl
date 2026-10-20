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

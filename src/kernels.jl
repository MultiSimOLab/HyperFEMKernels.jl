# Fused element kernels.
#
# Per cell and quadrature point: geometry, physical gradients of the shape functions, values and
# gradients of the unknowns, auxiliary values; then one call to the generated integrand with dual
# numbers on the test (and trial) slots gives the residual coefficients (and the tangent), which
# are contracted with the nodal shape functions. Sizes are type parameters so that loops have
# constant bounds.

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

"Weights of node `n` at point `q` for the derivative indices of a slot spec."
@inline wvec(::SlotSpec{NC,LV,true,true}, N, g) where {NC,LV} = SVector{LV,Float64}(N, Tuple(g)...)
@inline wvec(::SlotSpec{NC,LV,true,false}, N, g) where {NC,LV} = SVector{1,Float64}(N)
@inline wvec(::SlotSpec{NC,LV,false,true}, N, g) where {NC,LV} = SVector{LV,Float64}(Tuple(g))
@inline wvec(::SlotSpec{NC,0,false,false}, N, g) where {NC} = SVector{0,Float64}()

"Physical gradients of the nodal shape functions at point `q` into `buf`."
@inline function phys_grads!(buf::Vector{VectorValue{D,Float64}}, t::FieldTable{D,T,NN}, q::Int, iJt) where {D,T,NN}
    @inbounds for n in 1:NN
        buf[n] = iJt ⋅ t.dN[q, n]
    end
    buf
end

"Value and gradient (Gridap convention) of a field at point `q` from its local dof values."
@inline function field_point_values(t::FieldTable{D,Float64,NN}, xl, off::Int, q::Int, dN) where {D,NN}
    u = 0.0
    g = zero(VectorValue{D,Float64})
    @inbounds for n in 1:NN
        a = xl[off+t.ldof[n, 1]]
        u += a * t.N[q, n]
        g += a * dN[n]
    end
    (u, g)
end

@inline function field_point_values(t::FieldTable{D,VectorValue{NC,Float64},NN,NC}, xl, off::Int, q::Int, dN) where {D,NN,NC}
    u = ntuple(c -> begin
        s = 0.0
        @inbounds for n in 1:NN
            s += xl[off+t.ldof[n, c]] * t.N[q, n]
        end
        s
    end, Val(NC))
    g = ntuple(i -> begin
        J = (i - 1) % D + 1
        c = (i - 1) ÷ D + 1
        s = 0.0
        @inbounds for n in 1:NN
            s += xl[off+t.ldof[n, c]] * dN[n][J]
        end
        s
    end, Val(D * NC))
    (VectorValue{NC,Float64}(u), TensorValue{D,NC,Float64,D * NC}(g))
end

"Residual contribution of one field: r[(n,c)] += dV Σ_jj W_n[jj] R[slot(c,jj)]."
@inline function contract_residual!(re, t::FieldTable{D,T,NN,NC}, loc::Int, sp::SlotSpec{NC,LV},
                                    R, q::Int, dN, dV) where {D,T,NN,NC,LV}
    LV == 0 && return nothing
    Rm = SMatrix{NC,LV,Float64,NC * LV}(ntuple(i -> R[sp.sm[i]], Val(NC * LV)))
    @inbounds for n in 1:NN
        W = wvec(sp, t.N[q, n], dN[n])
        r = Rm * W
        for c in 1:NC
            re[loc+t.ldof[n, c]] += dV * r[c]
        end
    end
    nothing
end

"""
Jacobian block (test field f, trial field g), stored transposed (`KeT[trial, test]`):
K[(n,c),(m,k)] += dV Σ_{jj,ll} W_n[jj] T[slot_f(c,jj), slot_g(k,ll)] W_m[ll].
"""
@inline function contract_jacobian!(KeT, tf::FieldTable{D,Tf,NNf,NCf}, locf::Int, spf::SlotSpec{NCf,LVf}, dNf,
                                    tg::FieldTable{D,Tg,NNg,NCg}, locg::Int, spg::SlotSpec{NCg,LVg}, dNg,
                                    T, q::Int, dV) where {D,Tf,NNf,NCf,LVf,Tg,NNg,NCg,LVg}
    (LVf == 0 || LVg == 0) && return nothing
    Wg = ntuple(m -> wvec(spg, tg.N[q, m], dNg[m]), Val(NNg))
    @inbounds for n in 1:NNf
        Wn = wvec(spf, tf.N[q, n], dNf[n])
        for c in 1:NCf
            # Z[k, ll] = Σ_jj W_n[jj] T[slot_f(c,jj), slot_g(k,ll)]
            Z = SMatrix{NCg,LVg,Float64,NCg * LVg}(ntuple(i -> begin
                    k = (i - 1) % NCg + 1
                    ll = (i - 1) ÷ NCg + 1
                    su = spg.sm[k, ll]
                    z = 0.0
                    for jj in 1:LVf
                        z += Wn[jj] * T[spf.sm[c, jj], su]
                    end
                    z
                end, Val(NCg * LVg)))
            i = locf + tf.ldof[n, c]
            for m in 1:NNg
                zm = Z * Wg[m]
                for k in 1:NCg
                    KeT[locg+tg.ldof[m, k], i] += dV * zm[k]
                end
            end
        end
    end
    nothing
end

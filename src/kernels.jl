# Fused element kernels.
#
# Per cell and quadrature point: geometry, physical gradients of the shape functions, values and
# gradients of the unknowns, auxiliary values; then one call to the generated integrand with dual
# numbers on the test (and trial) slots gives the residual coefficients (and the tangent), which
# are contracted with the nodal shape functions.

"Physical gradients of the nodal shape functions of `t` at point `q` into `buf`."
@inline function phys_grads!(buf::Vector{VectorValue{D,Float64}}, t::FieldTable{D}, q::Int, iJt) where {D}
    @inbounds for n in 1:t.nn
        buf[n] = iJt ⋅ t.dN[q, n]
    end
    buf
end

"Value and gradient (Gridap convention) of a field at point `q` from its local dof values."
@inline function field_point_values(t::FieldTable{D,Float64}, xl, off::Int, q::Int, dN) where {D}
    u = 0.0
    g = zero(VectorValue{D,Float64})
    @inbounds for n in 1:t.nn
        a = xl[off+t.ldof[n, 1]]
        u += a * t.N[q, n]
        g += a * dN[n]
    end
    (u, g)
end

@inline function field_point_values(t::FieldTable{D,VectorValue{NC,Float64}}, xl, off::Int, q::Int, dN) where {D,NC}
    u = ntuple(c -> begin
        s = 0.0
        @inbounds for n in 1:t.nn
            s += xl[off+t.ldof[n, c]] * t.N[q, n]
        end
        s
    end, Val(NC))
    g = ntuple(i -> begin
        J = (i - 1) % D + 1
        c = (i - 1) ÷ D + 1
        s = 0.0
        @inbounds for n in 1:t.nn
            s += xl[off+t.ldof[n, c]] * dN[n][J]
        end
        s
    end, Val(D * NC))
    (VectorValue{NC,Float64}(u), TensorValue{D,NC,Float64,D * NC}(g))
end

"Residual contribution of one field: r[(n,c)] += dV (N_n R[value slot c] + ∂_J N_n R[slot (J,c)])."
@inline function contract_residual!(re, t::FieldTable{D}, loc::Int, smap::Matrix{Int}, R, q::Int, dN, dV) where {D}
    @inbounds for c in 1:t.nc
        s0 = smap[c, 1]
        for n in 1:t.nn
            acc = 0.0
            s0 > 0 && (acc += t.N[q, n] * R[s0])
            g = dN[n]
            for J in 1:D
                s = smap[c, J+1]
                s > 0 && (acc += g[J] * R[s])
            end
            re[loc+t.ldof[n, c]] += dV * acc
        end
    end
    nothing
end

"Jacobian block (test field f, trial field g): K[(n,c),(m,k)] += dV Σ_{J,L} W_n[J] T[sv(c,J), su(k,L)] W_m[L]."
@inline function contract_jacobian!(Ke, tf::FieldTable{D}, locf::Int, smapf::Matrix{Int}, dNf,
                                    tg::FieldTable{D}, locg::Int, smapg::Matrix{Int}, dNg,
                                    T, q::Int, dV, Z::Matrix{Float64}) where {D}
    ncg = tg.nc
    @inbounds for n in 1:tf.nn
        Wn0 = tf.N[q, n]
        Wn = dNf[n]
        for c in 1:tf.nc
            # Z[k, L+1] = Σ_J W_n[J] T[sv(c,J), su(k,L)]
            any_nz = false
            for k in 1:ncg, L in 0:D
                su = smapg[k, L+1]
                z = 0.0
                if su > 0
                    sv = smapf[c, 1]
                    sv > 0 && (z += Wn0 * T[sv, su])
                    for J in 1:D
                        sv = smapf[c, J+1]
                        sv > 0 && (z += Wn[J] * T[sv, su])
                    end
                end
                Z[k, L+1] = z
                any_nz |= z != 0.0
            end
            any_nz || continue
            i = locf + tf.ldof[n, c]
            for m in 1:tg.nn
                Wm0 = tg.N[q, m]
                Wm = dNg[m]
                for k in 1:ncg
                    acc = Z[k, 1] * Wm0
                    for L in 1:D
                        acc += Z[k, L+1] * Wm[L]
                    end
                    Ke[i, locg+tg.ldof[m, k]] += dV * acc
                end
            end
        end
    end
    nothing
end

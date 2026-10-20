# Matrix-free application of the Jacobian (partial assembly).
#
# `prepare!(mf, x)` evaluates, per cell and quadrature point, the tangent T (slot space, from the
# linear maps or dual numbers) and the geometry (dV, Jt⁻¹) at the state `x`; `mul!(y, mf, dx)`
# then applies the Jacobian without forming it: per cell, interpolate dx to the points (sum
# factorisation on n-cubes, dense weights otherwise), multiply by T, contract back. Dirichlet dofs
# carry zero increments and their rows are not written, so the operator acts on free dofs only.

struct GroupData{D}
    T::Array{Float64,3}          # nsv*nsu × np × ncells (column-major: T[sv + (su-1)*nsv, q, cell])
    iJt::Array{Float64,3}        # D*D × np × ncells
    dV::Matrix{Float64}          # np × ncells
end

"""
    MatrixFreeJacobian(op::KernelOperator)

Linear operator `y = J(x) dx` applied without assembling. Call [`prepare!`](@ref) at each new
state `x`, then use `mul!`/`*` (e.g. inside an iterative solver). Sum factorisation is used for
fields on quadrilaterals/hexahedra with tensor-product quadrature; other fields use the dense
shape-function weights.
"""
mutable struct MatrixFreeJacobian{OP,TP,GD}
    op::OP
    tps::TP                  # per Jacobian group: tuple of TPTable or nothing per field
    data::GD                 # per Jacobian group: GroupData
    nthreads::Int
    Afb::Union{Nothing,SparseMatrixCSC{Float64,Int}}   # Gridap-assembled part (hybrid fallback)
end

function MatrixFreeJacobian(op::KernelOperator)
    tps = map(op.jgroups) do g
        map(t -> TPTable(t, g.xq), g.tables)
    end
    data = map(op.jgroups) do g
        np, nc = g.geo.np, length(g.cellmap)
        D = num_cell_dims(g.trian)
        nsv, nsu = nslots(g)
        GroupData{D}(zeros(nsv * nsu, np, nc), zeros(D * D, np, nc), zeros(np, nc))
    end
    Afb = isempty(op.fallback.jtrians) ? nothing : fallback_pattern(op.fallback, op.U, op.V, op.jac)
    MatrixFreeJacobian(op, Tuple(tps), Tuple(data), op.nthreads, Afb)
end

nslots(::Group{NSV,NSU}) where {NSV,NSU} = (NSV, NSU)

Base.size(mf::MatrixFreeJacobian) = size(mf.op.A0)
Base.size(mf::MatrixFreeJacobian, i::Int) = size(mf.op.A0, i)
Base.eltype(::MatrixFreeJacobian) = Float64
Base.:*(mf::MatrixFreeJacobian, dx::AbstractVector) = LinearAlgebra.mul!(zeros(size(mf, 1)), mf, dx)

"Evaluate tangents and geometry at the state `x` (once per Newton iteration)."
function prepare!(mf::MatrixFreeJacobian, x::AbstractVector)
    op = mf.op
    dvals = [get_dirichlet_dof_values(s) for s in field_spaces(op.U)]
    for (gi, g) in enumerate(op.jgroups)
        auxd = aux_data(op, g)
        gd = mf.data[gi]
        cells = 1:length(g.cellmap)
        nchunks = clamp(length(cells) ÷ 16, 1, mf.nthreads)
        @sync for ch in Iterators.partition(cells, cld(length(cells), nchunks))
            Threads.@spawn prepare_cells!(gd, g, op, x, dvals, auxd, ch)
        end
    end
    if mf.Afb !== nothing
        fill!(nonzeros(mf.Afb), 0.0)
        fallback_jacobian!(mf.Afb, op, x)
    end
    mf
end

function prepare_cells!(gd::GroupData{D}, g::Group{NSV,NSU,PF,TT,AUX,SV,SU,D,NF,LF}, op, x, dvals, auxd, cells) where {D,NSV,NSU,PF,TT,AUX,SV,SU,NF,LF}
    tables, locs, pf = g.tables, g.locs, g.pf
    xl = zeros(op.ndl)
    Wb = map(t -> zeros(t.nn, D + 1), tables)
    zsv = zero(SVector{NSV,Float64}); zsu = zero(SVector{NSU,Float64})
    for cell in cells
        sc = g.cellmap[cell]
        gather_cell!(xl, op.fields, x, dvals, sc)
        for q in 1:g.geo.np
            dV, iJt = point_geometry(g.geo, q, cell)
            map((t, W) -> point_weights!(W, t, q, iJt), tables, Wb)
            FV = map((t, l, W) -> field_point_values(t, xl, l, W), tables, locs, Wb)
            Aq = map(a -> @inbounds(a[q, cell]), auxd)
            T = if LF === Nothing
                ForwardDiff.jacobian(r -> ForwardDiff.gradient(s -> pf.core(pf.ops, pf.coefs, FV, Aq, s, r), zsv), zsu)
            else
                SMatrix{NSV,NSU,Float64,NSV * NSU}(apply_maps(g.lin, g.lin.state(g.lin.ops, FV, Aq), Val(NSV * NSU)))
            end
            @inbounds for i in 1:NSV*NSU
                gd.T[i, q, cell] = T[i]
            end
            @inbounds for i in 1:D*D
                gd.iJt[i, q, cell] = iJt[i]
            end
            gd.dV[q, cell] = dV
        end
    end
    nothing
end

function LinearAlgebra.mul!(y::AbstractVector, mf::MatrixFreeJacobian, dx::AbstractVector)
    fill!(y, 0.0)
    op = mf.op
    for (gi, g) in enumerate(op.jgroups)
        for color in g.colors
            nchunks = clamp(length(color) ÷ 16, 1, mf.nthreads)
            @sync for ch in Iterators.partition(color, cld(length(color), nchunks))
                Threads.@spawn apply_cells!(y, mf.data[gi], mf.tps[gi], g, op, dx, ch)
            end
        end
    end
    mf.Afb === nothing || mul!(y, mf.Afb, dx, 1.0, 1.0)
    y
end

"Scratch buffers for the apply of one group."
function apply_buffers(g::Group{NSV,NSU,PF,TT,AUX,SV,SU,D}, tps, ndl::Int) where {NSV,NSU,PF,TT,AUX,SV,SU,D}
    np = g.geo.np
    (xl=zeros(ndl), re=zeros(ndl),
     su=zeros(NSU, np), sv=zeros(NSV, np),
     Wb=map(t -> zeros(t.nn, D + 1), g.tables),
     uc=map(t -> zeros(t.nn), g.tables), rc=map(t -> zeros(t.nn), g.tables),
     pts=map(t -> zeros(np, D + 1), g.tables), coef=map(t -> zeros(np, D + 1), g.tables),
     tmp=map(tp -> tp === nothing ? zeros(0, 0) : tp_scratch(tp), tps))
end

function apply_cells!(y, gd::GroupData{D}, tps, g::Group{NSV,NSU,PF,TT,AUX,SV,SU,D}, op, dx, cells) where {NSV,NSU,PF,TT,AUX,SV,SU,D}
    b = apply_buffers(g, tps, op.ndl)
    tables, locs = g.tables, g.locs
    np = g.geo.np
    fields = op.fields
    for cell in cells
        sc = g.cellmap[cell]
        # gather the increment (zero at Dirichlet dofs)
        for fl in fields
            @inbounds for j in 1:fl.nd
                id = fl.trial_ids[j, sc]
                b.xl[fl.loc+j] = id > 0 ? dx[fl.trial_off+id] : 0.0
            end
        end
        fill!(b.su, 0.0)
        # trial slots at every point, per field
        map(tables, locs, g.smu, tps, b.uc, b.pts, b.Wb, b.tmp) do t, loc, sp, tp, uc, pts, W, tmp
            interpolate_slots!(b.su, t, loc, sp, tp, b.xl, uc, pts, W, tmp, gd, cell, np)
        end
        # sv = T * su at every point
        @inbounds for q in 1:np
            for i in 1:NSV
                s = 0.0
                for j in 1:NSU
                    s += gd.T[i+(j-1)*NSV, q, cell] * b.su[j, q]
                end
                b.sv[i, q] = s * gd.dV[q, cell]
            end
        end
        fill!(b.re, 0.0)
        map(tables, locs, g.smv, tps, b.rc, b.coef, b.Wb, b.tmp) do t, loc, sp, tp, rc, coef, W, tmp
            contract_slots!(b.re, t, loc, sp, tp, b.sv, rc, coef, W, tmp, gd, cell, np)
        end
        # scatter (free test rows only)
        for fl in fields
            @inbounds for j in 1:fl.nd
                id = fl.test_ids[j, sc]
                id > 0 && (y[fl.test_off+id] += b.re[fl.loc+j])
            end
        end
    end
    nothing
end

# --- interpolation of the trial increment into slot space -----------------------------------

function interpolate_slots!(su, t::FieldTable{D,T,NN,NC}, loc, sp::SlotSpec{NC,LV,HV,HG}, tp::TPTable, xl, uc, pts, W, tmp,
                            gd::GroupData{D}, cell, np) where {D,T,NN,NC,LV,HV,HG}
    LV == 0 && return nothing
    @inbounds for c in 1:NC
        for n in 1:NN
            uc[n] = xl[loc+t.ldof[tp.nodeperm[n], c]]
        end
        tp_interpolate!(pts, tp, uc, tmp)
        for pl in 1:np
            q = tp.pointperm[pl]
            HV && (su[sp.sm[c, 1], q] = pts[pl, 1])
            if HG
                # physical gradient: g_J = Σ_K iJt[J,K] ∂u/∂ξ_K
                for J in 1:D
                    s = 0.0
                    for K in 1:D
                        s += gd.iJt[J+(K-1)*D, q, cell] * pts[pl, 1+K]
                    end
                    su[sp.sm[c, HV ? 1 + J : J], q] = s
                end
            end
        end
    end
    nothing
end

function interpolate_slots!(su, t::FieldTable{D,T,NN,NC}, loc, sp::SlotSpec{NC,LV,HV,HG}, ::Nothing, xl, uc, pts, W, tmp,
                            gd::GroupData{D}, cell, np) where {D,T,NN,NC,LV,HV,HG}
    LV == 0 && return nothing
    @inbounds for q in 1:np
        iJt = TensorValue{D,D,Float64,D * D}(ntuple(i -> gd.iJt[i, q, cell], Val(D * D)))
        point_weights!(W, t, q, iJt)
        for c in 1:NC
            if HV
                v = 0.0
                @simd for n in 1:NN
                    v += xl[loc+ldofidx(t, n, c)] * W[n, 1]
                end
                su[sp.sm[c, 1], q] = v
            end
            if HG
                for J in 1:D
                    gJ = 0.0
                    @simd for n in 1:NN
                        gJ += xl[loc+ldofidx(t, n, c)] * W[n, 1+J]
                    end
                    su[sp.sm[c, HV ? 1 + J : J], q] = gJ
                end
            end
        end
    end
    nothing
end

# --- contraction of the test coefficients back to the nodes ---------------------------------

function contract_slots!(re, t::FieldTable{D,T,NN,NC}, loc, sp::SlotSpec{NC,LV,HV,HG}, tp::TPTable, sv, rc, coef, W, tmp,
                         gd::GroupData{D}, cell, np) where {D,T,NN,NC,LV,HV,HG}
    LV == 0 && return nothing
    @inbounds for c in 1:NC
        for pl in 1:np
            q = tp.pointperm[pl]
            coef[pl, 1] = HV ? sv[sp.sm[c, 1], q] : 0.0
            for K in 1:D
                # reference coefficient: Σ_J g_J iJt[J,K]
                s = 0.0
                if HG
                    for J in 1:D
                        s += sv[sp.sm[c, HV ? 1 + J : J], q] * gd.iJt[J+(K-1)*D, q, cell]
                    end
                end
                coef[pl, 1+K] = s
            end
        end
        tp_contract!(rc, tp, coef, tmp)
        for n in 1:NN
            re[loc+t.ldof[tp.nodeperm[n], c]] += rc[n]
        end
    end
    nothing
end

function contract_slots!(re, t::FieldTable{D,T,NN,NC}, loc, sp::SlotSpec{NC,LV,HV,HG}, ::Nothing, sv, rc, coef, W, tmp,
                         gd::GroupData{D}, cell, np) where {D,T,NN,NC,LV,HV,HG}
    LV == 0 && return nothing
    @inbounds for q in 1:np
        iJt = TensorValue{D,D,Float64,D * D}(ntuple(i -> gd.iJt[i, q, cell], Val(D * D)))
        point_weights!(W, t, q, iJt)
        for c in 1:NC
            if HV
                r0 = sv[sp.sm[c, 1], q]
                @simd for n in 1:NN
                    re[loc+ldofidx(t, n, c)] += r0 * W[n, 1]
                end
            end
            if HG
                for J in 1:D
                    rJ = sv[sp.sm[c, HV ? 1 + J : J], q]
                    @simd for n in 1:NN
                        re[loc+ldofidx(t, n, c)] += rJ * W[n, 1+J]
                    end
                end
            end
        end
    end
    nothing
end

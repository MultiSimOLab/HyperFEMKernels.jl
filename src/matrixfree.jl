# Matrix-free application of the Jacobian (partial assembly).
#
# `prepare!(mf, x)` evaluates, per cell and quadrature point, the tangent T (slot space, from the
# linear maps or dual numbers) and the geometry (dV, Jt⁻¹) at the state `x`; `mul!(y, mf, dx)`
# then applies the Jacobian without forming it: per cell, interpolate dx to the points (sum
# factorisation on n-cubes, dense weights otherwise), multiply by T, contract back. Dirichlet dofs
# carry zero increments and their rows are not written, so the operator acts on free dofs only.
# H(div) fields use the dense path with the stored Piola geometry (Jt and Jt⁻¹) and the dof signs
# at gather/scatter.
# Skeleton groups do the same per facet with the increments of both cells (dense weights, the
# Jt⁻¹ of each side; the T and dS arrays are shared by the two `GroupData` of a class).

struct GroupData{D}
    T::Array{Float64,3}          # nsv*nsu × np × ncells (column-major: T[sv + (su-1)*nsv, q, cell])
    iJt::Array{Float64,3}        # D*D × np × ncells
    Jt::Array{Float64,3}         # D*D × np × ncells (Piola map of H(div) fields)
    dV::Matrix{Float64}          # np × ncells
    Phi::Array{Float64,3}        # D*D*D × np × ncells: curvature of the cell map (`point_curvature`), only for
                                 # groups with Hessians on non-affine cells; 0 × np × ncells otherwise
end
GroupData{D}(T, iJt, dV) where {D} = GroupData{D}(T, iJt, similar(iJt), dV, zeros(0, size(dV)...))
GroupData{D}(T, iJt, Jt, dV) where {D} = GroupData{D}(T, iJt, Jt, dV, zeros(0, size(dV)...))

"""
    MatrixFreeJacobian(op::KernelOperator)

Linear operator `y = J(x) dx` applied without assembling. Call [`prepare!`](@ref) at each new
state `x`, then use `mul!`/`*` (e.g. inside an iterative solver). Sum factorisation is used for
fields on quadrilaterals/hexahedra with tensor-product quadrature; other fields use the dense
shape-function weights.
"""
mutable struct MatrixFreeJacobian{TP,GD,BF}
    op::KernelOperator       # rebound by `jacobian!` when the operator is `update`d (same structure)
    tps::TP                  # per Jacobian group: tuple of TPTable or nothing per field
    data::GD                 # per Jacobian group: GroupData
    bufs::BF                 # per Jacobian group: one set of apply buffers per task (no allocation in `mul!`)
    nthreads::Int
    Afb::Union{Nothing,SparseMatrixCSC{Float64,Int}}   # Gridap-assembled part (hybrid fallback)
    x::Vector{Float64}       # copy of the state of the last `prepare!` (read by the preconditioner slots; a copy, so that
                             # the Newton updating its state in place does not make it disagree with the tangents)
end

function MatrixFreeJacobian(op::KernelOperator)
    tps = map(op.jgroups) do g
        # sum factorisation for nodal fields on n-cubes without Hessian slots (the Hessian columns of W
        # take the dense path); nothing on skeletons
        g isa SkeletonGroup ? map(_ -> nothing, g.tables) :
            map((t, sv, su) -> hess_weights(sv, su) isa Val{true} ? nothing : TPTable(t, g.xq), g.tables, g.smv, g.smu)
    end
    data = map(op.jgroups) do g
        np = g.geo.np
        D = group_dim(g)
        nsv, nsu = nslots(g)
        if g isa SkeletonGroup
            # one tangent and measure per facet of the class; Jt⁻¹ of the plus and of the minus cell
            n = length(g.facets)
            T, dV = zeros(nsv * nsu, np, n), zeros(np, n)
            (GroupData{D}(T, zeros(D * D, np, n), dV), GroupData{D}(T, zeros(D * D, np, n), dV))
        else
            nc = length(g.cellmap)
            curved = hess_use(g) isa Val{true} && !g.geo.affine
            GroupData{D}(zeros(nsv * nsu, np, nc), zeros(D * D, np, nc), zeros(D * D, np, nc), zeros(np, nc),
                         zeros(curved ? D * D * D : 0, np, nc))
        end
    end
    Afb = isempty(op.fallback.jtrians) ? nothing : fallback_pattern(op.fallback, op.U, op.V, op.jac)
    bufs = map((g, tp) -> [apply_buffers(g, tp, op.ndl) for _ in 1:op.nthreads], Tuple(op.jgroups), Tuple(tps))
    MatrixFreeJacobian(op, Tuple(tps), Tuple(data), bufs, op.nthreads, Afb, Float64[])
end

nslots(::Group{NSV,NSU}) where {NSV,NSU} = (NSV, NSU)
nslots(::SkeletonGroup{NSV,NSU}) where {NSV,NSU} = (NSV, NSU)
group_dim(::Group{NSV,NSU,PF,TT,AUX,SV,SU,D}) where {NSV,NSU,PF,TT,AUX,SV,SU,D} = D
group_dim(::SkeletonGroup{NSV,NSU,PF,TT,AUX,SV,SU,D}) where {NSV,NSU,PF,TT,AUX,SV,SU,D} = D

Base.size(mf::MatrixFreeJacobian) = (sum(num_free_dofs, field_spaces(mf.op.V)), sum(num_free_dofs, field_spaces(mf.op.U)))
Base.size(mf::MatrixFreeJacobian, i::Int) = size(mf)[i]
Base.eltype(::MatrixFreeJacobian) = Float64
Algebra.allocate_in_domain(mf::MatrixFreeJacobian) = zeros(Float64, size(mf, 2))
Algebra.allocate_in_range(mf::MatrixFreeJacobian) = zeros(Float64, size(mf, 1))
Base.:*(mf::MatrixFreeJacobian, dx::AbstractVector) = LinearAlgebra.mul!(zeros(size(mf, 1)), mf, dx)

"""
    jacobian!(mf::MatrixFreeJacobian, op, x)

Gridap's `jacobian!` for an operator with `jacobian = :matrixfree`: prepares `mf` at the state `x`
(no matrix is formed). If `op` is an `update` of the operator `mf` was built for (or the operator
of a Dirichlet projection on the increment space, with the same free dofs), `mf` is rebound to it:
its buffers and tensor-product tables are reused when every Jacobian group keeps the tables,
quadrature points, slots and cells of the one it replaces and the hybrid-fallback triangulations
are the same objects (what `update` guarantees for a form of the same structure), and rebuilt
otherwise.
"""
function Algebra.jacobian!(mf::MatrixFreeJacobian, op::KernelOperator, x::AbstractVector)
    rebind!(mf, op)
    prepare!(mf, x)
end

"Whether a Jacobian group `g` can use the buffers, tensor-product tables and data built for `g0`."
function same_group(g::Group, g0::Group)
    g.tables === g0.tables && g.xq == g0.xq && nslots(g) == nslots(g0) && g.geo.np == g0.geo.np &&
        length(g.cellmap) == length(g0.cellmap)
end
function same_group(g::SkeletonGroup, g0::SkeletonGroup)
    g.tables === g0.tables && nslots(g) == nslots(g0) && g.geo.np == g0.geo.np && length(g.facets) == length(g0.facets)
end
same_group(::AbstractGroup, ::AbstractGroup) = false

function rebind!(mf::MatrixFreeJacobian, op::KernelOperator)
    mf.op === op && return mf
    # spaces with the same free dofs (U, or the increment space of a Dirichlet projection): same size and element layout
    (sum(num_free_dofs, field_spaces(op.V)), sum(num_free_dofs, field_spaces(op.U))) == size(mf) && op.ndl == mf.op.ndl ||
        error("HyperFEMKernels: the MatrixFreeJacobian was built for an operator on other FE spaces")
    fb, fb0 = op.fallback.jtrians, mf.op.fallback.jtrians
    same = length(op.jgroups) == length(mf.op.jgroups) &&
           all(((g, g0),) -> same_group(g, g0), zip(op.jgroups, mf.op.jgroups)) &&
           length(fb) == length(fb0) && all(t -> any(s -> s === t, fb0), fb)
    if same
        mf.op = op
    else
        new = MatrixFreeJacobian(op)
        mf.op, mf.tps, mf.data, mf.bufs, mf.Afb = new.op, new.tps, new.data, new.bufs, new.Afb
    end
    mf
end

function Algebra.residual_and_jacobian!(b::AbstractVector, mf::MatrixFreeJacobian, op::KernelOperator, x::AbstractVector)
    residual!(b, op, x)
    jacobian!(mf, op, x)
    (b, mf)
end

Algebra.symbolic_setup(ls::Algebra.LinearSolver, ::MatrixFreeJacobian) =
    error("HyperFEMKernels: $(nameof(typeof(ls))) needs an assembled matrix; with jacobian = :matrixfree use a solver that applies the operator (MatrixFreeSolver), or jacobian = :assembled")

"Evaluate tangents and geometry at the state `x` (once per Newton iteration)."
function prepare!(mf::MatrixFreeJacobian, x::AbstractVector)
    op = mf.op
    length(mf.x) == length(x) || (mf.x = zeros(length(x)))
    copyto!(mf.x, x)
    dvals = [get_dirichlet_dof_values(s) for s in field_spaces(op.U)]
    for (gi, g) in enumerate(op.jgroups)
        auxd = aux_data(op, g)
        gd = mf.data[gi]
        cells = reduce(vcat, g.colors)
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

function prepare_cells!(gds::Tuple{GroupData{D},GroupData{D}}, g::SkeletonGroup{NSV,NSU,PF,TT,AUX,SV,SU,D,NF2,LF}, op, x, dvals, auxd,
                        lfacets) where {D,NSV,NSU,PF,TT,AUX,SV,SU,NF2,LF}
    gp, gm = gds
    tables, locs, pf = g.tables, g.locs, g.pf
    ndl = op.ndl
    xl = zeros(2ndl)
    zsv = zero(SVector{NSV,Float64}); zsu = zero(SVector{NSU,Float64})
    for lf in lfacets
        f = g.facets[lf]
        gather_cell!(xl, op.fields, x, dvals, g.skel.cells[1, f])
        gather_cell!(xl, op.fields, x, dvals, g.skel.cells[2, f], ndl)
        for q in 1:g.geo.np
            dS, iJp, Jp = point_geometry(g.geo, q, lf)
            _, iJm, Jm = point_geometry(g.geom, q, lf)
            iJs = side_tuple(iJp, iJm, Val(NF2))
            FV = field_values_sides(tables, locs, g.use, xl, q, iJs)
            Aq = map(a -> @inbounds(a[q, lf]), auxd)
            T = if LF === Nothing
                ForwardDiff.jacobian(r -> ForwardDiff.gradient(s -> pf.core(pf.ops, pf.coefs, FV, Aq, s, r), zsv), zsu)
            else
                SMatrix{NSV,NSU,Float64,NSV * NSU}(apply_maps(g.lin, g.lin.state(g.lin.ops, FV, Aq), Val(NSV * NSU)))
            end
            @inbounds for i in 1:NSV*NSU
                gp.T[i, q, lf] = T[i]
            end
            @inbounds for i in 1:D*D
                gp.iJt[i, q, lf] = iJp[i]
                gm.iJt[i, q, lf] = iJm[i]
                gp.Jt[i, q, lf] = Jp[i]
                gm.Jt[i, q, lf] = Jm[i]
            end
            gp.dV[q, lf] = dS
        end
    end
    nothing
end

function prepare_cells!(gd::GroupData{D}, g::Group{NSV,NSU,PF,TT,AUX,SV,SU,D,NF,LF,JF}, op, x, dvals, auxd, cells) where {D,NSV,NSU,PF,TT,AUX,SV,SU,NF,LF,JF}
    tables, locs, pf = g.tables, g.locs, g.pf
    xl = zeros(op.ndl)
    sg = zeros(op.ndl)
    zsv = zero(SVector{NSV,Float64}); zsu = zero(SVector{NSU,Float64})
    hu = hess_use(g)
    curved = size(gd.Phi, 1) > 0
    for cell in cells
        sc = g.cellmap[cell]
        gather_cell!(xl, op.fields, x, dvals, sc)
        op.signed && apply_signs!(xl, cell_signs!(sg, op.fields, sc), op.ndl)   # state of H(div) fields with the dof signs
        for q in 1:g.geo.np
            dV, iJt, Jt = point_geometry(g.geo, q, cell)
            Φ = point_curvature(g.geo, q, cell, hu)
            FV = field_values(tables, locs, g.use, xl, q, iJt, Jt, Φ)
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
                gd.Jt[i, q, cell] = Jt[i]
            end
            curved && store_curvature!(gd, Φ, q, cell)
            gd.dV[q, cell] = dV
        end
    end
    nothing
end

"Store the curvature `Φ` (D matrices D × D) of a point in `gd.Phi[:, q, cell]` (c outer, then column-major entries)."
@inline function store_curvature!(gd::GroupData{D}, Φ::Curvature{D}, q::Int, cell::Integer) where {D}
    @inbounds for c in 1:D, i in 1:D*D
        gd.Phi[i+(c-1)*D*D, q, cell] = Φ[c][i]
    end
    nothing
end
@inline store_curvature!(gd::GroupData, ::Nothing, q::Int, cell::Integer) = nothing

"Curvature of the cell map at a point of `gd` for a field whose weights need Hessians (`nothing` otherwise)."
@inline curvature_at(gd::GroupData{D}, q::Int, cell::Integer, ::Val{false}) where {D} = nothing
@inline function curvature_at(gd::GroupData{D}, q::Int, cell::Integer, ::Val{true}) where {D}
    size(gd.Phi, 1) == 0 && return ntuple(_ -> zero(SMatrix{D,D,Float64,D * D}), Val(D))    # affine cells
    ntuple(c -> SMatrix{D,D,Float64,D * D}(ntuple(i -> @inbounds(gd.Phi[i+(c-1)*D*D, q, cell]), Val(D * D))), Val(D))
end

function LinearAlgebra.mul!(y::AbstractVector, mf::MatrixFreeJacobian, dx::AbstractVector)
    fill!(y, 0.0)
    op = mf.op
    for (gi, g) in enumerate(op.jgroups)
        bufs = mf.bufs[gi]
        for color in g.colors
            nchunks = clamp(length(color) ÷ 16, 1, mf.nthreads)
            @sync for (i, ch) in enumerate(Iterators.partition(color, cld(length(color), nchunks)))
                Threads.@spawn apply_cells!(y, mf.data[gi], mf.tps[gi], g, op, dx, ch, bufs[i])
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
     Wb=map(t -> zeros(t.nn, wcols(t)), g.tables),
     uc=map(t -> zeros(t.nn), g.tables), rc=map(t -> zeros(t.nn), g.tables),
     pts=map(t -> zeros(np, D + 1), g.tables), coef=map(t -> zeros(np, D + 1), g.tables),
     tmp=map(tp -> tp === nothing ? zeros(0, 0) : tp_scratch(tp), tps))
end

"Scratch buffers for the apply of one skeleton class (dense weights, element vector of both cells)."
function apply_buffers(g::SkeletonGroup{NSV,NSU,PF,TT,AUX,SV,SU,D}, tps, ndl::Int) where {NSV,NSU,PF,TT,AUX,SV,SU,D}
    np = g.geo.np
    (xl=zeros(2ndl), re=zeros(2ndl),
     su=zeros(NSU, np), sv=zeros(NSV, np),
     Wb=map(t -> zeros(t.nn, D + 1), g.tables),
     uc=map(t -> zeros(t.nn), g.tables), rc=map(t -> zeros(t.nn), g.tables),
     pts=map(t -> zeros(np, D + 1), g.tables), coef=map(t -> zeros(np, D + 1), g.tables),
     tmp=map(_ -> zeros(0, 0), g.tables))
end

"Gather the increment of `cell` (zero at Dirichlet dofs) from position `shift` of `xl`."
@inline function gather_increment!(xl, fields::Vector{FieldLayout}, dx, cell::Integer, shift::Int=0)
    for fl in fields
        @inbounds for j in 1:fl.nd
            id = fl.trial_ids[j, cell]
            xl[shift+fl.loc+j] = id > 0 ? dx[fl.trial_off+id] : 0.0
        end
    end
    xl
end

"Add the free test rows of `re` (from position `shift`) of `cell` to `y`."
@inline function scatter_increment!(y, fields::Vector{FieldLayout}, re, cell::Integer, shift::Int=0)
    for fl in fields
        @inbounds for j in 1:fl.nd
            id = fl.test_ids[j, cell]
            id > 0 && (y[fl.test_off+id] += re[shift+fl.loc+j])
        end
    end
    y
end

function apply_cells!(y, gds::Tuple{GroupData{D},GroupData{D}}, tps, g::SkeletonGroup{NSV,NSU,PF,TT,AUX,SV,SU,D,NF2}, op, dx,
                      lfacets, b) where {NSV,NSU,PF,TT,AUX,SV,SU,D,NF2}
    tables, locs = g.tables, g.locs
    np = g.geo.np
    ndl = op.ndl
    gp = gds[1]
    gdsides = side_tuple(gds[1], gds[2], Val(NF2))
    for lf in lfacets
        f = g.facets[lf]
        cp, cm = g.skel.cells[1, f], g.skel.cells[2, f]
        gather_increment!(b.xl, op.fields, dx, cp)
        gather_increment!(b.xl, op.fields, dx, cm, ndl)
        fill!(b.su, 0.0)
        map(tables, locs, g.smu, tps, b.uc, b.pts, b.Wb, b.tmp, gdsides) do t, loc, sp, tp, uc, pts, W, tmp, gd
            interpolate_slots!(b.su, t, loc, sp, tp, b.xl, uc, pts, W, tmp, gd, lf, np)
        end
        @inbounds for q in 1:np
            for i in 1:NSV
                s = 0.0
                for j in 1:NSU
                    s += gp.T[i+(j-1)*NSV, q, lf] * b.su[j, q]
                end
                b.sv[i, q] = s * gp.dV[q, lf]
            end
        end
        fill!(b.re, 0.0)
        map(tables, locs, g.smv, tps, b.rc, b.coef, b.Wb, b.tmp, gdsides) do t, loc, sp, tp, rc, coef, W, tmp, gd
            contract_slots!(b.re, t, loc, sp, tp, b.sv, rc, coef, W, tmp, gd, lf, np)
        end
        scatter_increment!(y, op.fields, b.re, cp)
        scatter_increment!(y, op.fields, b.re, cm, ndl)
    end
    nothing
end

function apply_cells!(y, gd::GroupData{D}, tps, g::Group{NSV,NSU,PF,TT,AUX,SV,SU,D}, op, dx, cells, b) where {NSV,NSU,PF,TT,AUX,SV,SU,D}
    tables, locs = g.tables, g.locs
    np = g.geo.np
    fields = op.fields
    for cell in cells
        sc = g.cellmap[cell]
        # gather the increment (zero at Dirichlet dofs; with the dof signs of H(div) fields)
        for fl in fields
            if has_signs(fl)
                @inbounds for j in 1:fl.nd
                    id = fl.trial_ids[j, sc]
                    b.xl[fl.loc+j] = id > 0 ? fl.signs[j, sc] * dx[fl.trial_off+id] : 0.0
                end
            else
                @inbounds for j in 1:fl.nd
                    id = fl.trial_ids[j, sc]
                    b.xl[fl.loc+j] = id > 0 ? dx[fl.trial_off+id] : 0.0
                end
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
        # scatter (free test rows only; with the dof signs of H(div) fields)
        for fl in fields
            if has_signs(fl)
                @inbounds for j in 1:fl.nd
                    id = fl.test_ids[j, sc]
                    id > 0 && (y[fl.test_off+id] += fl.signs[j, sc] * b.re[fl.loc+j])
                end
            else
                @inbounds for j in 1:fl.nd
                    id = fl.test_ids[j, sc]
                    id > 0 && (y[fl.test_off+id] += b.re[fl.loc+j])
                end
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

function interpolate_slots!(su, t::FieldTable{D,T,NN,NC}, loc, sp::SlotSpec{NC,LV,HV,HG,HH}, ::Nothing, xl, uc, pts, W, tmp,
                            gd::GroupData{D}, cell, np) where {D,T,NN,NC,LV,HV,HG,HH}
    LV == 0 && return nothing
    oh = (HV ? 1 : 0) + (HG ? D : 0)
    @inbounds for q in 1:np
        iJt = TensorValue{D,D,Float64,D * D}(ntuple(i -> gd.iJt[i, q, cell], Val(D * D)))
        point_weights!(W, t, q, iJt, nothing, curvature_at(gd, q, cell, Val(HH)), Val(HH))
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
            if HH
                for p in 1:nhess(D)
                    hp = 0.0
                    @simd for n in 1:NN
                        hp += xl[loc+ldofidx(t, n, c)] * W[n, 1+D+p]
                    end
                    su[sp.sm[c, oh+p], q] = hp
                end
            end
        end
    end
    nothing
end

"H(div) field: value and gradient at every point from the reference tables and the stored Piola geometry."
function interpolate_slots!(su, t::PiolaTable{D,NN}, loc, sp::SlotSpec{1,LV,HV,HG}, ::Nothing, xl, uc, pts, W, tmp,
                            gd::GroupData{D}, cell, np) where {D,NN,LV,HV,HG}
    LV == 0 && return nothing
    @inbounds for q in 1:np
        iJt = TensorValue{D,D,Float64,D * D}(ntuple(i -> gd.iJt[i, q, cell], Val(D * D)))
        Jt = TensorValue{D,D,Float64,D * D}(ntuple(i -> gd.Jt[i, q, cell], Val(D * D)))
        u, g = field_point_values(t, xl, loc, q, iJt, Jt, FieldUse{HV,HG,false}())
        HV && for c in 1:D
            su[sp.sm[1, c], q] = u[c]
        end
        HG && for i in 1:D*D
            su[sp.sm[1, (HV ? D : 0)+i], q] = g[i]
        end
    end
    nothing
end

# --- contraction of the test coefficients back to the nodes ---------------------------------

"H(div) field: the slot coefficients of every point pulled back to the reference tables (`contract_residual!`)."
function contract_slots!(re, t::PiolaTable{D,NN}, loc, sp::SlotSpec{1,LV}, ::Nothing, sv, rc, coef, W, tmp,
                         gd::GroupData{D}, cell, np) where {D,NN,LV}
    LV == 0 && return nothing
    @inbounds for q in 1:np
        iJt = TensorValue{D,D,Float64,D * D}(ntuple(i -> gd.iJt[i, q, cell], Val(D * D)))
        Jt = TensorValue{D,D,Float64,D * D}(ntuple(i -> gd.Jt[i, q, cell], Val(D * D)))
        contract_residual!(re, t, loc, sp, view(sv, :, q), q, iJt, Jt, 1.0)
    end
    nothing
end

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

function contract_slots!(re, t::FieldTable{D,T,NN,NC}, loc, sp::SlotSpec{NC,LV,HV,HG,HH}, ::Nothing, sv, rc, coef, W, tmp,
                         gd::GroupData{D}, cell, np) where {D,T,NN,NC,LV,HV,HG,HH}
    LV == 0 && return nothing
    oh = (HV ? 1 : 0) + (HG ? D : 0)
    @inbounds for q in 1:np
        iJt = TensorValue{D,D,Float64,D * D}(ntuple(i -> gd.iJt[i, q, cell], Val(D * D)))
        point_weights!(W, t, q, iJt, nothing, curvature_at(gd, q, cell, Val(HH)), Val(HH))
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
            if HH
                for p in 1:nhess(D)
                    rp = sv[sp.sm[c, oh+p], q]
                    @simd for n in 1:NN
                        re[loc+ldofidx(t, n, c)] += rp * W[n, 1+D+p]
                    end
                end
            end
        end
    end
    nothing
end

# Operator and assembly.

const EMPTY_SLOTS = SVector{0,Float64}()
const MIN_CELLS_PER_TASK = 8

"Global numbering of one field: cell dof ids of test and trial spaces and offsets."
struct FieldLayout
    test_ids::Matrix{Int32}     # ndofs × ncells, ids in the test space (negative: Dirichlet)
    trial_ids::Matrix{Int32}    # same, trial space
    test_off::Int               # offset of the field in the global free vector (test)
    trial_off::Int              # same, trial
    loc::Int                    # offset of the field in the element vector
    nd::Int
end

"A compiled group of terms: same triangulation and quadrature."
struct Group{NSV,NSU,PF,TT,AUX,SV,SU,D,NF}
    pf::PF
    locs::NTuple{NF,Int}  # offset of each field in the element vector
    tables::TT          # FieldTable per field
    geo::GeoTable{D}
    aux::AUX            # Gridap arrays, evaluated per cell
    smv::SV             # slot specs (test) per field
    smu::SU             # slot specs (trial) per field
    trian::Triangulation
    xq::Vector          # reference quadrature points
    auxcache::Base.RefValue{Any}   # dense auxiliary values when cache_aux = true
end

function Group(terms::Vector{Term}, trian::Triangulation, Uspaces, locs, D::Int, previous=Any[])
    pf = PointForm(terms)
    ls = vcat(pf.test.leaves, pf.trial.leaves, leaves(terms, :unknown))
    isempty(ls) && error("HyperFEMKernels: integral without unknown, test or trial functions")
    xq = uniform_value(get_data(ls[1].points), "the quadrature points")
    w = uniform_value(terms[1].weights, "the quadrature weights")
    # reuse geometry and shape-function tables of a previous operator with the same quadrature
    k = findfirst(g -> g.trian === trian && g.xq == xq && g.geo.w == w, previous)
    if k === nothing
        geo = GeoTable(trian, xq, w)
        tables = Tuple(FieldTable(s, xq, D) for s in Uspaces)
    else
        geo, tables = previous[k].geo, previous[k].tables
    end
    smv = Tuple(SlotSpec(slot_map(pf.test, f, t.nc, D), D) for (f, t) in enumerate(tables))
    smu = Tuple(SlotSpec(slot_map(pf.trial, f, t.nc, D), D) for (f, t) in enumerate(tables))
    aux = Tuple(a.array for a in pf.aux)
    NF = length(tables)
    Group{pf.test.nslots,pf.trial.nslots,typeof(pf),typeof(tables),typeof(aux),typeof(smv),typeof(smu),D,NF}(
        pf, NTuple{NF,Int}(locs), tables, geo, aux, smv, smu, trian, collect(xq), Ref{Any}(nothing))
end

"""
    KernelOperator(res, jac, U, V; nthreads = Threads.nthreads(), cache_aux = false)

Gridap `NonlinearOperator` that assembles `res(u, v)` and `jac(u, du, v)` with fused, threaded
kernels. `res` and `jac` are written as for Gridap's `FEOperator`. Use [`update`](@ref) to change
`res`/`jac` (e.g. a new load factor) while keeping tables, sparsity pattern and colouring.

With `cache_aux = true` the auxiliary data (fields that do not depend on the unknown, e.g. a
previous state) is evaluated once per operator instead of once per assembly. Only safe when that
data does not change while the operator is used (e.g. it must not be built from the FE function
that holds the current Newton iterate).
"""
struct KernelOperator{R,J,US,VS} <: NonlinearOperator
    res::R
    jac::J
    U::US
    V::VS
    fields::Vector{FieldLayout}
    ndl::Int
    rgroups::Vector{Any}
    jgroups::Vector{Any}
    A0::SparseMatrixCSC{Float64,Int}
    nzpos::Matrix{Int32}
    colors::Vector{Vector{Int32}}
    nthreads::Int
    cache_aux::Bool
end

function compile_groups(forms::Vector{Form}, U, locs, ncells::Int, D::Int, previous=Any[])
    groups = Any[]
    Us = field_spaces(U)
    for form in forms
        num_cells(form.trian) == ncells ||
            error("HyperFEMKernels: integrals on triangulations other than the one of the FE spaces are not supported yet")
        bykey = Dict{UInt,Vector{Term}}()
        order = UInt[]
        for t in form.terms
            k = objectid(t.weights)
            haskey(bykey, k) || (push!(order, k); bykey[k] = Term[])
            push!(bykey[k], t)
        end
        for k in order
            push!(groups, Group(bykey[k], form.trian, Us, locs, D, previous))
        end
    end
    groups
end

function KernelOperator(res, jac, U::FESpace, V::FESpace; nthreads::Int=Threads.nthreads(), cache_aux::Bool=false)
    Us, Vs = field_spaces(U), field_spaces(V)
    length(Us) == length(Vs) || error("HyperFEMKernels: test and trial spaces have different numbers of fields")
    if U isa MultiFieldFESpace
        MultiFieldStyle(U) isa ConsecutiveMultiFieldStyle ||
            error("HyperFEMKernels: only ConsecutiveMultiFieldStyle multi-field spaces are supported")
    end
    trian = get_triangulation(Us[1])
    D = num_cell_dims(trian)
    ncells = num_cells(trian)
    fields = FieldLayout[]
    toff, uoff, loc = 0, 0, 0
    for (su, sv) in zip(Us, Vs)
        nd = length(testitem(get_cell_dof_ids(sv)))
        push!(fields, FieldLayout(cell_dof_matrix(sv, nd), cell_dof_matrix(su, nd), toff, uoff, loc, nd))
        toff += num_free_dofs(sv)
        uoff += num_free_dofs(su)
        loc += nd
    end
    rforms = capture(res, probes(U, :unknown), probes(V, :test))
    jforms = capture(jac, probes(U, :unknown), probes(U, :trial), probes(V, :test))
    locs = Tuple(fl.loc for fl in fields)
    rgroups = compile_groups(rforms, U, locs, ncells, D)
    jgroups = compile_groups(jforms, U, locs, ncells, D)
    A0, nzpos = sparsity(fields, loc, ncells, toff, uoff)
    colors = color_cells(fields, ncells, toff)
    KernelOperator(res, jac, U, V, fields, loc, rgroups, jgroups, A0, nzpos, colors, nthreads, cache_aux)
end

"""
    update(op, res, jac)

New operator with other `res`/`jac` (same spaces and structure), reusing the sparsity pattern,
positions and colouring of `op`.
"""
function update(op::KernelOperator, res, jac)
    D = num_cell_dims(get_triangulation(field_spaces(op.U)[1]))
    ncells = size(op.nzpos, 2)
    locs = Tuple(fl.loc for fl in op.fields)
    prev = vcat(op.rgroups, op.jgroups)
    rgroups = compile_groups(capture(res, probes(op.U, :unknown), probes(op.V, :test)), op.U, locs, ncells, D, prev)
    jgroups = compile_groups(capture(jac, probes(op.U, :unknown), probes(op.U, :trial), probes(op.V, :test)), op.U, locs, ncells, D, prev)
    KernelOperator(res, jac, op.U, op.V, op.fields, op.ndl, rgroups, jgroups, op.A0, op.nzpos, op.colors, op.nthreads, op.cache_aux)
end

# ---------------------------------------------------------------------------------------------
# Sparsity pattern and nonzero positions

function sparsity(fields::Vector{FieldLayout}, ndl::Int, ncells::Int, nrows::Int, ncols::Int)
    grow = zeros(Int32, ndl, ncells)
    gcol = zeros(Int32, ndl, ncells)
    for fl in fields, cell in 1:ncells, j in 1:fl.nd
        r = fl.test_ids[j, cell]
        c = fl.trial_ids[j, cell]
        grow[fl.loc+j, cell] = r > 0 ? fl.test_off + r : 0
        gcol[fl.loc+j, cell] = c > 0 ? fl.trial_off + c : 0
    end
    keys = UInt64[]
    sizehint!(keys, ncells * ndl * ndl)
    for cell in 1:ncells, jj in 1:ndl
        c = gcol[jj, cell]
        c > 0 || continue
        for ii in 1:ndl
            r = grow[ii, cell]
            r > 0 && push!(keys, (UInt64(c) << 32) | UInt64(r))
        end
    end
    sort!(keys)
    unique!(keys)
    cnt = zeros(Int, ncols)
    rowval = Vector{Int}(undef, length(keys))
    for (k, key) in enumerate(keys)
        cnt[Int(key >> 32)] += 1
        rowval[k] = Int(key & 0xffffffff)
    end
    colptr = Vector{Int}(undef, ncols + 1)
    colptr[1] = 1
    for c in 1:ncols
        colptr[c+1] = colptr[c] + cnt[c]
    end
    A0 = SparseMatrixCSC(nrows, ncols, colptr, rowval, zeros(length(keys)))
    nzpos = zeros(Int32, ndl * ndl, ncells)
    Threads.@threads for cell in 1:ncells
        for jj in 1:ndl
            c = gcol[jj, cell]
            c > 0 || continue
            lo, hi = colptr[c], colptr[c+1] - 1
            for ii in 1:ndl
                r = grow[ii, cell]
                r > 0 || continue
                p = searchsortedfirst(view(rowval, lo:hi), r) + lo - 1
                nzpos[ii+(jj-1)*ndl, cell] = p
            end
        end
    end
    A0, nzpos
end

# ---------------------------------------------------------------------------------------------
# Algebra interface

Algebra.allocate_residual(op::KernelOperator, x::AbstractVector) = zeros(sum(num_free_dofs, field_spaces(op.V)))
Algebra.allocate_jacobian(op::KernelOperator, x::AbstractVector) = copy(op.A0)
Algebra.zero_initial_guess(op::KernelOperator) = zeros(sum(num_free_dofs, field_spaces(op.U)))

function Algebra.residual!(b::AbstractVector, op::KernelOperator, x::AbstractVector)
    fill!(b, 0.0)
    dvals = [get_dirichlet_dof_values(s) for s in field_spaces(op.U)]
    for g in op.rgroups
        run_cells!(Val(:res), b, nothing, op, g, x, dvals)
    end
    b
end

function Algebra.jacobian!(A::SparseMatrixCSC, op::KernelOperator, x::AbstractVector)
    fill!(nonzeros(A), 0.0)
    dvals = [get_dirichlet_dof_values(s) for s in field_spaces(op.U)]
    for g in op.jgroups
        run_cells!(Val(:jac), nothing, A, op, g, x, dvals)
    end
    A
end

function Algebra.residual_and_jacobian!(b::AbstractVector, A::AbstractMatrix, op::KernelOperator, x::AbstractVector)
    residual!(b, op, x)
    jacobian!(A, op, x)
    (b, A)
end

# ---------------------------------------------------------------------------------------------
# Auxiliary data: dense values at the quadrature points
#
# Auxiliary data (fields that do not depend on the unknown/test/trial functions: previous states,
# interpolated fields, functions of x, constants) is evaluated by Gridap once per assembly (or
# once per operator with `cache_aux = true`) into dense `np × ncells` arrays, so the element
# kernels only read plain arrays (requirement for GPU execution).

"Dense values (np × ncells) of the auxiliary arrays of a group."
function materialize_aux(g::Group, nthreads::Int)
    np, ncells = g.geo.np, size(g.geo.coords, 2)
    map(g.aux) do a
        T = eltype(eltype(a))
        M = Matrix{T}(undef, np, ncells)
        chunks = Iterators.partition(1:ncells, cld(ncells, max(1, nthreads)))
        @sync for ch in chunks
            Threads.@spawn begin
                cache = array_cache(a)
                for cell in ch
                    v = getindex!(cache, a, cell)
                    @inbounds for q in 1:np
                        M[q, cell] = v[q]
                    end
                end
            end
        end
        M
    end
end

function aux_data(op::KernelOperator, g::Group)
    if op.cache_aux
        g.auxcache[] === nothing && (g.auxcache[] = materialize_aux(g, op.nthreads))
        return g.auxcache[]
    end
    materialize_aux(g, op.nthreads)
end

# ---------------------------------------------------------------------------------------------
# Element computations (no assembly): gather, element residual/Jacobian, scatter.
# The element functions only read dense arrays; the scatter is the part that changes for MPI
# (local numbering + communication) and GPU (atomics or colours).

"Local values of the unknowns of `cell` (free values and Dirichlet values)."
@inline function gather_cell!(xl, fields::Vector{FieldLayout}, x, dvals, cell::Integer)
    for (f, fl) in enumerate(fields)
        off = fl.trial_off
        dv = dvals[f]
        @inbounds for j in 1:fl.nd
            id = fl.trial_ids[j, cell]
            xl[fl.loc+j] = id > 0 ? x[off+id] : dv[-id]
        end
    end
    xl
end

"Geometry at point `q` of `cell`: (dV, inverse of Jt)."
@inline function point_geometry(geo::GeoTable{D}, q::Int, cell::Integer) where {D}
    Jt = zero(TensorValue{D,D,Float64,D * D})
    @inbounds for n in 1:geo.ng
        Jt += outer(geo.dM[q, n], geo.coords[n, cell])
    end
    (geo.w[q] * meas(Jt), inv(Jt))
end

"Element residual of `cell` into `re` (local numbering of the element)."
function element_residual!(re, g::Group{NSV,NSU,PF,TT,AUX,SV,SU,D,NF}, xl, auxd, cell::Integer,
                           dNb) where {NSV,NSU,PF,TT,AUX,SV,SU,D,NF}
    fill!(re, 0.0)
    tables, locs, pf = g.tables, g.locs, g.pf
    zsv = zero(SVector{NSV,Float64})
    for q in 1:g.geo.np
        dV, iJt = point_geometry(g.geo, q, cell)
        map((t, buf) -> phys_grads!(buf, t, q, iJt), tables, dNb)
        FV = map((t, l, buf) -> field_point_values(t, xl, l, q, buf), tables, locs, dNb)
        Aq = map(a -> @inbounds(a[q, cell]), auxd)
        R = ForwardDiff.gradient(s -> pf.core(pf.ops, pf.coefs, FV, Aq, s, EMPTY_SLOTS), zsv)
        map((t, l, sm, buf) -> contract_residual!(re, t, l, sm, R, q, buf, dV), tables, locs, g.smv, dNb)
    end
    re
end

"Element Jacobian of `cell` into `KeT` (transposed: KeT[trial, test], local numbering)."
function element_jacobian!(KeT, g::Group{NSV,NSU,PF,TT,AUX,SV,SU,D,NF}, xl, auxd, cell::Integer,
                           dNb) where {NSV,NSU,PF,TT,AUX,SV,SU,D,NF}
    fill!(KeT, 0.0)
    tables, locs, pf = g.tables, g.locs, g.pf
    zsv = zero(SVector{NSV,Float64})
    zsu = zero(SVector{NSU,Float64})
    for q in 1:g.geo.np
        dV, iJt = point_geometry(g.geo, q, cell)
        map((t, buf) -> phys_grads!(buf, t, q, iJt), tables, dNb)
        FV = map((t, l, buf) -> field_point_values(t, xl, l, q, buf), tables, locs, dNb)
        Aq = map(a -> @inbounds(a[q, cell]), auxd)
        T = ForwardDiff.jacobian(s -> ForwardDiff.gradient(r -> pf.core(pf.ops, pf.coefs, FV, Aq, s, r), zsu), zsv)
        map((tf, lf, smf, bf) -> map((tg, lg, smg, bg) ->
                contract_jacobian!(KeT, tf, lf, smf, bf, tg, lg, smg, bg, T, q, dV),
                tables, locs, g.smu, dNb),
            tables, locs, g.smv, dNb)
    end
    KeT
end

"Add an element residual to the global vector (serial numbering)."
@inline function scatter_residual!(b, fields::Vector{FieldLayout}, re, cell::Integer)
    for fl in fields
        off = fl.test_off
        @inbounds for j in 1:fl.nd
            id = fl.test_ids[j, cell]
            id > 0 && (b[off+id] += re[fl.loc+j])
        end
    end
    b
end

"Add an element Jacobian to the nonzeros of the global matrix with precomputed positions."
@inline function scatter_jacobian!(nzv, nzpos::Matrix{Int32}, KeT, ndl::Int, cell::Integer)
    @inbounds for jj in 1:ndl, ii in 1:ndl
        p = nzpos[ii+(jj-1)*ndl, cell]
        p > 0 && (nzv[p] += KeT[jj, ii])
    end
    nzv
end

# ---------------------------------------------------------------------------------------------
# Threaded CPU driver: colours, then chunks of cells per task.

function run_cells!(kind::Val, b, A, op::KernelOperator, g::Group, x, dvals)
    auxd = aux_data(op, g)
    for color in op.colors
        nchunks = clamp(length(color) ÷ MIN_CELLS_PER_TASK, 1, op.nthreads)
        if nchunks == 1
            cells!(kind, b, A, op, g, x, dvals, auxd, color)
        else
            @sync for ch in Iterators.partition(color, cld(length(color), nchunks))
                Threads.@spawn cells!(kind, b, A, op, g, x, dvals, auxd, ch)
            end
        end
    end
    nothing
end

function cells!(::Val{K}, b, A, op::KernelOperator, g::Group{NSV,NSU,PF,TT,AUX,SV,SU,D,NF},
                x, dvals, auxd, cells) where {K,NSV,NSU,PF,TT,AUX,SV,SU,D,NF}
    ndl = op.ndl
    xl = zeros(ndl)
    dNb = map(t -> Vector{VectorValue{D,Float64}}(undef, t.nn), g.tables)
    if K === :res
        re = zeros(ndl)
        for cell in cells
            gather_cell!(xl, op.fields, x, dvals, cell)
            element_residual!(re, g, xl, auxd, cell, dNb)
            scatter_residual!(b, op.fields, re, cell)
        end
    else
        KeT = zeros(ndl, ndl)
        nzv = nonzeros(A)
        for cell in cells
            gather_cell!(xl, op.fields, x, dvals, cell)
            element_jacobian!(KeT, g, xl, auxd, cell, dNb)
            scatter_jacobian!(nzv, op.nzpos, KeT, ndl, cell)
        end
    end
    nothing
end

# Operator and assembly.

const EMPTY_SLOTS = SVector{0,Float64}()
const MIN_CELLS_PER_TASK = 64

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
        pf, NTuple{NF,Int}(locs), tables, geo, aux, smv, smu, trian, collect(xq))
end

"""
    KernelOperator(res, jac, U, V; nthreads = Threads.nthreads())

Gridap `NonlinearOperator` that assembles `res(u, v)` and `jac(u, du, v)` with fused, threaded
kernels. `res` and `jac` are written as for Gridap's `FEOperator`. Use [`update`](@ref) to change
`res`/`jac` (e.g. a new load factor) while keeping tables, sparsity pattern and colouring.
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

function KernelOperator(res, jac, U::FESpace, V::FESpace; nthreads::Int=Threads.nthreads())
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
    KernelOperator(res, jac, U, V, fields, loc, rgroups, jgroups, A0, nzpos, colors, nthreads)
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
    KernelOperator(res, jac, op.U, op.V, op.fields, op.ndl, rgroups, jgroups, op.A0, op.nzpos, op.colors, op.nthreads)
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
# Cell loops

function run_cells!(kind::Val, b, A, op::KernelOperator, g::Group, x, dvals)
    for color in op.colors
        nchunks = clamp(length(color) ÷ MIN_CELLS_PER_TASK, 1, op.nthreads)
        chunks = Iterators.partition(color, cld(length(color), nchunks))
        if nchunks == 1
            cells!(kind, b, A, op, g, x, dvals, color)
        else
            @sync for ch in chunks
                Threads.@spawn cells!(kind, b, A, op, g, x, dvals, ch)
            end
        end
    end
    nothing
end

function cells!(kind::Val{K}, b, A, op::KernelOperator, g::Group{NSV,NSU,PF,TT,AUX,SV,SU,D,NF},
                x, dvals, cells) where {K,NSV,NSU,PF,TT,AUX,SV,SU,D,NF}
    ndl = op.ndl
    fields = op.fields
    tables = g.tables
    geo = g.geo
    pf = g.pf
    xl = zeros(ndl)
    re = K === :res ? zeros(ndl) : Float64[]
    KeT = K === :jac ? zeros(ndl, ndl) : zeros(0, 0)   # transposed: KeT[trial, test]
    locs = g.locs
    dNb = map(t -> Vector{VectorValue{D,Float64}}(undef, t.nn), tables)
    caches = map(array_cache, g.aux)
    zsv = zero(SVector{NSV,Float64})
    zsu = zero(SVector{NSU,Float64})
    nzv = A === nothing ? Float64[] : nonzeros(A)
    for cell in cells
        # gather unknowns (free values and Dirichlet values)
        for (f, fl) in enumerate(fields)
            off = fl.trial_off
            dv = dvals[f]
            @inbounds for j in 1:fl.nd
                id = fl.trial_ids[j, cell]
                xl[fl.loc+j] = id > 0 ? x[off+id] : dv[-id]
            end
        end
        K === :res ? fill!(re, 0.0) : fill!(KeT, 0.0)
        auxcell = map((c, a) -> getindex!(c, a, cell), caches, g.aux)
        for q in 1:geo.np
            Jt = zero(TensorValue{D,D,Float64,D * D})
            @inbounds for n in 1:geo.ng
                Jt += outer(geo.dM[q, n], geo.coords[n, cell])
            end
            dV = geo.w[q] * meas(Jt)
            iJt = inv(Jt)
            map((t, buf) -> phys_grads!(buf, t, q, iJt), tables, dNb)
            FV = map((t, fl, buf) -> field_point_values(t, xl, fl, q, buf), tables, locs, dNb)
            Aq = map(v -> v[q], auxcell)
            if K === :res
                R = ForwardDiff.gradient(s -> pf.core(pf.ops, pf.coefs, FV, Aq, s, EMPTY_SLOTS), zsv)
                map((t, fl, sm, buf) -> contract_residual!(re, t, fl, sm, R, q, buf, dV),
                    tables, locs, g.smv, dNb)
            else
                T = ForwardDiff.jacobian(s -> ForwardDiff.gradient(r -> pf.core(pf.ops, pf.coefs, FV, Aq, s, r), zsu), zsv)
                map((tf, flf, smf, bf) -> map((tg, flg, smg, bg) ->
                        contract_jacobian!(KeT, tf, flf, smf, bf, tg, flg, smg, bg, T, q, dV),
                        tables, locs, g.smu, dNb),
                    tables, locs, g.smv, dNb)
            end
        end
        # scatter
        if K === :res
            for fl in fields
                off = fl.test_off
                @inbounds for j in 1:fl.nd
                    id = fl.test_ids[j, cell]
                    id > 0 && (b[off+id] += re[fl.loc+j])
                end
            end
        else
            @inbounds for jj in 1:ndl, ii in 1:ndl
                p = op.nzpos[ii+(jj-1)*ndl, cell]
                p > 0 && (nzv[p] += KeT[jj, ii])
            end
        end
    end
    nothing
end

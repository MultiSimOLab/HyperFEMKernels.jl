# Operator and assembly.

const EMPTY_SLOTS = SVector{0,Float64}()
const MATRIX_PRODUCTS = get(ENV, "HFK_MATRIX_PRODUCTS", "1") == "1"
const WORK_PER_TASK = parse(Int, get(ENV, "HFK_WORK_PER_TASK", "20000"))

"Global numbering of one field: cell dof ids of test and trial spaces and offsets."
struct FieldLayout
    test_ids::Matrix{Int32}     # ndofs × ncells, ids in the test space (negative: Dirichlet)
    trial_ids::Matrix{Int32}    # same, trial space
    test_off::Int               # offset of the field in the global free vector (test)
    trial_off::Int              # same, trial
    loc::Int                    # offset of the field in the element vector
    nd::Int
end

"Triangulations assembled by Gridap (`[:all]` = every contribution) and the Gridap assembler."
struct Fallback
    rtrians::Vector{Any}
    jtrians::Vector{Any}
    assem::SparseMatrixAssembler
end

"A compiled group of terms: same triangulation and quadrature."
struct Group{NSV,NSU,PF,TT,AUX,SV,SU,D,NF,LF,JF}
    pf::PF
    lin::LF               # LinearForm (no dual numbers) or nothing
    locs::NTuple{NF,Int}  # offset of each field in the element vector
    tables::TT          # FieldTable per field
    geo::GeoTable{D,JF}
    aux::Tuple          # Gridap arrays, evaluated per cell (lazy; their types are not part of the group's type,
                        # so that a changed auxiliary expression does not recompile the kernels)
                        # AUX: type of the dense values of `aux` at the points (NTuple of Matrix{T}, see materialize_aux)
    smv::SV             # slot specs (test) per field
    smu::SU             # slot specs (trial) per field
    trian::Triangulation
    xq::Vector          # reference quadrature points
    auxcache::Base.RefValue{Any}   # dense auxiliary values when cache_aux = true
    cellmap::Vector{Int32}         # integration cell → cell of the FE spaces
    colors::Vector{Vector{Int32}}  # integration cells per colour
end

function Group(terms::Vector{Term}, trian::Triangulation, Uspaces, locs, D::Int, cellmap, colors, previous=Any[];
               linear_maps::Bool=true, boundary=nothing)
    pf = PointForm(terms)
    ls = vcat(pf.test.leaves, pf.trial.leaves, leaves(terms, :unknown))
    isempty(ls) && error("HyperFEMKernels: integral without unknown, test or trial functions")
    w = uniform_value(terms[1].weights, "the quadrature weights")
    # quadrature points in the reference cell (for a boundary group: the face points mapped to the cell)
    xq = boundary === nothing ? uniform_value(get_data(ls[1].points), "the quadrature points") : boundary.xq
    # reuse geometry and shape-function tables of a previous operator with the same quadrature
    k = findfirst(g -> g.trian === trian && g.xq == xq && g.geo.w == w, previous)
    if k === nothing
        strian = get_triangulation(Uspaces[1])
        geo = boundary === nothing ? GeoTable(strian, xq, w; cells=cellmap) : GeoTable(strian, xq, w; cells=cellmap, Jf=boundary.Jf)
        tables = Tuple(FieldTable(s, xq, D) for s in Uspaces)
    else
        geo, tables = previous[k].geo, previous[k].tables
    end
    smv = Tuple(SlotSpec(slot_map(pf.test, f, t.nc, D), D) for (f, t) in enumerate(tables))
    smu = Tuple(SlotSpec(slot_map(pf.trial, f, t.nc, D), D) for (f, t) in enumerate(tables))
    aux = Tuple(a.array for a in pf.aux)
    AUX = Tuple{(Matrix{eltype(eltype(a))} for a in aux)...}
    NF = length(tables)
    lin = linear_maps ? LinearForm(terms, pf, tables, aux, pf.trial.nslots == 0 ? :res : :jac) : nothing
    Group{pf.test.nslots,pf.trial.nslots,typeof(pf),typeof(tables),AUX,typeof(smv),typeof(smu),D,NF,typeof(lin),typeof(geo.Jf)}(
        pf, lin, NTuple{NF,Int}(locs), tables, geo, aux, smv, smu, trian, collect(xq), Ref{Any}(nothing), cellmap, colors)
end

"""
    KernelOperator(res, jac, U, V; nthreads = Threads.nthreads(), cache_aux = false)

Gridap `NonlinearOperator` that assembles `res(u, v)` and `jac(u, du, v)` with fused, threaded
kernels. `res` and `jac` are written as for Gridap's `FEOperator`. Use [`update`](@ref) to change
`res`/`jac` (e.g. a new load factor) while keeping tables, sparsity pattern and colouring.

With `linear_maps = true` (default) residual coefficients and tangents are obtained by applying
precomputed linear maps to the state inputs of each term (no dual numbers) whenever the weak
form allows it (see `linearmaps.jl`); otherwise, and with `linear_maps = false`, by exact
differentiation with dual numbers.

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
    linear_maps::Bool
    fallback::Fallback
end

"""
    integration_cells(trian, space_trian) -> Vector{Int32}

Cell of the FE spaces for each cell of the integration triangulation `trian` (a subset of the
cells of the spaces, e.g. a subdomain or, in MPI, the owned cells of a part).
"""
function integration_cells(trian::Triangulation, space_trian::Triangulation)
    n = num_cells(trian)
    trian === space_trian && return Int32.(1:n)
    D = num_cell_dims(space_trian)
    glue = Geometry.get_glue(trian, Val(D))
    glue isa Geometry.FaceToFaceGlue ||
        error("HyperFEMKernels: integrals on this triangulation are not supported yet (no glue to the cells of the FE spaces)")
    t2m = glue.tface_to_mface
    s2m = Geometry.get_glue(space_trian, Val(D)).tface_to_mface
    m2s = Dict{Int,Int32}()
    for (sc, m) in enumerate(s2m)
        m2s[m] = sc
    end
    cmap = Vector{Int32}(undef, n)
    for (tc, m) in enumerate(t2m)
        sc = get(m2s, m, Int32(0))
        sc > 0 || error("HyperFEMKernels: the integration domain contains cells where the FE spaces are not defined")
        cmap[tc] = sc
    end
    cmap
end

"""
Integration cells grouped by the colour of their FE-space cell. Several integration cells may
map to the same FE cell (faces of a boundary): they are put in different colours.
"""
function group_colors(colors::Vector{Vector{Int32}}, cmap::Vector{Int32}, nspace::Int; subset=nothing)
    color_of = zeros(Int32, nspace)
    for (c, cells) in enumerate(colors), cell in cells
        color_of[cell] = c
    end
    out = Dict{Tuple{Int32,Int},Vector{Int32}}()
    seen = Dict{Int32,Int}()
    for tc in (subset === nothing ? eachindex(cmap) : subset)
        sc = cmap[tc]
        k = get(seen, sc, 0) + 1
        seen[sc] = k
        push!(get!(out, (color_of[sc], k), Int32[]), Int32(tc))
    end
    [out[k] for k in sort!(collect(keys(out)))]
end

"""
Compile the groups of the forms; forms whose compilation fails (unsupported triangulation,
quadrature or operation) are added to `fallback` with the reason and are assembled by Gridap.
"""
function compile_groups(forms::Vector{Form}, U, locs, D::Int, colors, previous=Any[];
                        linear_maps::Bool=true, fallback=Tuple{Any,String}[])
    groups = Any[]
    Us = field_spaces(U)
    strian = get_triangulation(Us[1])
    for form in forms
        fgroups = Any[]
        try
            # the cell map and the colouring of a triangulation only depend on the triangulation: reuse those of
            # a previous operator (`update`) instead of recomputing them (both are O(cells))
            prev = findfirst(g -> g.trian === form.trian, previous)
            cmap = prev === nothing ? integration_cells(form.trian, strian) : previous[prev].cellmap
            bykey = Dict{UInt,Vector{Term}}()
            order = UInt[]
            for t in form.terms
                k = objectid(t.weights)
                haskey(bykey, k) || (push!(order, k); bykey[k] = Term[])
                push!(bykey[k], t)
            end
            if num_cell_dims(form.trian) == D
                gcolors = prev === nothing ? group_colors(colors, cmap, num_cells(strian)) : previous[prev].colors
                for k in order
                    push!(fgroups, Group(bykey[k], form.trian, Us, locs, D, cmap, gcolors, previous; linear_maps))
                end
            else
                # boundary: one group per class of faces with the same points in the reference cell
                for k in order, b in boundary_classes(form.trian, bykey[k], D)
                    gcolors = group_colors(colors, cmap, num_cells(strian); subset=b.faces)
                    push!(fgroups, Group(bykey[k], form.trian, Us, locs, D, cmap, gcolors, previous; linear_maps, boundary=b))
                end
            end
        catch err
            err isa InterruptException && rethrow()
            push!(fallback, (form.trian, sprint(showerror, err)))
            continue
        end
        append!(groups, fgroups)
    end
    groups
end

# ---------------------------------------------------------------------------------------------
# Boundary integrals: faces of a lower-dimensional triangulation glued to the cells of the spaces.
#
# The quadrature points of a face are mapped to the reference cell through the glue; faces with
# the same mapped points (same local face and cell type) form a class and share the tables. The
# normal and any field of the volume evaluated on the faces are ordinary auxiliary data (Gridap
# arrays, materialised per face), so no new node type is needed.

"Faces with the same reference-cell points: (faces, xq, Jf) per class."
function boundary_classes(trian::Triangulation, terms::Vector{Term}, D::Int)
    pts = get_data(_first_points(terms))
    glue = Geometry.get_glue(trian, Val(D))
    cpts = lazy_map(evaluate, glue.tface_to_mface_map, pts)
    jf = lazy_map(∇, glue.tface_to_mface_map)
    classes = Dict{Vector{Float64},Vector{Int32}}()
    data = Dict{Vector{Float64},Any}()
    cache = array_cache(cpts); jcache = array_cache(jf)
    for face in 1:num_cells(trian)
        x = getindex!(cache, cpts, face)
        key = round.(reduce(vcat, [collect(p.data) for p in x]); digits=10)
        if !haskey(classes, key)
            classes[key] = Int32[]
            Jf = evaluate(getindex!(jcache, jf, face), pts[face])[1]
            data[key] = (xq=collect(x), Jf=Jf)
        end
        push!(classes[key], Int32(face))
    end
    [(faces=classes[k], xq=data[k].xq, Jf=data[k].Jf) for k in sort!(collect(keys(classes)))]
end

function _first_points(terms::Vector{Term})
    for t in terms
        l = _first_leaf(t.expr)
        l === nothing || return l.points
    end
    error("HyperFEMKernels: integral without unknown, test or trial functions")
end
_first_leaf(n::Leaf) = n
_first_leaf(n::Op) = (for a in n.args; l = _first_leaf(a); l === nothing || return l; end; nothing)
_first_leaf(n) = nothing

# ---------------------------------------------------------------------------------------------
# Hybrid fallback: contributions the engine cannot translate are assembled by Gridap.
#
# The user's `res`/`jac` are evaluated with Gridap's own arguments (FEFunction and bases); of the
# resulting `DomainContribution` only the triangulations in the fallback list are kept (the others
# are lazy and never evaluated) and assembled with Gridap's `SparseMatrixAssembler` into the same
# vector/matrix, whose sparsity pattern is the union of the kernel's and Gridap's.

"Keep the contributions of `dc` on `trians` (matched by identity; `[:all]` keeps everything)."
function restrict(dc::DomainContribution, trians::Vector{Any})
    any(t -> t === :all, trians) && return dc
    out = DomainContribution()
    for (trian, arr) in dc.dict
        any(t -> t === trian, trians) && (out.dict[trian] = arr)
    end
    out
end

fallback_residual_dc(op, x) = restrict(op.res(FEFunction(op.U, x), get_fe_basis(op.V)), op.fallback.rtrians)
fallback_jacobian_dc(op, x) = restrict(op.jac(FEFunction(op.U, x), get_trial_fe_basis(op.U), get_fe_basis(op.V)), op.fallback.jtrians)

function fallback_residual!(b, op, x)
    isempty(op.fallback.rtrians) && return b
    assemble_vector_add!(b, op.fallback.assem, collect_cell_vector(op.V, fallback_residual_dc(op, x)))
    b
end
function fallback_jacobian!(A, op, x)
    isempty(op.fallback.jtrians) && return A
    assemble_matrix_add!(A, op.fallback.assem, collect_cell_matrix(op.U, op.V, fallback_jacobian_dc(op, x)))
    A
end

"Sparsity pattern of the Gridap-assembled part of the Jacobian (nothing when there is none)."
function fallback_pattern(fb::Fallback, U, V, jac)
    isempty(fb.jtrians) && return nothing
    x = zeros(sum(num_free_dofs, field_spaces(U)))
    dc = restrict(jac(FEFunction(U, x), get_trial_fe_basis(U), get_fe_basis(V)), fb.jtrians)
    allocate_matrix(fb.assem, collect_cell_matrix(U, V, dc))
end

function warn_fallback(kind, list)
    for (trian, why) in list
        where = trian === :all ? "whole $kind is" : "$kind on a $(nameof(typeof(trian))) is"
        msg = "HyperFEMKernels: the $where assembled by Gridap (hybrid fallback): $why"
        @warn msg maxlog = 1 _id = hash(msg)
    end
end

function KernelOperator(res, jac, U::FESpace, V::FESpace; nthreads::Int=Threads.nthreads(), cache_aux::Bool=false,
                        linear_maps::Bool=true)
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
    rforms, rfb = capture(res, probes(U, :unknown), probes(V, :test))
    jforms, jfb = capture(jac, probes(U, :unknown), probes(U, :trial), probes(V, :test))
    locs = Tuple(fl.loc for fl in fields)
    colors = color_cells(fields, ncells, toff)
    rgroups = compile_groups(rforms, U, locs, D, colors; linear_maps, fallback=rfb)
    jgroups = compile_groups(jforms, U, locs, D, colors; linear_maps, fallback=jfb)
    warn_fallback("residual", rfb); warn_fallback("Jacobian", jfb)
    fb = Fallback(Any[t for (t, _) in rfb], Any[t for (t, _) in jfb], SparseMatrixAssembler(U, V))
    A0, nzpos = sparsity(fields, loc, ncells, toff, uoff, fallback_pattern(fb, U, V, jac))
    KernelOperator(res, jac, U, V, fields, loc, rgroups, jgroups, A0, nzpos, colors, nthreads, cache_aux, linear_maps, fb)
end

"""
    update(op, res, jac)

New operator with other `res`/`jac` (same spaces and structure), reusing the sparsity pattern,
positions and colouring of `op`.
"""
function update(op::KernelOperator, res, jac)
    D = num_cell_dims(get_triangulation(field_spaces(op.U)[1]))
    locs = Tuple(fl.loc for fl in op.fields)
    prev = vcat(op.rgroups, op.jgroups)
    lm = op.linear_maps
    rforms, rfb = capture(res, probes(op.U, :unknown), probes(op.V, :test))
    jforms, jfb = capture(jac, probes(op.U, :unknown), probes(op.U, :trial), probes(op.V, :test))
    rgroups = compile_groups(rforms, op.U, locs, D, op.colors, prev; linear_maps=lm, fallback=rfb)
    jgroups = compile_groups(jforms, op.U, locs, D, op.colors, prev; linear_maps=lm, fallback=jfb)
    warn_fallback("residual", rfb); warn_fallback("Jacobian", jfb)
    fb = Fallback(Any[t for (t, _) in rfb], Any[t for (t, _) in jfb], op.fallback.assem)
    same = length(fb.jtrians) == length(op.fallback.jtrians) && all(t -> any(s -> s === t, op.fallback.jtrians), fb.jtrians)
    A0, nzpos = same ? (op.A0, op.nzpos) : begin
        ncells = num_cells(get_triangulation(field_spaces(op.U)[1]))
        sparsity(op.fields, op.ndl, ncells, sum(num_free_dofs, field_spaces(op.V)), sum(num_free_dofs, field_spaces(op.U)),
                 fallback_pattern(fb, op.U, op.V, jac))
    end
    KernelOperator(res, jac, op.U, op.V, op.fields, op.ndl, rgroups, jgroups, A0, nzpos, op.colors, op.nthreads, op.cache_aux, op.linear_maps, fb)
end

# ---------------------------------------------------------------------------------------------
# Sparsity pattern and nonzero positions

function sparsity(fields::Vector{FieldLayout}, ndl::Int, ncells::Int, nrows::Int, ncols::Int, extra=nothing)
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
    if extra !== nothing   # union with the pattern of the Gridap-assembled part
        rv, cp = rowvals(extra), SparseArrays.getcolptr(extra)
        for c in 1:ncols, p in cp[c]:cp[c+1]-1
            push!(keys, (UInt64(c) << 32) | UInt64(rv[p]))
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
    fallback_residual!(b, op, x)
end

function Algebra.jacobian!(A::SparseMatrixCSC, op::KernelOperator, x::AbstractVector)
    fill!(nonzeros(A), 0.0)
    dvals = [get_dirichlet_dof_values(s) for s in field_spaces(op.U)]
    for g in op.jgroups
        run_cells!(Val(:jac), nothing, A, op, g, x, dvals)
    end
    fallback_jacobian!(A, op, x)
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

"Dense values (np × ncells) of the auxiliary arrays of a group (a tuple of type `AUX`)."
function materialize_aux(g::Group{NSV,NSU,PF,TT,AUX}, nthreads::Int) where {NSV,NSU,PF,TT,AUX}
    np, ncells = g.geo.np, size(g.geo.coords, 2)
    chunks = collect(Iterators.partition(1:ncells, cld(ncells, max(1, nthreads))))
    vals = ntuple(length(g.aux)) do k
        a = g.aux[k]                                   # not inferred: function barrier below
        M = Matrix{eltype(eltype(a))}(undef, np, ncells)
        @sync for ch in chunks
            Threads.@spawn _materialize!(M, a, ch, np)
        end
        M
    end
    AUX(vals)::AUX                                     # the assertion keeps the callers' inference precise (Julia 1.10)
end
function _materialize!(M::Matrix, a, cells, np::Int)
    cache = array_cache(a)
    for cell in cells
        v = getindex!(cache, a, cell)
        @inbounds for q in 1:np
            M[q, cell] = v[q]
        end
    end
    M
end

function aux_data(op::KernelOperator, g::Group{NSV,NSU,PF,TT,AUX}) where {NSV,NSU,PF,TT,AUX}
    if op.cache_aux
        g.auxcache[] === nothing && (g.auxcache[] = materialize_aux(g, op.nthreads))
        return g.auxcache[]::AUX
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
    (geo.w[q] * measure(Jt, geo.Jf), inv(Jt))
end
@inline measure(Jt, ::Nothing) = meas(Jt)           # volume
@inline measure(Jt, Jf) = meas(Jf ⋅ Jt)             # surface: Jacobian of the face map composed with the cell map

"Element residual of `cell` into `re` (local numbering of the element)."
function element_residual!(re, g::Group{NSV,NSU,PF,TT,AUX,SV,SU,D,NF,LF,JF}, xl, auxd, cell::Integer,
                           dNb) where {NSV,NSU,PF,TT,AUX,SV,SU,D,NF,LF,JF}
    fill!(re, 0.0)
    tables, locs, pf = g.tables, g.locs, g.pf
    zsv = zero(SVector{NSV,Float64})
    for q in 1:g.geo.np
        dV, iJt = point_geometry(g.geo, q, cell)
        map((t, W) -> point_weights!(W, t, q, iJt), tables, dNb)
        FV = map((t, l, W) -> field_point_values(t, xl, l, W), tables, locs, dNb)
        Aq = map(a -> @inbounds(a[q, cell]), auxd)
        R = if LF === Nothing
            ForwardDiff.gradient(s -> pf.core(pf.ops, pf.coefs, FV, Aq, s, EMPTY_SLOTS), zsv)
        else
            apply_maps(g.lin, g.lin.state(g.lin.ops, FV, Aq), Val(NSV))
        end
        map((t, l, sm, W) -> contract_residual!(re, t, l, sm, R, W, dV), tables, locs, g.smv, dNb)
    end
    re
end

"Element Jacobian of `cell` into `KeT` (transposed: KeT[trial, test], local numbering)."
function element_jacobian!(KeT, g::Group{NSV,NSU,PF,TT,AUX,SV,SU,D,NF,LF,JF}, xl, auxd, cell::Integer,
                           dNb) where {NSV,NSU,PF,TT,AUX,SV,SU,D,NF,LF,JF}
    fill!(KeT, 0.0)
    tables, locs, pf = g.tables, g.locs, g.pf
    zsv = zero(SVector{NSV,Float64})
    zsu = zero(SVector{NSU,Float64})
    for q in 1:g.geo.np
        dV, iJt = point_geometry(g.geo, q, cell)
        map((t, W) -> point_weights!(W, t, q, iJt), tables, dNb)
        FV = map((t, l, W) -> field_point_values(t, xl, l, W), tables, locs, dNb)
        Aq = map(a -> @inbounds(a[q, cell]), auxd)
        T = if LF === Nothing
            ForwardDiff.jacobian(r -> ForwardDiff.gradient(s -> pf.core(pf.ops, pf.coefs, FV, Aq, s, r), zsv), zsu)   # T[test slot, trial slot]
        else
            SMatrix{NSV,NSU,Float64,NSV * NSU}(apply_maps(g.lin, g.lin.state(g.lin.ops, FV, Aq), Val(NSV * NSU)))
        end
        map((tf, lf, smf, bf) -> map((tg, lg, smg, bg) ->
                contract_jacobian!(KeT, tf, lf, smf, bf, tg, lg, smg, bg, T, dV),
                tables, locs, g.smu, dNb),
            tables, locs, g.smv, dNb)
    end
    KeT
end

"""
Element Jacobian of `cell` as matrix products (requires node-contiguous dofs); buffers from
`mm_buffers`. The quadrature points are processed in blocks of `POINT_BLOCK[]` points so that
the G and B buffers of a block stay in L1 cache during the products (0 = all points at once).
"""
function element_jacobian_mm!(KeT, g::Group{NSV,NSU,PF,TT,AUX,SV,SU,D,NF,LF,JF}, xl, auxd, cell::Integer,
                              Wb, Bb, Gb) where {NSV,NSU,PF,TT,AUX,SV,SU,D,NF,LF,JF}
    fill!(KeT, 0.0)
    tables, locs, pf = g.tables, g.locs, g.pf
    np = g.geo.np
    qb = point_block(np)
    zsv = zero(SVector{NSV,Float64})
    zsu = zero(SVector{NSU,Float64})
    for q0 in 1:qb:np
        nq = min(qb, np - q0 + 1)
        for ql in 1:nq
            q = q0 + ql - 1
            dV, iJt = point_geometry(g.geo, q, cell)
            map((t, W) -> point_weights!(W, t, q, iJt), tables, Wb)
            FV = map((t, l, W) -> field_point_values(t, xl, l, W), tables, locs, Wb)
            Aq = map(a -> @inbounds(a[q, cell]), auxd)
            T = if LF === Nothing
                ForwardDiff.jacobian(r -> ForwardDiff.gradient(s -> pf.core(pf.ops, pf.coefs, FV, Aq, s, r), zsv), zsu)   # T[test slot, trial slot]
            else
                SMatrix{NSV,NSU,Float64,NSV * NSU}(apply_maps(g.lin, g.lin.state(g.lin.ops, FV, Aq), Val(NSV * NSU)))
            end
            map((t, sm, W, B) -> store_test_weights!(B, sm, W, ql, t.nn), tables, g.smv, Wb, Bb)
            map((smf, Gf) -> map((tg, smg, Wg, G) -> store_trial_block!(G, smf, smg, Wg, T, ql, dV, tg.nn),
                                 tables, g.smu, Wb, Gf), g.smv, Gb)
        end
        map((tf, lf, smf, B, Gf) -> map((tg, lg, smg, G) -> block_products!(KeT, tf, lf, smf, B, tg, lg, smg, G, nq),
                                        tables, locs, g.smu, Gf), tables, locs, g.smv, Bb, Gb)
    end
    KeT
end

"Points per block of the matrix-product path (`HFK_POINT_BLOCK`, 0 = all points of the cell)."
const POINT_BLOCK = Ref(parse(Int, get(ENV, "HFK_POINT_BLOCK", "12")))
point_block(np::Int) = (qb = POINT_BLOCK[]; qb <= 0 ? np : min(qb, np))

"Buffers for the matrix-product path: weights per field, B per field, G per block (f, g)."
function mm_buffers(g::Group{NSV,NSU,PF,TT,AUX,SV,SU,D}) where {NSV,NSU,PF,TT,AUX,SV,SU,D}
    np = point_block(g.geo.np)
    lv(sp::SlotSpec{NC,LV}) where {NC,LV} = LV
    Wb = map(t -> zeros(t.nn, D + 1), g.tables)
    Bb = map((t, sm) -> zeros(np * lv(sm), t.nn), g.tables, g.smv)
    Gb = map((tf, smf) -> map((tg, smg) -> zeros(tg.nn, np * lv(smf), tg.nc, tf.nc), g.tables, g.smu), g.tables, g.smv)
    (Wb, Bb, Gb)
end

"True if every field of the group has node-contiguous dofs (matrix-product path available)."
contiguous(g::Group) = all(t -> t isa FieldTable{<:Any,<:Any,<:Any,<:Any,true}, g.tables)

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
    for color in g.colors
        # tasks of at least WORK_PER_TASK "units" (points × element-matrix entries per cell)
        work = length(color) * g.geo.np * op.ndl^2
        nchunks = clamp(work ÷ WORK_PER_TASK, 1, min(op.nthreads, length(color)))
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

function cells!(::Val{K}, b, A, op::KernelOperator, g::Group{NSV,NSU,PF,TT,AUX,SV,SU,D},
                x, dvals, auxd, cells) where {K,NSV,NSU,PF,TT,AUX,SV,SU,D}
    ndl = op.ndl
    xl = zeros(ndl)
    dNb = map(t -> zeros(t.nn, D + 1), g.tables)
    if K === :res
        re = zeros(ndl)
        for cell in cells
            sc = g.cellmap[cell]
            gather_cell!(xl, op.fields, x, dvals, sc)
            element_residual!(re, g, xl, auxd, cell, dNb)
            scatter_residual!(b, op.fields, re, sc)
        end
    else
        KeT = zeros(ndl, ndl)
        nzv = nonzeros(A)
        mm = MATRIX_PRODUCTS && contiguous(g)
        Wb, Bb, Gb = mm ? mm_buffers(g) : (nothing, nothing, nothing)
        for cell in cells
            sc = g.cellmap[cell]
            gather_cell!(xl, op.fields, x, dvals, sc)
            if mm
                element_jacobian_mm!(KeT, g, xl, auxd, cell, Wb, Bb, Gb)
            else
                element_jacobian!(KeT, g, xl, auxd, cell, dNb)
            end
            scatter_jacobian!(nzv, op.nzpos, KeT, ndl, sc)
        end
    end
    nothing
end

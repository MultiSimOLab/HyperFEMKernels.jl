# Internal variables (states) at the quadrature points, updated by fused, threaded kernels.
#
# Gridap's `update_state!(updater, f...)` evaluates every argument lazily, cell by cell, and calls
# `updater` at each point to overwrite the `CellState`s among the arguments. Here the same call
# is executed by the engine: the arguments are read once as a lazy tree (like a weak form), the
# FE functions in it are evaluated from their dof values with the basis tables, the states are
# read and written in place, and `updater` is called from a generated pointwise function in a
# threaded loop over cells. The user's code does not change: the updater and the arguments are
# the ones of Gridap's call.
#
# Two storages are supported: Gridap's `CellState` (one vector per cell) and `KernelState`, a
# dense `np × ncells` matrix (the layout for GPU and MPI). Both can be used in weak forms.

# ---------------------------------------------------------------------------------------------
# KernelState: dense storage

"""
    KernelState(v, dΩ)
    KernelState(v, points::CellPoint)
    KernelState{T}(undef, dΩ)
    KernelState(s::CellState)

Internal variable with the value `v` (a number or tensor) at every quadrature point of `dΩ`,
stored as a dense `np × ncells` matrix (`values[q, cell]`). It is a `CellField` that can be
evaluated at the points it was created from (so it can appear in weak forms and in
`update_state!`), like Gridap's `CellState`.
"""
struct KernelState{T,P<:CellPoint} <: CellField
    points::P
    values::Matrix{T}
end

function KernelState{T}(::UndefInitializer, points::CellPoint) where {T}
    np = length(testitem(get_data(points)))
    KernelState{T,typeof(points)}(points, Matrix{T}(undef, np, length(get_data(points))))
end
KernelState{T}(::UndefInitializer, a) where {T} = KernelState{T}(undef, get_cell_points(a))
function KernelState(v::Number, points::CellPoint)
    s = KernelState{typeof(v)}(undef, points)
    fill!(s.values, v)
    s
end
KernelState(v::Number, a) = KernelState(v, get_cell_points(a))
function KernelState(s::CellState{T}) where {T}
    k = KernelState{T}(undef, s.points)
    cache = array_cache(s.values)
    for cell in axes(k.values, 2)
        v = getindex!(cache, s.values, cell)
        for q in axes(k.values, 1)
            k.values[q, cell] = v[q]
        end
    end
    k
end

CellData.get_triangulation(s::KernelState) = get_triangulation(s.points)
CellData.DomainStyle(::Type{KernelState{T,P}}) where {T,P} = CellData.DomainStyle(P)
CellData.get_data(::KernelState) = error("HyperFEMKernels: get_data cannot be called on a KernelState (project it to an FE space first)")

"Lazy per-cell view of a `KernelState` (cell `i` → `view(M, :, i)`), what Gridap sees when it evaluates one."
struct StateColumns{T} <: AbstractVector{SubArray{T,1,Matrix{T},Tuple{Base.Slice{Base.OneTo{Int}},Int},true}}
    M::Matrix{T}
end
Base.size(a::StateColumns) = (size(a.M, 2),)
Base.IndexStyle(::Type{<:StateColumns}) = IndexLinear()
Base.getindex(a::StateColumns, i::Int) = view(a.M, :, i)

function Arrays.evaluate!(cache, s::KernelState, x::CellPoint)
    same_points(s.points, x) && return StateColumns(s.values)
    error("HyperFEMKernels: a KernelState can only be evaluated at the points it was created from")
end

"""
Same quadrature points: the same object, or the same triangulation and reference points (then the
physical points are the same too; comparing them, as `==` on `CellPoint` does, is O(cells)).
"""
same_points(p::CellPoint, x::CellPoint) = p === x ||
    (get_triangulation(p) === get_triangulation(x) && p.cell_ref_point == x.cell_ref_point) || p == x

# ---------------------------------------------------------------------------------------------
# Storage read/written by the kernels: `s[q, cell]`

"Gridap's `CellState` storage (one vector per cell) indexed as `s[q, cell]`."
struct NestedState{T}
    v::Vector{Vector{T}}
end
Base.@propagate_inbounds Base.getindex(s::NestedState, q::Int, cell::Int) = s.v[cell][q]
Base.@propagate_inbounds Base.setindex!(s::NestedState, val, q::Int, cell::Int) = (s.v[cell][q] = val)
Base.eltype(::NestedState{T}) where {T} = T

storage(s::KernelState) = s.values
function storage(s::CellState{T}) where {T}
    s.values isa Vector{Vector{T}} || error("HyperFEMKernels: unsupported CellState storage $(typeof(s.values))")
    NestedState{T}(s.values)
end

is_state(s) = s isa CellState || s isa KernelState
"True if `a` is the array Gridap returns when evaluating the state `s` at its points."
is_state_array(s::CellState, a) = a === s.values
is_state_array(s::KernelState, a) = a isa StateColumns && a.M === s.values

# ---------------------------------------------------------------------------------------------
# Reading the arguments: FE functions, states, auxiliary data

"A state read at the point: `S[k]`."
struct StateLeaf <: Node
    k::Int
    T::DataType
end
emit!(st::CodegenState, n::StateLeaf) = :(S[$(n.k)])
_walk_leaves!(found, role, n::StateLeaf) = nothing
_walk_aux!(found, n::StateLeaf) = nothing

"An FE function (value, `deriv = 0`, or gradient, `deriv = 1`) read at the point: `FV[k]`."
struct FELeaf <: Node
    k::Int
    deriv::Int
    T::DataType
end
emit!(st::CodegenState, n::FELeaf) = :(FV[$(n.k)])
_walk_leaves!(found, role, n::FELeaf) = nothing
_walk_aux!(found, n::FELeaf) = nothing

"""
Dof values and basis of an FE function at the points, read from Gridap's lazy tree: the cell dof
ids (a `Table`, with an optional map from the cells of the points to the cells of the space),
the free and Dirichlet values, and the uniform `np × nd` matrix of basis values (or reference
gradients) at the points.
"""
struct FEData{T,DV,VF,VD,N,C,P,I}
    vals::Any                 # Gridap's cell dof values (identity key within one capture)
    free::VF                  # free values (a view of the block vector for a field of a multi-field function)
    diri::VD
    ptrs::P                   # Table of cell dof ids
    data::I
    cellmap::C                # nothing, or point cell → space cell
    N::Matrix{N}              # np × nd basis at the points (values, DV = 0, or reference gradients, DV = 1)
    nd::Int
end
Base.eltype(::FEData{T}) where {T} = T
fe_deriv(::FEData{T,DV}) where {T,DV} = DV

const StateContext = @NamedTuple{x::CellPoint, states::Vector{Any}, fes::Vector{FEData}, aux::Vector{Aux}, D::Int}

function state_node(a::AbstractArray, ctx::StateContext)
    k = findfirst(s -> is_state_array(s, a), ctx.states)
    k === nothing || return StateLeaf(k, eltype(eltype(a)))
    if a isa LazyArray && a.maps isa Fill && a.maps.value isa Fields.ConstantField && is_constant_value(a.maps.value.value)
        return Const(a.maps.value.value)
    end
    if a isa Fill && a.value isa AbstractVector && !isempty(a.value) && is_constant_value(a.value[1]) &&
       all(v -> v == a.value[1], a.value)
        return Const(a.value[1])
    end
    if a isa LazyArray
        k = first(a.maps)
        if k isa Fields.LinearCombinationMap && k.column isa Colon
            n = fe_leaf(a, 0, ctx)
            n === nothing || return n
        elseif k isa BroadcastingFieldOpMap
            # physical gradient of an FE function: inv(Jt) ⋅ (reference gradient of the linear combination)
            if k.op === dot && length(a.args) == 2 && is_inverse_jacobian(a.args[1], ctx.D)
                n = fe_leaf(a.args[2], 1, ctx)
                n === nothing || return n
            end
            return Op(k.op, Node[state_node(b, ctx) for b in a.args])
        end
    end
    Aux(a)
end

"The single matrix of a cell-wise array that is uniform across cells (`Fill`, or `CompressedArray` with one value)."
function uniform_matrix(a)
    a isa Fill && a.value isa Matrix && return a.value
    a isa CompressedArray && length(a.values) == 1 && a.values[1] isa Matrix && return a.values[1]
    nothing
end

"The array `pinvJt(∇(cell_map))` at the points (a `Fill` on affine meshes, lazy otherwise)."
function is_inverse_jacobian(a, D::Int)
    a isa Fill && return eltype(a.value) <: TensorValue{D,D}
    a isa LazyArray || return false
    k = first(a.maps)
    (k isa BroadcastingFieldOpMap && k.op === Fields.pinvJt) || (k isa Broadcasting && k.f === Fields.pinvJt)
end

function fe_leaf(a::LazyArray, deriv::Int, ctx::StateContext)
    first(a.maps) isa Fields.LinearCombinationMap || return nothing
    length(a.args) == 2 || return nothing
    vals, bx = a.args
    vals isa LazyArray && first(vals.maps) isa Broadcasting && first(vals.maps).f isa Arrays.PosNegReindex || return nothing
    N = uniform_matrix(bx)          # basis at the points: the same matrix for every cell (one cell type)
    N === nothing && return nothing
    ids = vals.args[1]
    if ids isa Arrays.Table
        table, cellmap = ids, nothing
    elseif ids isa LazyArray && first(ids.maps) isa Arrays.Reindex && first(ids.maps).values isa Arrays.Table
        table, cellmap = first(ids.maps).values, ids.args[1]
    else
        return nothing
    end
    np = length(testitem(get_data(ctx.x)))
    size(N, 1) == np || return nothing
    # the same FE function appears with different bases (values, reference gradients): one leaf per basis
    k = findfirst(f -> f.vals === vals && f.N === N, ctx.fes)
    T = eltype(a)
    if k === nothing
        pnr = first(vals.maps).f
        free, diri = pnr.values_pos, pnr.values_neg
        eltype(free) == eltype(diri) || return nothing
        nd = size(N, 2)
        push!(ctx.fes, FEData{eltype(T),deriv,typeof(free),typeof(diri),eltype(N),typeof(cellmap),typeof(table.ptrs),typeof(table.data)}(
            vals, free, diri, table.ptrs, table.data, cellmap, N, nd))
        k = length(ctx.fes)
    end
    FELeaf(k, deriv, eltype(T))
end

# ---------------------------------------------------------------------------------------------
# Compiled update

"""
    StateUpdate

A compiled `update_state!` call: the generated pointwise function, the FE functions, states and
auxiliary data it reads, and the states it may write (the first `NW` of `states` are the state
arguments in the order of the call; the updater overwrites the first `length(new)` of them).
"""
struct StateUpdate{F,O,FE,ST,NW,AUX,G}
    core::F
    ops::O
    fes::FE                 # tuple of FEData
    states::ST              # tuple of storages (written ones first)
    aux::Vector{Aux}        # Gridap arrays (evaluated per cell)
    geo::G                  # GeoTable, or nothing when no gradient is read
    x::CellPoint
    np::Int
    ncells::Int
    expr::Expr
end
nwritten(::StateUpdate{F,O,FE,ST,NW}) where {F,O,FE,ST,NW} = NW

function StateUpdate(updater, args::Tuple, x::CellPoint, written::Vector{Int}, nthreads::Int)
    trian = get_triangulation(x)
    D = num_cell_dims(trian)
    D == num_point_dims(trian) || error("HyperFEMKernels: states on a boundary are not supported yet")
    xdata = get_data(x)
    np = length(testitem(xdata))
    ncells = length(xdata)
    # states: the written ones first (in order), then the others referenced by the arguments
    sts = Any[args[i] for i in written]
    for a in args
        is_state(a) && !any(s -> s === a, sts) && push!(sts, a)
    end
    ctx = StateContext((x, sts, FEData[], Aux[], D))
    arrs = map(a -> evaluate_arg(a, x), args)
    nodes = Node[state_node(a, ctx) for a in arrs]
    for n in nodes
        _walk_aux!(ctx.aux, n)
    end
    # code: (ops, FV, A, S) -> updater(args...)
    st = CodegenState(SlotLayout(Leaf[]), SlotLayout(Leaf[]), ctx.aux)
    push!(st.ops, updater)
    argex = [emit_node!(st, n) for n in nodes]
    ex = :((ops, FV, A, S) -> begin
        $(st.body...)
        return ops[1]($(argex...))
    end)
    core = RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, ex)
    needs_geo = any(f -> fe_deriv(f) == 1, ctx.fes)
    geo = needs_geo ? geo_table(trian, uniform_value(xdata, "the quadrature points")) : nothing
    fes = Tuple(ctx.fes)
    storages = Tuple(storage(s) for s in sts)
    AUX = Tuple{(Matrix{eltype(eltype(a.array))} for a in ctx.aux)...}
    StateUpdate{typeof(core),typeof(Tuple(st.ops)),typeof(fes),typeof(storages),length(written),AUX,typeof(geo)}(
        core, Tuple(st.ops), fes, storages, ctx.aux, geo, x, np, ncells, ex)
end

# a state is its storage (Gridap's evaluate! compares the points with ==, O(cells)); anything else goes through Gridap
evaluate_arg(a::CellState, x::CellPoint) = same_points(a.points, x) ? a.values : evaluate(a, x)
evaluate_arg(a::KernelState, x::CellPoint) = same_points(a.points, x) ? StateColumns(a.values) : evaluate(a, x)
evaluate_arg(a::CellField, x::CellPoint) = evaluate(a, x)

# geometry tables are O(cells): keep the last few, by triangulation and points
const GEO_CACHE = Any[]
const GEO_LOCK = ReentrantLock()
function geo_table(trian::Triangulation, xq)
    lock(GEO_LOCK) do
        k = findfirst(e -> e[1] === trian && e[2] == xq, GEO_CACHE)
        k === nothing || return GEO_CACHE[k][3]
        geo = GeoTable(trian, xq, ones(length(xq)))
        length(GEO_CACHE) >= 8 && popfirst!(GEO_CACHE)
        push!(GEO_CACHE, (trian, xq, geo))
        geo
    end
end

"Show the generated pointwise function (debugging)."
show_code(su::StateUpdate) = println(Base.remove_linenums!(deepcopy(su.expr)))

# ---------------------------------------------------------------------------------------------
# Kernels

"Local dof values of the FE function in `cell` (a cell of the points)."
@inline function gather_fe!(xl, fe::FEData, cell::Int)
    sc = fe.cellmap === nothing ? cell : Int(fe.cellmap[cell])
    @inbounds begin
        p0 = fe.ptrs[sc]
        for j in 1:fe.nd
            id = fe.data[p0+j-1]
            xl[j] = id > 0 ? fe.free[id] : fe.diri[-id]
        end
    end
    xl
end

"Value (or physical gradient, `inv(Jt) ⋅` reference gradient) of the FE function at point `q`."
@inline function fe_point_value(fe::FEData{T,0}, xl, q::Int, iJt) where {T}
    linear_combination_at(fe, xl, q)
end
@inline function fe_point_value(fe::FEData{T,1}, xl, q::Int, iJt) where {T}
    iJt ⋅ linear_combination_at(fe, xl, q)
end
@inline function linear_combination_at(fe::FEData{T,DV,VF,VD,N}, xl, q::Int) where {T,DV,VF,VD,N}
    s = zero(N) * zero(eltype(xl))
    @inbounds for j in 1:fe.nd
        s += xl[j] * fe.N[q, j]
    end
    s
end

"Update the states of one cell (no allocation; the part that runs on each device)."
function update_cell!(su::StateUpdate{F,O,FE,ST,NW}, xls, auxd, cell::Int) where {F,O,FE,ST,NW}
    map((fe, xl) -> gather_fe!(xl, fe, cell), su.fes, xls)
    for q in 1:su.np
        iJt = su.geo === nothing ? nothing : point_geometry(su.geo, q, cell)[2]
        FV = map((fe, xl) -> fe_point_value(fe, xl, q, iJt), su.fes, xls)
        Aq = map(a -> @inbounds(a[q, cell]), auxd)
        Sq = map(s -> @inbounds(s[q, cell]), su.states)
        need, new = first_and_tail(su.core(su.ops, FV, Aq, Sq))
        if need
            # as in Gridap: the first length(new) state arguments are overwritten, the rest are inputs
            length(new) <= NW || error("HyperFEMKernels: the updater returned $(length(new)) states for $NW CellState/KernelState arguments")
            write_states!(su.states, new, q, cell)
        end
    end
    nothing
end

@inline function write_states!(S::Tuple, new::NTuple{NR,Any}, q::Int, cell::Int) where {NR}
    ntuple(i -> (@inbounds S[i][q, cell] = new[i]; nothing), Val(NR))
    nothing
end

"Dense values of the auxiliary arrays (np × ncells each)."
function materialize_state_aux(su::StateUpdate{F,O,FE,ST,NW,AUX}, nthreads::Int) where {F,O,FE,ST,NW,AUX}
    vals = ntuple(length(su.aux)) do k
        a = su.aux[k].array
        a isa StateColumns && return a.M
        M = Matrix{eltype(eltype(a))}(undef, su.np, su.ncells)
        chunks = collect(Iterators.partition(1:su.ncells, cld(su.ncells, max(1, nthreads))))
        @sync for ch in chunks
            Threads.@spawn _materialize!(M, a, ch, su.np)
        end
        M
    end
    AUX(vals)::AUX
end

function run_update!(su::StateUpdate, nthreads::Int)
    auxd = materialize_state_aux(su, nthreads)
    # the updater (a return mapping) costs ~1 µs per point: tasks of at least 16 cells
    nchunks = clamp(su.ncells ÷ 16, 1, min(nthreads, su.ncells))
    if nchunks == 1
        update_cells!(su, auxd, 1:su.ncells)
    else
        @sync for ch in Iterators.partition(1:su.ncells, cld(su.ncells, nchunks))
            Threads.@spawn update_cells!(su, auxd, ch)
        end
    end
    nothing
end

function update_cells!(su::StateUpdate, auxd, cells)
    xls = map(fe -> zeros(eltype(fe.free), fe.nd), su.fes)
    for cell in cells
        update_cell!(su, xls, auxd, cell)
    end
    nothing
end

# ---------------------------------------------------------------------------------------------
# API

"""
    update_states!(updater, args::CellField...; nthreads = Threads.nthreads()) -> StateUpdate

Gridap's `update_state!(updater, args...)` executed by the engine: at every quadrature point
`need, states... = updater(args_at_point...)` and, when `need`, the `CellState`/`KernelState`
arguments are overwritten with `states`, in order. The arguments are read once (FE functions from
their dof values, states in place, other data through Gridap) and the loop over cells is threaded.
Arguments the engine cannot read are evaluated by Gridap per cell; if the call cannot be compiled
at all, Gridap's algorithm is used (with a warning).
"""
function update_states!(updater, args::CellField...; nthreads::Int=Threads.nthreads())
    written = findall(is_state, args)
    isempty(written) && error("HyperFEMKernels: at least one CellState or KernelState has to be given to update_states!")
    x = args[written[1]].points
    all(i -> same_points(args[i].points, x), written) ||
        error("HyperFEMKernels: all the states given to update_states! need to be defined on the same points")
    su = try
        StateUpdate(updater, args, x, written, nthreads)
    catch err
        err isa InterruptException && rethrow()
        msg = "HyperFEMKernels: update_state! is executed by Gridap's algorithm (not compiled): $(sprint(showerror, err))"
        @warn msg maxlog = 1 _id = hash(msg)
        generic_update!(updater, args, x, written)
        return nothing
    end
    run_update!(su, nthreads)
    su
end

"Gridap's algorithm (cell by cell, lazy evaluation), for both storages."
function generic_update!(updater, args, x::CellPoint, written::Vector{Int})
    fx = map(a -> evaluate_arg(a, x), args)
    caches = map(array_cache, fx)
    sts = map(i -> storage(args[i]), Tuple(written))
    np = length(testitem(get_data(x)))
    for cell in 1:length(get_data(x))
        fxi = map((c, f) -> getindex!(c, f, cell), caches, fx)
        for q in 1:np
            need, new = first_and_tail(updater(map(f -> f[q], fxi)...))
            need || continue
            for i in eachindex(new)
                sts[i][q, cell] = new[i]
            end
        end
    end
    nothing
end

"""
    KernelUpdater(f)

Wrapper that makes `update_state!(KernelUpdater(f), args...)` run with the engine
([`update_states!`](@ref)) instead of Gridap's loop; `update_state!(f, s::KernelState, args...)`
does so as well.
"""
struct KernelUpdater{F}
    f::F
end
(u::KernelUpdater)(args...) = u.f(args...)
CellData.update_state!(u::KernelUpdater, f::CellField...) = (update_states!(u.f, f...); nothing)
CellData.update_state!(updater::Function, s::KernelState, f::CellField...) = (update_states!(updater, s, f...); nothing)

"""
    allocations(su::StateUpdate) -> Int

Bytes allocated by the update of one cell (after a warm-up call, the states of that cell are left
as they were); must be 0 unless the updater itself allocates.
"""
function allocations(su::StateUpdate)
    auxd = materialize_state_aux(su, 1)
    xls = map(fe -> zeros(eltype(fe.free), fe.nd), su.fes)
    saved = map(s -> [s[q, 1] for q in 1:su.np], su.states)     # the states of the cell are restored afterwards
    update_cell!(su, xls, auxd, 1)
    bytes = @allocated update_cell!(su, xls, auxd, 1)
    map((s, v) -> (for q in 1:su.np; s[q, 1] = v[q]; end), su.states, saved)
    bytes
end

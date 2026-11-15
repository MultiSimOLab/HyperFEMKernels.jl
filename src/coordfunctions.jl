# Functions of the coordinates in the weak form: `CellField(f, Ω)`, or a Julia function `f` used
# directly in an operation (`f * v`, `Operation(g)(f, uh)`), which Gridap turns into the same thing.
#
# Gridap evaluates such a field at the quadrature points as
#   lazy_map(GenericField(f), physical points)
# where the physical points are the cell map evaluated at the reference points (a `CartesianMap`
# on Cartesian grids, `Broadcasting(AffineMap)` on simplices and boundary faces, a linear
# combination of the cell nodes on other cells, a `FaceCompressedVector` for a field of the cells
# evaluated on a boundary). The reader recognises only the outer map (`coord_array`); the engine
# calls `f` on the physical points (`materialize_coord!`), without evaluating Gridap's lazy tree and
# without allocations.
#
# The physical points are Gridap's own (`coord_points!`), not recomputed: `f` then sees exactly the
# same `x` as in Gridap, which matters for a function that is discontinuous on a face (`x[1] >= L` for
# a load on a side, a material indicator with a jump on a mesh line), where a round-off difference in
# `x` would change the value. They are read only after checking, on a few sample columns, that the
# group's geometry gives the same points to round-off (`coord_points_match`); otherwise (other points,
# other cells) the array is evaluated by Gridap as before. The points only depend on the cells and the
# quadrature, but Gridap may build them in more than one way for the same faces (the face map for
# `CellField(f, Γ)`, the cell map for a field of the cells on `Γ`), equal only to round-off; so the
# geometry stores a list of point sets (`GeoTable.points`), and an array uses a stored set only if its
# own points equal it bit for bit on sample columns, otherwise a new set is read. In practice one or two
# sets per group, shared by its functions of x and by later captures (`update`).

"""
    CoordArray

An auxiliary array of a weak form that is a Julia function `f` of the physical coordinates at the
quadrature points. It behaves as Gridap's array (`getindex`), and the assembly materialises it with
the engine (`materialize_coord!`): `f(x)` at Gridap's physical points, stored with the geometry of
the group (`set`: index of its point set in `GeoTable.points`, assigned by `engine_aux`).

The type of `f` is part of the type (and of the structural signature of a capture, see
`signature`): a new closure type means a new structure (`update` recompiles), while the same
function, or the same closure type with other captured values, is rebound and the new values are
used at the next assembly (as for the operations of the form; not with `cache_aux = true`, which
never refills the auxiliary buffers).
"""
mutable struct CoordArray{T,F} <: AbstractVector{Vector{T}}   # mutable: `===` is the identity of the leaf (see `same_aux`)
    lazy::AbstractArray      # Gridap's array (the same values, evaluated lazily)
    f::F
    set::Int                 # point set in the geometry of the group (a group stores its own copy, `engine_aux`); 0 in the tree
end
CoordArray{T,F}(lazy, f) where {T,F} = CoordArray{T,F}(lazy, f, 0)
Base.size(a::CoordArray) = size(a.lazy)
Base.IndexStyle(::Type{<:CoordArray}) = IndexLinear()
Base.getindex(a::CoordArray, i::Int) = a.lazy[i]
Arrays.array_cache(a::CoordArray) = array_cache(a.lazy)
Arrays.getindex!(c, a::CoordArray, i::Int) = getindex!(c, a.lazy, i)
Arrays.testitem(a::CoordArray) = testitem(a.lazy)
# Two uses of a function of x share a slot only when the result cannot depend on captured values: the
# same function type without fields (a named function, a closure capturing nothing: its type decides
# equality) at the same points (Gridap's points array, compared with `===`, which on its immutable lazy
# arrays compares contents: a false "different" only costs a slot). A closure with captures is never
# merged: whether two of them are equal depends on the captured values, which are not part of the
# structural signature, so merging would make the slot layout change between captures (`update`).
same_aux(a::CoordArray, b::CoordArray) =
    a === b || (typeof(a) == typeof(b) && Base.issingletontype(typeof(a.f)) && length(a) == length(b) &&
                a.lazy.args[1] === b.lazy.args[1])

"""
    coord_array(a) -> CoordArray or nothing

`a` recognised as a function of the coordinates at the points: `lazy_map(GenericField(f), points)`
(or a non-void `VoidField` of it) with one argument whose cells hold physical points, and a
concrete value type.
"""
function coord_array(a)
    a isa LazyArray && a.maps isa Fill && length(a.args) == 1 || return nothing
    f = coord_function(a.maps.value)
    f === nothing && return nothing
    eltype(a.args[1]) <: AbstractVector{<:VectorValue} || return nothing
    T = eltype(eltype(a))
    isconcretetype(T) || return nothing
    CoordArray{T,typeof(f)}(a, f)
end
# the function of a `GenericField`, also wrapped in a non-void `VoidField` (a field of a triangulation
# restricted to the cells of a sub-triangulation, e.g. `CellField(f, Ωs)` with fields of `Ω`)
coord_function(m::Fields.GenericField) = m.object
coord_function(m::Fields.VoidField) = (!m.isvoid && m.field isa Fields.GenericField) ? m.field.object : nothing
coord_function(m) = nothing

"Physical coordinates of point `q` of column `cell` of the geometry (cell nodes times geometric shape functions)."
@inline function point_coordinate(geo::GeoTable{D}, q::Int, cell::Integer) where {D}
    x = zero(VectorValue{D,Float64})
    @inbounds for n in 1:geo.ng
        x += geo.Mv[q, n] * geo.coords[n, cell]
    end
    x
end

"Whether `HFK_DEBUG_COORD=1`: print why a function of x is left to Gridap."
debug_coord() = get(ENV, "HFK_DEBUG_COORD", "0") == "1"

"Sample positions (in `cols`) used to compare point arrays: first, last and a spread of up to 16."
sample_positions(n::Int) = n == 0 ? Int[] : unique!(sort!(vcat([1, n], round.(Int, range(1, n; length=min(n, 16))))))

"""
    coord_points_match(a, geo, cols, idx) -> Bool

Whether the geometry `geo` reproduces Gridap's physical points of `a` (to round-off) on sample
columns among `cols` (the columns the group integrates: for a boundary group, the faces of its
class, whose reference points are those of `geo`); column `i` of `geo` holds the cell of entry
`idx[i]` of `a`. `HFK_DEBUG_COORD=1` prints the reason of a mismatch.
"""
function coord_points_match(a::CoordArray, geo::GeoTable, cols, idx)
    pts = a.lazy.args[1]
    try
        cache = array_cache(pts)
        for j in sample_positions(length(cols))
            i = cols[j]
            p = getindex!(cache, pts, idx[i])
            if length(p) != geo.np
                debug_coord() && @info "HyperFEMKernels: function of x left to Gridap: $(length(p)) points per cell, the group has $(geo.np)"
                return false
            end
            h = maximum(norm(geo.coords[k, i] - geo.coords[1, i]) for k in 1:geo.ng)
            for q in 1:geo.np
                x = point_coordinate(geo, q, i)
                if !(norm(x - p[q]) <= 1e-10 * h + 1e-13 * norm(p[q]))
                    debug_coord() && @info "HyperFEMKernels: function of x left to Gridap: point $q of column $i is $(p[q]), the group's geometry gives $x"
                    return false
                end
            end
        end
    catch err
        err isa InterruptException && rethrow()
        debug_coord() && @info "HyperFEMKernels: function of x left to Gridap: reading its points failed" exception = err
        return false
    end
    true
end

"Whether the points of `a` equal, bit for bit on the sample columns, the stored point set `X` (columns `cols`)."
function same_points(a::CoordArray, X::Matrix, cols, idx)
    pts = a.lazy.args[1]
    cache = array_cache(pts)
    for j in sample_positions(length(cols))
        p = getindex!(cache, pts, idx[cols[j]])
        all(q -> p[q] === X[q, j], 1:size(X, 1)) || return false
    end
    true
end

"""
    coord_points!(geo, a, cols, idx) -> Int or nothing

Index of the point set of `geo` (`geo.points[][2]`) holding Gridap's physical points of `a` at the
columns `cols` (`X[q, j]` for column `cols[j]`, whose cell is entry `idx[cols[j]]` of `a`): a stored set
equal to the points of `a` bit for bit on the sample columns, or a new one read from `a`. `nothing`
if reading them fails (the array is then left to Gridap).
"""
function coord_points!(geo::GeoTable{D}, a::CoordArray, cols, idx) where {D}
    sets = geo.points[][2]
    try
        k = findfirst(X -> same_points(a, X, cols, idx), sets)
        k === nothing || return k
        X = Matrix{VectorValue{D,Float64}}(undef, geo.np, length(cols))
        pts = a.lazy.args[1]
        cache = array_cache(pts)
        for (j, i) in enumerate(cols)
            p = getindex!(cache, pts, idx[i])
            for q in 1:geo.np
                X[q, j] = p[q]
            end
        end
        push!(sets, X)
        return length(sets)
    catch err
        err isa InterruptException && rethrow()
        debug_coord() && @info "HyperFEMKernels: function of x left to Gridap: reading its points failed" exception = err
        return nothing
    end
end

"""
    engine_aux(a, geo, cols, idx)

The auxiliary array a group stores: a `CoordArray` whose points the group's geometry reproduces
(`coord_points_match`) is kept, with its point set (`coord_points!`), and materialised by the engine;
otherwise its Gridap array (the same values, evaluated by Gridap). Other arrays are returned
unchanged. `cols` is a function returning the columns the group integrates, called only for a
function of x and only the first time (stored in `geo.points`).
"""
engine_aux(a, geo, cols, idx) = a
function engine_aux(a::CoordArray, geo::GeoTable{D}, cols, idx) where {D}
    geo.points[] === nothing && (geo.points[] = (Vector{Int32}(cols()), Matrix{VectorValue{D,Float64}}[]))
    cs = geo.points[][1]
    coord_points_match(a, geo, cs, idx) || return a.lazy
    k = coord_points!(geo, a, cs, idx)
    k === nothing && return a.lazy
    typeof(a)(a.lazy, a.f, k)        # a copy per group: the leaf of the tree is shared by the groups of all boundary classes
end

"Columns a group integrates (all its colours), in increasing order."
group_columns(colors) = sort!(reduce(vcat, colors; init=Int32[]))

"""
Dense values `M[q, i]` of a function of the coordinates at the physical points `X[q, j]` of the columns
`i = cols[j]`, `j ∈ js` (only the columns the group integrates). No allocation (if `f` does not allocate).
"""
function materialize_coord!(M::Matrix, a::CoordArray, js, cols::Vector{Int32}, np::Int, X::Matrix)
    f = a.f
    @inbounds for j in js
        i = cols[j]
        for q in 1:np
            M[q, i] = f(X[q, j])
        end
    end
    M
end
"The same, with the point set of `a` stored in the geometry (`coord_points!`)."
function materialize_coord!(M::Matrix, a::CoordArray, js, np::Int, geo::GeoTable)
    cols, sets = geo.points[]
    materialize_coord!(M, a, js, cols, np, sets[a.set])
end

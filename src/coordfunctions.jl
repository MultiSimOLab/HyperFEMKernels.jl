# Functions of the coordinates in the weak form: `CellField(f, Ω)`, or a Julia function `f` used
# directly in an operation (`f * v`, `Operation(g)(f, uh)`), which Gridap turns into the same thing.
#
# Gridap evaluates such a field at the quadrature points as
#   lazy_map(GenericField(f), physical points)
# where the physical points are the cell map evaluated at the reference points (a `CartesianMap`
# on Cartesian grids, `Broadcasting(AffineMap)` on simplices and boundary faces, a linear
# combination of the cell nodes on other cells, a `FaceCompressedVector` for a field of the cells
# evaluated on a boundary). The reader recognises only the outer map (`coord_array`) and the engine
# recomputes the physical points from the nodes of the cells and the geometric shape functions of
# the group (`point_coordinate`, the same for volume, boundary and skeleton groups) and calls `f`
# on them (`materialize_coord!`): no evaluation of Gridap's lazy tree and no allocation per cell.
#
# A group uses the engine only if its geometry reproduces Gridap's points on a few sample cells
# (`engine_aux`); otherwise (other points, other cells) the array is evaluated by Gridap as before.

"""
    CoordArray

An auxiliary array of a weak form that is a Julia function `f` of the physical coordinates at the
quadrature points. It behaves as Gridap's array (`getindex`), and the assembly materialises it with
the engine (`materialize_coord!`): `f(x)` at the points recomputed from the cell nodes.

The type of `f` is part of the type (and of the structural signature of a capture, see
`signature`): a new closure type means a new structure (`update` recompiles), while the same
function, or the same closure type with other captured values, is rebound and the new values are
used at the next assembly (as for the operations of the form).
"""
struct CoordArray{T,F} <: AbstractVector{Vector{T}}
    lazy::AbstractArray      # Gridap's array (the same values, evaluated lazily)
    f::F
end
Base.size(a::CoordArray) = size(a.lazy)
Base.IndexStyle(::Type{<:CoordArray}) = IndexLinear()
Base.getindex(a::CoordArray, i::Int) = a.lazy[i]
Arrays.array_cache(a::CoordArray) = array_cache(a.lazy)
Arrays.getindex!(c, a::CoordArray, i::Int) = getindex!(c, a.lazy, i)
Arrays.testitem(a::CoordArray) = testitem(a.lazy)
same_aux(a::CoordArray, b::CoordArray) = a === b || (typeof(a) == typeof(b) && a.f === b.f && length(a) == length(b))

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

"""
    coord_points_match(a, geo, cols, idx) -> Bool

Whether the geometry `geo` reproduces Gridap's physical points of `a` (to round-off) on a few
sample columns among `cols` (the columns the group integrates: for a boundary group, the faces of
its class, whose reference points are those of `geo`); column `i` of `geo` holds the cell of entry
`idx[i]` of `a`.
"""
function coord_points_match(a::CoordArray, geo::GeoTable, cols, idx)
    n = length(cols)
    n == 0 && return true
    pts = a.lazy.args[1]
    try
        cache = array_cache(pts)
        for j in unique!([1, cld(n, 2), n])
            i = cols[j]
            p = getindex!(cache, pts, idx[i])
            length(p) == geo.np || return false
            h = maximum(norm(geo.coords[k, i] - geo.coords[1, i]) for k in 1:geo.ng)
            for q in 1:geo.np
                x = point_coordinate(geo, q, i)
                norm(x - p[q]) <= 1e-10 * h + 1e-13 * norm(p[q]) || return false
            end
        end
    catch err
        err isa InterruptException && rethrow()
        return false
    end
    true
end

"""
    engine_aux(a, geo, cols, idx)

The auxiliary array a group stores: a `CoordArray` whose points the group's geometry reproduces
(`coord_points_match`) is kept (materialised by the engine); otherwise its Gridap array (the same
values, evaluated by Gridap). Other arrays are returned unchanged.
"""
engine_aux(a, geo, cols, idx) = a
engine_aux(a::CoordArray, geo::GeoTable, cols, idx) = coord_points_match(a, geo, cols, idx) ? a : a.lazy

"Columns a group integrates (all its colours), in increasing order."
group_columns(colors) = sort!(reduce(vcat, colors; init=Int32[]))

"""
Dense values `M[q, i]` of a function of the coordinates at the points of the columns `cols` of
`geo`. No allocation (if `f` does not allocate).
"""
function materialize_coord!(M::Matrix, a::CoordArray, cols, np::Int, geo::GeoTable)
    f = a.f
    @inbounds for i in cols
        for q in 1:np
            M[q, i] = f(point_coordinate(geo, q, i))
        end
    end
    M
end

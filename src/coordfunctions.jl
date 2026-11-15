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
# The physical points are Gridap's own (read once per group, `coord_points!`), not recomputed: `f`
# then sees exactly the same `x` as in Gridap, which matters for a function that is discontinuous on
# a face (`x[1] >= L` for a load on a side, a material indicator with a jump on a mesh line), where a
# round-off difference in `x` would change the value. They are read only after checking, on a few
# sample columns, that the group's geometry gives the same points to round-off (`coord_points_match`):
# the points are then those of the group's cells and quadrature, which do not change while the
# operator lives, so they are stored with the geometry and shared by all the functions of x of the
# group and by later captures (`update`). Otherwise (other points, other cells) the array is
# evaluated by Gridap as before.

"""
    CoordArray

An auxiliary array of a weak form that is a Julia function `f` of the physical coordinates at the
quadrature points. It behaves as Gridap's array (`getindex`), and the assembly materialises it with
the engine (`materialize_coord!`): `f(x)` at Gridap's physical points, stored with the geometry of
the group.

The type of `f` is part of the type (and of the structural signature of a capture, see
`signature`): a new closure type means a new structure (`update` recompiles), while the same
function, or the same closure type with other captured values, is rebound and the new values are
used at the next assembly (as for the operations of the form; not with `cache_aux = true`, which
never refills the auxiliary buffers).
"""
mutable struct CoordArray{T,F} <: AbstractVector{Vector{T}}   # mutable: `===` is identity (see `same_aux`)
    lazy::AbstractArray      # Gridap's array (the same values, evaluated lazily)
    f::F
end
Base.size(a::CoordArray) = size(a.lazy)
Base.IndexStyle(::Type{<:CoordArray}) = IndexLinear()
Base.getindex(a::CoordArray, i::Int) = a.lazy[i]
Arrays.array_cache(a::CoordArray) = array_cache(a.lazy)
Arrays.getindex!(c, a::CoordArray, i::Int) = getindex!(c, a.lazy, i)
Arrays.testitem(a::CoordArray) = testitem(a.lazy)
# Two uses of a function of x are never merged into one slot: whether two closures of the same type
# (or two of Gridap's immutable lazy arrays) are `===` depends on their captured values, which are not
# part of the structural signature, so merging them would make the slot layout change between captures
# of the same structure (`update`). A `CoordArray` is mutable, so `===` is the identity of the leaf.
same_aux(a::CoordArray, b::CoordArray) = a === b

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

"""
    coord_points_match(a, geo, cols, idx) -> Bool

Whether the geometry `geo` reproduces Gridap's physical points of `a` (to round-off) on a few
sample columns among `cols` (the columns the group integrates: for a boundary group, the faces of
its class, whose reference points are those of `geo`); column `i` of `geo` holds the cell of entry
`idx[i]` of `a`. `HFK_DEBUG_COORD=1` prints the reason of a mismatch.
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

"""
    coord_points!(geo, a, cols, idx) -> (cols, X)

The columns a group integrates and Gridap's physical points at them (`X[q, i]`, column `i` of `geo`
holding the cell of entry `idx[i]` of `a`), read from the points of `a` the first time and then
stored with the geometry (`geo.points`): they only depend on the cells and the quadrature.
"""
function coord_points!(geo::GeoTable{D}, a::CoordArray, cols, idx) where {D}
    c = geo.points[]
    c === nothing || return c
    X = Matrix{VectorValue{D,Float64}}(undef, geo.np, size(geo.coords, 2))
    pts = a.lazy.args[1]
    cache = array_cache(pts)
    for i in cols
        p = getindex!(cache, pts, idx[i])
        for q in 1:geo.np
            X[q, i] = p[q]
        end
    end
    geo.points[] = (Vector{Int32}(cols), X)
end

"""
    engine_aux(a, geo, cols, idx)

The auxiliary array a group stores: a `CoordArray` whose points the group's geometry reproduces
(`coord_points_match`) is kept (materialised by the engine at Gridap's points, `coord_points!`);
otherwise its Gridap array (the same values, evaluated by Gridap). Other arrays are returned
unchanged. `cols` is a function returning the columns the group integrates (only called for a
function of x, and only the first time, see `coord_points!`).
"""
engine_aux(a, geo, cols, idx) = a
function engine_aux(a::CoordArray, geo::GeoTable, cols, idx)
    c = geo.points[]
    cs = c === nothing ? cols() : c[1]
    coord_points_match(a, geo, cs, idx) || return a.lazy
    coord_points!(geo, a, cs, idx)
    a
end

"Columns a group integrates (all its colours), in increasing order."
group_columns(colors) = sort!(reduce(vcat, colors; init=Int32[]))

"""
Dense values `M[q, i]` of a function of the coordinates at the stored physical points `X` of the
columns `cols` (only the columns the group integrates). No allocation (if `f` does not allocate).
"""
function materialize_coord!(M::Matrix, a::CoordArray, cols, np::Int, X::Matrix)
    f = a.f
    @inbounds for i in cols
        for q in 1:np
            M[q, i] = f(X[q, i])
        end
    end
    M
end
"The same, with the points stored in the geometry (`coord_points!`)."
materialize_coord!(M::Matrix, a::CoordArray, cols, np::Int, geo::GeoTable) = materialize_coord!(M, a, cols, np, geo.points[][2])

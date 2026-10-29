# FE functions in the weak form (not the unknown): `uh⁻`, interpolated fields, their gradients.
#
# Gridap evaluates an FE function at the points as the linear combination of its cell dof values
# with the basis at the points (`LinearCombinationMap`), and a physical gradient as
# `inv(Jt) ⋅ (reference gradient)`. Both are recognised in the lazy tree and evaluated by the
# engine from the current nodal values: in the state updates (`states.jl`, read at the point by
# the compiled function) and in the assembly (`FEArray`, materialised into the dense auxiliary
# buffers of the group by `materialize_fe!`, see `assembly.jl`). Reading the vectors at every
# evaluation keeps the data current when the FE function is updated in place (a previous state
# per load step, a field computed by another physics in a staggered scheme, interpolated
# boundary data).

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

"""
    fe_data(a, deriv, np) -> FEData or nothing

`a` recognised as an FE function evaluated at `np` points per cell: the linear combination of its
cell dof values (`PosNegReindex` of the free and Dirichlet values over a `Table` of ids, directly
or through a `Reindex` from the cells of the points to the cells of the space) with a uniform
`np × nd` basis matrix (values with `deriv = 0`, reference gradients with `deriv = 1`).
"""
function fe_data(a::LazyArray, deriv::Int, np::Int)
    first(a.maps) isa Fields.LinearCombinationMap || return nothing
    length(a.args) == 2 || return nothing
    vals, bx = a.args
    vals isa LazyArray && first(vals.maps) isa Broadcasting && first(vals.maps).f isa Arrays.PosNegReindex || return nothing
    N = uniform_matrix(bx)          # basis at the points: the same matrix for every cell (one cell type)
    N === nothing && return nothing
    size(N, 1) == np || return nothing
    ids = vals.args[1]
    if ids isa Arrays.Table
        table, cellmap = ids, nothing
    elseif ids isa LazyArray && first(ids.maps) isa Arrays.Reindex && first(ids.maps).values isa Arrays.Table
        table, cellmap = first(ids.maps).values, ids.args[1]
    else
        return nothing
    end
    T = eltype(a)
    pnr = first(vals.maps).f
    free, diri = pnr.values_pos, pnr.values_neg
    eltype(free) == eltype(diri) || return nothing
    nd = size(N, 2)
    FEData{eltype(T),deriv,typeof(free),typeof(diri),eltype(N),typeof(cellmap),typeof(table.ptrs),typeof(table.data)}(
        vals, free, diri, table.ptrs, table.data, cellmap, N, nd)
end

"""
    fe_array(a) -> FEArray or nothing

An FE function at the points (value, or physical gradient `inv(Jt) ⋅` reference gradient), read
from Gridap's lazy tree, for the assembly: see `FEArray`.
"""
function fe_array(a)
    a isa LazyArray || return nothing
    k = first(a.maps)
    if k isa Fields.LinearCombinationMap && k.column isa Colon && length(a.args) == 2
        N = uniform_matrix(a.args[2])
        N === nothing && return nothing
        fe = fe_data(a, 0, size(N, 1))
        fe === nothing || return FEArray(a, fe)
    elseif k isa BroadcastingFieldOpMap && k.op === dot && length(a.args) == 2
        b = a.args[2]
        if b isa LazyArray && first(b.maps) isa Fields.LinearCombinationMap && first(b.maps).column isa Colon
            D = _tensor_dim(eltype(eltype(a)))        # a is cell-wise: Vector of point values
            D === nothing && return nothing
            is_inverse_jacobian(a.args[1], D) || return nothing
            N = uniform_matrix(b.args[2])
            N === nothing && return nothing
            fe = fe_data(b, 1, size(N, 1))
            fe === nothing || return FEArray(a, fe)
        end
    end
    nothing
end
_tensor_dim(::Type{<:TensorValue{D}}) where {D} = D
_tensor_dim(::Type{<:VectorValue{D}}) where {D} = D
_tensor_dim(::Type) = nothing

"""
    FEArray

An auxiliary array of a weak form that is an FE function at the quadrature points (`uh⁻`, an
interpolated field, their gradients). It behaves as Gridap's array (`getindex`), and the
assembly materialises it with the engine (`materialize_fe!`): the current free and Dirichlet
values are read at every assembly, without evaluating Gridap's lazy tree.
"""
struct FEArray{T,FE} <: AbstractVector{Vector{T}}
    lazy::AbstractArray      # Gridap's array (the same values, evaluated lazily)
    fe::FE                   # FEData
end
FEArray(a, fe::FEData) = FEArray{eltype(fe),typeof(fe)}(a, fe)
Base.size(f::FEArray) = size(f.lazy)
Base.IndexStyle(::Type{<:FEArray}) = IndexLinear()
Base.getindex(f::FEArray, i::Int) = f.lazy[i]
Arrays.array_cache(f::FEArray) = array_cache(f.lazy)
Arrays.getindex!(c, f::FEArray, i::Int) = getindex!(c, f.lazy, i)
Arrays.testitem(f::FEArray) = testitem(f.lazy)
fe_deriv(::FEArray{T,FE}) where {T,FE} = fe_deriv(FE)
fe_deriv(::Type{<:FEData{T,DV}}) where {T,DV} = DV
same_aux(a::FEArray, b::FEArray) = a === b || (a.fe.vals === b.fe.vals && fe_deriv(a) == fe_deriv(b) &&
                                              _same_data(a.fe.cellmap, b.fe.cellmap) && _same_data(a.fe.N, b.fe.N))
_same_data(a, b) = a === b || (typeof(a) == typeof(b) && size(a) == size(b) && a == b)
_same_data(::Nothing, ::Nothing) = true

"""
Dense values `M[q, cell]` of an FE function at the points of `cells`, by the engine: local dof
values gathered from the current free/Dirichlet vectors and combined with the basis table (and
`inv(Jt)` from `geo` for a gradient). No allocation beyond the local vector `xl`.
"""
function materialize_fe!(M::Matrix, a::FEArray{T,FE}, cells, np::Int, geo) where {T,FE}
    fe = a.fe
    xl = zeros(eltype(fe.free), fe.nd)
    if fe_deriv(FE) == 0
        @inbounds for cell in cells
            gather_fe!(xl, fe, cell)
            for q in 1:np
                M[q, cell] = fe_point_value(fe, xl, q, nothing)
            end
        end
    else
        @inbounds for cell in cells
            gather_fe!(xl, fe, cell)
            for q in 1:np
                iJt = point_geometry(geo, q, cell)[2]
                M[q, cell] = fe_point_value(fe, xl, q, iJt)
            end
        end
    end
    M
end

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


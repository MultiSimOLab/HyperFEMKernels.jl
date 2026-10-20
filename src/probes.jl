# Probe leaves.
#
# The weak form is captured by calling the user's `res(u, v)` / `jac(u, du, v)` with probe
# `CellField`s instead of FE functions and bases. Gridap builds its usual operation tree on top
# of them and, inside `∫(...)dΩ`, evaluates it lazily at the quadrature points: the probes return
# a `ProbeArray`, so the result is a tree of `LazyArray`s whose leaves are probes (unknown, test
# and trial functions) or ordinary Gridap arrays (auxiliary data). Nothing is evaluated.

const ROLES = (:unknown, :test, :trial)

"""
    ProbeField(role, field, deriv, T, D, trian)

Placeholder for the unknown (`role = :unknown`), a test function (`:test`) or a trial function
(`:trial`) of field `field`, differentiated `deriv` times (0 or 1). `T` is the value type of the
field (e.g. `Float64`, `VectorValue{3,Float64}`) and `D` the spatial dimension.
"""
struct ProbeField <: CellField
    role::Symbol
    field::Int
    deriv::Int
    T::DataType
    D::Int
    trian::Triangulation
end

CellData.get_triangulation(p::ProbeField) = p.trian
CellData.DomainStyle(::Type{ProbeField}) = ReferenceDomain()
CellData.change_domain(p::ProbeField, ::Triangulation, ::ReferenceDomain) = p
CellData.change_domain(p::ProbeField, ::ReferenceDomain, ::Triangulation, ::ReferenceDomain) = p

function Fields.gradient(p::ProbeField)
    p.deriv == 0 || error("HyperFEMKernels: second derivatives of the unknown/test/trial functions are not supported yet")
    ProbeField(p.role, p.field, 1, p.T, p.D, p.trian)
end
Fields.symmetric_gradient(p::ProbeField) = Operation(symmetric_part)(gradient(p))
Fields.divergence(p::ProbeField) = Operation(tr)(gradient(p))

CellData.get_data(p::ProbeField) = error("""
    HyperFEMKernels: a differential operator was applied to an expression that contains the
    unknown, test or trial function (e.g. ∇(u*w)). Write it in terms of ∇(u) instead.""")

"Value type of a probe at a quadrature point (Gridap's convention for gradients)."
function probe_valtype(p::ProbeField)
    p.deriv == 0 && return p.T
    p.T <: Real && return VectorValue{p.D,Float64}
    p.T <: VectorValue || error("HyperFEMKernels: gradients of $(p.T) fields are not supported yet")
    N = length(p.T)
    return TensorValue{p.D,N,Float64,p.D * N}
end

"""
    ProbeArray

Cell-wise array returned when a probe is evaluated at the quadrature points. Its items are
mock values with the shapes Gridap expects (points for the unknown, points × 1 basis for test
functions, points × 1 × 1 for trial functions), used only for return-type inference.
"""
struct ProbeArray{T,N} <: AbstractVector{Array{T,N}}
    probe::ProbeField
    points::CellPoint
    ncells::Int
    np::Int
end

Base.size(a::ProbeArray) = (a.ncells,)
Base.IndexStyle(::Type{<:ProbeArray}) = IndexLinear()
function Base.getindex(a::ProbeArray{T,N}, i::Int) where {T,N}
    N == 1 ? zeros(T, a.np) : N == 2 ? zeros(T, a.np, 1) : zeros(T, a.np, 1, 1)
end

function Arrays.evaluate!(cache, p::ProbeField, x::CellPoint)
    np = length(testitem(get_data(x)))
    N = p.role === :unknown ? 1 : p.role === :test ? 2 : 3
    ProbeArray{probe_valtype(p),N}(p, x, length(get_data(x)), np)
end

contains_probe(a::ProbeArray) = true
contains_probe(a::LazyArray) = any(contains_probe, a.args)
contains_probe(a) = false

"""
    probes(space, role)

Probe (or tuple of probes, for a multi-field space) for every field of `space`.
"""
function probes(space::FESpace, role::Symbol)
    fs = field_spaces(space)
    ps = map(enumerate(fs)) do (i, s)
        trian = get_triangulation(s)
        ProbeField(role, i, 0, field_valtype(s), num_cell_dims(trian), trian)
    end
    length(ps) == 1 ? ps[1] : Tuple(ps)
end

field_spaces(s::SingleFieldFESpace) = [s]
field_spaces(s::MultiFieldFESpace) = s.spaces

"Value type of the shape functions of a single-field space."
function field_valtype(s::SingleFieldFESpace)
    fields = testitem(get_data(get_fe_basis(s)))
    x = testitem(get_data(get_cell_points(get_triangulation(s))))
    eltype(evaluate(fields, x))
end

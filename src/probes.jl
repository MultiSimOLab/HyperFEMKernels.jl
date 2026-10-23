# Probe leaves.
#
# The weak form is captured by calling the user's `res(u, v)` / `jac(u, du, v)` with probe
# `CellField`s instead of FE functions and bases. Gridap builds its usual operation tree on top
# of them and, inside `∫(...)dΩ`, evaluates it lazily at the quadrature points: the probes return
# a `ProbeArray`, so the result is a tree of `LazyArray`s whose leaves are probes (unknown, test
# and trial functions) or ordinary Gridap arrays (auxiliary data). Nothing is evaluated.

const ROLES = (:unknown, :test, :trial)

"""
    ProbeField(role, field, deriv, T, D, trian, side = :cell)

Placeholder for the unknown (`role = :unknown`), a test function (`:test`) or a trial function
(`:trial`) of field `field`, differentiated `deriv` times (0 or 1). `T` is the value type of the
field (e.g. `Float64`, `VectorValue{3,Float64}`) and `D` the spatial dimension. On a skeleton
`side` is the cell the trace is taken from (`:plus` or `:minus`, from `u.plus` / `u.minus`);
`:cell` elsewhere.
"""
struct ProbeField <: CellField
    role::Symbol
    field::Int
    deriv::Int
    T::DataType
    D::Int
    trian::Triangulation
    side::Symbol
end
ProbeField(role, field, deriv, T, D, trian) = ProbeField(role, field, deriv, T, D, trian, :cell)

CellData.get_triangulation(p::ProbeField) = p.trian
CellData.DomainStyle(::Type{ProbeField}) = ReferenceDomain()
# moved to another triangulation (e.g. a boundary) the probe reports that triangulation, as a
# real basis would, so that Gridap picks the right common domain in e.g. ∇(u) ⋅ n
CellData.change_domain(p::ProbeField, t::Triangulation, ::ReferenceDomain) = ProbeField(p.role, p.field, p.deriv, p.T, p.D, t, p.side)
CellData.change_domain(p::ProbeField, ::ReferenceDomain, t::Triangulation, ::ReferenceDomain) = ProbeField(p.role, p.field, p.deriv, p.T, p.D, t, p.side)

# Skeleton: `u.plus` / `u.minus` wrap the probe in Gridap's `CellFieldAt{:plus/:minus}`. When the
# integrand is moved to the skeleton, Gridap restricts each side with the glue of that side and
# would read the cell-wise data of the probe; the probe is instead relabelled with the side (the
# kernels evaluate it in the plus or minus cell of the facet, see `skeleton.jl`).
function CellData.change_domain_ref_ref(a::CellData.CellFieldAt{S,ProbeField}, ttrian::Triangulation,
                                        sglue::Geometry.FaceToFaceGlue, tglue::Geometry.FaceToFaceGlue) where {S}
    S in (:plus, :minus) || error("HyperFEMKernels: unknown skeleton side $S")
    p = a.parent
    ProbeField(p.role, p.field, p.deriv, p.T, p.D, ttrian, S)
end

function Fields.gradient(p::ProbeField)
    p.deriv == 0 || error("HyperFEMKernels: second derivatives of the unknown/test/trial functions are not supported yet")
    ProbeField(p.role, p.field, 1, p.T, p.D, p.trian, p.side)
end
Fields.symmetric_gradient(p::ProbeField) = Operation(symmetric_part)(gradient(p))
Fields.divergence(p::ProbeField) = Operation(tr)(gradient(p))

# A differential operator applied to an expression with probes (∇(u*w)), or a probe used on a
# skeleton without choosing a side, makes Gridap ask for the cell-wise data of the probe. There is
# none: the whole weak form then goes to the hybrid fallback (see `capture`). Returning mock cell
# data instead was tried and rejected: Gridap then builds a tree the reader misparses silently.
CellData.get_data(p::ProbeField) = error("""
    HyperFEMKernels: a differential operator was applied to an expression that contains the
    unknown, test or trial function (e.g. ∇(u*w)), or the function is used on a skeleton without
    a side (use plus/minus, jump or mean). Write it in terms of ∇(u), or let the hybrid fallback
    assemble it with Gridap.""")

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

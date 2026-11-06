# Probe leaves.
#
# The weak form is captured by calling the user's `res(u, v)` / `jac(u, du, v)` with probe
# `CellField`s instead of FE functions and bases. Gridap builds its usual operation tree on top
# of them and, inside `∫(...)dΩ`, evaluates it lazily at the quadrature points: the probes return
# a `ProbeArray`, so the result is a tree of `LazyArray`s whose leaves are probes (unknown, test
# and trial functions) or ordinary Gridap arrays (auxiliary data). Nothing is evaluated.

const ROLES = (:unknown, :test, :trial)

"""
    ProbeField(role, field, deriv, T, D, trian, side = :cell, data = nothing)

Placeholder for the unknown (`role = :unknown`), a test function (`:test`) or a trial function
(`:trial`) of field `field`, differentiated `deriv` times (0, 1 or 2). `T` is the value type of the
field (e.g. `Float64`, `VectorValue{3,Float64}`) and `D` the spatial dimension. On a skeleton
`side` is the cell the trace is taken from (`:plus` or `:minus`, from `u.plus` / `u.minus`);
`:cell` elsewhere. `data` is the FE space of the field (`nothing` if unknown): the real FE function or
basis it stands for is built from it only when Gridap asks for the cell-wise data of the probe (see
`get_data`).
"""
struct ProbeField <: CellField
    role::Symbol
    field::Int
    deriv::Int
    T::DataType
    D::Int
    trian::Triangulation
    side::Symbol
    data::Any
end
ProbeField(role, field, deriv, T, D, trian) = ProbeField(role, field, deriv, T, D, trian, :cell, nothing)
ProbeField(role, field, deriv, T, D, trian, side) = ProbeField(role, field, deriv, T, D, trian, side, nothing)
"Copy of a probe with other `deriv`, `trian` or `side`."
reprobe(p::ProbeField; deriv=p.deriv, trian=p.trian, side=p.side) = ProbeField(p.role, p.field, deriv, p.T, p.D, trian, side, p.data)

CellData.get_triangulation(p::ProbeField) = p.trian
CellData.DomainStyle(::Type{ProbeField}) = ReferenceDomain()
# moved to another triangulation (e.g. a boundary) the probe reports that triangulation, as a
# real basis would, so that Gridap picks the right common domain in e.g. ∇(u) ⋅ n
CellData.change_domain(p::ProbeField, t::Triangulation, ::ReferenceDomain) = reprobe(p; trian=t)
CellData.change_domain(p::ProbeField, ::ReferenceDomain, t::Triangulation, ::ReferenceDomain) = reprobe(p; trian=t)

# Skeleton: `u.plus` / `u.minus` wrap the probe in Gridap's `CellFieldAt{:plus/:minus}`. When the
# integrand is moved to the skeleton, Gridap restricts each side with the glue of that side and
# would read the cell-wise data of the probe; the probe is instead relabelled with the side (the
# kernels evaluate it in the plus or minus cell of the facet, see `skeleton.jl`).
function CellData.change_domain_ref_ref(a::CellData.CellFieldAt{S,ProbeField}, ttrian::Triangulation,
                                        sglue::Geometry.FaceToFaceGlue, tglue::Geometry.FaceToFaceGlue) where {S}
    S in (:plus, :minus) || error("HyperFEMKernels: unknown skeleton side $S")
    reprobe(a.parent; trian=ttrian, side=S)
end

# Derivatives: ∇(u) and ∇∇(u) (also ∇(∇(u)), Δ(u) = tr(∇∇(u))) are probes of a higher `deriv`.
# Gridap's `∇∇(::CellField)` reads the cell data, so it is overridden for the probes.
function Fields.gradient(p::ProbeField)
    p.deriv < 2 && return reprobe(p; deriv=p.deriv + 1)
    # third derivatives: not supported by the kernels; Gridap builds the expression on the real
    # data (the integrals with it go to the hybrid fallback, see `get_data`)
    gradient(GenericCellField(get_data(p), p.trian, ReferenceDomain()))
end
Fields.∇∇(p::ProbeField) = gradient(gradient(p))
Fields.symmetric_gradient(p::ProbeField) = Operation(symmetric_part)(gradient(p))
Fields.divergence(p::ProbeField) = Operation(tr)(gradient(p))

"""
    ProbeCellData

Cell-wise data of a probe (the data of the real FE function, or the first function of the real
basis, wrapped so that the reader still sees the probe). Gridap asks for it when a differential
operator is applied to an expression with probes (∇(u*w)), when a probe is used on a skeleton
without choosing a side, or for a third derivative: it then builds its ordinary lazy arrays over
this data, which the reader does not translate (`to_ops`), so only the integrals on that
triangulation go to the hybrid fallback (see `capture`) instead of the whole weak form. A basis is
reduced to one function so that its arrays have the shapes of the probes' mocks (`ProbeArray`:
points × 1, points × 1 × 1) and combine with them. Returning plain mock data was tried and
rejected: Gridap then builds a tree the reader misparses silently; the wrapper makes every such
array detectable (`contains_probe`).
"""
struct ProbeCellData{A,T} <: AbstractVector{T}
    data::A
    probe::ProbeField
end
ProbeCellData(data::AbstractArray, p::ProbeField) = ProbeCellData{typeof(data),eltype(data)}(data, p)
Base.size(a::ProbeCellData) = size(a.data)
Base.IndexStyle(::Type{<:ProbeCellData}) = IndexLinear()
Base.getindex(a::ProbeCellData, i::Int) = a.data[i]
Arrays.array_cache(a::ProbeCellData) = array_cache(a.data)
Arrays.getindex!(c, a::ProbeCellData, i::Int) = getindex!(c, a.data, i)
Arrays.testitem(a::ProbeCellData) = testitem(a.data)

function CellData.get_data(p::ProbeField)
    p.data === nothing && error("""
        HyperFEMKernels: a differential operator was applied to an expression that contains the
        unknown, test or trial function (e.g. ∇(u*w)), or the function is used on a skeleton without
        a side (use plus/minus, jump or mean). Write it in terms of ∇(u), or let the hybrid fallback
        assemble it with Gridap.""")
    real = probe_object(p)
    data = p.role === :unknown ? get_data(real) : lazy_map(single_function, get_data(real))
    f = GenericCellField(ProbeCellData(data, p), get_triangulation(real), ReferenceDomain())
    for _ in 1:p.deriv
        f = gradient(f)
    end
    get_data(change_domain(f, p.trian, ReferenceDomain()))
end

"The real object a probe stands for: the zero FE function of its space (unknown) or its test/trial basis."
probe_object(p::ProbeField) = p.role === :unknown ? zero(p.data) : p.role === :test ? get_fe_basis(p.data) : get_trial_fe_basis(p.data)

"The first function of a cell basis, keeping its layout (test: a vector of fields; trial: a 1 × n matrix)."
single_function(b::AbstractVector{<:Field}) = b[1:1]
single_function(b::AbstractMatrix{<:Field}) = b[:, 1:1]

"Number of unique entries of a symmetric `D × D` Hessian (the pairs `(a, b)` with `a ≤ b`)."
nhess(D::Int) = D * (D + 1) ÷ 2

"Value type of a probe at a quadrature point (Gridap's convention for gradients and Hessians)."
function probe_valtype(p::ProbeField)
    p.deriv == 0 && return p.T
    D = p.D
    if p.T <: Real
        p.deriv == 1 && return VectorValue{D,Float64}
        return TensorValue{D,D,Float64,D * D}
    end
    p.T <: VectorValue || error("HyperFEMKernels: gradients of $(p.T) fields are not supported yet")
    N = length(p.T)
    p.deriv == 1 && return TensorValue{D,N,Float64,D * N}
    return ThirdOrderTensorValue{D,D,N,Float64,D * D * N}     # H[a, b, k] = ∂²u_k/∂x_a∂x_b
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

# Whether an array holds a probe: a `ProbeArray`, a `ProbeCellData`, or a lazy array built over one:
# in its arguments, in its array of maps (`lazy_map(evaluate, fields, points)` takes the field array
# as the maps), or in a reindexing map that holds the data (`Reindex`, `PosNegReindex`).
contains_probe(a::ProbeArray) = true
contains_probe(a::ProbeCellData) = true
contains_probe(a::LazyArray) = any(contains_probe, a.args) || contains_probe(a.maps)
contains_probe(a::Arrays.AppendedArray) = contains_probe(a.a) || contains_probe(a.b)
contains_probe(a::Fields.MemoArray) = contains_probe(a.parent)
contains_probe(a::Arrays.Fill) = contains_probe_map(a.value)
contains_probe(a) = false
contains_probe_map(k::Arrays.Reindex) = contains_probe(k.values)
contains_probe_map(k::Arrays.PosNegReindex) = contains_probe(k.values_pos) || contains_probe(k.values_neg)
contains_probe_map(k::Broadcasting) = contains_probe_map(k.f)
contains_probe_map(k) = false

"""
    probes(space, role)

Probe (or tuple of probes, for a multi-field space) for every field of `space`.
"""
function probes(space::FESpace, role::Symbol)
    fs = field_spaces(space)
    ps = map(enumerate(fs)) do (i, s)
        trian = get_triangulation(s)
        ProbeField(role, i, 0, field_valtype(s), num_point_dims(trian), trian, :cell, s)   # gradients live in the physical space (also on a manifold triangulation)
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

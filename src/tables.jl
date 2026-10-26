# Reference tables, generated from Gridap objects.
#
# One cell type per triangulation. Two kinds of field tables:
# - `FieldTable`: Lagrangian (nodal) spaces, whose vector shape functions are scalar functions times
#   unit vectors, dof (n, c) = N_n e_c. Tables store the scalar functions per node, so contractions
#   work per node and not per vector dof.
# - `PiolaTable`: H(div)-conforming (Raviart–Thomas) spaces, whose shape functions are vector-valued
#   and mapped from the reference cell by the contravariant Piola map, with a sign per cell dof.
#   Tables store the reference vector shape functions and their reference gradients per dof; the
#   Piola map is applied per point (to the field and to the residual coefficients, like the chain
#   rule of the gradients of a `FieldTable`) and the signs at gather/scatter (`cell_signs`).

"Field tables share the layout parameters: dimension, value type, dofs-or-nodes, components, contiguity."
abstract type AbstractFieldTable{D,T,NN,NC,CONTIG} end

"Return the single value of a cell-wise array that is uniform across cells, or fail."
function uniform_value(a::AbstractArray, what::String)
    if a isa Fill
        return a.value
    elseif a isa CompressedArray
        length(a.values) == 1 && return a.values[1]
    end
    v = a[1]
    error("HyperFEMKernels: $what differs between cells (mixed cell types are not supported yet)")
end

"""
Geometry: gradients of the geometric shape functions at the quadrature points, weights,
coordinates. For a boundary group `Jf` is the (constant) Jacobian of the map from the reference
face to the reference cell, so that the surface measure is `w * meas(Jf ⋅ Jt)`; `nothing` for
volume groups.
"""
struct GeoTable{D,JF}
    np::Int
    ng::Int
    dM::Matrix{VectorValue{D,Float64}}      # np × ng, reference gradients (at the cell points)
    w::Vector{Float64}                       # np
    coords::Matrix{VectorValue{D,Float64}}   # ng × ncells (of the integration triangulation)
    Jf::JF
end

"""
    GeoTable(trian, xq, w; cells = 1:num_cells(trian), Jf = nothing)

Geometry of the cells `cells` of `trian` (the FE-space triangulation) at the reference points
`xq`: column `i` of `coords` holds the nodes of cell `cells[i]` (a face of a boundary group maps
to its cell).
"""
function GeoTable(trian::Triangulation, xq::AbstractVector, w::AbstractVector; cells=1:num_cells(trian), Jf=nothing)
    reffes = Geometry.get_reffes(trian)
    length(reffes) == 1 || error("HyperFEMKernels: mixed cell types are not supported yet")
    M = get_shapefuns(reffes[1])
    dM = evaluate(Broadcasting(∇)(M), xq)
    D = num_point_dims(trian)
    cc = get_cell_coordinates(trian)
    ng = size(dM, 2)
    coords = Matrix{VectorValue{D,Float64}}(undef, ng, length(cells))
    cache = array_cache(cc)
    for (i, cell) in enumerate(cells)
        xc = getindex!(cache, cc, cell)
        for n in 1:ng
            coords[n, i] = xc[n]
        end
    end
    GeoTable{D,typeof(Jf)}(length(xq), ng, Matrix{VectorValue{D,Float64}}(dM), collect(Float64, w), coords, Jf)
end

"Scalar nodal shape functions of one field at the quadrature points and the local dof layout.
`T` is the value type of the field (`Float64` or `VectorValue{nc,Float64}`)."
struct FieldTable{D,T,NN,NC,CONTIG} <: AbstractFieldTable{D,T,NN,NC,CONTIG}
    nn::Int                                   # nodes
    nc::Int                                   # components
    N::Matrix{Float64}                        # np × nn
    dN::Matrix{VectorValue{D,Float64}}        # np × nn, reference gradients
    ldof::Matrix{Int}                         # nn × nc → local dof
    Nt::Matrix{Float64}                       # nn × np (contiguous over nodes)
    dNt::Array{Float64,3}                     # nn × D × np, reference gradients (contiguous over nodes)
end
# CONTIG: the dofs of each component are consecutive (ldof[n, c] = ldof[1, c] + n - 1), which
# makes the node loops contiguous in memory (vectorisable on CPU, coalesced on GPU).

function FieldTable(space::SingleFieldFESpace, xq::AbstractVector, D::Int)
    is_piola(space) && error("HyperFEMKernels: Piola-mapped (H(div)) spaces are not supported on skeletons yet")
    fields = uniform_value(get_data(get_fe_basis(space)), "the FE basis")
    # extra points make node identification robust when quadrature points are few
    probe_pts = [Point(ntuple(i -> 0.1 + 0.07 * (k + i) % 0.6, D)) for k in 1:7]
    pts = vcat(collect(xq), probe_pts)
    vals = evaluate(fields, pts)
    grads = evaluate(Broadcasting(∇)(fields), pts)
    nd = size(vals, 2)
    T = eltype(vals)
    nc = ncomps(T)
    np = length(xq)
    comp = zeros(Int, nd)
    for j in 1:nd
        if nc == 1
            comp[j] = 1
        else
            comp[j] = findfirst(c -> any(q -> abs(vals[q, j][c]) > 0, axes(vals, 1)), 1:nc)
        end
    end
    scalar(q, j) = nc == 1 ? vals[q, j] : vals[q, j][comp[j]]
    sgrad(q, j) = nc == 1 ? grads[q, j] : VectorValue(ntuple(J -> grads[q, j][J, comp[j]], D))
    keyof(j) = [scalar(q, j) for q in axes(vals, 1)]
    reps = Vector{Float64}[]          # one representative key per node (approximate matching:
    node = zeros(Int, nd)             # Gridap evaluates each component with its own round-off)
    for j in 1:nd
        k = keyof(j)
        n = findfirst(r -> isapprox(r, k; atol=1e-7, rtol=0.0), reps)   # Q3 values carry ~1e-9 round-off
        n === nothing && (push!(reps, k); n = length(reps))
        node[j] = n
    end
    nn = length(reps)
    nn * nc == nd || error("HyperFEMKernels: the FE space is not a nodal (Lagrangian) space")
    ldof = zeros(Int, nn, nc)
    first_dof = zeros(Int, nn)
    for j in 1:nd
        ldof[node[j], comp[j]] = j
        first_dof[node[j]] == 0 && (first_dof[node[j]] = j)
    end
    N = [scalar(q, first_dof[n]) for q in 1:np, n in 1:nn]
    dN = VectorValue{D,Float64}[sgrad(q, first_dof[n]) for q in 1:np, n in 1:nn]
    Nt = permutedims(N)
    dNt = [dN[q, n][K] for n in 1:nn, K in 1:D, q in 1:np]
    contig = all(c -> ldof[:, c] == ldof[1, c] .+ (0:nn-1), 1:nc)
    FieldTable{D,T,nn,nc,contig}(nn, nc, N, dN, ldof, Nt, dNt)
end

ndofs(t::AbstractFieldTable) = t.nn * t.nc

"Columns of the physical weight matrix W of a table (`point_weights!`)."
wcols(::FieldTable{D}) where {D} = D + 1

# ---------------------------------------------------------------------------------------------
# H(div) (Raviart–Thomas) spaces
#
# Gridap builds the cell basis of a div-conforming space as the contravariant Piola map of the
# reference shape functions with a sign flip on the dofs of the facets whose cell is the slave:
#   φ_j(x) = σ_j / |det Jt| · φ̂_j(ξ) ⋅ Jt,   ∇φ_j = Jt⁻¹ ⋅ (σ_j / |det Jt| · ∇̂φ̂_j ⋅ Jt)
# (Gridap 0.19 takes the Jacobian as constant in the gradient; the formula above is reproduced
# exactly, also on non-affine cells, so that kernel and Gridap agree to round-off). The reference
# functions are the same in every cell; the signs are a dense ±1 per (dof, cell).

"The cell basis of a div-conforming space (the lazy Piola-map array), or `nothing`."
function piola_basis(space::SingleFieldFESpace)
    a = get_data(get_fe_basis(space))
    (a isa LazyArray && a.maps isa Fill && a.maps.value isa Broadcasting{<:Operation{ContraVariantPiolaMap}}) ? a : nothing
end
is_piola(space::SingleFieldFESpace) = piola_basis(space) !== nothing

"""
Reference vector shape functions of an H(div) field at the quadrature points, per dof: `Vt[n, K, q]`
= φ̂_n(ξ_q)[K] and `dVt[n, K, L, q]` = ∂φ̂_n[L]/∂ξ_K (Gridap convention). One pseudo-component (`nc = 1`,
`ldof[n, 1] = n`), always node-contiguous.
"""
struct PiolaTable{D,NN} <: AbstractFieldTable{D,VectorValue{D,Float64},NN,1,true}
    nn::Int                                   # dofs
    nc::Int                                   # 1
    Vt::Array{Float64,3}                      # nn × D × np
    dVt::Array{Float64,4}                     # nn × D × D × np
    ldof::Matrix{Int}                         # nn × 1
end

function PiolaTable(space::SingleFieldFESpace, xq::AbstractVector, D::Int)
    a = piola_basis(space)
    a === nothing && error("HyperFEMKernels: not a Piola-mapped space")
    ref = uniform_value(a.args[1], "the reference H(div) basis")
    vals = evaluate(ref, xq)                         # np × nd, VectorValue{D}
    grads = evaluate(Broadcasting(∇)(ref), xq)       # np × nd, TensorValue{D,D}
    np, nd = size(vals)
    eltype(vals) == VectorValue{D,Float64} || error("HyperFEMKernels: unexpected value type $(eltype(vals)) of the H(div) basis")
    Vt = [vals[q, n][K] for n in 1:nd, K in 1:D, q in 1:np]
    dVt = [grads[q, n][K, L] for n in 1:nd, K in 1:D, L in 1:D, q in 1:np]
    PiolaTable{D,nd}(nd, 1, Vt, dVt, reshape(collect(1:nd), nd, 1))
end

wcols(::PiolaTable{D}) where {D} = D + D * D   # values (D) then gradient [J,c] column-major (D²)

"Field table of a space: `PiolaTable` for H(div) spaces, `FieldTable` (nodal) otherwise."
field_table(space::SingleFieldFESpace, xq::AbstractVector, D::Int) =
    is_piola(space) ? PiolaTable(space, xq, D) : FieldTable(space, xq, D)

"""
    cell_signs(space, nd) -> Matrix{Int8} (nd × ncells)

Sign of every cell dof of an H(div) space (−1 on the facet dofs of the slave cell, read from the
sign flips of Gridap's cell basis); a 0 × 0 matrix for other spaces (no signs).
"""
function cell_signs(space::SingleFieldFESpace, nd::Int)
    a = piola_basis(space)
    a === nothing && return Matrix{Int8}(undef, 0, 0)
    flips = a.args[4].args[1]                        # Bool per cell dof (lazy, per cell)
    nc = length(flips)
    s = Matrix{Int8}(undef, nd, nc)
    cache = array_cache(flips)
    for cell in 1:nc
        f = getindex!(cache, flips, cell)
        length(f) == nd || error("HyperFEMKernels: cells with different numbers of dofs are not supported yet")
        for j in 1:nd
            s[j, cell] = f[j] ? Int8(-1) : Int8(1)
        end
    end
    s
end

"Cell dof ids of a single-field space as a dense matrix (ndofs × ncells); negative = Dirichlet."
function cell_dof_matrix(space::SingleFieldFESpace, nd::Int)
    ids = get_cell_dof_ids(space)
    nc = length(ids)
    m = Matrix{Int32}(undef, nd, nc)
    cache = array_cache(ids)
    for cell in 1:nc
        c = getindex!(cache, ids, cell)
        length(c) == nd || error("HyperFEMKernels: cells with different numbers of dofs are not supported yet")
        for j in 1:nd
            m[j, cell] = c[j]
        end
    end
    m
end

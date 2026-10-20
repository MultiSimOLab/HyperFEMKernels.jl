# Reference tables, generated from Gridap objects.
#
# Phase 1 assumes one cell type per triangulation and Lagrangian (nodal) spaces, whose vector
# shape functions are scalar functions times unit vectors: dof (n, c) = N_n e_c. Tables store the
# scalar functions per node, so contractions work per node and not per vector dof.

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

"Geometry: gradients of the geometric shape functions at the quadrature points, weights, coordinates."
struct GeoTable{D}
    np::Int
    ng::Int
    dM::Matrix{VectorValue{D,Float64}}      # np × ng, reference gradients
    w::Vector{Float64}                       # np
    coords::Matrix{VectorValue{D,Float64}}   # ng × ncells
end

function GeoTable(trian::Triangulation, xq::AbstractVector, w::AbstractVector)
    reffes = Geometry.get_reffes(trian)
    length(reffes) == 1 || error("HyperFEMKernels: mixed cell types are not supported yet")
    M = get_shapefuns(reffes[1])
    dM = evaluate(Broadcasting(∇)(M), xq)
    D = num_point_dims(trian)
    cc = get_cell_coordinates(trian)
    ng = size(dM, 2)
    coords = Matrix{VectorValue{D,Float64}}(undef, ng, num_cells(trian))
    cache = array_cache(cc)
    for cell in 1:num_cells(trian)
        xc = getindex!(cache, cc, cell)
        for n in 1:ng
            coords[n, cell] = xc[n]
        end
    end
    GeoTable{D}(length(xq), ng, Matrix{VectorValue{D,Float64}}(dM), collect(Float64, w), coords)
end

"Scalar nodal shape functions of one field at the quadrature points and the local dof layout.
`T` is the value type of the field (`Float64` or `VectorValue{nc,Float64}`)."
struct FieldTable{D,T,NN,NC,CONTIG}
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
        n = findfirst(r -> isapprox(r, k; atol=1e-10, rtol=0.0), reps)
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

ndofs(t::FieldTable) = t.nn * t.nc

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

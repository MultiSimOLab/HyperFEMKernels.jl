# Transfers between nested Lagrangian spaces on the same mesh (p-multigrid, design/solvers-s4a-plan.md §3.2).
#
# The coarse space Q_{p'} (p' < p) is a subspace of the fine one on every cell, so the prolongation of a
# coarse nodal field is exact: on each cell the fine nodal values are E · (coarse nodal values), with
# E[f, c] = (coarse shape function c at fine node f), the same matrix in every cell. The restriction is
# the transpose, applied cell by cell with the multiplicity of the shared fine dofs so that every fine
# dof is counted once. Both act on the free dofs only (homogeneous corrections): Dirichlet dofs are
# read as zero and never written. The injection of a state (free and Dirichlet values of a fine FE
# function interpolated at the coarse nodes) is what the coarse levels linearise at.
#
# E is dense and static per field (nf × nc per component); at Q2 → Q1 in 3D it is 27 × 8, far below the
# cost of one matrix-free product, so the tensor-product form of the plan is not needed (measured in S4-A
# step 1; it stays an option if a transfer ever exceeds ~0.1 products).

"One field of a `Prolongation`: the reference embedding and the dof layouts of both levels."
struct TransferField{NF,NC}
    E::Matrix{Float64}              # NF × NC: fine node f, coarse node c (scalar shape functions). Plain matrices: a
                                    # static 64 × 27 (Q3 → Q1) takes minutes to compile, the loops have static bounds anyway
    ldof_f::Matrix{Int}             # NF × ncomp → local dof of the fine field
    ldof_c::Matrix{Int}             # NC × ncomp → local dof of the coarse field
    ids_f::Matrix{Int32}            # nd_f × ncells, free ids of the fine field (negative: Dirichlet)
    ids_c::Matrix{Int32}            # nd_c × ncells, coarse
    off_f::Int                      # offset of the field in the fine free vector
    off_c::Int                      # same, coarse
    Einj::Matrix{Float64}           # NC × NF: fine shape function f at coarse node c (nodal interpolation)
    ncomp::Int
    cells_of::Vector{Int32}         # multiplicity of every fine free dof of the field (cells containing it)
    ndir_f::Int                     # Dirichlet dofs of the fine field (offsets of the concatenated Dirichlet values)
    ndir_c::Int
end

"""
    Prolongation(Uc, Uf; nthreads = Threads.nthreads())

Exact prolongation from the coarse trial space `Uc` to the fine one `Uf` (same mesh, same fields, the same closed
Dirichlet boundary — tags with their edges and vertices, as HyperFEM's `⁺` Cartesian tags —, nested Lagrangian
reference elements: Q1 ⊂ Q2 ⊂ Q3, P1 ⊂ P2): `mul!(yf, P, xc)`,
`mul!(yc, transpose(P), yf)`, `prolongate!`, `restrict!` on the free vectors (Dirichlet dofs read as zero,
never written), `inject!` for the state. Cells are processed by colour (no two cells of a colour share a
fine dof), so both transfers are threaded; the cell kernels do not allocate (the tasks of the colour sweeps do).
"""
struct Prolongation{F<:Tuple}
    fields::F
    colors::Vector{Vector{Int32}}
    nf::Int
    nc::Int
    nthreads::Int
    bufs::Vector{Tuple{Vector{Float64},Vector{Float64}}}   # per task: coarse and fine local values
end

Base.size(P::Prolongation) = (P.nf, P.nc)
Base.size(P::Prolongation, i::Int) = size(P)[i]
Base.eltype(::Prolongation) = Float64

"Reference nodes of a Lagrangian single-field space (the nodes of its cell dof basis)."
function reference_nodes(space::SingleFieldFESpace)
    db = testitem(get_data(get_fe_dof_basis(space)))
    db isa LagrangianDofBasis || error("HyperFEMKernels: p-multigrid transfers need Lagrangian (nodal) spaces, got $(nameof(typeof(db)))")
    collect(db.nodes)
end

function TransferField(sc::SingleFieldFESpace, sf::SingleFieldFESpace, off_c::Int, off_f::Int, D::Int)
    (is_piola(sc) || is_piola(sf)) && error("HyperFEMKernels: p-multigrid transfers do not support H(div) fields")
    nodes = reference_nodes(sf)
    tf = FieldTable(sf, nodes, D)          # fine basis at its own nodes: a permuted identity → node ↔ point
    tc = FieldTable(sc, nodes, D)          # coarse basis at the fine nodes: the embedding
    tf.nc == tc.nc || error("HyperFEMKernels: the two levels have different numbers of components")
    NF, NC = tf.nn, tc.nn
    NF >= NC || error("HyperFEMKernels: the coarse space has more nodes per cell than the fine one")
    point_of = zeros(Int, NF)              # fine table node → reference node (point) index
    for n in 1:NF
        q = findfirst(q -> isapprox(tf.N[q, n], 1.0; atol=1e-6), 1:length(nodes))
        q === nothing && error("HyperFEMKernels: the fine space is not nodal at its dof-basis nodes")
        all(qq -> qq == q || abs(tf.N[qq, n]) < 1e-6, 1:length(nodes)) || error("HyperFEMKernels: the fine space is not nodal at its dof-basis nodes")
        point_of[n] = q
    end
    E = [tc.N[point_of[f], c] for f in 1:NF, c in 1:NC]
    # nodal interpolation of a fine field at the coarse nodes (the injection of the state): the fine basis at the
    # coarse nodes. With nested nodes (Q1 ⊂ Q2, Q1 ⊂ Q3) it selects the coinciding fine node; Q2 ⊂ Q3 is nested as a
    # space but not in its equispaced nodes (ξ = 1/2 is not a Q3 node), and the interpolation handles both.
    cnodes = reference_nodes(sc)
    tfc = FieldTable(sf, cnodes, D)
    tcc = FieldTable(sc, cnodes, D)
    cpoint_of = zeros(Int, NC)
    for n in 1:NC
        q = findfirst(q -> isapprox(tcc.N[q, n], 1.0; atol=1e-6), 1:length(cnodes))
        q === nothing && error("HyperFEMKernels: the coarse space is not nodal at its dof-basis nodes")
        all(qq -> qq == q || abs(tcc.N[qq, n]) < 1e-6, 1:length(cnodes)) || error("HyperFEMKernels: the coarse space is not nodal at its dof-basis nodes")
        cpoint_of[n] = q
    end
    tfc.ldof == tf.ldof || error("HyperFEMKernels: inconsistent node numbering of the fine space")
    Einj = [tfc.N[cpoint_of[c], f] for c in 1:NC, f in 1:NF]
    # the bases come from inverting Vandermonde matrices (Gridap): entries that are 0 or 1 carry round-off (~5e-10 at
    # Q3); snap them, so the Dirichlet check below and the sparsity of the cell kernels see exact zeros
    snap(e) = abs(e) < 1e-7 ? 0.0 : abs(e - 1) < 1e-7 ? 1.0 : e
    E .= snap.(E); Einj .= snap.(Einj)
    ndf, ndc = NF * tf.nc, NC * tc.nc
    ids_f = cell_dof_matrix(sf, ndf); ids_c = cell_dof_matrix(sc, ndc)
    size(ids_f, 2) == size(ids_c, 2) || error("HyperFEMKernels: the two levels live on different meshes")
    cf, cc = get_cell_coordinates(get_triangulation(sf)), get_cell_coordinates(get_triangulation(sc))
    for cell in unique((1, size(ids_f, 2) ÷ 2 + 1, size(ids_f, 2)))      # the same cells: a sample of their vertices
        cf[cell] ≈ cc[cell] || error("HyperFEMKernels: the two levels live on different meshes (cell $cell differs)")
    end
    # same Dirichlet boundary on both levels: a fine Dirichlet node is a zero of every coarse function that is free there
    # (so the prolongation may skip it), and a coarse node is Dirichlet iff the fine function is Dirichlet there (so the
    # injection reads the right value). Checked through the embedding: a fine Dirichlet dof must receive no weight from a
    # free coarse dof, and a coarse Dirichlet dof must coincide with (or interpolate only from) fine Dirichlet dofs.
    @inbounds for cell in axes(ids_f, 2), comp in 1:tf.nc
        for f in 1:NF
            ids_f[tf.ldof[f, comp], cell] > 0 && continue
            for c in 1:NC
                E[f, c] != 0 && ids_c[tc.ldof[c, comp], cell] > 0 &&
                    error("HyperFEMKernels: the two levels have different Dirichlet boundaries (a fine Dirichlet dof is interpolated from a free coarse dof; use the same closed Dirichlet set — e.g. tags with their edges and vertices — on every level)")
            end
        end
        for c in 1:NC
            ids_c[tc.ldof[c, comp], cell] > 0 && continue
            for f in 1:NF
                Einj[c, f] != 0 && ids_f[tf.ldof[f, comp], cell] > 0 &&
                    error("HyperFEMKernels: the two levels have different Dirichlet boundaries (a coarse Dirichlet dof is interpolated from a free fine dof)")
            end
        end
    end
    mult = zeros(Int32, num_free_dofs(sf))
    @inbounds for cell in axes(ids_f, 2), j in 1:ndf
        id = ids_f[j, cell]
        id > 0 && (mult[id] += 1)
    end
    TransferField{NF,NC}(E, tf.ldof, tc.ldof, ids_f, ids_c, off_f, off_c, Einj, tf.nc, mult,
                                 num_dirichlet_dofs(sf), num_dirichlet_dofs(sc))
end

function Prolongation(Uc::FESpace, Uf::FESpace; nthreads::Int=Threads.nthreads())
    Ucs, Ufs = field_spaces(Uc), field_spaces(Uf)
    length(Ucs) == length(Ufs) || error("HyperFEMKernels: the two levels have different numbers of fields")
    for U in (Uc, Uf)
        U isa MultiFieldFESpace && !(MultiFieldStyle(U) isa ConsecutiveMultiFieldStyle) &&
            error("HyperFEMKernels: only ConsecutiveMultiFieldStyle multi-field spaces are supported")
    end
    D = num_point_dims(get_triangulation(Ufs[1]))
    fields = TransferField[]
    off_c = off_f = 0
    for (sc, sf) in zip(Ucs, Ufs)
        push!(fields, TransferField(sc, sf, off_c, off_f, D))
        off_c += num_free_dofs(sc); off_f += num_free_dofs(sf)
    end
    # colouring by the fine dofs: cells sharing a coarse dof share it as a fine dof too, so the same
    # colours serve the scatter-add of the restriction
    layouts = [FieldLayout(tf.ids_f, tf.ids_f, tf.off_f, tf.off_f, 0, size(tf.ids_f, 1)) for tf in fields]
    colors = color_cells(layouts, size(fields[1].ids_f, 2), off_f)
    ndc = maximum(size(tf.ids_c, 1) for tf in fields); ndf = maximum(size(tf.ids_f, 1) for tf in fields)
    bufs = [(zeros(ndc), zeros(ndf)) for _ in 1:nthreads]
    Prolongation(Tuple(fields), colors, off_f, off_c, nthreads, bufs)
end

# --- cell kernels (static sizes from the field type) -------------------------------------------

"Fine local values of the coarse local values (both in local-dof order): `yl = E · xl` per component."
@inline function prolongate_cell!(yl, tf::TransferField{NF,NC}, xl) where {NF,NC}
    E = tf.E
    @inbounds for comp in 1:tf.ncomp
        for f in 1:NF
            s = 0.0
            for c in 1:NC
                s += E[f, c] * xl[tf.ldof_c[c, comp]]
            end
            yl[tf.ldof_f[f, comp]] = s
        end
    end
    yl
end

"Coarse local values of the fine local values: `xl = Eᵀ · yl` per component."
@inline function restrict_cell!(xl, tf::TransferField{NF,NC}, yl) where {NF,NC}
    E = tf.E
    @inbounds for comp in 1:tf.ncomp
        for c in 1:NC
            s = 0.0
            for f in 1:NF
                s += E[f, c] * yl[tf.ldof_f[f, comp]]
            end
            xl[tf.ldof_c[c, comp]] = s
        end
    end
    xl
end

function prolongate_cells!(yf, tf::TransferField, xc, cells, (xl, yl))
    ids_c, ids_f = tf.ids_c, tf.ids_f
    ndc, ndf = size(ids_c, 1), size(ids_f, 1)
    for cell in cells
        @inbounds for j in 1:ndc
            id = ids_c[j, cell]
            xl[j] = id > 0 ? xc[tf.off_c+id] : 0.0
        end
        prolongate_cell!(yl, tf, xl)
        @inbounds for j in 1:ndf                      # insertion: every cell writes the same value at a shared node
            id = ids_f[j, cell]
            id > 0 && (yf[tf.off_f+id] = yl[j])
        end
    end
    nothing
end

function restrict_cells!(yc, tf::TransferField, yf, cells, (xl, yl))
    ids_c, ids_f = tf.ids_c, tf.ids_f
    ndc, ndf = size(ids_c, 1), size(ids_f, 1)
    mult = tf.cells_of
    for cell in cells
        @inbounds for j in 1:ndf                      # each fine dof once over the cells that contain it
            id = ids_f[j, cell]
            yl[j] = id > 0 ? yf[tf.off_f+id] / mult[id] : 0.0
        end
        restrict_cell!(xl, tf, yl)
        @inbounds for j in 1:ndc                      # scatter-add (cells of a colour share no coarse dof)
            id = ids_c[j, cell]
            id > 0 && (yc[tf.off_c+id] += xl[j])
        end
    end
    nothing
end

function run_colored(f, P::Prolongation)
    for color in P.colors
        nchunks = clamp(length(color) ÷ 16, 1, P.nthreads)
        @sync for (i, ch) in enumerate(Iterators.partition(color, cld(length(color), nchunks)))
            Threads.@spawn f(ch, P.bufs[i])
        end
    end
    nothing
end

"""
    prolongate!(yf, P, xc)

Fine free vector of the coarse free vector `xc` (Dirichlet dofs read as zero, never written): `yf = P xc`.
"""
function prolongate!(yf::AbstractVector, P::Prolongation, xc::AbstractVector)
    length(yf) == P.nf && length(xc) == P.nc || throw(DimensionMismatch("HyperFEMKernels: prolongation of a vector of length $(length(xc)) into $(length(yf)) (expected $(P.nc) → $(P.nf))"))
    fill!(yf, 0.0)
    for tf in P.fields
        run_colored((cells, b) -> prolongate_cells!(yf, tf, xc, cells, b), P)
    end
    yf
end

"""
    restrict!(yc, P, yf)

Coarse free vector of the fine free vector `yf`: `yc = Pᵀ yf` (the transpose of `prolongate!`).
"""
function restrict!(yc::AbstractVector, P::Prolongation, yf::AbstractVector)
    length(yf) == P.nf && length(yc) == P.nc || throw(DimensionMismatch("HyperFEMKernels: restriction of a vector of length $(length(yf)) into $(length(yc)) (expected $(P.nf) → $(P.nc))"))
    fill!(yc, 0.0)
    for tf in P.fields
        run_colored((cells, b) -> restrict_cells!(yc, tf, yf, cells, b), P)
    end
    yc
end

"The transpose of a `Prolongation` (the restriction), as an operator: `mul!(yc, transpose(P), yf)`."
struct Restriction{P<:Prolongation}
    parent::P
end
Base.size(R::Restriction) = reverse(size(R.parent))
Base.size(R::Restriction, i::Int) = size(R)[i]
Base.eltype(::Restriction) = Float64
LinearAlgebra.transpose(P::Prolongation) = Restriction(P)
LinearAlgebra.adjoint(P::Prolongation) = Restriction(P)
LinearAlgebra.transpose(R::Restriction) = R.parent
LinearAlgebra.adjoint(R::Restriction) = R.parent
LinearAlgebra.mul!(y::AbstractVector, P::Prolongation, x::AbstractVector) = prolongate!(y, P, x)
LinearAlgebra.mul!(y::AbstractVector, R::Restriction, x::AbstractVector) = restrict!(y, R.parent, x)
Base.:*(P::Prolongation, x::AbstractVector) = prolongate!(zeros(P.nf), P, x)
Base.:*(R::Restriction, x::AbstractVector) = restrict!(zeros(R.parent.nc), R.parent, x)

"""
    inject!(xc, dc, P, xf, df)

State of the coarse level from the fine one: the free values `xc` and the Dirichlet values `dc` of the coarse
fields are the nodal interpolation of the fine FE function (free `xf`, Dirichlet `df`) at the coarse nodes — the
fine value at the coinciding node when the nodes are nested (Q1 → Q2, Q1 → Q3), no system solved. `dc`/`df` are
the concatenated `dirichlet_values` of the fields of each level (`HyperFEMKernels.dirichlet_values(U)`).
"""
function inject!(xc::AbstractVector, dc::AbstractVector, P::Prolongation, xf::AbstractVector, df::AbstractVector)
    length(xf) == P.nf && length(xc) == P.nc || throw(DimensionMismatch("HyperFEMKernels: inject! with free vectors of length $(length(xf)) → $(length(xc)) (expected $(P.nf) → $(P.nc))"))
    doff_c = doff_f = 0
    for tf in P.fields
        inject_field!(xc, dc, tf, xf, df, doff_c, doff_f)
        doff_c += tf.ndir_c; doff_f += tf.ndir_f
    end
    xc, dc
end

function inject_field!(xc, dc, tf::TransferField{NF,NC}, xf, df, doff_c, doff_f) where {NF,NC}
    ids_c, ids_f, Ei = tf.ids_c, tf.ids_f, tf.Einj
    @inbounds for cell in axes(ids_c, 2), comp in 1:tf.ncomp, c in 1:NC
        v = 0.0
        for f in 1:NF
            w = Ei[c, f]
            w == 0.0 && continue
            idf = ids_f[tf.ldof_f[f, comp], cell]
            v += w * (idf > 0 ? xf[tf.off_f+idf] : df[doff_f-idf])
        end
        idc = ids_c[tf.ldof_c[c, comp], cell]
        if idc > 0
            xc[tf.off_c+idc] = v
        else
            dc[doff_c-idc] = v
        end
    end
    nothing
end

"Concatenated Dirichlet values of the fields of a trial space (the layout `inject!` uses)."
dirichlet_values(U::FESpace) = reduce(vcat, (get_dirichlet_dof_values(s) for s in field_spaces(U)); init=Float64[])

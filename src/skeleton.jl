# Skeleton integrals: interior facets with a cell on each side (plus, minus).
#
# A facet of a skeleton (a `SkeletonTriangulation`, or the skeleton of a sub-triangulation, which Gridap
# builds as a composite triangulation; both are recognised by their `SkeletonPair` glue) couples the dofs
# of its two cells. The quadrature points of
# the facet are mapped to the reference cell of each side through the glue of that side (the minus
# map includes the permutation of the face), so the shape functions of the plus and minus cells are
# tabulated at their own points. Facets with the same pair of mapped points form a class and share
# the tables (`skeleton_classes`). The element vector of a facet holds the dofs of the plus cell
# and then those of the minus cell (2·ndl); its matrix is scattered with positions precomputed per
# facet (`SkeletonData.nzpos`, the cross blocks are added to the sparsity pattern). The pointwise
# integrand sees the unknown of both sides (`FV = (plus fields..., minus fields...)`) and test/trial
# slots per side (`Leaf.side`); the measure is that of the plus face (`w · meas(Jf⁺ ⋅ Jt⁺)`).

"FE-space cell of the plus and minus side of each facet (2 × nfacets)."
function skeleton_cells(trian::Triangulation, space_trian::Triangulation)
    D = num_cell_dims(space_trian)
    glue = Geometry.get_glue(trian, Val(D))
    glue isa Geometry.SkeletonPair || error("HyperFEMKernels: unexpected glue of the skeleton triangulation")
    s2m = Geometry.get_glue(space_trian, Val(D)).tface_to_mface
    m2s = Dict{Int,Int32}()
    for (sc, m) in enumerate(s2m)
        m2s[m] = sc
    end
    n = num_cells(trian)
    cells = Matrix{Int32}(undef, 2, n)
    for (side, g) in enumerate((glue.plus, glue.minus)), f in 1:n
        sc = get(m2s, g.tface_to_mface[f], Int32(0))
        sc > 0 || error("HyperFEMKernels: the skeleton has facets whose cells are outside the FE spaces")
        cells[side, f] = sc
    end
    cells
end

"Facet colouring: two facets of the same colour share no free (test) dof of their four cells."
function color_facets(fields::Vector{FieldLayout}, cells::Matrix{Int32}, nrows::Int)
    masks = zeros(UInt128, nrows)
    nf = size(cells, 2)
    color = zeros(Int, nf)
    for f in 1:nf
        used = UInt128(0)
        for side in 1:2, fl in fields, j in 1:fl.nd
            r = fl.test_ids[j, cells[side, f]]
            r > 0 && (used |= masks[fl.test_off+r])
        end
        c = trailing_ones(used)
        c < 128 || error("HyperFEMKernels: more than 128 colours needed for the skeleton facets")
        color[f] = c + 1
        bit = UInt128(1) << c
        for side in 1:2, fl in fields, j in 1:fl.nd
            r = fl.test_ids[j, cells[side, f]]
            r > 0 && (masks[fl.test_off+r] |= bit)
        end
    end
    ncol = maximum(color; init=0)
    groups = [Int32[] for _ in 1:ncol]
    for f in 1:nf
        push!(groups[color[f]], f)
    end
    groups
end

"Data of a skeleton for the FE spaces on `space_trian`: cells per facet and facet colouring (positions come with the pattern)."
function skeleton_data(trian::Triangulation, space_trian::Triangulation, fields::Vector{FieldLayout}, nrows::Int)
    cells = skeleton_cells(trian, space_trian)
    SkeletonData(trian, cells, color_facets(fields, cells, nrows), Matrix{Int32}(undef, 0, 0))
end

"""
Facets with the same quadrature points in the reference cells of both sides:
`(facets, xqp, xqm, Jf, key)` per class, with the Jacobian of the plus face map (measure).
"""
function skeleton_classes(trian::Triangulation, terms::Vector{Term}, D::Int)
    pts = get_data(_first_points(terms))
    glue = Geometry.get_glue(trian, Val(D))
    cp = lazy_map(evaluate, glue.plus.tface_to_mface_map, pts)
    cm = lazy_map(evaluate, glue.minus.tface_to_mface_map, pts)
    jf = lazy_map(∇, glue.plus.tface_to_mface_map)
    classes = Dict{Vector{Float64},Vector{Int32}}()
    data = Dict{Vector{Float64},Any}()
    pcache = array_cache(cp); mcache = array_cache(cm); jcache = array_cache(jf)
    flat(x) = reduce(vcat, [collect(p.data) for p in x])
    for facet in 1:num_cells(trian)
        xp = getindex!(pcache, cp, facet)
        xm = getindex!(mcache, cm, facet)
        key = round.(vcat(flat(xp), flat(xm)); digits=10)
        if !haskey(classes, key)
            classes[key] = Int32[]
            Jf = evaluate(getindex!(jcache, jf, facet), pts[facet])[1]
            data[key] = (xqp=collect(xp), xqm=collect(xm), Jf=Jf)
        end
        push!(classes[key], Int32(facet))
    end
    [(facets=classes[k], xqp=data[k].xqp, xqm=data[k].xqm, Jf=data[k].Jf, key=k) for k in sort!(collect(keys(classes)))]
end

"A compiled class of facets of a skeleton: same points on both sides."
struct SkeletonGroup{NSV,NSU,PF,TT,AUX,SV,SU,D,NF2,LF,JF,UU} <: AbstractGroup
    pf::PF
    lin::LF                        # LinearForm or nothing
    locs::NTuple{NF2,Int}          # offset of each (side, field) in the facet element vector: plus fields, then minus (+ ndl)
    tables::TT                     # FieldTable per (side, field): plus fields at xqp, then minus fields at xqm
    use::UU                        # FieldUse per (side, field)
    geo::GeoTable{D,JF}            # plus cells at xqp, with Jf (facet measure)
    geom::GeoTable{D,Nothing}      # minus cells at xqm (only Jt⁻¹ is used)
    aux::Tuple                     # Gridap arrays on the skeleton, materialised per facet of the class (AUX: dense types)
    smv::SV
    smu::SU
    trian::Triangulation
    key::Vector{Float64}           # class key (points of both sides), for reuse in `update`
    auxcache::Base.RefValue{Any}
    facets::Vector{Int32}          # facets of the class (indices in the skeleton)
    skel::SkeletonData
    colors::Vector{Vector{Int32}}  # facets of the class per colour (local indices, 1:length(facets))
end

function SkeletonGroup(terms::Vector{Term}, trian::Triangulation, Uspaces, locs, D::Int, ndl::Int, skel::SkeletonData, cls,
                       previous=Any[]; linear_maps::Bool=true)
    NF = length(Uspaces)
    pf = PointForm(terms; nf=NF)
    ls = vcat(pf.test.leaves, pf.trial.leaves, leaves(terms, :unknown))
    isempty(ls) && error("HyperFEMKernels: integral without unknown, test or trial functions")
    any(l -> l.side === :cell, ls) && error("HyperFEMKernels: a function is used on the skeleton without a side (use plus/minus, jump or mean)")
    w = uniform_value(terms[1].weights, "the quadrature weights")
    k = findfirst(g -> g isa SkeletonGroup && g.trian === trian && g.key == cls.key && g.geo.w == w && g.facets == cls.facets, previous)
    if k === nothing
        strian = get_triangulation(Uspaces[1])
        geo = GeoTable(strian, cls.xqp, w; cells=skel.cells[1, cls.facets], Jf=cls.Jf)
        geom = GeoTable(strian, cls.xqm, w; cells=skel.cells[2, cls.facets])
        tables = (Tuple(FieldTable(s, cls.xqp, D) for s in Uspaces)..., Tuple(FieldTable(s, cls.xqm, D) for s in Uspaces)...)
    else
        geo, geom, tables = previous[k].geo, previous[k].geom, previous[k].tables
    end
    NF2 = 2NF
    locs2 = (locs..., Tuple(l + ndl for l in locs)...)
    sides = (ntuple(_ -> :plus, NF)..., ntuple(_ -> :minus, NF)...)
    fieldof = (ntuple(identity, NF)..., ntuple(identity, NF)...)
    smv = Tuple(SlotSpec(slot_map(pf.test, fieldof[i], tables[i].nc, D; side=sides[i]), D) for i in 1:NF2)
    smu = Tuple(SlotSpec(slot_map(pf.trial, fieldof[i], tables[i].nc, D; side=sides[i]), D) for i in 1:NF2)
    use = Tuple(field_use(terms, fieldof[i]; side=sides[i]) for i in 1:NF2)
    aux = Tuple(a.array for a in pf.aux)
    any(a -> a isa StateColumns, aux) && error("HyperFEMKernels: internal variables (KernelState) on a skeleton are not supported yet")
    AUX = Tuple{(Matrix{eltype(eltype(a))} for a in aux)...}
    lin = linear_maps ? LinearForm(terms, pf, tables, aux, pf.trial.nslots == 0 ? :res : :jac) : nothing
    # colours of the skeleton restricted to the class, in local indices
    pos = zeros(Int32, num_cells(trian))
    for (i, f) in enumerate(cls.facets)
        pos[f] = i
    end
    colors = [Int32[pos[f] for f in col if pos[f] > 0] for col in skel.colors]
    filter!(!isempty, colors)
    SkeletonGroup{pf.test.nslots,pf.trial.nslots,typeof(pf),typeof(tables),AUX,typeof(smv),typeof(smu),D,NF2,typeof(lin),typeof(geo.Jf),typeof(use)}(
        pf, lin, NTuple{NF2,Int}(locs2), tables, use, geo, geom, aux, smv, smu, trian, cls.key, Ref{Any}(nothing), cls.facets, skel, colors)
end

# ---------------------------------------------------------------------------------------------
# Auxiliary data of a class: dense values (np × facets of the class)

function materialize_aux(g::SkeletonGroup{NSV,NSU,PF,TT,AUX}, nthreads::Int) where {NSV,NSU,PF,TT,AUX}
    np, n = g.geo.np, length(g.facets)
    vals = ntuple(k -> Matrix{eltype(eltype(g.aux[k]))}(undef, np, n), length(g.aux))
    materialize_aux!(AUX(vals)::AUX, g, nthreads)
end
function materialize_aux!(vals::AUX, g::SkeletonGroup{NSV,NSU,PF,TT,AUX}, nthreads::Int) where {NSV,NSU,PF,TT,AUX}
    np, n = g.geo.np, length(g.facets)
    chunks = collect(Iterators.partition(1:n, cld(n, max(1, nthreads))))
    for k in 1:length(g.aux)
        a = g.aux[k]; M = vals[k]
        @sync for ch in chunks
            Threads.@spawn _materialize_facets!(M, a, g.facets, ch, np)
        end
    end
    vals
end
function _materialize_facets!(M::Matrix, a, facets, range, np::Int)
    cache = array_cache(a)
    for i in range
        v = getindex!(cache, a, facets[i])
        @inbounds for q in 1:np
            M[q, i] = v[q]
        end
    end
    M
end

function aux_data(op::KernelOperator, g::SkeletonGroup{NSV,NSU,PF,TT,AUX}) where {NSV,NSU,PF,TT,AUX}
    if g.auxcache[] === nothing
        g.auxcache[] = materialize_aux(g, op.nthreads)
    elseif !op.cache_aux
        materialize_aux!(g.auxcache[]::AUX, g, op.nthreads)
    end
    g.auxcache[]::AUX
end

# ---------------------------------------------------------------------------------------------
# Element computations of a facet (local index `lf` in the class)

"Same value for the plus half of the tuple and for the minus half."
@inline side_tuple(a, b, ::Val{N}) where {N} = ntuple(i -> 2i <= N ? a : b, Val(N))

function element_residual!(re, g::SkeletonGroup{NSV,NSU,PF,TT,AUX,SV,SU,D,NF2,LF}, xl, auxd,
                           lf::Integer) where {NSV,NSU,PF,TT,AUX,SV,SU,D,NF2,LF}
    fill!(re, 0.0)
    tables, locs, pf = g.tables, g.locs, g.pf
    zsv = zero(SVector{NSV,Float64})
    for q in 1:g.geo.np
        dS, iJp = point_geometry(g.geo, q, lf)
        _, iJm = point_geometry(g.geom, q, lf)
        iJs = side_tuple(iJp, iJm, Val(NF2))
        FV = field_values_sides(tables, locs, g.use, xl, q, iJs)
        Aq = map(a -> @inbounds(a[q, lf]), auxd)
        R = if LF === Nothing
            ForwardDiff.gradient(s -> pf.core(pf.ops, pf.coefs, FV, Aq, s, EMPTY_SLOTS), zsv)
        else
            apply_maps(g.lin, g.lin.state(g.lin.ops, FV, Aq), Val(NSV))
        end
        map((t, l, sm, iJ) -> contract_residual!(re, t, l, sm, R, q, iJ, dS), tables, locs, g.smv, iJs)
    end
    re
end

function element_jacobian!(KeT, g::SkeletonGroup{NSV,NSU,PF,TT,AUX,SV,SU,D,NF2,LF}, xl, auxd, lf::Integer,
                           Wb) where {NSV,NSU,PF,TT,AUX,SV,SU,D,NF2,LF}
    fill!(KeT, 0.0)
    tables, locs, pf = g.tables, g.locs, g.pf
    zsv = zero(SVector{NSV,Float64})
    zsu = zero(SVector{NSU,Float64})
    for q in 1:g.geo.np
        dS, iJp = point_geometry(g.geo, q, lf)
        _, iJm = point_geometry(g.geom, q, lf)
        iJs = side_tuple(iJp, iJm, Val(NF2))
        map((t, W, iJ) -> point_weights!(W, t, q, iJ), tables, Wb, iJs)
        FV = field_values_sides(tables, locs, g.use, xl, q, iJs)
        Aq = map(a -> @inbounds(a[q, lf]), auxd)
        T = if LF === Nothing
            ForwardDiff.jacobian(r -> ForwardDiff.gradient(s -> pf.core(pf.ops, pf.coefs, FV, Aq, s, r), zsv), zsu)   # T[test slot, trial slot]
        else
            SMatrix{NSV,NSU,Float64,NSV * NSU}(apply_maps(g.lin, g.lin.state(g.lin.ops, FV, Aq), Val(NSV * NSU)))
        end
        map((tf, lf_, smf, bf) -> map((tg, lg, smg, bg) ->
                contract_jacobian!(KeT, tf, lf_, smf, bf, tg, lg, smg, bg, T, dS),
                tables, locs, g.smu, Wb),
            tables, locs, g.smv, Wb)
    end
    KeT
end

"Threaded driver of a class: gather both cells, facet element residual/Jacobian, scatter."
function cells!(::Val{K}, b, A, op::KernelOperator, g::SkeletonGroup{NSV,NSU,PF,TT,AUX,SV,SU,D},
                x, dvals, auxd, lfacets) where {K,NSV,NSU,PF,TT,AUX,SV,SU,D}
    ndl = op.ndl
    n2 = 2ndl
    xl = zeros(n2)
    Wb = map(t -> zeros(t.nn, D + 1), g.tables)
    skel, facets = g.skel, g.facets
    if K === :res
        re = zeros(n2)
        for lf in lfacets
            f = facets[lf]
            cp, cm = skel.cells[1, f], skel.cells[2, f]
            gather_cell!(xl, op.fields, x, dvals, cp)
            gather_cell!(xl, op.fields, x, dvals, cm, ndl)
            element_residual!(re, g, xl, auxd, lf)
            scatter_residual!(b, op.fields, re, cp)
            scatter_residual!(b, op.fields, re, cm, ndl)
        end
    else
        KeT = zeros(n2, n2)
        nzv = nonzeros(A)
        size(skel.nzpos, 2) == size(skel.cells, 2) || error("HyperFEMKernels: the facet positions of the skeleton were not computed")
        for lf in lfacets
            f = facets[lf]
            cp, cm = skel.cells[1, f], skel.cells[2, f]
            gather_cell!(xl, op.fields, x, dvals, cp)
            gather_cell!(xl, op.fields, x, dvals, cm, ndl)
            element_jacobian!(KeT, g, xl, auxd, lf, Wb)
            scatter_jacobian!(nzv, skel.nzpos, KeT, n2, f)
        end
    end
    nothing
end

# ---------------------------------------------------------------------------------------------
# Sparsity: the cross blocks (plus rows × minus columns and the reverse) of every facet, and the
# positions of the whole facet matrix (2ndl × 2ndl) in the nonzeros.

"Add the keys `(col << 32) | row` of the cross blocks of the facets of `sd`."
function skeleton_keys!(keys::Vector{UInt64}, sd::SkeletonData, grow::Matrix{Int32}, gcol::Matrix{Int32}, ndl::Int)
    for f in 1:size(sd.cells, 2)
        cp, cm = sd.cells[1, f], sd.cells[2, f]
        for (crow, ccol) in ((cp, cm), (cm, cp)), jj in 1:ndl
            c = gcol[jj, ccol]
            c > 0 || continue
            for ii in 1:ndl
                r = grow[ii, crow]
                r > 0 && push!(keys, (UInt64(c) << 32) | UInt64(r))
            end
        end
    end
    keys
end

"Positions in the nonzeros of the (2ndl)² entries of each facet matrix (0 = not stored)."
function skeleton_positions!(sd::SkeletonData, grow::Matrix{Int32}, gcol::Matrix{Int32}, ndl::Int, colptr, rowval)
    nf = size(sd.cells, 2)
    n2 = 2ndl
    nzpos = zeros(Int32, n2 * n2, nf)
    Threads.@threads for f in 1:nf
        cells = (sd.cells[1, f], sd.cells[2, f])
        for jj in 1:n2
            cj, j = jj <= ndl ? (cells[1], jj) : (cells[2], jj - ndl)
            c = gcol[j, cj]
            c > 0 || continue
            lo, hi = colptr[c], colptr[c+1] - 1
            for ii in 1:n2
                ci, i = ii <= ndl ? (cells[1], ii) : (cells[2], ii - ndl)
                r = grow[i, ci]
                r > 0 || continue
                p = searchsortedfirst(view(rowval, lo:hi), r) + lo - 1
                nzpos[ii+(jj-1)*n2, f] = p
            end
        end
    end
    sd.nzpos = nzpos
    sd
end

"Element allocations of a skeleton group (first facet of the class), see `allocations`."
function _element_allocations(kind::Symbol, op::KernelOperator, g::SkeletonGroup{NSV,NSU,PF,TT,AUX,SV,SU,D}, x, dvals) where {NSV,NSU,PF,TT,AUX,SV,SU,D}
    auxd = materialize_aux(g, 1)
    ndl = op.ndl
    xl = zeros(2ndl)
    f = g.facets[1]
    gather_cell!(xl, op.fields, x, dvals, g.skel.cells[1, f])
    gather_cell!(xl, op.fields, x, dvals, g.skel.cells[2, f], ndl)
    Wb = map(t -> zeros(t.nn, D + 1), g.tables)
    if kind === :residual
        re = zeros(2ndl)
        element_residual!(re, g, xl, auxd, 1)
        return @allocated element_residual!(re, g, xl, auxd, 1)
    else
        KeT = zeros(2ndl, 2ndl)
        element_jacobian!(KeT, g, xl, auxd, 1, Wb)
        return @allocated element_jacobian!(KeT, g, xl, auxd, 1, Wb)
    end
end

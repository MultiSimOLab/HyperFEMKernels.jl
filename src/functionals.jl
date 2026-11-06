# Scalar functionals of the unknown evaluated by fused, threaded kernels (no assembly): energies
# (`∫(Ψ ∘ …)dΩ`, summed) and pointwise extrema over the quadrature points (the injectivity step of
# a line search, `min_q α(∇u, ∇du)`). The functional is captured like a residual without test
# function: its terms are compiled into a pointwise function of the field values and the auxiliary
# data (FE functions in the form read their nodal values at every evaluation, so an increment
# stored in an FE function can change between calls without recompiling).

"""
    KernelFunctional(f, U; reduce = :sum, nthreads = Threads.nthreads())

Scalar functional of the unknown `u ∈ U` evaluated by the kernels. `f(u)` returns a
`DomainContribution` as a residual does, but without test function: `u -> ∫(Ψ ∘ (F ∘ ∇(u)'))dΩ`
(an energy), `u -> ∫(g ∘ (∇(u)', ∇(duh)'))dΩ` with `duh` an FE function on `U` (a pointwise
quantity of the state and an increment). The functional is called on the free values:
`fun(x) -> Float64`.

- `reduce = :sum`: the integral, `Σ_cells Σ_q f(x_q) dV_q` (the energy of a Newton line search).
- `reduce = :min` / `:max`: the extremum of the integrand over the quadrature points of the
  measure, without weights (the injectivity step `min_q α(∇u_q, ∇du_q)` of `InjectivityLS`).

Contributions the kernels cannot read are summed by Gridap (`:sum` only; `:min`/`:max` need every
contribution in the kernels). [`update`](@ref) gives a functional with another `f` on the same
space reusing the tables.
"""
struct KernelFunctional{F,US}
    f::F
    U::US
    fields::Vector{FieldLayout}
    ndl::Int
    groups::Vector{Any}
    reduce::Symbol
    nthreads::Int
    cache_aux::Bool
    fallback::Vector{Any}     # triangulations whose contributions Gridap sums (`:sum`)
    signed::Bool
end

function KernelFunctional(f, U::FESpace; reduce::Symbol=:sum, nthreads::Int=Threads.nthreads())
    reduce in (:sum, :min, :max) || throw(ArgumentError("HyperFEMKernels: KernelFunctional: reduce must be :sum, :min or :max (got :$reduce)"))
    Us = field_spaces(U)
    if U isa MultiFieldFESpace
        MultiFieldStyle(U) isa ConsecutiveMultiFieldStyle ||
            error("HyperFEMKernels: only ConsecutiveMultiFieldStyle multi-field spaces are supported")
    end
    trian = get_triangulation(Us[1])
    D = num_point_dims(trian)
    ncells = num_cells(trian)
    fields = FieldLayout[]
    uoff, loc = 0, 0
    for su in Us
        nd = length(testitem(get_cell_dof_ids(su)))
        ids = cell_dof_matrix(su, nd)
        push!(fields, FieldLayout(ids, ids, uoff, uoff, loc, nd, cell_signs(su, nd)))
        uoff += num_free_dofs(su)
        loc += nd
    end
    forms, fb = capture(f, probes(U, :unknown))
    colors = [Int32.(1:ncells)]                     # a reduction has no races: one colour
    groups = compile_groups(forms, U, Tuple(fl.loc for fl in fields), D, colors; linear_maps=false, fallback=fb, fields, ndl=loc)
    _check_functional(groups, fb, reduce)
    KernelFunctional(f, U, fields, loc, groups, reduce, nthreads, false, Any[t for (t, _) in fb], any(has_signs, fields))
end

function _check_functional(groups, fb, reduce)
    for g in groups
        g isa Group || error("HyperFEMKernels: KernelFunctional does not support skeleton integrals yet")
        nslots(g) == (0, 0) || error("HyperFEMKernels: a KernelFunctional has no test or trial functions (got a form with them)")
    end
    if !isempty(fb)
        reduce === :sum || error("HyperFEMKernels: KernelFunctional with reduce = :$reduce needs every contribution in the kernels: " *
                                 join(("$(t): $why" for (t, why) in fb), "; "))
        warn_fallback("functional", fb)
    end
end

"""
    update(fun::KernelFunctional, f) -> KernelFunctional

Functional with another `f` on the same space (e.g. a new load factor), reusing the tables and
geometry of `fun`.
"""
function update(fun::KernelFunctional, f)
    U = fun.U
    D = num_point_dims(get_triangulation(field_spaces(U)[1]))
    forms, fb = capture(f, probes(U, :unknown))
    colors = [Int32.(1:num_cells(get_triangulation(field_spaces(U)[1])))]
    groups = compile_groups(forms, U, Tuple(fl.loc for fl in fun.fields), D, colors, fun.groups; linear_maps=false, fallback=fb,
                            fields=fun.fields, ndl=fun.ndl)
    _check_functional(groups, fb, fun.reduce)
    KernelFunctional(f, U, fun.fields, fun.ndl, groups, fun.reduce, fun.nthreads, fun.cache_aux, Any[t for (t, _) in fb], fun.signed)
end

_init(::Val{:sum}) = 0.0
_init(::Val{:min}) = Inf
_init(::Val{:max}) = -Inf
@inline _combine(::Val{:sum}, a, b) = a + b
@inline _combine(::Val{:min}, a, b) = min(a, b)
@inline _combine(::Val{:max}, a, b) = max(a, b)

"Evaluate the functional at the free values `x`."
function (fun::KernelFunctional)(x::AbstractVector)
    fun.reduce === :sum ? _evaluate(Val(:sum), fun, x) : fun.reduce === :min ? _evaluate(Val(:min), fun, x) : _evaluate(Val(:max), fun, x)
end

function _evaluate(R::Val, fun::KernelFunctional, x::AbstractVector)
    dvals = [get_dirichlet_dof_values(s) for s in field_spaces(fun.U)]
    acc = _init(R)
    for g in fun.groups
        auxd = aux_data(fun, g)
        for cells in g.colors                 # a reduction has no races: the colours are only a partition
            isempty(cells) && continue
            work = length(cells) * g.geo.np * fun.ndl
            nchunks = clamp(work ÷ WORK_PER_TASK, 1, min(fun.nthreads, length(cells)))
            if nchunks == 1
                acc = _combine(R, acc, functional_cells(R, fun, g, x, dvals, auxd, cells))
            else
                parts = Vector{Float64}(undef, nchunks)
                @sync for (i, ch) in enumerate(Iterators.partition(cells, cld(length(cells), nchunks)))
                    Threads.@spawn parts[i] = functional_cells(R, fun, g, x, dvals, auxd, ch)
                end
                for p in parts
                    acc = _combine(R, acc, p)
                end
            end
        end
    end
    if !isempty(fun.fallback)
        dc = fun.f(FEFunction(fun.U, x))
        any(t -> t === :all, fun.fallback) || (dc = restrict(dc, fun.fallback))
        acc = _combine(R, acc, sum(dc))
    end
    acc
end

function functional_cells(R::Val, fun::KernelFunctional, g::Group{NSV,NSU,PF,TT,AUX,SV,SU,D}, x, dvals, auxd,
                          cells) where {NSV,NSU,PF,TT,AUX,SV,SU,D}
    ndl = fun.ndl
    xl = zeros(ndl)
    sg = zeros(ndl)
    signed = fun.signed
    tables, locs, pf = g.tables, g.locs, g.pf
    acc = _init(R)
    for cell in cells
        sc = g.cellmap[cell]
        gather_cell!(xl, fun.fields, x, dvals, sc)
        signed && apply_signs!(xl, cell_signs!(sg, fun.fields, sc), ndl)
        for q in 1:g.geo.np
            dV, iJt, Jt = point_geometry(g.geo, q, cell)
            FV = field_values(tables, locs, g.use, xl, q, iJt, Jt)
            Aq = map(a -> @inbounds(a[q, cell]), auxd)
            v = pf.core(pf.ops, pf.coefs, FV, Aq, EMPTY_SLOTS, EMPTY_SLOTS)::Float64
            acc = _combine(R, acc, R isa Val{:sum} ? v * dV : v)
        end
    end
    acc
end

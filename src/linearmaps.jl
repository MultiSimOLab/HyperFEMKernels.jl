# Linear-map shortcut: residual coefficients and tangents without dual numbers.
#
# Each term of a residual is linear in the test slots, and each term of a Jacobian is bilinear in
# (test, trial) slots. The part of a term that touches the slots depends on the state through its
# "state inputs": the maximal subexpressions that do not contain test/trial functions (e.g.
# ∂Ψ∂FF(F, E) in ∇(v)' ⊙ (∂Ψ∂FF ∘ (...) ⊙ ∇(du)'), or u, uh⁻ and υh in (2/Δt²(u - uh⁻) - 2/Δt υh)⋅v).
# Numbers of the integrand are parameters, not inputs (see `emit!(::CodegenState, ::Const)`). When the
# slot coefficients are a linear
# function of the concatenated components of the inputs (the common case: the weak form is linear
# in stresses, tangents and states),
#
#     coeff[i] = Σ_term C_term Σ_(g,k) M_term[i, (g,k)] · s_g[k]
#
# M is computed when the operator is built (with dual numbers on unit inputs) and checked on random
# inputs; at run time only the inputs are evaluated (plain numbers) and M is applied. Terms that
# are not linear in their inputs make the whole group fall back to the dual-number path.
#
# M depends on the parameters, so its values are data of the LinearForm, and the generated `apply`
# depends only on the positions of the nonzeros. Positions and the linearity check also use random
# parameter values: neither the code nor the choice of path depend on the values (a parameter that is
# zero, e.g. a load factor at the first step, does not remove entries or hide a nonlinear term), so
# `update` with other parameter values reuses the compiled functions.

dependent(n::Leaf) = n.role !== :unknown
dependent(n::Aux) = false
dependent(n::Const) = false
dependent(n::Op) = any(dependent, n.args)

"Maximal non-dependent subexpressions of `n` (constants excluded)."
function state_inputs!(acc::Vector{Node}, n::Node)
    n isa Op || return acc
    for a in n.args
        if dependent(a)
            state_inputs!(acc, a)
        elseif !(a isa Const) && !any(b -> b === a, acc)
            push!(acc, a)
        end
    end
    acc
end

"Sparse linear map: target slot index (linear in R or T), global input index, component, value."
const SparseMap = Vector{Tuple{Int32,Int32,Int32,Float64}}
const DEBUG_LINEAR = get(ENV, "HFK_DEBUG_LINEAR", "0") == "1"
_nolin(why) = (DEBUG_LINEAR && println("HyperFEMKernels: linear-map shortcut not used: ", why); nothing)

struct LinearForm{F,O,C,ST,AP,VT}
    state::F            # (ops, FV, A) -> tuple of all state inputs of the group
    ops::O
    coefs::C            # one coefficient per term
    maps::Vector{SparseMap}
    apply::AP           # (V, S) -> SVector of slot coefficients (generated from the positions of `maps`)
    vals::VT            # term coefficient × value of each entry of `maps`, in order (NTuple{N,Float64})
    expr::Expr
end

comp(x::Real, k) = x
comp(x::MultiValue, k) = x.data[k]
ncomp(::Type{<:Real}) = 1
ncomp(::Type{T}) where {T<:MultiValue} = length(T)
unit_value(::Type{<:Real}, k) = 1.0
unit_value(::Type{T}, k) where {T<:MultiValue} = T(ntuple(i -> i == k ? 1.0 : 0.0, length(T)))
rand_value(::Type{<:Real}) = rand()
rand_value(::Type{T}) where {T<:MultiValue} = T(ntuple(_ -> rand(), length(T)))

field_types(::FieldTable{D,Float64}) where {D} = (Float64, VectorValue{D,Float64})
field_types(::FieldTable{D,VectorValue{NC,Float64}}) where {D,NC} = (VectorValue{NC,Float64}, TensorValue{D,NC,Float64,D * NC})

# Split a slot-dependent expression into additive parts: [(expr, sign), ...].
function split_sum(n::Node, sgn::Float64=1.0)
    if n isa Op && all(dependent, n.args)
        n.f === (+) && return reduce(vcat, [split_sum(a, sgn) for a in n.args])
        n.f === (-) && length(n.args) == 1 && return split_sum(n.args[1], -sgn)
        n.f === (-) && length(n.args) == 2 && return vcat(split_sum(n.args[1], sgn), split_sum(n.args[2], -sgn))
    end
    [(n, sgn)]
end

"""
    LinearForm(terms, pf, tables, aux, kind) -> LinearForm or nothing

Build the linear-map shortcut for a group, or `nothing` if some term does not qualify.
`kind` is `:res` (linear in test slots) or `:jac` (bilinear in test and trial slots).
"""
function LinearForm(terms0::Vector{Term}, pf::PointForm, tables, aux::Tuple, kind::Symbol)
    terms = Term[]
    for t in terms0, (e, sg) in split_sum(t.expr)
        push!(terms, Term(t.coef * sg, e, t.weights, t.jacobians))
    end
    coefs = Tuple(t.coef for t in terms)
    all(t -> dependent(t.expr), terms) || return _nolin("a term does not depend on test/trial functions")
    # all state inputs of the group (shared between terms when identical), one state function
    inputs = Node[]
    term_inputs = [Int[] for _ in terms]
    for (i, t) in enumerate(terms)
        for n in state_inputs!(Node[], t.expr)
            k = findfirst(m -> m === n, inputs)
            k === nothing && (push!(inputs, n); k = length(inputs))
            push!(term_inputs[i], k)
        end
    end
    st = CodegenState(pf.test, pf.trial, pf.aux)
    vals = Any[emit_node!(st, n) for n in inputs]
    ex = :((ops, FV, A) -> begin
        $(st.body...)
        return ($(vals...),)
    end)
    state = RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, ex)
    ops = Tuple(st.ops)
    # concrete types of the inputs: evaluate once on sample data (small random state, first cell)
    FVs = map(t -> map(T -> 0.01 * rand_value(T), field_types(t)), tables)
    As = map(a -> a[1][1], aux)
    ST = try
        typeof(state(ops, FVs, As))
    catch err
        return _nolin("state function failed on sample data: $(sprint(showerror, err))")
    end
    all(T -> isconcretetype(T) && (T <: Real || T <: MultiValue), ST.parameters) ||
        return _nolin("state inputs have unsupported types: $ST")
    nsv, nsu = pf.test.nslots, pf.trial.nslots
    maps = SparseMap[]
    for (i, t) in enumerate(terms)
        gidx = term_inputs[i]
        lst = CodegenState(pf.test, pf.trial, pf.aux)
        for (j, g) in enumerate(gidx)
            lst.subst[inputs[g]] = :(S[$j])
        end
        r = emit_node!(lst, t.expr)
        lex = :((ops, S, sv, su) -> begin
            $(lst.body...)
            return $r
        end)
        lin = RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, lex)
        lops = Tuple(lst.ops)
        types = [ST.parameters[g] for g in gidx]
        # the actual parameters, and random ones: positions and linearity must not depend on the values
        rops = random_params(lops, lst.consts)
        coeffs(S, o) = slot_coeffs(lin, o, S, nsv, nsu, kind)
        m = try
            term_map(coeffs, rops === nothing ? Any[lops] : Any[lops, rops], types, gidx)
        catch err
            (rops === nothing || err isa InterruptException) && rethrow()
            # random parameters not admissible (e.g. outside the domain of a function): actual values only
            term_map(coeffs, Any[lops], types, gidx)
        end
        m isa String && return _nolin("term $i $m")
        push!(maps, m)
    end
    apply = generate_apply(maps, ST, kind === :res ? nsv : nsv * nsu)
    vals = map_values(maps, coefs)
    LinearForm{typeof(state),typeof(ops),typeof(coefs),ST,typeof(apply),typeof(vals)}(state, ops, coefs, maps, apply, vals, ex)
end

"""
    term_map(coeffs, opsets, types, gidx) -> SparseMap or String (reason it is not linear)

Linear map of one term: entries at the positions that are nonzero for any of the parameter sets
`opsets` (actual parameters first), with the values of the actual parameters; linearity in the
state inputs is checked on random inputs with every parameter set.
"""
function term_map(coeffs, opsets, types, gidx)
    m = SparseMap()
    zS = Tuple(zero(T) for T in types)
    cz = [coeffs(zS, o) for o in opsets]
    if isempty(gidx)
        for idx in eachindex(cz[1])
            any(c -> c[idx] != 0.0, cz) && push!(m, (Int32(idx), Int32(0), Int32(0), cz[1][idx]))
        end
        return m
    end
    any(c -> any(!=(0.0), c), cz) && return "has a slot coefficient that does not vanish for zero state inputs"
    cols = Vector{Vector{Float64}}[Vector{Float64}[] for _ in opsets]   # coefficients for each unit input
    units = Tuple{Int32,Int32}[]
    for (j, g) in enumerate(gidx), k in 1:ncomp(types[j])
        S = Tuple(jj == j ? unit_value(types[j], k) : zero(types[jj]) for jj in eachindex(types))
        cu = [coeffs(S, o) for o in opsets]
        foreach((q, c) -> push!(cols[q], c), eachindex(opsets), cu)
        push!(units, (Int32(g), Int32(k)))
        for idx in eachindex(cu[1])
            any(c -> c[idx] != 0.0, cu) && push!(m, (Int32(idx), Int32(g), Int32(k), cu[1][idx]))
        end
    end
    # joint linearity check on random inputs
    Sr = Tuple(rand_value(T) for T in types)
    sr = [comp(Sr[findfirst(==(g), gidx)], k) for (g, k) in units]
    for (q, o) in enumerate(opsets)
        direct = coeffs(Sr, o)
        mapped = reduce(+, (cols[q][u] .* sr[u] for u in eachindex(sr)))
        err = norm(direct - mapped) / max(norm(direct), 1.0)
        err <= 1e-12 || return "is not linear in its state inputs (rel. error $err$(q > 1 ? ", random parameters" : ""))"
    end
    m
end

"Term coefficient × value of the entries of the maps, in the order used by `generate_apply`."
map_values(maps::Vector{SparseMap}, coefs) = Tuple(coefs[i] * v for (i, m) in enumerate(maps) for (_, _, _, v) in m)

"Copy of `ops` with the parameters at `consts` replaced by random values of the same type (nothing if none)."
function random_params(ops::Tuple, consts::Vector{Int})
    isempty(consts) && return nothing
    Tuple(i in consts ? random_param(o) : o for (i, o) in enumerate(ops))
end
random_param(x::AbstractFloat) = typeof(x)(0.5 + rand())
random_param(x::MultiValue) = eltype(x) <: AbstractFloat ? typeof(x)(Tuple(0.5 + rand() for _ in 1:length(x.data))) : x
random_param(x) = x            # integers (e.g. exponents), booleans: structural, kept

"""
Generated function (V, S) -> SVector{N}: static application of the sparse maps, with the values of their
entries (times the term coefficients) read from `V` (see `map_values`), so the code depends only on the
positions.
"""
function generate_apply(maps::Vector{SparseMap}, ST, N::Int)
    sums = [Any[] for _ in 1:N]
    n = 0
    for (i, m) in enumerate(maps), (idx, g, k, _) in m
        n += 1
        if g == 0
            push!(sums[idx], :(V[$n]))
        else
            sg = ST.parameters[Int(g)] <: Real ? :(S[$(Int(g))]) : :(S[$(Int(g))].data[$(Int(k))])
            push!(sums[idx], :(V[$n] * $sg))
        end
    end
    comps = [isempty(sm) ? 0.0 : Expr(:call, :+, sm...) for sm in sums]
    ex = :((V, S) -> SVector{$N,Float64}(($(comps...),)))
    RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, ex)
end

"Slot coefficients (vector for :res, vec of the matrix for :jac) of a linear part for inputs `S`."
function slot_coeffs(lin, lops, S, nsv, nsu, kind)
    zsv = zeros(SVector{nsv,Float64})
    if kind === :res
        return collect(ForwardDiff.gradient(v -> lin(lops, S, v, SVector{0,Float64}()), zsv))
    end
    zsu = zeros(SVector{nsu,Float64})
    vec(collect(ForwardDiff.jacobian(w -> ForwardDiff.gradient(v -> lin(lops, S, v, w), zsv), zsu)))   # [test slot, trial slot]
end

"Apply the linear maps: coefficients (length nsv, or nsv*nsu) for the state inputs at a point."
@inline apply_maps(lf::LinearForm, svals::Tuple, ::Val{N}) where {N} = lf.apply(lf.vals, svals)::SVector{N,Float64}

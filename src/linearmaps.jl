# Linear-map shortcut: residual coefficients and tangents without dual numbers.
#
# Each term of a residual is linear in the test slots, and each term of a Jacobian is bilinear in
# (test, trial) slots. In most weak forms (all of HyperFEM's) the part of the term that touches
# the slots depends on the state through a single "state input": a subexpression that does not
# contain test/trial functions, e.g. ∂Ψ∂FF(F, E) in ∇(v)' ⊙ (∂Ψ∂FF ∘ (...) ⊙ ∇(du)'). The slot
# coefficients are then a *fixed linear map* of the components of that state input:
#
#     T[i, j] = Σ_k M[(i, j), k] · s_k
#
# M is computed once (at compile time, with dual numbers on unit state inputs) and stored as a
# sparse list. At run time only the state inputs are evaluated (plain numbers) and M is applied.
# Terms with no state input (constant coefficients) have M applied to s = 1. If a term has more
# than one state input, the group falls back to the dual-number path.

dependent(n::Leaf) = n.role !== :unknown
dependent(n::Aux) = false
dependent(n::Op) = any(dependent, n.args)

"State inputs of the slot-dependent part of `n` (arguments that do not depend on the slots)."
function state_inputs!(acc::Vector{Node}, n::Node)
    n isa Op || return acc
    for a in n.args
        dependent(a) ? state_inputs!(acc, a) : push!(acc, a)
    end
    acc
end

"Sparse linear map: target slot index (linear in R or T), state component, value."
const SparseMap = Vector{Tuple{Int32,Int32,Float64}}

struct LinearForm{F,O,C,ST}
    state::F            # (ops, FV, A) -> tuple of state inputs, one per term (1.0 if none)
    ops::O
    coefs::C
    maps::Vector{SparseMap}
    expr::Expr
end

ncomps_val(x::Real) = 1
ncomps_val(x::MultiValue) = length(x)
comp(x::Real, k) = x
comp(x::MultiValue, k) = x.data[k]

unit_value(::Type{<:Real}, k) = 1.0
unit_value(::Type{T}, k) where {T<:MultiValue} = T(ntuple(i -> i == k ? 1.0 : 0.0, length(T)))
rand_value(::Type{<:Real}) = rand()
rand_value(::Type{T}) where {T<:MultiValue} = T(ntuple(_ -> rand(), length(T)))

"Value types at a point of the unknowns of each field (value, gradient)."
fv_types(tables) = Tuple{map(t -> Tuple{field_types(t)...}, tables)...}
field_types(::FieldTable{D,Float64}) where {D} = (Float64, VectorValue{D,Float64})
field_types(::FieldTable{D,VectorValue{NC,Float64}}) where {D,NC} = (VectorValue{NC,Float64}, TensorValue{D,NC,Float64,D * NC})

"""
    LinearForm(terms, pf, tables, aux, kind) -> LinearForm or nothing

Build the linear-map shortcut for a group, or `nothing` if some term does not qualify.
`kind` is `:res` (linear in test slots) or `:jac` (bilinear in test and trial slots).
"""
"Split a slot-dependent expression into additive parts: [(expr, sign), ...]."
function split_sum(n::Node, sgn::Float64=1.0)
    if n isa Op && all(dependent, n.args)
        n.f === (+) && return reduce(vcat, [split_sum(a, sgn) for a in n.args])
        n.f === (-) && length(n.args) == 1 && return split_sum(n.args[1], -sgn)
        n.f === (-) && length(n.args) == 2 && return vcat(split_sum(n.args[1], sgn), split_sum(n.args[2], -sgn))
    end
    [(n, sgn)]
end

function LinearForm(terms0::Vector{Term}, pf::PointForm, tables, aux::Tuple, kind::Symbol)
    # additive parts of each term become separate terms with their own state input
    terms = Term[]
    for t in terms0, (e, sg) in split_sum(t.expr)
        push!(terms, Term(t.coef * sg, e, t.weights, t.jacobians))
    end
    coefs = Tuple(t.coef for t in terms)
    for t in terms
        dependent(t.expr) || return nothing
        length(state_inputs!(Node[], t.expr)) <= 1 || return nothing
    end
    # state function: one value per term (1.0 if the term has no state input), shared CSE
    st = CodegenState(pf.test, pf.trial, pf.aux)
    vals = Any[]
    for t in terms
        acc = state_inputs!(Node[], t.expr)
        push!(vals, isempty(acc) ? 1.0 : emit_node!(st, acc[1]))
    end
    ex = :((ops, FV, A) -> begin
        $(st.body...)
        return ($(vals...),)
    end)
    state = RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, ex)
    ops = Tuple(st.ops)
    auxT = Tuple{map(a -> eltype(eltype(a)), aux)...}
    ST = Core.Compiler.return_type(state, Tuple{typeof(ops),fv_types(tables),auxT})
    (ST isa DataType && ST <: Tuple && all(T -> isconcretetype(T) && (T <: Real || T <: MultiValue), ST.parameters)) ||
        return nothing
    nsv, nsu = pf.test.nslots, pf.trial.nslots
    maps = SparseMap[]
    for (i, t) in enumerate(terms)
        S = ST.parameters[i]
        acc = state_inputs!(Node[], t.expr)
        lst = CodegenState(pf.test, pf.trial, pf.aux)
        isempty(acc) || (lst.subst[acc[1]] = :S)
        r = emit_node!(lst, t.expr)
        lex = :((ops, S, sv, su) -> begin
            $(lst.body...)
            return $r
        end)
        lin = RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, lex)
        lops = Tuple(lst.ops)
        coeffs(s) = slot_coeffs(lin, lops, s, nsv, nsu, kind)
        m = SparseMap()
        ns = S <: Real ? 1 : length(S)
        for k in 1:ns
            Ek = isempty(acc) ? 1.0 : unit_value(S, k)
            c = coeffs(Ek)
            for (idx, v) in enumerate(c)
                v != 0.0 && push!(m, (Int32(idx), Int32(k), v))
            end
        end
        # check linearity in the state input with a random value
        if !isempty(acc)
            sr = rand_value(S)
            direct = coeffs(sr)
            mapped = zeros(length(direct))
            for (idx, k, v) in m
                mapped[idx] += v * comp(sr, k)
            end
            norm(direct - mapped) <= 1e-12 * max(norm(direct), 1.0) || return nothing
        end
        push!(maps, m)
    end
    LinearForm{typeof(state),typeof(ops),typeof(coefs),ST}(state, ops, coefs, maps, ex)
end

"Slot coefficients (vector for :res, vec of the matrix for :jac) of a linear part for state value `s`."
function slot_coeffs(lin, lops, s, nsv, nsu, kind)
    zsv = zeros(SVector{nsv,Float64})
    if kind === :res
        return collect(ForwardDiff.gradient(v -> lin(lops, s, v, SVector{0,Float64}()), zsv))
    end
    zsu = zeros(SVector{nsu,Float64})
    vec(collect(ForwardDiff.jacobian(v -> ForwardDiff.gradient(w -> lin(lops, s, v, w), zsu), zsv)))
end

"Apply the linear maps: coefficients (length nsv, or nsv*nsu) for the state inputs at a point."
@inline function apply_maps(lf::LinearForm, svals::Tuple, ::Val{N}) where {N}
    out = zero(MVector{N,Float64})
    _apply!(out, lf.maps, lf.coefs, svals, 1)
    SVector(out)
end
@inline _apply!(out, maps, coefs, ::Tuple{}, i) = nothing
@inline function _apply!(out, maps, coefs, svals::Tuple, i)
    s = first(svals)
    c = coefs[i]
    @inbounds for (idx, k, v) in maps[i]
        out[idx] += c * v * comp(s, k)
    end
    _apply!(out, maps, coefs, Base.tail(svals), i + 1)
end

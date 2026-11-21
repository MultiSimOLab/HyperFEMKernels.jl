# Test × trial factorisation of the Jacobian at a point.
#
# The integrand of a Jacobian is bilinear in the test slots `s` and the trial slots `r`; its tangent
# T[i, j] = ∂²f/∂s_i∂r_j is computed by default with nested dual numbers (16 × 16 slots for u–p P1: every
# operation that touches the slots costs ~17² operations). When the linear-map shortcut does not apply (a term
# not linear in its state inputs: SUPG, (∇v'⋅w)⋅(∇du'⋅w), has w twice), most Jacobian terms are still a product
# of a factor that depends on the test functions only and a factor that depends on the trial functions only:
#
#     f = c · W(N(A(s), B(r)))
#
# with N ∈ {⋅, ⊙, *} (`dot`, `inner`, `*`), A test-only, B trial-only (any operations, state included, inside),
# and W a chain of multiplications (or divisions) by state values. Both factors are linear in their slots, so
#
#     T[i, j] = c · W(N(∂A/∂s_i, ∂B/∂r_j))
#
# and each factor needs one evaluation with single dual numbers seeded on its own slots only, followed by small
# products. Additive parts (sums, differences, also inside W) are taken one by one; parts without test or trial
# functions contribute nothing to T. A group is factorised only when every part has this form, proved on the
# tree (no other operation is accepted between the factors), and when the result reproduces the integrand on
# sample points (`check_factorisation`; `FactorForm` returns `nothing` otherwise and the group keeps the nested
# dual-number path).
# `HFK_FACTORISE=0` disables it; `HFK_DEBUG_FACTORISE=1` prints why a group is not factorised.

const FACTORISE = Ref(true)
debug_factorise() = get(ENV, "HFK_DEBUG_FACTORISE", "0") == "1"
_nofac(why) = (debug_factorise() && println("HyperFEMKernels: factorisation not used: ", why); nothing)

"Tag of the dual numbers of the factors."
struct FactorTag end

has_role(n::Leaf, r::Symbol) = n.role === r
has_role(n::Op, r::Symbol) = any(a -> has_role(a, r), n.args)
has_role(n::Node, r::Symbol) = false
function role_kind(n::Node)
    t, u = has_role(n, :test), has_role(n, :trial)
    t && u ? :mixed : t ? :test : u ? :trial : :state
end
is_product(f) = f === LinearAlgebra.dot || f === Gridap.TensorValues.inner || f === (*)

"One factorised part: c · W(N(A, B)), W = `wraps` (outermost first: the operation and the position of the factor)."
struct FactorPart
    sign::Float64
    wraps::Vector{Tuple{Op,Int}}
    prod::Op
    test::Node
    trial::Node
    test_first::Bool
end

"""
Split `n` (a Jacobian integrand, or part of one under the wrappers `wraps`) into factorised parts, appended to
`parts`; a String with the reason when some part is not of the accepted form.
"""
function factor_parts!(parts::Vector{FactorPart}, n::Node, wraps::Vector{Tuple{Op,Int}}, sg::Float64)
    role_kind(n) === :mixed || return nothing              # no test or no trial function: ∂²/∂s∂r = 0
    n isa Op || return "unexpected node $(typeof(n))"
    f, args = n.f, n.args
    if f === (+)
        for a in args
            r = factor_parts!(parts, a, wraps, sg); r isa String && return r
        end
        return nothing
    elseif f === (-) && length(args) == 1
        return factor_parts!(parts, args[1], wraps, -sg)
    elseif f === (-) && length(args) == 2
        r = factor_parts!(parts, args[1], wraps, sg); r isa String && return r
        return factor_parts!(parts, args[2], wraps, -sg)
    end
    kinds = map(role_kind, args)
    nmixed = count(==(:mixed), kinds)
    if (f === (*) || f === (/)) && nmixed == 1 && count(==(:state), kinds) == length(args) - 1
        i = findfirst(==(:mixed), kinds)
        f === (/) && i != 1 && return "a division by an expression of test and trial functions"
        return factor_parts!(parts, args[i], vcat(wraps, [(n, i)]), sg)
    end
    if is_product(f) && length(args) == 2 && Set(kinds) == Set([:test, :trial])
        tf = kinds[1] === :test
        push!(parts, FactorPart(sg, wraps, n, tf ? args[1] : args[2], tf ? args[2] : args[1], tf))
        return nothing
    end
    "an operation `$(nameof(f))` that does not separate test and trial functions"
end

"Seed vector of `N` dual numbers with unit partials at the slots `S` (zero elsewhere)."
@generated function seed_slots(::Val{N}, ::Val{S}) where {N,S}
    M = length(S)
    DT = ForwardDiff.Dual{FactorTag,Float64,M}
    comps = map(1:N) do i
        k = findfirst(==(i), S)
        p = ntuple(j -> j == k ? 1.0 : 0.0, M)
        :($DT(0.0, ForwardDiff.Partials{$M,Float64}($p)))
    end
    :(SVector{$N,$DT}(($(comps...),)))
end

"Partial `k` of a value of dual numbers (a number or a tensor), as plain numbers of the same shape."
@inline dpart(x::ForwardDiff.Dual, k) = ForwardDiff.partials(x, k)
@inline dpart(x::Real, k) = zero(x)
@inline dpart(x::MultiValue, k) = Gridap.TensorValues.change_eltype(typeof(x), Float64)(map(d -> Float64(dpart(d, k)), x.data))
"The `M` partials of a factor (its derivatives with respect to its seeded slots)."
@inline factor_partials(x, ::Val{M}) where {M} = ntuple(k -> dpart(x, k), Val(M))

"Generated point tangent of a factorised group, with its data."
struct FactorForm{F,O,C}
    fun::F             # (ops, C, FV, A) -> SMatrix{nsv,nsu}
    ops::O
    coefs::C           # term coefficient × sign of each part
    expr::Expr
    consts::Vector{Int}
end

"Position of the operation `f` in `st.ops` (added if new), as `emit!(::Op)` does."
function op_index!(st::CodegenState, f)
    k = findfirst(i -> !(i in st.consts) && st.ops[i] === f, eachindex(st.ops))
    k === nothing && (push!(st.ops, f); k = length(st.ops))
    k
end

"Slots of the leaves of `role` in `n` (positions in the slot layout `layout`), sorted."
function factor_slots(n::Node, role::Symbol, layout::SlotLayout)
    ls = Leaf[]
    _walk_leaves!(ls, role, n)
    sort!(unique!(reduce(vcat, [collect(layout.offsets[findfirst(==(l), layout.leaves)] .+ (1:nslots(l))) for l in ls])))
end

"Generated expression, operations and coefficients of the point tangent of the parts of `terms`, or a String."
function factor_code(terms::Vector{Term}, pf::PointForm)
    parts = FactorPart[]
    pcoefs = Float64[]
    for t in terms, (e, sg) in split_sum(t.expr)
        n0 = length(parts)
        r = factor_parts!(parts, e, Tuple{Op,Int}[], sg)
        r isa String && return r
        for p in parts[n0+1:end]
            push!(pcoefs, t.coef * p.sign)
        end
    end
    isempty(parts) && return "no term with test and trial functions"
    nsv, nsu = pf.test.nslots, pf.trial.nslots
    st = CodegenState(pf.test, pf.trial, pf.aux, pf.nf)
    seeds = Dict{Tuple{Symbol,Vector{Int}},Symbol}()
    pre = Expr[]
    factor_syms = Dict{Any,Tuple{Symbol,Vector{Int}}}()
    "Emit a factor with its own seeds; returns (symbol of its partials, its slots)."
    function emit_factor!(n::Node, role::Symbol)
        layout = role === :test ? st.test : st.trial
        S = factor_slots(n, role, layout)
        key = (role, S)
        sd = get!(seeds, key) do
            s = Symbol("seed", length(seeds) + 1)
            push!(pre, :($s = seed_slots(Val($(role === :test ? nsv : nsu)), Val($(Tuple(S))))))
            s
        end
        ls = Leaf[]
        _walk_leaves!(ls, role, n)
        for l in ls
            st.subst[l] = slot_expr(l, sd, layout.offsets[findfirst(==(l), layout.leaves)])
        end
        r = emit_node!(st, n)
        foreach(l -> delete!(st.subst, l), ls)
        get!(factor_syms, (r, key)) do
            p = Symbol("P", length(factor_syms) + 1)
            push!(st.body, :($p = factor_partials($r, Val($(length(S))))))
            (p, S)
        end
    end
    acc = Expr[]
    for (ip, p) in enumerate(parts)
        PA, SA = emit_factor!(p.test, :test)
        PB, SB = emit_factor!(p.trial, :trial)
        kN = op_index!(st, p.prod.f)
        x = p.test_first ? :(ops[$kN]($PA[a], $PB[b])) : :(ops[$kN]($PB[b], $PA[a]))
        for (w, i) in reverse(p.wraps)
            wargs = Any[j == i ? x : emit_node!(st, w.args[j]) for j in eachindex(w.args)]
            x = :(ops[$(op_index!(st, w.f))]($(wargs...)))
        end
        push!(acc, quote
            for b in 1:$(length(SB)), a in 1:$(length(SA))
                @inbounds T[$(Tuple(SA))[a], $(Tuple(SB))[b]] += C[$ip] * $x
            end
        end)
    end
    ex = :((ops, C, FV, A) -> begin
        $(pre...)
        $(st.body...)
        T = zero(MMatrix{$nsv,$nsu,Float64,$(nsv * nsu)})
        $(acc...)
        return SMatrix{$nsv,$nsu,Float64,$(nsv * nsu)}(T)
    end)
    (ex, Tuple(st.ops), Tuple(pcoefs), st.consts, st)
end

"Copies of the auxiliary values of a point with random values of the same types (signs included)."
rand_aux(As) = map(x -> x isa Union{AbstractFloat,MultiValue} ? (rand(Bool) ? 1 : -1) * rand_value(typeof(x)) : x, As)

"""
    FactorForm(terms, pf, tables, aux) -> FactorForm or nothing

The factorised point tangent of a Jacobian group, or `nothing` when some part is not a product of a test factor
and a trial factor (see the top of this file) or when it does not reproduce the integrand on sample points
(`check_factorisation`).
"""
function FactorForm(terms::Vector{Term}, pf::PointForm, tables, aux::Tuple)
    code = factor_code(terms, pf)
    code isa String && return _nofac(code)
    ex, ops, coefs, consts, st = code
    ff = FactorForm(generated(ex), ops, coefs, ex, consts)
    ok = try
        with_check_rng(() -> check_factorisation(ff, st, terms, pf, tables, aux))
    catch err
        err isa InterruptException && rethrow()
        "the check failed: $(first(sprint(showerror, err), 200))"
    end
    ok === true || return _nofac(ok)
    ff
end

"""
    with_check_rng(f)

Run `f()` (the construction check) with the random generator of the task seeded to a fixed value, and restore
its state afterwards: the check is reproducible and leaves the random numbers of the user's script unchanged.
"""
function with_check_rng(f)
    rng = Random.default_rng()
    saved = copy(rng)
    Random.seed!(rng, 0x66616374)
    try
        f()
    finally
        copy!(rng, saved)
    end
end

"""
Check a factorised tangent T against the integrand itself: the integrand f(s, r) of a Jacobian is bilinear in the
test and trial slots, so f(a, b) = aᵀ T b for any slot vectors a, b, and for random ones a wrong T fails with
probability one. The integrand is evaluated with plain numbers (no dual numbers to compile). Four sample points
(small random states; the auxiliary values of the first point, then random ones of the same types and random
signs), three random pairs (a, b) each, with the actual parameters and with random ones (the same random value
for a number of the integrand in both forms, so that a parameter that is zero now hides nothing); relative 1e-12
of Σ|a_i||T_ij||b_j|. True, or the reason.
"""
function check_factorisation(ff::FactorForm, st::CodegenState, terms::Vector{Term}, pf::PointForm, tables, aux::Tuple)
    NV, NU = pf.test.nslots, pf.trial.nslots
    st0 = CodegenState(pf.test, pf.trial, pf.aux, pf.nf)     # the emission of the point form: nodes of its parameters
    foreach(t -> emit_node!(st0, t.expr), terms)
    rvals = IdDict{Any,Any}()
    rops(o, s) = Tuple((k = findfirst(==(i), s.consts); k === nothing ? o[i] : get!(() -> random_param(o[i]), rvals, s.constnodes[k]))
                       for i in eachindex(o))
    paramsets = Any[(pf.ops, ff.ops, false)]
    !isempty(st0.consts) && length(st0.ops) == length(pf.ops) && push!(paramsets, (rops(pf.ops, st0), rops(ff.ops, st), true))
    As0 = map(a -> a[1][1], aux)
    for sample in 1:4
        FV = map(t -> map(T -> 0.01 * rand_value(T), field_types(t)), tables)
        As = sample <= 2 ? As0 : rand_aux(As0)
        for (po, fo, random) in paramsets
            Tf = ff.fun(fo, ff.coefs, FV, As)
            for _ in 1:3
                a = SVector{NV,Float64}(ntuple(_ -> 2rand() - 1, NV))
                b = SVector{NU,Float64}(ntuple(_ -> 2rand() - 1, NU))
                f = try
                    pf.core(po, pf.coefs, FV, As, a, b)
                catch err
                    random && !(err isa InterruptException) && break     # random parameters outside a function's domain
                    rethrow()
                end
                scale = max(sum(abs.(a) .* abs.(Tf) .* abs.(b')), abs(f))
                abs(f - a' * Tf * b) <= 1e-12 * scale || (scale == 0 && f == 0) ||
                    return "does not reproduce the integrand (relative $(abs(f - a' * Tf * b) / max(scale, floatmin())), sample $sample$(random ? ", random parameters" : ""))"
            end
        end
    end
    true
end

"""
    FactorForm(terms, pf, old::FactorForm) -> FactorForm

The factorised tangent of `terms` reusing the compiled function of `old` (same structure: `update`): operations,
parameters and coefficients re-read from the new tree; no new code and no check.
"""
function FactorForm(terms::Vector{Term}, pf::PointForm, old::FactorForm)
    code = factor_code(terms, pf)
    code isa String && error("HyperFEMKernels: the weak form changed its structure (rebinding a factorised tangent)")
    ex, ops, coefs, consts, _ = code
    ex == old.expr && typeof(ops) == typeof(old.ops) && consts == old.consts && length(coefs) == length(old.coefs) ||
        error("HyperFEMKernels: the weak form changed its structure (rebinding a factorised tangent of another structure)")
    FactorForm(old.fun, ops, coefs, ex, consts)
end

# --- the tangent at a point, whatever the path -----------------------------------------------------------------
"Tangent T[test slot, trial slot] of a Jacobian group at a point: nested dual numbers, linear maps or factorised."
@inline point_tangent(::Nothing, pf, FV, Aq, ::Val{NV}, ::Val{NU}) where {NV,NU} =
    ForwardDiff.jacobian(r -> ForwardDiff.gradient(s -> pf.core(pf.ops, pf.coefs, FV, Aq, s, r), zeros(SVector{NV,Float64})), zeros(SVector{NU,Float64}))
@inline point_tangent(lf::LinearForm, pf, FV, Aq, ::Val{NV}, ::Val{NU}) where {NV,NU} =
    SMatrix{NV,NU,Float64,NV * NU}(apply_maps(lf, lf.state(lf.ops, FV, Aq), Val(NV * NU)))
@inline point_tangent(ff::FactorForm, pf, FV, Aq, ::Val{NV}, ::Val{NU}) where {NV,NU} =
    ff.fun(ff.ops, ff.coefs, FV, Aq)::SMatrix{NV,NU,Float64,NV * NU}

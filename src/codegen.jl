# Code generation: from a captured integrand tree to a pointwise Julia function.
#
# For a group of terms sharing triangulation and quadrature, we generate
#
#     core(ops, C, FV, A, sv, su) -> Number
#
# where `ops` is the tuple of pointwise operations (user/HyperFEM functions, ⊙, ', ...) followed by the
# numbers of the integrand (runtime parameters, see `emit!(::CodegenState, ::Const)`), `C` the
# tuple of term coefficients, `FV[f] = (u_f, ∇u_f)` the unknown values at the point, `A[k]` the
# auxiliary values at the point, and `sv`/`su` static vectors with the components ("slots") of the
# test and trial leaves. The integrand is linear in `sv` and in `su`, so residual coefficients
# and tangents follow from exact differentiation with respect to the slots.

ncomps(::Type{<:Real}) = 1
ncomps(::Type{T}) where {T<:MultiValue} = length(T)

"Slot layout of the test (or trial) leaves: leaf => first slot index."
struct SlotLayout
    leaves::Vector{Leaf}
    offsets::Vector{Int}
    nslots::Int
end
function SlotLayout(ls::Vector{Leaf})
    offs = Int[]
    n = 0
    for l in ls
        push!(offs, n)
        n += ncomps(l.T)
    end
    SlotLayout(ls, offs, n)
end

"""
    slot_map(layout, field, nc, D) -> Matrix{Int}

`m[c, J+1]` = slot of component `c` of the value (`J = 0`) or of derivative ∂/∂x_J (`J ≥ 1`)
of `field` in `layout`, or 0 if that leaf does not appear.
"""
function slot_map(layout::SlotLayout, field::Int, nc::Int, D::Int)
    m = zeros(Int, nc, D + 1)
    for (l, off) in zip(layout.leaves, layout.offsets)
        l.field == field || continue
        if l.deriv == 0
            for c in 1:nc
                m[c, 1] = off + c
            end
        else  # Gridap gradient: TensorValue{D,nc}[J,c] stored column-major, or VectorValue{D} if scalar
            for c in 1:nc, J in 1:D
                m[c, J+1] = off + J + D * (c - 1)
            end
        end
    end
    m
end

function slot_expr(T::Type, v::Symbol, off::Int)
    n = ncomps(T)
    comps = [:($v[$(off + i)]) for i in 1:n]
    T <: Real && return comps[1]
    T <: VectorValue && return :(VectorValue(($(comps...),)))
    T <: TensorValue && return :(TensorValue{$(size(T)[1]),$(size(T)[2])}(($(comps...),)))
    error("HyperFEMKernels: unsupported leaf type $T")
end

struct CodegenState
    ops::Vector{Any}
    body::Vector{Expr}
    memo::Dict{Any,Symbol}
    test::SlotLayout
    trial::SlotLayout
    aux::Vector{Aux}
    subst::IdDict{Any,Any}   # nodes replaced by a given expression (e.g. a symbol)
    consts::Vector{Int}      # positions in `ops` that hold numbers of the integrand (parameters)
end
CodegenState(test, trial, aux) = CodegenState(Any[], Expr[], Dict{Any,Symbol}(), test, trial, aux, IdDict{Any,Any}(), Int[])

"Emit a node, honouring substitutions."
emit_node!(st::CodegenState, n::Node) = haskey(st.subst, n) ? st.subst[n] : emit!(st, n)

function emit!(st::CodegenState, n::Leaf)
    if n.role === :unknown
        return :(FV[$(n.field)][$(n.deriv + 1)])
    end
    layout, v = n.role === :test ? (st.test, :sv) : (st.trial, :su)
    k = findfirst(==(n), layout.leaves)
    slot_expr(n.T, v, layout.offsets[k])
end

"""
Numbers of the integrand (`Λ * v`, `2μ * ε(u) ⊙ ε(v)`, `Pc(t) * (v ⋅ n)`) are runtime parameters, not
literals: they are stored in `ops` and read as `ops[k]`, so the generated code depends on the structure
of the weak form only, and `update` with other values reuses the compiled functions. One slot per
occurrence: slots are never merged by value, which would make the code depend on the values again.
"""
function emit!(st::CodegenState, n::Const)
    push!(st.ops, n.value)
    k = length(st.ops)
    push!(st.consts, k)
    :(ops[$k])
end

function emit!(st::CodegenState, n::Aux)
    k = findfirst(a -> a.array === n.array, st.aux)
    :(A[$k])
end

function emit!(st::CodegenState, n::Op)
    args = [emit_node!(st, c) for c in n.args]
    key = (objectid(n.f), args)
    haskey(st.memo, key) && return st.memo[key]
    k = findfirst(i -> !(i in st.consts) && st.ops[i] === n.f, eachindex(st.ops))
    if k === nothing
        push!(st.ops, n.f)
        k = length(st.ops)
    end
    s = Symbol("t", length(st.body) + 1)
    push!(st.body, :($s = ops[$k]($(args...))))
    st.memo[key] = s
    s
end

"Generated pointwise integrand for a group of terms, with its data."
struct PointForm{F,O,C}
    core::F
    ops::O
    coefs::C
    test::SlotLayout
    trial::SlotLayout
    aux::Vector{Aux}
    consts::Vector{Int}   # positions of the parameters (numbers of the integrand) in `ops`
    expr::Expr
end

function PointForm(terms::Vector{Term})
    test = SlotLayout(leaves(terms, :test))
    trial = SlotLayout(leaves(terms, :trial))
    aux = aux_nodes(terms)
    st = CodegenState(test, trial, aux)
    results = [emit_node!(st, t.expr) for t in terms]
    total = Expr(:call, :+, [:(C[$i] * $(r)) for (i, r) in enumerate(results)]...)
    ex = :((ops, C, FV, A, sv, su) -> begin
        $(st.body...)
        return $total
    end)
    core = RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, ex)
    coefs = Tuple(t.coef for t in terms)
    PointForm(core, Tuple(st.ops), coefs, test, trial, aux, st.consts, ex)
end

"Show the generated pointwise integrand (debugging)."
show_code(pf::PointForm) = println(Base.remove_linenums!(deepcopy(pf.expr)))

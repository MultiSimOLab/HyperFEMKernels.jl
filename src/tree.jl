# Reading the captured lazy tree.
#
# A residual (or Jacobian) captured with probes is a `DomainContribution`: for each
# triangulation, a lazy array that combines integrals with `Broadcasting(+/-/*)`. Each integral is
# `lazy_map(IntegrationMap(), integrand_at_points, weights, Jt_at_points)` and the integrand is a
# tree of `BroadcastingFieldOpMap(op)` nodes over probe leaves and auxiliary Gridap arrays.

abstract type Node end

"Unknown/test/trial function of `field`, value (`deriv = 0`) or gradient (`deriv = 1`)."
struct Leaf <: Node
    role::Symbol
    field::Int
    deriv::Int
    T::DataType
    points::CellPoint
end
Base.:(==)(a::Leaf, b::Leaf) = a.role == b.role && a.field == b.field && a.deriv == b.deriv
Base.hash(a::Leaf, h::UInt) = hash((a.role, a.field, a.deriv), h)

"Data that does not depend on the unknown, test or trial functions: evaluated by Gridap per cell."
struct Aux <: Node
    array::AbstractArray
end

"Pointwise operation `f(args...)`."
struct Op <: Node
    f::Any
    args::Vector{Node}
end

"One integral: `coef * ∫(expr)` with Gridap's quadrature `weights` and geometry `Jt`."
struct Term
    coef::Float64
    expr::Node
    weights::AbstractArray
    jacobians::AbstractArray
end

"Weak form captured on one triangulation: a sum of terms."
struct Form
    trian::Triangulation
    terms::Vector{Term}
end

const Fill = Arrays.Fill

"""
    capture(f, args...) -> Vector{Form}

Call `f(args...)` with probe arguments and read the resulting tree.
"""
function capture(f, args...)
    dc = f(args...)
    dc isa DomainContribution || error("HyperFEMKernels: the weak form did not return integrals (got $(typeof(dc)))")
    [Form(trian, collect_terms(arr, 1.0)) for (trian, arr) in dc.dict]
end

function collect_terms(a::LazyArray, coef::Float64)
    k = first(a.maps)
    if k isa IntegrationMap
        length(a.args) == 3 || error("HyperFEMKernels: only integrals in the physical domain with a reference-domain integrand are supported")
        return [Term(coef, to_node(a.args[1]), a.args[2], a.args[3])]
    elseif k isa Broadcasting && k.f === (+)
        return reduce(vcat, [collect_terms(b, coef) for b in a.args])
    elseif k isa Broadcasting && k.f === (-)
        length(a.args) == 1 && return collect_terms(a.args[1], -coef)
        return vcat(collect_terms(a.args[1], coef), reduce(vcat, [collect_terms(b, -coef) for b in a.args[2:end]]))
    elseif k isa Broadcasting && k.f === (*)
        s, b = a.args
        if s isa Fill && s.value isa Real
            return collect_terms(b, coef * s.value)
        elseif b isa Fill && b.value isa Real
            return collect_terms(s, coef * b.value)
        end
    end
    error("HyperFEMKernels: unsupported combination of integrals ($(typeof(k)))")
end
collect_terms(a, coef) = error("HyperFEMKernels: unsupported contribution of type $(typeof(a))")

function to_node(a::AbstractArray)
    a isa ProbeArray && return Leaf(a.probe.role, a.probe.field, a.probe.deriv, probe_valtype(a.probe), a.points)
    contains_probe(a) || return Aux(a)
    if a isa LazyArray
        k = first(a.maps)
        if k isa BroadcastingFieldOpMap
            return Op(k.op, Node[to_node(b) for b in a.args])
        end
        error("HyperFEMKernels: unsupported operation in the integrand: $(typeof(k))")
    end
    error("HyperFEMKernels: unsupported array in the integrand: $(typeof(a))")
end

"Leaves of each role appearing in a set of forms, in a stable order."
function leaves(terms::Vector{Term}, role::Symbol)
    found = Leaf[]
    for t in terms
        _walk_leaves!(found, role, t.expr)
    end
    sort!(found, by = l -> (l.field, l.deriv))
end
_walk_leaves!(found, role, n::Leaf) = (n.role === role && !(n in found)) && push!(found, n)
_walk_leaves!(found, role, n::Op) = foreach(c -> _walk_leaves!(found, role, c), n.args)
_walk_leaves!(found, role, n::Aux) = nothing

"Auxiliary nodes of a set of terms (identical arrays counted once)."
function aux_nodes(terms::Vector{Term})
    found = Aux[]
    for t in terms
        _walk_aux!(found, t.expr)
    end
    found
end
_walk_aux!(found, n::Aux) = any(a -> a.array === n.array, found) || push!(found, n)
_walk_aux!(found, n::Op) = foreach(c -> _walk_aux!(found, c), n.args)
_walk_aux!(found, n::Leaf) = nothing

function show_tree(io::IO, n::Node, depth=0)
    pad = "  "^depth
    if n isa Leaf
        println(io, pad, n.role, "[", n.field, "]", n.deriv == 1 ? " ∇" : "", " :: ", n.T)
    elseif n isa Aux
        println(io, pad, "aux :: ", eltype(eltype(n.array)))
    else
        println(io, pad, nameof_op(n.f))
        foreach(c -> show_tree(io, c, depth + 1), n.args)
    end
end
nameof_op(f::Function) = string(nameof(f))
nameof_op(f) = string(typeof(f))

function Base.show(io::IO, ::MIME"text/plain", f::Form)
    println(io, "Form on ", typeof(f.trian).name.name, " with ", length(f.terms), " term(s)")
    for t in f.terms
        println(io, "  coef = ", t.coef)
        show_tree(io, t.expr, 2)
    end
end

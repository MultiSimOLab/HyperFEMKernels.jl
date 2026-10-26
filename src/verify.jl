# Verification against Gridap.

"""
    check(res, jac, U, V, x; op = KernelOperator(res, jac, U, V))

Assemble residual and Jacobian at the free values `x` with Gridap and with `op`, and return the
relative differences `(res = ‖b_k - b_g‖/‖b_g‖, jac = ‖A_k - A_g‖/‖A_g‖)`.
"""
function check(res, jac, U, V, x::AbstractVector; op=KernelOperator(res, jac, U, V))
    gop = get_algebraic_operator(FEOperator(res, jac, U, V))
    bg = Algebra.residual(gop, x)
    Ag = Algebra.jacobian(gop, x)
    bk = Algebra.residual(op, x)
    Ak = Algebra.jacobian(op, x)
    rel(a, b) = norm(b) == 0 ? norm(a - b) : norm(a - b) / norm(b)
    (res=rel(bk, bg), jac=rel(Ak, Ag))
end

"""
    allocations(op, x) -> Vector{NamedTuple}

Bytes allocated by one call of the element residual and element Jacobian of each compiled group,
evaluated on the first cell at the free values `x` (after a warm-up call). Element kernels must
not allocate (requirement for GPU execution); a nonzero value points to the pointwise functions
of the weak form (e.g. a constitutive law that builds dense matrices).
"""
function allocations(op::KernelOperator, x::AbstractVector)
    dvals = [get_dirichlet_dof_values(s) for s in field_spaces(op.U)]
    out = NamedTuple{(:kind, :group, :bytes),Tuple{Symbol,Int,Int}}[]
    for (kind, groups) in ((:residual, op.rgroups), (:jacobian, op.jgroups)), (i, g) in enumerate(groups)
        push!(out, (kind=kind, group=i, bytes=_element_allocations(kind, op, g, x, dvals)))
    end
    out
end

function _element_allocations(kind::Symbol, op::KernelOperator, g::Group{NSV,NSU,PF,TT,AUX,SV,SU,D}, x, dvals) where {NSV,NSU,PF,TT,AUX,SV,SU,D}
    auxd = materialize_aux(g, 1)
    xl = zeros(op.ndl)
    gather_cell!(xl, op.fields, x, dvals, g.cellmap[1])
    dNb = map(t -> zeros(t.nn, wcols(t)), g.tables)
    if kind === :residual
        re = zeros(op.ndl)
        if MATRIX_PRODUCTS && contiguous(g)
            Cb, Mb = rm_buffers(g)
            element_residual_mm!(re, g, xl, auxd, 1, Cb, Mb)
            return @allocated element_residual_mm!(re, g, xl, auxd, 1, Cb, Mb)
        end
        element_residual!(re, g, xl, auxd, 1)
        return @allocated element_residual!(re, g, xl, auxd, 1)
    else
        KeT = zeros(op.ndl, op.ndl)
        element_jacobian!(KeT, g, xl, auxd, 1, dNb)
        return @allocated element_jacobian!(KeT, g, xl, auxd, 1, dNb)
    end
end

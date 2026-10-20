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

# Generic fused element kernels.
#
# Planned content (phase 1):
# - `element_kernel!` dispatched on the table (dimension, nodes, points) and on
#   the model;
# - one pass per quadrature point: kinematics and constitutive law evaluated
#   once, then contracted with all test/trial basis functions;
# - flux extraction by linearity: the residual integrand is linear in the test
#   function, r = v⋅A + ∇v⊙B; A and B (and the Jacobian blocks) are obtained by
#   exact differentiation of the pointwise integrand, or taken directly from
#   the analytical derivatives provided by HyperFEM (∂ΨF, ∂ΨFF, ∂ΨE, ...).

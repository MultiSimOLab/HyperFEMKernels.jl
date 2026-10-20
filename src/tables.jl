# Per-cell-type reference tables generated from Gridap FESpaces and Measures.
#
# Planned content (phase 1):
# - shape-function values and reference gradients at quadrature points,
#   quadrature weights, geometric map from the geometric ReferenceFE;
# - local DOF layout from `get_dof_to_node` / `get_dof_to_comp`;
# - sizes (dimension, number of nodes, number of points, components) as type
#   parameters (StaticArrays) so that the compiler specialises the kernels.
# No hand-written element types: everything is derived from Gridap objects.

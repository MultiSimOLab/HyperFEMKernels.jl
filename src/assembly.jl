# Sparse assembly.
#
# Planned content (phase 1):
# - sparsity pattern and positions in `nonzeros(K)` precomputed once;
# - Dirichlet lifting inside the assembly loop;
# - per-thread buffers, no allocations inside the cell loop.

# Tests

`julia --project -e 'using Pkg; Pkg.test()'` (or `julia -t auto --project=benchmark dev/runtests.jl`
in the benchmark environment). Every test compares against Gridap or against explicit sums; the
tolerance is 1e-12 relative.

| File | What it checks |
|---|---|
| `common.jl` | shared problems (Poisson 2D, St Venant–Kirchhoff 3D, coupled u–φ) and helpers |
| `test_probes.jl` | probe leaves: roles, fields, value types, gradients, evaluation shapes |
| `test_tree.jl` | reading Gridap's lazy tree: terms, coefficients (sums, differences, scalings), leaves, auxiliary nodes, several measures |
| `test_codegen.jl` | slot layout and maps, generated pointwise integrand against a hand evaluation, CSE |
| `test_tables.jl` | geometry table (Jacobians, volume), field tables against Gridap's basis, dof layout, contiguity, cell dof matrix |
| `test_kernels.jl` | slot specs, shape-function weights and field values on an affine field, contractions against explicit sums, matrix-product path, `gemm_add!` tiles |
| `test_linearmaps.jl` | state-input detection, additive splitting, linear maps vs dual numbers, where the shortcut applies / falls back, `apply_maps` |
| `test_assembly.jl` | sparsity pattern and positions vs Gridap, colouring, threads vs serial, Algebra interface, Dirichlet re-read, `update`, auxiliary data and `cache_aux`, subdomains, matrix-product vs loop Jacobian, zero allocations |
| `test_errors.jl` | clear errors for unsupported inputs; `check` |
| `test_gridap_equality.jl` | end-to-end equality with Gridap on all problem types |

Set `HFK_DEBUG_LINEAR=1` to print why the linear-map shortcut is not used for a weak form.

# Tests

`julia --project -e 'using Pkg; Pkg.test()'`. `HFK_TEST_GROUP=hdiv` runs only `test_hdiv.jl`, `HFK_TEST_GROUP=core` everything else (default `all`); CI runs the two groups as separate jobs. Every test compares against Gridap or against explicit sums; the
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
| `test_fallback.jl` | hybrid fallback: unsupported operations (differential operators on products with the unknown) assembled by Gridap and added to the kernel vector/matrix, union of sparsity patterns, residual-only fallback, matrix-free, `update`; every case against pure Gridap |
| `test_boundary.jl` | boundary integrals in the engine: Neumann/Robin on quads and triangles (orders 1–3), whole boundary with several faces per cell (threads, colouring), normal vector and ∇u on Γ, 3D follower pressure load on hexahedra and tetrahedra, multi-field, volume fields on Γ as auxiliary data, matrix-free, `update`, zero allocations; every case against pure Gridap |
| `test_hdiv.jl` | H(div) (Raviart–Thomas) fields in the engine: Piola tables against Gridap's basis (values, gradients, dof signs; 2D/3D, n-cubes/simplices, orders 0–1), mixed Darcy RT_k–P_k/Q_k (orders 0–2, non-affine meshes), Biot u–W (Lagrangian u + RT W: Biot coupling through the divergences, mass couplings, deformation-dependent permeability, ∇W term; linear maps on/off, threads, `update`, zero allocations), matrix-product vs loop paths, boundary integrals with w ⋅ n on curved faces, matrix-free Jacobian (dense weights with the Piola geometry and the dof signs), skeleton terms with RT sent to the fallback with a clear reason; every case against pure Gridap |
| `test_cfd.jl` | weak form of the aorta CFD example (Oseen–BDF2 Navier–Stokes, P2–P1, SUPG/PSPG/LSIC, elementwise Laplacian of the previous step, backflow and Windkessel Robin terms on the outlet) on a small tetrahedral channel: backward Euler and BDF2, linear system with lifted Dirichlet values (A = J, b = −r(0)), data overwritten in place between assemblies, `update` after the first step; every case against pure Gridap |
| `test_skeleton.jl` | skeleton integrals in the engine: interior penalty on quads/triangles (orders 1–3), sparsity pattern with the cross blocks and facet positions, facet colouring and threads, `mean`/`jump` of vector fields with the normal on hexahedra and tetrahedra, the weighted jump of ∇u of third-medium contact (nonlinear weight on both sides, facet size as data, non-symmetric tangent, linear maps on/off), skeleton of a subdomain with the space on a sub-triangulation, multi-field, `update` (parameters, reuse of tables and positions, pattern recomputed when a skeleton appears), skeleton term only in the residual, matrix-free Jacobian with facets (scalar/vector, 2D/3D, multi-field, threads), errors; every case against pure Gridap, zero allocations |
| `test_subdomains.jl` | configuration of the hMRE + vacuum example: FE spaces on a sub-triangulation with integrals on its parts and different laws, the boundary of a subdomain as internal interface, an auxiliary FE function living on that interface, Dirichlet data on the interface changed in place; every case against pure Gridap |
| `test_states.jl` | internal variables updated by the engine (`update_states!`, `KernelState`): FE functions read from their dofs (values, gradients, transposes, several spaces, sub-triangulations, nonzero Dirichlet values), both storages, several states written, states read as inputs, constants, auxiliary data, fallback to Gridap's loop, threads, `KernelUpdater`, zero allocations; every case against Gridap's `update_state!` |
| `test_matrixfree.jl` | 1D Lagrange basis, tensor-product tables vs ND tables (Q1–Q3, 2D/3D, simplices rejected), sum-factorised interpolation/contraction vs dense, matrix-free Jacobian vs assembled (n-cubes and dense fallback, several states, non-symmetric forms) |
| `test_errors.jl` | clear errors for unsupported inputs; `check` |
| `test_gridap_equality.jl` | end-to-end equality with Gridap on all problem types |

Set `HFK_DEBUG_LINEAR=1` to print why the linear-map shortcut is not used for a weak form.

## Tests on HyperFEM's weak forms

`test/hyperfem/runtests.jl`: EM monolithic, hMRE with masks and auxiliary fields, the three staggered problems of the hMRE + vacuum example, and a
visco-hyperelastic dynamic problem with `CellState` internal variables, value+gradient test/trial
terms and a time-stepping comparison. They need the HyperFEM.jl package cloned
next to this package; the meshes are in `test/models` (see its README). The CI job `hyperfem` clones
HyperFEM.jl and runs `test/hyperfem/setup.jl` then the
tests; locally: `julia test/hyperfem/setup.jl` once, then
`julia -t auto --project=test/hyperfem test/hyperfem/runtests.jl` (or the benchmark environment).

HyperFEM's simulation tests (`test/SimulationsTests` of HyperFEM.jl, branch `kernel-backend`) are included at the
end of `test/hyperfem/runtests.jl`: nine simulations (HyperFEM's seven, a Turek DFG 2D-1 Navier-Stokes test and the
hMRE + vacuum staggered example reduced to half the potential and four steps), each one run with `:gridap` and
`:kernel`, reference values checked with both and the kernel compared with Gridap. They live in HyperFEM, their
final place; with a HyperFEM without the kernel backend they are skipped with a message. The CI job `hyperfem`
clones HyperFEM's `kernel-backend`.

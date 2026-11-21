# Introspection for the deterministic performance checks (test/test_perf_invariants.jl) and the performance gate
# (benchmark/gate): which path each group takes, and counters of the work done by operator-level calls. Not exported.
#
# The counters are incremented by operator-level calls (construction, `update`, pattern build, code generation,
# assemblies of the Gridap part), never inside the loops over cells or points.

const COUNTER_NAMES = (:update_fast, :update_full, :patterns_built, :codegen, :linear_maps_evaluated, :fallback_assemblies)
const COUNTERS = NamedTuple{COUNTER_NAMES}(ntuple(_ -> Threads.Atomic{Int}(0), length(COUNTER_NAMES)))

"Increment the counter `name` (one of `COUNTER_NAMES`)."
@inline bump!(name::Symbol) = (Threads.atomic_add!(getfield(COUNTERS, name), 1); nothing)

"""
    HyperFEMKernels.counters() -> NamedTuple

Work done since the last `reset_counters!()` (process-wide, all operators):

- `update_fast`, `update_full`: calls of `update` that rebound the groups (same structure) or compiled them again;
- `patterns_built`: sparsity patterns built (construction, `update` with a new pattern, first assembled use of a
  matrix-free operator);
- `codegen`: functions generated (element kernels, linear maps, state updaters);
- `linear_maps_evaluated`: values of the linear maps of a group evaluated again in `update` (one per group with
  linear maps and rebind);
- `fallback_assemblies`: assemblies of a part of the weak form by Gridap (hybrid fallback).
"""
counters() = NamedTuple{COUNTER_NAMES}(map(c -> c[], values(COUNTERS)))

"Set every counter of `counters()` to zero."
reset_counters!() = (foreach(c -> (c[] = 0), values(COUNTERS)); nothing)

"Function generated at run time from the expression `ex` (counted by `counters().codegen`)."
generated(ex) = (bump!(:codegen); RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, ex))

"""
    HyperFEMKernels.work_summary(op) -> NamedTuple

What the operator does, for checks that do not depend on timings:

- `groups`: one entry per compiled group, `(kind, group, domain, cells, points, path)` with `kind` `:residual` or
  `:jacobian`, `domain` `:volume`, `:boundary` or `:skeleton`, `cells` the integration cells (facets on a
  skeleton), `points` the quadrature points per cell and `path` `:linear_maps` (coefficients from linear maps of
  the state inputs), `:factorised` (Jacobian: test factor × trial factor, `factorforms.jl`) or `:duals` (dual
  numbers: one level for a residual, nested for a Jacobian);
- `fallback_residual`, `fallback_jacobian`: triangulations assembled by Gridap;
- `jacobian`: `:assembled` or `:matrixfree`;
- `nnz`, `pattern_bytes`: nonzeros of the sparsity pattern and bytes of the pattern and positions (`-1` when the
  pattern is not built, as for a matrix-free operator never assembled).
"""
function work_summary(op::KernelOperator)
    groups = NamedTuple{(:kind, :group, :domain, :cells, :points, :path),Tuple{Symbol,Int,Symbol,Int,Int,Symbol}}[]
    Dc = num_cell_dims(get_triangulation(field_spaces(op.U)[1]))
    for (kind, gs) in ((:residual, op.rgroups), (:jacobian, op.jgroups)), (i, g) in enumerate(gs)
        domain = g isa SkeletonGroup ? :skeleton : num_cell_dims(g.trian) == Dc ? :volume : :boundary
        cells = g isa SkeletonGroup ? length(g.facets) : length(g.cellmap)
        push!(groups, (kind=kind, group=i, domain=domain, cells=cells, points=g.geo.np, path=g.lin === nothing ? :duals : g.lin isa FactorForm ? :factorised : :linear_maps))
    end
    p = op.pattern
    built = p.A0 !== nothing
    pbytes = built ? Base.summarysize(p.A0) + sizeof(p.nzpos) + sum(sd -> sizeof(sd.nzpos), op.skel; init=0) : -1
    (groups=groups, fallback_residual=length(op.fallback.rtrians), fallback_jacobian=length(op.fallback.jtrians),
     jacobian=op.jacobian, nnz=built ? nnz(p.A0) : -1, pattern_bytes=pbytes)
end

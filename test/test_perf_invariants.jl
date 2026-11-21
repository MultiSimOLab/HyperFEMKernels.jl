# Deterministic performance checks (no timings): which path each compiled group takes, allocations of the
# assemblies, type stability of the element kernels and the work done by `update`. A change of path or a new
# allocation shows up here as a test failure (and in the diff of the expected tables), before any timing.
# Introspection: `HyperFEMKernels.work_summary(op)`, `HyperFEMKernels.counters()` (src/introspection.jl).
# Uses problems of other test files that run before this one: `oseen_channel` (test_cfd.jl), `product_gradient_problem`
# and `fallback_op` (test_fallback.jl).

paths(op) = [(g.kind, g.domain, g.path) for g in HK.work_summary(op).groups]

@testset "paths of the compiled groups" begin
    p = poisson_2d()
    @test paths(KernelOperator(p.res, p.jac, p.U, p.V)) == [(:residual, :volume, :linear_maps), (:jacobian, :volume, :linear_maps)]
    for p in (elasticity_3d(), coupled_3d())        # St Venant–Kirchhoff tangent: not linear in its state inputs
        @test paths(KernelOperator(p.res, p.jac, p.U, p.V)) == [(:residual, :volume, :linear_maps), (:jacobian, :volume, :duals)]
    end
    # Oseen step (SUPG/PSPG/LSIC, backflow, Robin outlet): the stabilised volume terms and the backflow term go by
    # dual numbers (products of state inputs), the Robin term by linear maps
    s = oseen_channel()
    op = KernelOperator(s.res, s.jac, s.U, s.V)
    @test paths(op) == [(:residual, :volume, :duals), (:residual, :boundary, :duals), (:residual, :boundary, :duals),
                        (:jacobian, :volume, :duals), (:jacobian, :boundary, :linear_maps), (:jacobian, :boundary, :duals)]
    w = HK.work_summary(op)
    @test w.fallback_residual == 0 && w.fallback_jacobian == 0 && w.jacobian === :assembled
    @test w.nnz == nnz(HK.sparsity_pattern(op)) && w.pattern_bytes > 0
    @test all(g -> g.cells > 0 && g.points > 0, w.groups)
    mf = KernelOperator(s.res, s.jac, s.U, s.V; jacobian=:matrixfree)
    @test HK.work_summary(mf).nnz == -1 && HK.work_summary(mf).jacobian === :matrixfree
end

@testset "allocations of repeated assemblies (1 thread): none, or a fixed budget" begin
    s = oseen_channel()
    # budgets (bytes per call, residual and Jacobian): 0 when every auxiliary array is filled by the engine; the
    # Oseen step reads data evaluated by Gridap at every assembly (the elementwise Laplacian of uⁿ, through the
    # discontinuous space, and τ(uⁿ, h)): 167 424 and 142 368 B measured (Julia 1.13, macOS, 2026-11-21); the budget
    # (+50 %) leaves room for other Julia versions and systems in CI and still catches a change of order
    cases = [((p.res, p.jac, p.U, p.V), (0, 0)) for p in (poisson_2d(), elasticity_3d(), coupled_3d())]
    push!(cases, ((s.res, s.jac, s.U, s.V), (250_000, 215_000)))
    for ((res, jac, U, V), (rb, jb)) in cases
        op = KernelOperator(res, jac, U, V; nthreads=1)
        x = randx(U)
        b = allocate_residual(op, x); A = allocate_jacobian(op, x)
        residual!(b, op, x); jacobian!(A, op, x)
        @test (@allocated residual!(b, op, x)) <= rb
        @test (@allocated jacobian!(A, op, x)) <= jb
        @test all(a -> a.bytes == 0, allocations(op, x))      # the element kernels themselves never allocate
    end
end

@testset "type-stable element kernels" begin
    for p in (poisson_2d(), elasticity_3d())
        op = KernelOperator(p.res, p.jac, p.U, p.V)
        x = randx(p.U)
        dvals = HK.refresh_dirichlet_values!(op)
        for g in op.rgroups
            auxd = HK.materialize_aux(g, 1); xl = zeros(op.ndl)
            HK.gather_cell!(xl, op.fields, x, dvals, g.cellmap[1])
            @test (@inferred HK.element_residual!(zeros(op.ndl), g, xl, auxd, 1)) isa Vector{Float64}
        end
        for g in op.jgroups
            auxd = HK.materialize_aux(g, 1); xl = zeros(op.ndl)
            HK.gather_cell!(xl, op.fields, x, dvals, g.cellmap[1])
            dNb = map(t -> zeros(t.nn, HK.wcols(t)), g.tables)
            @test (@inferred HK.element_jacobian!(zeros(op.ndl, op.ndl), g, xl, auxd, 1, dNb)) isa Matrix{Float64}
        end
    end
end

@testset "update: same structure takes the fast path (no code generated, no pattern built)" begin
    s = oseen_channel()
    op = KernelOperator(s.res, s.jac, s.U, s.V)
    nlin = count(g -> g.lin !== nothing, vcat(op.rgroups, op.jgroups))
    HK.reset_counters!()
    op2 = update(op, s.res, s.jac)
    c = HK.counters()
    @test c.update_fast == 1 && c.update_full == 0 && c.codegen == 0 && c.patterns_built == 0
    @test c.linear_maps_evaluated == nlin              # the values of the maps are evaluated again, exactly
    @test op2.pattern === op.pattern
    HK.reset_counters!(); s.Pc[] = 0.9                  # a number of the form: still the fast path
    op3 = update(op2, s.res, s.jac)
    c = HK.counters()
    @test c.update_fast == 1 && c.codegen == 0 && c.patterns_built == 0
    HK.reset_counters!(); s.bdf2[] = true               # another advecting velocity (a new operation): full path
    op4 = update(op3, s.res, s.jac)
    c = HK.counters()
    @test c.update_full == 1 && c.update_fast == 0 && c.codegen > 0 && c.patterns_built == 0   # same blocks: pattern kept
    @test op4.pattern === op.pattern
end

@testset "counters: construction, matrix-free operators, Gridap-assembled part" begin
    p = elasticity_3d()
    HK.reset_counters!()
    op = KernelOperator(p.res, p.jac, p.U, p.V)
    c = HK.counters()
    @test c.patterns_built == 1 && c.codegen > 0 && c.update_fast == 0 && c.update_full == 0
    HK.reset_counters!()
    mf = KernelOperator(p.res, p.jac, p.U, p.V; jacobian=:matrixfree)
    @test HK.counters().patterns_built == 0
    x = randx(p.U); b = allocate_residual(op, x); A = allocate_jacobian(op, x)
    HK.reset_counters!()
    residual!(b, op, x); jacobian!(A, op, x)
    @test HK.counters() == (update_fast=0, update_full=0, patterns_built=0, codegen=0, linear_maps_evaluated=0, fallback_assemblies=0)
    # a weak form with a Gridap-assembled part: one fallback assembly per residual and per Jacobian
    s = product_gradient_problem(; order=1)
    fop = fallback_op(KernelOperator, s.res, s.jac, s.p.U, s.p.V)
    xs = randx(s.p.U); bs = allocate_residual(fop, xs); As = allocate_jacobian(fop, xs)
    HK.reset_counters!()
    residual!(bs, fop, xs); jacobian!(As, fop, xs)
    @test HK.counters().fallback_assemblies == 2
end

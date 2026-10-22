# Where the time goes in the hMRE + vacuum example, and what allocates in its kernels.
#   julia -t auto --project=benchmark dev/prof_hmre_vacuum.jl
using Printf
const HV_ARGS = copy(ARGS)
empty!(ARGS); append!(ARGS, ["mode=kernel", "nsteps=1", "nsubsteps=1"])
include(joinpath(@__DIR__, "..", "examples", "hmre_vacuum_2d", "hmre_vacuum_2d.jl"))
const HK = HyperFEMKernels
best(f; n=3) = (f(); minimum(@elapsed(f()) for _ in 1:n))

s = build(:kernel)
args = (Dict(:stepping => (nsteps=1, maxbisec=5), :ProjectDirichlet => true), Dict(:stepping => (nsteps=1, maxbisec=5)),
        Dict(:stepping => (nsteps=1, maxbisec=5), :ProjectDirichlet => true))
solve!(s.comp_model; stepping=(nsteps=2, nsubsteps=1, maxbisec=1), kargsolve=args, verbose=false)   # a deformed state

println("\nPer problem at a perturbed deformed state (best of 3):")
for p in s.problems
    x = get_free_dof_values(p.xh) .* (1 .+ 1e-3 .* randn(num_free_dofs(p.U)))
    res, jac = p.res(1.0), p.jac(1.0)
    tcap = @elapsed (op = KernelOperator(res, jac, p.U, p.V))
    tupd = best(() -> HK.update(op, p.res(1.0), p.jac(1.0)))
    tres_fun = best(() -> p.res(1.0))                        # the residual builder itself (side effects in res_mech)
    gop = get_algebraic_operator(FEOperator(res, jac, p.U, p.V))
    b = Gridap.Algebra.allocate_residual(op, x); A = Gridap.Algebra.allocate_jacobian(op, x)
    bg = Gridap.Algebra.allocate_residual(gop, x); Ag = Gridap.Algebra.allocate_jacobian(gop, x)
    tgr = best(() -> Gridap.Algebra.residual!(bg, gop, x)); tgj = best(() -> Gridap.Algebra.jacobian!(Ag, gop, x))
    tkr = best(() -> Gridap.Algebra.residual!(b, op, x)); tkj = best(() -> Gridap.Algebra.jacobian!(A, op, x))
    c = check(res, jac, p.U, p.V, x; op)
    @printf("  %-17s %6d DOFs | Gridap res %.4f jac %.4f | kernel res %.4f jac %.4f | update %.4f (first build %.1f) | res builder %.4f | check res %.1e jac %.1e\n",
            p.name, num_free_dofs(p.U), tgr, tgj, tkr, tkj, tupd, tcap, tres_fun, c.res, c.jac)
    for (kind, groups) in ((:res, op.rgroups), (:jac, op.jgroups)), (i, g) in enumerate(groups)
        a = HK._element_allocations(kind === :res ? :residual : :jacobian, op, g, x, [get_dirichlet_dof_values(u) for u in HK.field_spaces(op.U)])
        @printf("      %s group %d on %-26s %s, %d bytes/element\n", kind, i, nameof(typeof(g.trian)), g.lin === nothing ? "dual" : "lin ", a)
        if a > 0 && g.lin !== nothing
            auxd = HK.materialize_aux(g, 1)
            FVs = map(t -> map(T -> 0.01 * HK.rand_value(T), HK.field_types(t)), g.tables)
            Aq = map(m -> m[1, 1], auxd)
            g.lin.state(g.lin.ops, FVs, Aq)
            @printf("         state function: %d bytes; ops: %s\n", @allocated(g.lin.state(g.lin.ops, FVs, Aq)),
                    join(unique(string.(nameof.(filter(o -> o isa Function, collect(g.lin.ops))))), ", "))
        end
    end
end

# constitutive functions of the example, one call each with typical arguments
println("\nConstitutive functions (bytes per call):")
μ0 = 4π * 1e-7
F2 = TensorValue(1.01, 0.02, -0.01, 0.99); H0 = VectorValue(100.0, -50.0); N = VectorValue(1.0, 0.0)
params = [10.0, 12480.64495232286, 1.0, 2.0, 5195.545287237134, 0.2602717127043121, 2.0]
modelmech = NonlinearMooneyRivlin2D(λ=(params[1] + params[2]) * 1e2, μ1=params[1] * 1e3, μ2=params[2], α1=params[3], α2=params[4]) +
            TransverseIsotropy2D(μ=params[5], α1=params[6], α2=params[7])
models = (("MagnetoMech (HardMagnetic2D + MR + TI)", MagnetoMechModel(magneto=HardMagnetic2D(μ0=μ0, αr=10e-3 / μ0, χe=0.0, χr=8.0), mechano=modelmech), (F2, H0, N)),
          ("IdealMagnetic2D", IdealMagnetic2D(μ0=μ0, χe=0.0), (F2, H0)),
          ("HessianRegularization(MR2D_CV)", HessianRegularization(mechano=NonlinearMooneyRivlin2D_CV(λ=1.0, μ1=1.0, μ2=0.0, α1=6.0, α2=1.0, γ=6.0), δ=1e-6), (F2,)),
          ("MR2D + TI2D", modelmech, (F2, N)))
for (name, m, a) in models
    fs = m()
    out = String[]
    for (k, f) in enumerate(fs)
        try
            f(a...); push!(out, "$k:$(@allocated f(a...))")
        catch err
            push!(out, "$k:n/a")
        end
    end
    println("  ", rpad(name, 40), join(out, "  "))
end
println("FIN PROF HV")

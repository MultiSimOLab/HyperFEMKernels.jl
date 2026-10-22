# Cost of updating the Dirichlet values (TrialFESpace!) of each space of the hMRE + vacuum example.
using Printf
empty!(ARGS); append!(ARGS, ["mode=none"])
include(joinpath(@__DIR__, "..", "examples", "hmre_vacuum_2d", "hmre_vacuum_2d.jl"))
s = build(:gridap)
cm = s.comp_model.compmodels
best(f; n=3) = (f(); minimum(@elapsed(f()) for _ in 1:n))
for (name, m) in zip(("magnetic", "solid", "vacuum"), cm)
    U = m.spaces[1]; D = m.dirichlet
    t = best(() -> TrialFESpace!(U, D, 0.5))
    tags = HyperFEM.get_tags(D)
    @printf("%-9s %6d free, %5d Dirichlet dofs, tags %-45s TrialFESpace!: %.3f s  (%d bytes)\n", name, num_free_dofs(U),
            num_dirichlet_dofs(U), string(tags), t, @allocated(TrialFESpace!(U, D, 0.5)))
    # same values with Gridap's interpolation of plain constants (no HyperFEM prescriptions), for reference
    if name != "vacuum"
        vals = [x -> VectorValue(0.0, 0.0) for _ in tags]
        name == "magnetic" && (vals = [x -> 0.5 * 1040.0, x -> 0.0])
        tg = best(() -> TrialFESpace!(U, vals))
        @printf("          plain Gridap TrialFESpace! with Julia functions: %.3f s\n", tg)
    end
end
println("FIN BC")

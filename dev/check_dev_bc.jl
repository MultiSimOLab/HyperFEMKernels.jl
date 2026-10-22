# With HyperFEM 533b92c (both interfaces): the conditions of the dev scripts written with DirichletBC
# and with BoundaryConditions must give the same spaces and Dirichlet values (bit-identical).
using HyperFEM, Gridap, GridapGmsh, Gridap.FESpaces
K = normpath(joinpath(@__DIR__, ".."))
ok = true
function cmp(name, model, reffe, old, new; masks=nothing)
    Vo = masks === nothing ? TestFESpace(model, reffe, old, conformity=:H1) : TestFESpace(model, reffe, old, conformity=:H1, dirichlet_masks=masks)
    Vn = TestFESpace(model, reffe, new, conformity=:H1)
    same = num_free_dofs(Vo) == num_free_dofs(Vn) && get_dirichlet_dof_tag(Vo) == get_dirichlet_dof_tag(Vn)
    for Λ in (0.0, 0.3, 1.0)
        same &= get_dirichlet_dof_values(TrialFESpace(Vo, old, Λ)) == get_dirichlet_dof_values(TrialFESpace(Vn, new, Λ))
    end
    global ok &= same
    println(rpad(name, 32), same ? "IDENTICAL" : "DIFFERENT", "  (free $(num_free_dofs(Vn)), dirichlet $(num_dirichlet_dofs(Vn)))")
end
em = GmshDiscreteModel(joinpath(K, "test", "models", "test_static_EM.msh"))
cmp("EM 3D u (fixedup)", em, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1),
    DirichletBC(["fixedup"], [[0.0, 0.0, 0.0]], [Λ -> 1.0]),
    BoundaryConditions(BoundaryCondition("fixedup", [0.0, 0.0, 0.0])))
cmp("EM 3D φ (midsuf, topsuf)", em, ReferenceFE(lagrangian, Float64, 1),
    DirichletBC(["midsuf", "topsuf"], [0.0, 0.1], [Λ -> Λ, Λ -> Λ]),
    BoundaryConditions(BoundaryCondition("midsuf", 0.0, Λ -> Λ), BoundaryCondition("topsuf", 0.1, Λ -> Λ)))
sm = GmshDiscreteModel(joinpath(K, "test", "models", "Short_magnet.msh"))
cmp("hMRE 2D solid (masks)", Triangulation(sm, tags=["Beam"]), ReferenceFE(lagrangian, VectorValue{2,Float64}, 1),
    DirichletBC(["Fixed_Solid", "Fixed_Solid_x"], [[0.0, 0.0], [0.0, 0.0]]),
    BoundaryConditions(BoundaryCondition("Fixed_Solid", [0.0, 0.0]), BoundaryCondition("Fixed_Solid_x", [0.0, 0.0]; mask=(true, false)));
    masks=[(true, true), (true, false)])
for f in ("bench1.jl", "bench_cases.jl", "m1.jl", "prof_hmre.jl", "prof_hmre_vacuum_solve.jl")
    Meta.parseall(read(joinpath(K, "dev", f), String)); println("parses: dev/", f)
end
println(ok ? "ALL IDENTICAL" : "SOME DIFFER")

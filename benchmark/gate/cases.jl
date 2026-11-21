# Cases of the performance gate (benchmark/gate/README.md), loaded by worker.jl in each measuring process.
#
# A case is one call measured with both backends when it has a Gridap counterpart: `gridap` (lazy evaluation, the
# backend HyperFEM uses without compilation) and `kernel` (HyperFEMKernels). Groups of cases share their problem
# and are built together; `build_group!(name)` registers the cases of one group in `CASES`.
# Both processes of a comparison (origin/main and the branch) load this file from the branch, so that they solve
# the same problems; a case must therefore only use the public API, or check with `isdefined` what it reads.
#
# kind: :call (one assembly-level call: residual!, jacobian!, a pattern build...) or :simulation (a whole run).
# exact_bytes: the bytes of a call are deterministic (assemblies, products): the branch may allocate at most 256 B
# more; otherwise (builds, updates, simulations) a relative margin applies (stats.jl).
# value: a number of the result (a norm), compared between main and the branch: the optimisations under test must
# not change results beyond rounding.
using HyperFEMKernels, HyperFEM, Gridap, GridapGmsh, LinearAlgebra, SparseArrays, Printf, Random
using Gridap.FESpaces, Gridap.Algebra, Gridap.CellData
const HK = HyperFEMKernels
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

struct Case
    id::String
    kind::Symbol
    exact_bytes::Bool
    gridap::Union{Nothing,Function}
    kernel::Function
    value::Function            # () -> Float64 after a call of `kernel` (NaN: not compared)
    info::Dict{String,Any}     # deterministic facts (nnz, sizes, paths), reported side by side
end
const CASES = Dict{String,Case}()
const ORDER = String[]
function add!(id, kind, gridap, kernel; value=() -> NaN, info=Dict{String,Any}(), exact=false)
    CASES[id] = Case(id, kind, exact, gridap, kernel, value, info)
    push!(ORDER, id)
end
"Deterministic facts of a kernel operator (what `HyperFEMKernels.work_summary` reports, when this version has it)."
function op_info(op)
    d = Dict{String,Any}("bytes" => Base.summarysize(op))
    if isdefined(HK, :work_summary)
        w = HK.work_summary(op)
        d["nnz"] = w.nnz
        d["paths"] = join(("$(g.kind == :residual ? 'r' : 'j')$(g.group):$(g.domain):$(g.path)" for g in w.groups), " ")
        d["fallback"] = w.fallback_residual + w.fallback_jacobian
    elseif op.jacobian === :assembled
        d["nnz"] = nnz(HK.sparsity_pattern(op))
    end
    d
end
nrm(x) = norm(x)
"A global of module `m` defined after this code was compiled (an included file, a script): read in the newest world."
latest(m::Module, s::Symbol) = Base.invokelatest(getglobal, m, s)

# --- HyperFEM's simulation tests ----------------------------------------------------------------------------
const SIMDIR = joinpath(pkgdir(HyperFEM), "test", "SimulationsTests")
const SIMS = [("ViscoElastic", :visco_elastic_simulation, r -> r[2][end]),
              ("StaticMechanicalDirichlet", :static_mechanical_dirichlet_simulation, r -> norm(r)),
              ("StaticMechanicalNeumann", :static_mechanical_neumann_simulation, r -> norm(r[1])),
              ("StaggeredElectroMechanical", :staggered_electro_mechanical_simulation, r -> norm(r[2][1])),
              ("StaggeredViscoElectric", :staggered_visco_electric_simulation, r -> norm(r[2])),
              ("Stokes", :stokes_simulation, r -> norm(r)),
              ("MeshMovement", :mesh_movement_simulation, r -> norm(r)),
              ("TurekNavierStokes", :turek_navier_stokes_simulation, r -> r.cD),
              ("HMREVacuum", :hmre_vacuum_simulation, r -> norm(r.u_solid))]
function group_sims()
    for (name, fname, val) in SIMS
        if !isdefined(Main, fname)
            Base.include(Main, joinpath(SIMDIR, name * "Simulation.jl"))
        end
        sim = latest(Main, fname)
        last = Ref{Any}(nothing)
        add!("sims/$name", :simulation, () -> sim(backend=:lazy), () -> (last[] = sim(backend=:compiled));
             value=() -> Float64(val(last[])))
    end
end

# --- EM monolithic Q2 box and P1 plate (benchmark/bench_cases.jl) --------------------------------------------
function em_problem(model, order)
    physmodel = ElectroMechModel(IdealDielectric(ε=1.0), NeoHookean3D(λ=10.0, μ=1.0))
    k = (Kinematics(Mechano, Solid), Kinematics(Electro, Solid))
    Ω = Triangulation(model); dΩ = Measure(Ω, 2order)
    Du = BoundaryConditions(BoundaryCondition("fixedup", [0.0, 0.0, 0.0]))
    Dφ = BoundaryConditions(BoundaryCondition("midsuf", 0.0, Λ -> Λ), BoundaryCondition("topsuf", 0.1, Λ -> Λ))
    Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, order), Du, conformity=:H1)
    Vφ = TestFESpace(model, ReferenceFE(lagrangian, Float64, order), Dφ, conformity=:H1)
    V = MultiFieldFESpace([Vu, Vφ])
    U = MultiFieldFESpace([TrialFESpace(Vu, Du, 1.0), TrialFESpace(Vφ, Dφ, 1.0)])
    res((u, φ), (v, vφ)) = HyperFEM.residual(physmodel, k, (u, φ), (v, vφ), dΩ)
    jac((u, φ), (du, dφ), (v, vφ)) = HyperFEM.jacobian(physmodel, k, (u, φ), (du, dφ), (v, vφ), dΩ)
    res, jac, U, V
end
"Residual and Jacobian cases of a weak form at a random state, Gridap and kernel; the kernel operator."
function assembly_cases!(prefix, res, jac, U, V, x)
    gop = get_algebraic_operator(FEOperator(res, jac, U, V))
    kop = KernelOperator(res, jac, U, V)
    bg = allocate_residual(gop, x); Ag = allocate_jacobian(gop, x)
    bk = allocate_residual(kop, x); Ak = allocate_jacobian(kop, x)
    info = op_info(kop)
    add!("$prefix/residual", :call, () -> residual!(bg, gop, x), () -> residual!(bk, kop, x); value=() -> nrm(bk), info, exact=true)
    add!("$prefix/jacobian", :call, () -> jacobian!(Ag, gop, x), () -> jacobian!(Ak, kop, x); value=() -> nrm(nonzeros(Ak)), info, exact=true)
    kop, Ak
end
function em_box_model(nx, ny, nz)
    m = CartesianDiscreteModel((0.0, 0.1, 0.0, 0.01, 0.0, 0.002), (nx, ny, nz))
    labels = get_face_labeling(m)
    add_tag_from_tags!(labels, "fixedup", CartesianTags.face0YZ⁺)
    add_tag_from_tags!(labels, "topsuf", CartesianTags.faceXY1⁺)
    add_tag_from_vertex_filter!(labels, "midsuf", m, x -> abs(x[3] - 0.001) < 1e-9)
    m
end
function group_em()
    res, jac, U, V = em_problem(em_box_model(20, 4, 4), 2)
    x = 0.01 .* randn(MersenneTwister(1), num_free_dofs(U))
    kop, Ak = assembly_cases!("em_box_q2", res, jac, U, V, x)
    holder = Ref{Any}(kop)
    add!("em_box_q2/build", :call, nothing, () -> (holder[] = KernelOperator(res, jac, U, V)))
    add!("em_box_q2/update", :call, nothing, () -> update(kop, res, jac))
    # matrix-free product (prepare at the state, then apply)
    mf = MatrixFreeJacobian(kop); prepare!(mf, x); dx = randn(MersenneTwister(2), length(x)); y = mf * dx
    add!("em_box_q2/matrixfree_prepare", :call, nothing, () -> prepare!(mf, x); exact=true)
    add!("em_box_q2/matrixfree_apply", :call, nothing, () -> mul!(y, mf, dx); value=() -> nrm(y), exact=true)
    add!("em_box_q2/spmv", :call, nothing, () -> mul!(y, Ak, dx); exact=true)
    file = joinpath(ROOT, "benchmark", "meshes", "em_plate_n2.msh")
    isfile(file) || run(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $(joinpath(ROOT, "benchmark", "meshes", "generate.jl")) em 2`)
    rp, jp, Up, Vp = em_problem(GmshDiscreteModel(file), 1)     # other names: the closures above capture res, jac
    assembly_cases!("em_plate_p1", rp, jp, Up, Vp, 0.01 .* randn(MersenneTwister(1), num_free_dofs(Up)))
end

# --- visco-hyperelastic dynamic weak form with internal variables (benchmark/bench_visco.jl, bench_states.jl) --
function visco_problem()
    width, thick, xdiv, order = 0.2, 0.001, 7, 2
    geometry = CartesianDiscreteModel((-0.5width, 0.5width, -0.5width, 0.5width, 0.0, thick), (xdiv, xdiv, 1))
    model = GeneralizedMaxwell(NeoHookean3D(μ=13446.1, λ=0.0) + VolumetricEnergy(λ=2.5e6),
                               ViscousPolyconvex(μ=11053.7, τ=151.68), ViscousPolyconvex(μ=30480.1, τ=7.0731))
    F, _, _ = get_Kinematics(Kinematics(Mechano, Solid))
    Δt = 0.2 * thick / (order * 0.1)
    Ω = Triangulation(geometry); dΩ = Measure(Ω, 2order)
    Vu = TestFESpace(geometry, ReferenceFE(lagrangian, VectorValue{3,Float64}, order), conformity=:H1)
    Uu = TrialFESpace(Vu)
    rng = MersenneTwister(1)
    uh⁺ = FEFunction(Uu, 1e-5 .* randn(rng, num_free_dofs(Uu))); uh⁻ = FEFunction(TrialFESpace(Vu), 1e-5 .* randn(rng, num_free_dofs(Uu)))
    υh = FEFunction(TrialFESpace(Vu), 1e-3 .* randn(rng, num_free_dofs(Uu)))
    update_time_step!(model, Δt)
    (; model, F, Δt, dΩ, Uu, Vu, uh⁺, uh⁻, υh, rng)
end
function group_visco()
    p = visco_problem()
    (; model, F, Δt, dΩ, Uu, Vu, uh⁻, υh) = p
    Fh = F ∘ ∇(p.uh⁺)'; Fh⁻ = F ∘ ∇(uh⁻)'
    Ah = CellState(model, dΩ); Ph⁻ = CellState(zero(TensorValue{3,3,Float64}), dΩ)
    Ψ, ∂Ψ∂F, ∂∂Ψ∂FF = model()
    P_updater(P, F, Fn, A...) = (true, ∂Ψ∂F(F, Fn, A...))
    update_state!(P_updater, Ph⁻, Fh, Fh⁻, Ah...); update_state!(model, Ah, Fh, Fh⁻)
    ρ₀ = 960.0
    res(u, v) = ∫(ρ₀ * (2 / Δt^2 * (u - uh⁻) - 2 / Δt * υh) ⋅ v + ∇(v)' ⊙ (1 / 2 * (∂Ψ∂F ∘ (F ∘ ∇(u)', Fh⁻, Ah...) + Ph⁻)))dΩ
    jac(u, du, v) = ∫(ρ₀ * (2 / Δt^2 * du) ⋅ v + ∇(v)' ⊙ (1 / 2 * (∂∂Ψ∂FF ∘ (F ∘ ∇(u)', Fh⁻, Ah...)) ⊙ ∇(du)'))dΩ
    assembly_cases!("visco", res, jac, Uu, Vu, 1e-5 .* randn(p.rng, num_free_dofs(Uu)))
    # internal variables updated by Gridap (CellState) and by the engine (KernelState)
    T33 = TensorValue{3,3,Float64,9}
    branches = model.branches.branches
    branch_updater(b) = (A, F, Fn) -> (true, HyperFEM.return_mapping(b, F, Fn, A))
    Ahg = CellState(model, dΩ); Pg = CellState(zero(T33), dΩ)
    xq = get_cell_points(dΩ)
    Ak = Tuple(KernelState(one(T33), xq) for _ in branches); Pk = KernelState(zero(T33), xq)
    kernel!() = (update_states!(P_updater, Pk, Fh, Fh⁻, Ak...); foreach((b, A) -> update_states!(branch_updater(b), A, Fh, Fh⁻), branches, Ak))
    # the states advance at every call (return mapping from the previous state): no value compared (the number of
    # calls differs between processes)
    add!("visco/update_states", :call, () -> (update_state!(P_updater, Pg, Fh, Fh⁻, Ahg...); update_state!(model, Ahg, Fh, Fh⁻)), kernel!;
         exact=true)
end

# --- micro weak forms on 40³ hexahedra (residual only) ----------------------------------------------------------
form_H(r) = (tanh(1.5) + tanh(3(r - 0.5))) / (2tanh(1.5))
function group_forms()
    n = 40
    model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (n, n, n)); Ω = Triangulation(model); dΩ = Measure(Ω, 2)
    V0 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 0); conformity=:L2); U0 = TrialFESpace(V0)
    V1 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity=:H1)
    uf = FEFunction(V1, rand(MersenneTwister(1), num_free_dofs(V1)))
    cfun = CellField(x -> 1 + x[1], Ω); cfe = interpolate(x -> 1 + x[1], V1); ρH = Operation(form_H)(uf)
    forms = [("uf_v", v -> ∫(uf * v)dΩ), ("cfe_uf_v", v -> ∫(2 * cfe * uf * v)dΩ), ("cfun_uf_v", v -> ∫(2 * cfun * uf * v)dΩ),
             ("cfe_H_v", v -> ∫(2 * cfe * ρH * v)dΩ), ("cfun_H_v", v -> ∫(2 * cfun * ρH * v)dΩ)]
    x0 = zeros(num_free_dofs(U0)); assem = SparseMatrixAssembler(U0, V0)
    for (name, L) in forms
        op = KernelOperator((u, v) -> L(v), (u, du, v) -> ∫(0.0 * du * v)dΩ, U0, V0; jacobian=:matrixfree)
        bk = zeros(num_free_dofs(V0)); bg = zeros(num_free_dofs(V0))
        add!("forms/$name", :call, () -> assemble_vector!(L, bg, assem, V0), () -> residual!(bk, op, x0); value=() -> nrm(bk), exact=true)
    end
end

# --- sparsity pattern build (elasticity Q1 40³, u–p Q1–Q1 30³) --------------------------------------------------
function rebuild_pattern!(op)
    op.pattern.A0 = nothing
    HK.build_pattern!(op)
end
function group_pattern()
    m = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (40, 40, 40)); Ω = Triangulation(m); dΩ = Measure(Ω, 2)
    V = TestFESpace(m, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1); dirichlet_tags=[1]); U = TrialFESpace(V, VectorValue(0.0, 0.0, 0.0))
    op = KernelOperator((u, v) -> ∫(∇(v) ⊙ ∇(u))dΩ, (u, du, v) -> ∫(∇(v) ⊙ ∇(du))dΩ, U, V)
    add!("pattern/elasticity_q1_40", :call, nothing, () -> rebuild_pattern!(op); info=op_info(op))
    m2 = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (30, 30, 30)); Ω2 = Triangulation(m2); dΩ2 = Measure(Ω2, 2)
    Vu = TestFESpace(m2, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1); dirichlet_tags="boundary"); Vp = TestFESpace(m2, ReferenceFE(lagrangian, Float64, 1))
    X = MultiFieldFESpace([TrialFESpace(Vu, VectorValue(0.0, 0.0, 0.0)), TrialFESpace(Vp)]); Y = MultiFieldFESpace([Vu, Vp])
    op2 = KernelOperator(((u, p), (v, q)) -> ∫(∇(v) ⊙ ∇(u) - (∇ ⋅ v) * p + q * (∇ ⋅ u))dΩ2,
                         (_, (du, dp), (v, q)) -> ∫(∇(v) ⊙ ∇(du) - (∇ ⋅ v) * dp + q * (∇ ⋅ du))dΩ2, X, Y)
    add!("pattern/stokes_q1q1_30", :call, nothing, () -> rebuild_pattern!(op2); info=op_info(op2))
end

# --- examples: aorta P2–P1 (examples/aorta_cfd/aorta_flow.jl), hMRE + vacuum 2D (examples/hmre_vacuum_2d) -------
"""
Include the script `file` with `args` as its command-line arguments into a new module (with its own `include`, for
the files the script includes); returns the module.
"""
function include_script(name::Symbol, file, args)
    saved = copy(ARGS)
    empty!(ARGS); append!(ARGS, args)
    m = Module(name)
    Core.eval(m, :(include(x) = Base.include($m, x)))
    try
        Base.include(m, file)
    finally
        empty!(ARGS); append!(ARGS, saved)
    end
    m
end
function group_aorta_p2p1()
    # a copy of the example in a temporary folder (it writes results/ next to itself), with a link to its mesh; three
    # steps, so that the form is the BDF2 one of every later step for Gridap and for the operator
    src = joinpath(ROOT, "examples", "aorta_cfd")
    tmp = mktempdir()
    cp(joinpath(src, "aorta_flow.jl"), joinpath(tmp, "aorta_flow.jl"))
    for f in readdir(src)
        endswith(f, ".msh") && symlink(joinpath(src, f), joinpath(tmp, f))
    end
    M = include_script(:AortaP2P1, joinpath(tmp, "aorta_flow.jl"), ["mode=compare", "steps=3", "threads1=off"])
    g(s) = latest(M, s)
    Ag, bg, assem, gdata, kres, kjac = g(:Ag), g(:bg), g(:assem), g(:gridap_data), g(:kres), g(:kjac)
    op, rk, Ak, x0 = g(:op), g(:rk), g(:Ak), g(:x0)
    add!("aorta_p2p1/assembly", :call, () -> assemble_matrix_and_vector!(Ag, bg, assem, Base.invokelatest(gdata)),
         () -> (residual!(rk, op, x0); jacobian!(Ak, op, x0)); value=() -> nrm(nonzeros(Ak)), info=op_info(op), exact=true)
    add!("aorta_p2p1/update", :call, nothing, () -> HK.update(op, kres, kjac))      # same structure: the fast path
end
function group_hmre2d()
    M = include_script(:HMREVacuum2D, joinpath(ROOT, "examples", "hmre_vacuum_2d", "hmre_vacuum_2d.jl"), ["mode=none"])
    last = Ref{Any}(nothing)
    sim = latest(M, :simulate)
    add!("hmre_vacuum_2d/run", :simulation, () -> Base.invokelatest(sim, :lazy; nsteps=2, nsubsteps=1, quiet=true),
         () -> (last[] = Base.invokelatest(sim, :compiled; nsteps=2, nsubsteps=1, quiet=true)[1]);
         value=() -> norm(get_free_dof_values(last[].problems[2].xh)))
end

# --- aorta P1–P1 through HyperFEM's TransientModel (phases of a step) ---------------------------------------------
include(joinpath(@__DIR__, "aorta_transient.jl"))
function group_aorta_p1p1()
    g, k = aorta_p1p1_operator(:lazy), aorta_p1p1_operator(:compiled)
    add!("aorta_p1p1/jacobian", :call, () -> jacobian!(g.A, g.op, g.y), () -> jacobian!(k.A, k.op, k.y); value=() -> nrm(nonzeros(k.A)),
         info=k.info, exact=true)
    add!("aorta_p1p1/residual", :call, () -> residual!(g.b, g.op, g.y), () -> residual!(k.b, k.op, k.y); value=() -> nrm(k.b), exact=true)
    # LU of the kernel's matrix (its pattern decides the fill): against main only, Gridap's LU is the same code
    add!("aorta_p1p1/lu", :call, nothing, () -> lu(k.A))
end

include(joinpath(@__DIR__, "groups.jl"))
const GROUPS = [("sims", group_sims), ("em", group_em), ("visco", group_visco), ("forms", group_forms),
                ("pattern", group_pattern), ("aorta_p1p1", group_aorta_p1p1), ("aorta_p2p1", group_aorta_p2p1),
                ("hmre_vacuum_2d", group_hmre2d)]
@assert [g for (g, _) in GROUPS] == GROUP_NAMES

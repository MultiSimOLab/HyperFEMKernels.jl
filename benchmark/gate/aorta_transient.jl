# Aorta P1–P1, stabilised Navier–Stokes (Oseen, BDF2) with RCR Windkessel outlets and backflow stabilisation,
# written on HyperFEM's TransientModel (as in the tutorial being prepared; _work/aorta-p1-1thread/scratch/wk_p1.jl).
# `aorta_p1p1_operator(backend)` runs three steps and returns the operator the driver holds, with its matrix and
# vectors, to time the phases of a step (Jacobian, residual, LU) with each backend.
const AORTA_MESH = joinpath(ROOT, "examples", "aorta_cfd", "aorta_fluido_h1p5_3p0.msh")
const ρ_blood = 1050e-6; const μ_blood = 3.5e-3          # g/mm³, Pa s (constants: read by the pointwise functions)
τsupg_p1(u, h) = ((2 * norm(u) / h)^2 + 9 * (4μ_blood / (ρ_blood * h^2))^2)^(-1 / 2)
τpspg_p1(u, h) = τsupg_p1(u, h) / ρ_blood
τlsic_p1(u, h) = ρ_blood * norm(u) * h * 0.5
εsym(∇u) = 0.5 * (∇u + ∇u')
negpart_p1(x) = 0.5 * (x - abs(x))
function close_tags_p1!(model, names)
    labels = get_face_labeling(model); topo = Gridap.Geometry.get_grid_topology(model); D = num_cell_dims(model)
    for name in names
        tag = findfirst(==(name), labels.tag_to_name)
        faces = findall(Gridap.Geometry.get_face_mask(labels, name, D - 1))
        for d in 0:D-2
            f_to_d = Gridap.Geometry.get_faces(topo, D - 1, d); ents = labels.d_to_dface_to_entity[d+1]
            newent = maximum(maximum(e) for e in labels.d_to_dface_to_entity) + 1
            for f in faces, e in f_to_d[f]; ents[e] = newent; end
            push!(labels.tag_to_entities[tag], newent)
        end
    end
    model
end
function aorta_p1p1_operator(backend::Symbol; nsteps=3)
    ρ, μ = ρ_blood, μ_blood
    model = GmshDiscreteModel(AORTA_MESH)
    close_tags_p1!(model, ["inlet", "wall"])
    T, Ts, Vs = 1.0, 0.34, 83.3e3; Qmax = π * Vs / (2Ts)
    Qin = t -> (τ = mod(t, T); τ <= Ts ? Qmax * sin(π * τ / Ts) : 0.0)
    outs = ["outlet", "B1", "B2", "B3"]; SIR, SIC = 1e-9, 1e9
    Z = [1.5e7, 1.36e8, 2.46e8, 1.74e8] .* SIR; C = [1.05e-8, 4.83e-10, 2.92e-10, 3.93e-10] .* SIC; R = [2.5e8, 9.23e8, 1.53e9, 1.13e9] .* SIR
    Ω = Triangulation(model); dΩ = Measure(Ω, 3)
    he = CellField(map(v -> (6sqrt(2) * v)^(1 / 3), collect(get_cell_measure(Ω))), Ω)
    Γin = BoundaryTriangulation(model, tags="inlet"); dΓin = Measure(Γin, 2); nin = get_normal_vector(Γin)
    n̄ = sum(∫(nin)dΓin) / sum(∫(1.0)dΓin)
    Γo = [BoundaryTriangulation(model, tags=o) for o in outs]; dΓo = [Measure(Γ, 3) for Γ in Γo]; no = [get_normal_vector(Γ) for Γ in Γo]
    ZA = [Z[i] * sum(∫(1.0)dΓo[i]) for i in eachindex(outs)]
    reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, 1); reffep = ReferenceFE(lagrangian, Float64, 1)
    Vaux = TestFESpace(model, reffeu; conformity=:H1, dirichlet_tags=["inlet", "wall"])
    u1 = FEFunction(TrialFESpace(Vaux, [x -> -n̄, x -> zero(n̄)]), zeros(num_free_dofs(Vaux)))
    q1 = -sum(∫(u1 ⋅ nin)dΓin)
    Du = BoundaryConditions(BoundaryCondition("inlet", x -> -n̄ / q1, t -> Qin(t)), BoundaryCondition("wall", [0.0, 0.0, 0.0]))
    D_bc = MultiFieldBC([Du, NothingBC()])
    Vu = TestFESpace(model, reffeu, D_bc[1], conformity=:H1); Vp = TestFESpace(model, reffep, D_bc[2], conformity=:H1)
    V = MultiFieldFESpace([Vu, Vp]); U = MultiFieldFESpace([TrialFESpace(Vu, D_bc[1], 0.0), TrialFESpace(Vp, D_bc[2], 0.0)])
    Pcn = fill(1.0e4, 4); Pc = copy(Pcn)
    advecting(cu) = ((α, β) = cu.step == 1 ? (1.0, 0.0) : (2.0, 1.0); α * cu.u⁻ - β * cu.u⁻⁻)
    function a_form(w, un, (u, p), ut, (v, q))
        τs = τsupg_p1 ∘ (un, he); τp = τpspg_p1 ∘ (un, he); τl = τlsic_p1 ∘ (un, he)
        Ru = ρ * ut + ρ * (∇(u)' ⋅ w)
        mf = ∫(ρ * (ut ⋅ v))dΩ + ∫(2μ * (εsym ∘ (∇(u))) ⊙ (εsym ∘ (∇(v))))dΩ + ∫(ρ * ((∇(u)' ⋅ w) ⋅ v))dΩ -
            ∫(p * (∇ ⋅ v))dΩ - ∫(q * (∇ ⋅ u))dΩ +
            ∫(τs * ((∇(v)' ⋅ w) ⋅ Ru))dΩ + ∫(τs * ((∇(v)' ⋅ w) ⋅ ∇(p)))dΩ -
            ∫(τp * (∇(q) ⋅ Ru))dΩ - ∫(τp * (∇(q) ⋅ ∇(p)))dΩ + ∫(τl * (∇ ⋅ v) * (∇ ⋅ u))dΩ
        for i in eachindex(outs)
            mf = mf + ∫(ZA[i] * (u ⋅ no[i]) * (v ⋅ no[i]))dΓo[i] - ∫(ρ * (negpart_p1 ∘ (w ⋅ no[i])) * (u ⋅ v))dΓo[i]
        end
        mf
    end
    res(ctx) = (cu = ctx[1]; ((u, p), (v, q)) -> begin
        r = a_form(advecting(cu), cu.u⁻, (u, p), ∂t(cu, u), (v, q))
        for i in eachindex(outs); r = r + ∫(Pc[i] * (v ⋅ no[i]))dΓo[i]; end
        r
    end)
    jac(ctx) = (cu = ctx[1]; ((u, p), (du, dp), (v, q)) -> a_form(advecting(cu), cu.u⁻, (du, dp), δ∂t(cu, du), (v, q)))
    nls = NewtonRaphson(LUSolver(); verbose=0)
    flux(uh, i) = sum(∫(uh ⋅ no[i])dΓo[i])
    tm = TransientModel(res, jac, U, V, D_bc; scheme=BDF(2), tangent_update=:step, nls, backend, internal=(xh, xh⁻) -> (Pcn .= Pc))
    windkessel!(tn, Δt) = (un = get_state(tm)[1]; for i in eachindex(outs); Pc[i] = (Pcn[i] + Δt * flux(un, i) / C[i]) / (1 + Δt / (R[i] * C[i])); end)
    post = PostProcessor(tm, p -> nothing; is_vtk=false, filepath=mktempdir())
    solve!(tm; stepping=TimeStepping(; Δt=0.005, t_end=nsteps * 0.005), post, presolver=windkessel!)
    c = tm.caches; o = c.op
    mop = HyperFEM.ComputationalModels._model_operator(tm.hooks, tm.spaces, o, o.structured)
    A, bvec = c.nls_cache.A, c.nls_cache.b
    info = o.inner isa KernelOperator ? op_info(o.inner) : Dict{String,Any}()
    (op=mop, A=A, b=bvec, y=zeros(length(bvec)), info=info)
end

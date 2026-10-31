# Refinable cases of the solver baseline (design/solvers.md, S0): EM plate (Gmsh hexahedra, order 1),
# EM box (Cartesian hexahedra, order 2) and hMRE beam (Gmsh triangles, order 1), the reference
# problems of the solver block (D4). Same physics and boundary conditions as benchmark/phase1/solve.jl.
#
# Each builder returns a NamedTuple (res, jac, U, V, D, nls_kw, nsteps, extra) where `extra` are the
# keyword arguments of StaticNonlinearModel besides nls and backend (xh, xh⁻ for the hMRE case).
using HyperFEM, Gridap, GridapGmsh
using Gridap.FESpaces

const MESHDIR = normpath(joinpath(@__DIR__, "..", "meshes"))

function em_case(model, order, φtop)
    physmodel = ElectroMechModel(IdealDielectric(ε=1.0), NeoHookean3D(λ=10.0, μ=1.0))
    k = (Kinematics(Mechano, Solid), Kinematics(Electro, Solid))
    Ω = Triangulation(model); dΩ = Measure(Ω, 2order)
    Du = BoundaryConditions(BoundaryCondition("fixedup", [0.0, 0.0, 0.0]))
    Dφ = BoundaryConditions(BoundaryCondition("midsuf", 0.0, Λ -> Λ), BoundaryCondition("topsuf", φtop, Λ -> Λ))
    D = MultiFieldBC([Du, Dφ])
    Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, order), Du, conformity=:H1)
    Vφ = TestFESpace(model, ReferenceFE(lagrangian, Float64, order), Dφ, conformity=:H1)
    V = MultiFieldFESpace([Vu, Vφ])
    U = MultiFieldFESpace([TrialFESpace(Vu, Du, 1.0), TrialFESpace(Vφ, Dφ, 1.0)])
    res(Λ) = ((u, φ), (v, vφ)) -> HyperFEM.residual(physmodel, k, (u, φ), (v, vφ), dΩ)
    jac(Λ) = ((u, φ), (du, dφ), (v, vφ)) -> HyperFEM.jacobian(physmodel, k, (u, φ), (du, dφ), (v, vφ), dΩ)
    (; res, jac, U, V, D, nls_kw=(maxiter=20, atol=1e-10, rtol=1e-8), nsteps=5, extra=(;))
end

"EM plate of the HyperFEM README (test_static_EM geometry), Gmsh hexahedra (7n, 2n, 19n), order 1."
em_plate(n::Int) = em_case(GmshDiscreteModel(joinpath(MESHDIR, "em_plate_n$(n).msh")), 1, 0.1)

# EM plate with the electrodes constrained in the thickness direction (u_y = 0 on y = 0 and y = 0.8, besides the
# clamped end): no bending, the in-plane electrostatic deformation only. The S2 case where the u block is AMG-friendly
# (the free cantilever of em_plate has a bending-dominated u block on which GAMG needs 200-500 iterations).
function em_case_constrained(model, order, φtop)
    labels = get_face_labeling(model)
    add_tag_from_vertex_filter!(labels, "botsuf", model, x -> abs(x[2]) < 1e-9)
    physmodel = ElectroMechModel(IdealDielectric(ε=1.0), NeoHookean3D(λ=10.0, μ=1.0))
    k = (Kinematics(Mechano, Solid), Kinematics(Electro, Solid))
    Ω = Triangulation(model); dΩ = Measure(Ω, 2order)
    Du = BoundaryConditions(BoundaryCondition("fixedup", [0.0, 0.0, 0.0]),
                            BoundaryCondition("topsuf", [0.0, 0.0, 0.0]; mask=(false, true, false)),
                            BoundaryCondition("botsuf", [0.0, 0.0, 0.0]; mask=(false, true, false)))
    Dφ = BoundaryConditions(BoundaryCondition("midsuf", 0.0, Λ -> Λ), BoundaryCondition("topsuf", φtop, Λ -> Λ))
    D = MultiFieldBC([Du, Dφ])
    Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, order), Du, conformity=:H1)
    Vφ = TestFESpace(model, ReferenceFE(lagrangian, Float64, order), Dφ, conformity=:H1)
    V = MultiFieldFESpace([Vu, Vφ])
    U = MultiFieldFESpace([TrialFESpace(Vu, Du, 1.0), TrialFESpace(Vφ, Dφ, 1.0)])
    res(Λ) = ((u, φ), (v, vφ)) -> HyperFEM.residual(physmodel, k, (u, φ), (v, vφ), dΩ)
    jac(Λ) = ((u, φ), (du, dφ), (v, vφ)) -> HyperFEM.jacobian(physmodel, k, (u, φ), (du, dφ), (v, vφ), dΩ)
    (; res, jac, U, V, D, nls_kw=(maxiter=20, atol=1e-10, rtol=1e-8), nsteps=5, extra=(;))
end
em_plate_c(n::Int) = em_case_constrained(GmshDiscreteModel(joinpath(MESHDIR, "em_plate_n$(n).msh")), 1, 0.1)

"EM box 0.1 × 0.01 × 0.002, Cartesian (nx, ny, nz) hexahedra, order 2 (phase-0/1 case)."
function em_box(nx::Int, ny::Int, nz::Int)
    m = CartesianDiscreteModel((0.0, 0.1, 0.0, 0.01, 0.0, 0.002), (nx, ny, nz))
    labels = get_face_labeling(m)
    add_tag_from_tags!(labels, "fixedup", CartesianTags.face0YZ⁺)
    add_tag_from_tags!(labels, "topsuf", CartesianTags.faceXY1⁺)
    add_tag_from_vertex_filter!(labels, "midsuf", m, x -> abs(x[3] - 0.001) < 1e-9)
    em_case(m, 2, 5e-5)
end

"hMRE beam (Short_magnet geometry refined by r on the beam, rv elsewhere), mechanics with the magnetic load."
function hmre(r::Real, rv::Real=2)
    model = GmshDiscreteModel(joinpath(MESHDIR, "hmre_short_r$(Float64(r))_rv$(Float64(rv)).msh"))
    params = [10.0, 12480.64495232286, 1.0, 2.0, 5195.545287237134, 0.2602717127043121, 2.0]
    modelmech = NonlinearMooneyRivlin2D(λ=(params[1] + params[2]) * 1e2, μ1=params[1] * 1e3, μ2=params[2], α1=params[3], α2=params[4]) +
                TransverseIsotropy2D(μ=params[5], α1=params[6], α2=params[7])
    μ0 = 4π * 1e-7; Banorm = 80.0E-9; αr = 10e-3 / μ0
    reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
    Ωs = Triangulation(model, tags=["Beam"]); dΩs = Measure(Ωs, 2)
    Du = BoundaryConditions(BoundaryCondition("Fixed_Solid", [0.0, 0.0]), BoundaryCondition("Fixed_Solid_x", [0.0, 0.0]; mask=(true, false)))
    V = TestFESpace(Ωs, reffeu, Du, conformity=:H1)
    U = TrialFESpace(V, Du); U⁻ = TrialFESpace(V, Du)
    uh = FEFunction(U, zero_free_values(U)); uh⁻ = FEFunction(U⁻, zero_free_values(U⁻))
    _, ∂Ψs∂F, ∂Ψs∂FF = modelmech()
    F, _, _ = get_Kinematics(Kinematics(Mechano, Solid))
    V_N = TestFESpace(Ωs, reffeu)
    Nh = interpolate_everywhere(x -> VectorValue(1.0, 0.0), V_N)
    Bah = interpolate_everywhere(VectorValue([0.0, -Banorm]), V_N)
    # inverse of the right stretch tensor, closed form in 2D so that the element kernel does not allocate
    # (the matrix-sqrt form of the original script builds dense matrices; a local function with keyword
    # arguments is called dynamically by Julia and allocates too, see `allocations(op, x)`)
    function invU(F)
        C = transpose(F) ⋅ F + 1e-12 * I2
        s = sqrt(det(C))
        inv((C + s * I2) / sqrt(tr(C) + 2s))
    end
    ∂Ψmag∂F(Fn, N, Ba) = (-αr / μ0) * (Ba ⊗ ((invU(Fn)) * N))
    res(Λ) = (u, v) -> ∫((∇(v)' ⊙ (∂Ψs∂F ∘ (F ∘ (∇(u)'), Nh))))dΩs + Λ * ∫((∂Ψmag∂F ∘ (F ∘ (∇(uh⁻)'), Nh, Bah)) ⊙ (∇(v)'))dΩs
    jac(Λ) = (u, du, v) -> ∫(∇(v)' ⊙ ((∂Ψs∂FF ∘ (F ∘ (∇(u)'), Nh)) ⊙ (∇(du)')))dΩs
    (; res, jac, U, V, D=Du, nls_kw=(maxiter=20, atol=1e-8, rtol=1e-2), nsteps=20, extra=(xh=uh, xh⁻=uh⁻))
end

# Mechanical block alone (the clean S1 measurement, D4 refined 2026-10-30): the EM geometries with the
# neo-Hookean solid only, clamped at "fixedup" and sheared by a prescribed displacement of "topsuf".
function mech_case(model, order, utop)
    physmodel = NeoHookean3D(λ=10.0, μ=1.0)
    k = Kinematics(Mechano, Solid)
    Ω = Triangulation(model); dΩ = Measure(Ω, 2order)
    Du = BoundaryConditions(BoundaryCondition("fixedup", [0.0, 0.0, 0.0]), BoundaryCondition("topsuf", utop, Λ -> Λ))
    V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, order), Du, conformity=:H1)
    U = TrialFESpace(V, Du, 0.0)
    res(Λ) = (u, v) -> HyperFEM.residual(physmodel, k, u, v, dΩ)
    jac(Λ) = (u, du, v) -> HyperFEM.jacobian(physmodel, k, u, du, v, dΩ)
    (; res, jac, U, V, D=Du, nls_kw=(maxiter=20, atol=1e-10, rtol=1e-8), nsteps=5, extra=(;))
end

"Mechanical plate: the EM plate mesh (8 × 0.8 × 40), order 1, top face sheared by 0.4 along z (shear strain 0.5)."
mech_plate(n::Int) = mech_case(GmshDiscreteModel(joinpath(MESHDIR, "em_plate_n$(n).msh")), 1, [0.0, 0.0, 0.4])

"Mechanical box: the EM box, order 2, top face sheared by 0.001 along x (shear strain 0.5)."
function mech_box(nx::Int, ny::Int, nz::Int)
    m = CartesianDiscreteModel((0.0, 0.1, 0.0, 0.01, 0.0, 0.002), (nx, ny, nz))
    labels = get_face_labeling(m)
    add_tag_from_tags!(labels, "fixedup", CartesianTags.face0YZ⁺)
    add_tag_from_tags!(labels, "topsuf", CartesianTags.faceXY1⁺)
    mech_case(m, 2, [0.001, 0.0, 0.0])
end

# Periodic unit cell of a magneto-active composite (the "periodic cell" study of 2026-09, magnetoactive_unit_cell_2D.jl,
# adapted to the BoundaryConditions API and to both backends; 3D by the same construction): soft magnetically inert
# matrix (Mooney–Rivlin + IdealMagnetic, χ = 0), rigid inclusion (×100) with remanent magnetisation Mr along x
# (HardMagnetic), external field Hext = Λ·Hy along y by continuation, periodic faces (topology of the mesh), u = 0 and
# φ = 0 at the corner node. Monolithic u–φ. Mesh from `generate.jl cell dim h`.
function hmre_cell(dim::Int, h::Real, order::Int=1, R::Real=(dim == 2 ? 0.175 : 0.25); contrast::Real=100.0)
    model = GmshDiscreteModel(joinpath(MESHDIR, "cell$(dim)d_h$(h)" * (R == 0.25 ? "" : "_R$(R)") * ".msh"))
    μ0 = 4π * 1e-7
    Mr = 0.1 / μ0; Hy = 0.05 / μ0                        # μ0 Mr = 100 mT, μ0 Hy = 50 mT
    e(c) = VectorValue(ntuple(i -> i == c ? 1.0 : 0.0, dim))
    Mr_dir = e(1); Hext = Hy * e(2); N_incl = -Mr_dir    # HyperFEM's convention: m = -ℋᵣ, so N = -Mr_dir gives Mr along +x
    if dim == 2
        mech_m = MooneyRivlin2D(λ=100.0e3, μ1=10.0e3, μ2=0.0); mech_i = MooneyRivlin2D(λ=10.0e6, μ1=1.0e6, μ2=0.0)
        mag_m = IdealMagnetic2D(μ0=μ0, χe=0.0); mag_i = HardMagnetic2D(μ0=μ0, αr=Mr, χe=0.0)
    else
        mech_m = MooneyRivlin3D(λ=100.0e3, μ1=10.0e3, μ2=0.0); mech_i = MooneyRivlin3D(λ=contrast * 100.0e3, μ1=contrast * 10.0e3, μ2=0.0)   # inclusion/matrix stiffness contrast (100 = the original case)
        mag_m = IdealMagnetic(μ0=μ0, χe=0.0); mag_i = HardMagnetic(μ0=μ0, αr=Mr, χe=0.0)
    end
    _, ∂Ψu_m, ∂Ψφ_m, ∂Ψuu_m, ∂Ψφu_m, ∂Ψφφ_m = (mech_m + mag_m)()
    _, ∂Ψu_i, ∂Ψφ_i, ∂Ψuu_i, ∂Ψφu_i, ∂Ψφφ_i = (mech_i + mag_i)()
    F, _, _ = get_Kinematics(Kinematics(Mechano, Solid))
    kmag = EvolutiveKinematics(Magneto; H=(Λ) -> ((∇φ) -> Λ * Hext - ∇φ))
    Ωm = Triangulation(model, tags=["matriz"]); Ωi = Triangulation(model, tags=["inclusion"])
    dΩm = Measure(Ωm, 2order); dΩi = Measure(Ωi, 2order)
    Du = BoundaryConditions(BoundaryCondition("corner_dirichlet", zeros(dim)))
    Dφ = BoundaryConditions(BoundaryCondition("corner_dirichlet", 0.0))
    D = MultiFieldBC([Du, Dφ])
    Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{dim,Float64}, order), Du, conformity=:H1)
    Vφ = TestFESpace(model, ReferenceFE(lagrangian, Float64, order), Dφ, conformity=:H1)
    num_dirichlet_dofs(Vu) == dim && num_dirichlet_dofs(Vφ) == 1 || error("corner_dirichlet does not fix one vertex")
    V = MultiFieldFESpace([Vu, Vφ])
    U = MultiFieldFESpace([TrialFESpace(Vu, Du, 0.0), TrialFESpace(Vφ, Dφ, 0.0)])
    Nm = CellField(Mr_dir, Ωm); Ni = CellField(N_incl, Ωi)
    res_region(∂Ψu, ∂Ψφ, N, ℋ, (u, φ), (v, vφ), dΩr) =
        ∫(∇(v)' ⊙ (∂Ψu ∘ (F ∘ (∇(u)'), ℋ ∘ (∇(φ)), N)))dΩr - ∫(∇(vφ) ⋅ (∂Ψφ ∘ (F ∘ (∇(u)'), ℋ ∘ (∇(φ)), N)))dΩr
    jac_region(∂Ψuu, ∂Ψφu, ∂Ψφφ, N, ℋ, (u, φ), (du, dφ), (v, vφ), dΩr) =
        ∫(∇(v)' ⊙ ((∂Ψuu ∘ (F ∘ (∇(u)'), ℋ ∘ (∇(φ)), N)) ⊙ (∇(du)')))dΩr +
        ∫(∇(vφ) ⋅ ((∂Ψφφ ∘ (F ∘ (∇(u)'), ℋ ∘ (∇(φ)), N)) ⋅ ∇(dφ)))dΩr -
        ∫(∇(dφ) ⋅ ((∂Ψφu ∘ (F ∘ (∇(u)'), ℋ ∘ (∇(φ)), N)) ⊙ (∇(v)')))dΩr -
        ∫(∇(vφ) ⋅ ((∂Ψφu ∘ (F ∘ (∇(u)'), ℋ ∘ (∇(φ)), N)) ⊙ (∇(du)')))dΩr
    res(Λ) = (x, y) -> (ℋ = get_Kinematics(kmag; Λ=Λ);
        res_region(∂Ψu_m, ∂Ψφ_m, Nm, ℋ, x, y, dΩm) + res_region(∂Ψu_i, ∂Ψφ_i, Ni, ℋ, x, y, dΩi))
    jac(Λ) = (x, dx, y) -> (ℋ = get_Kinematics(kmag; Λ=Λ);
        jac_region(∂Ψuu_m, ∂Ψφu_m, ∂Ψφφ_m, Nm, ℋ, x, dx, y, dΩm) + jac_region(∂Ψuu_i, ∂Ψφu_i, ∂Ψφφ_i, Ni, ℋ, x, dx, y, dΩi))
    (; res, jac, U, V, D, nls_kw=(maxiter=20, atol=1e-10, rtol=1e-8), nsteps=10, extra=(;))
end

"Case from command-line words: em_plate n | em_box nx ny nz | hmre r [rv] | mech_plate n | mech_box nx ny nz | hmre_cell dim h [order] | hmre_cell_c dim h order contrast."
function build_case(words)
    kind = words[1]
    kind == "hmre_cell_c" && return hmre_cell(parse(Int, words[2]), parse(Float64, words[3]), parse(Int, words[4]); contrast=parse(Float64, words[5])), "hmre_cell$(words[2])d_h$(words[3])_p$(words[4])_c$(words[5])"
    kind == "hmre_cell" && return hmre_cell(parse(Int, words[2]), parse(Float64, words[3]), parse(Int, get(words, 4, "1"))), "hmre_cell$(words[2])d_h$(words[3])_p$(get(words, 4, "1"))"
    kind == "mech_plate" && return mech_plate(parse(Int, words[2])), "mech_plate_n$(words[2])"
    kind == "mech_box" && return mech_box(parse.(Int, words[2:4])...), "mech_box_$(join(words[2:4], 'x'))"
    kind == "em_plate" && return em_plate(parse(Int, words[2])), "em_plate_n$(words[2])"
    kind == "em_plate_c" && return em_plate_c(parse(Int, words[2])), "em_plate_c_n$(words[2])"
    kind == "em_box" && return em_box(parse.(Int, words[2:4])...), "em_box_$(join(words[2:4], 'x'))"
    kind == "hmre" && return hmre(parse(Float64, words[2]), parse(Float64, get(words, 3, "2"))), "hmre_r$(words[2])"
    error("unknown case $kind")
end

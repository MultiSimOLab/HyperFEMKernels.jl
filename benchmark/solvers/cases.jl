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
function hmre(r::Real, rv::Real=2; hybrid::Bool=false, Bfactor::Real=1.0)
    model = GmshDiscreteModel(joinpath(MESHDIR, "hmre_short_r$(Float64(r))_rv$(Float64(rv)).msh"))
    params = [10.0, 12480.64495232286, 1.0, 2.0, 5195.545287237134, 0.2602717127043121, 2.0]
    modelmech = NonlinearMooneyRivlin2D(λ=(params[1] + params[2]) * 1e2, μ1=params[1] * 1e3, μ2=params[2], α1=params[3], α2=params[4]) +
                TransverseIsotropy2D(μ=params[5], α1=params[6], α2=params[7])
    # hybrid = true (S5): the mechanical model in HessianRegularization with a TangentSwitch (the Newton's HybridTangent);
    # Bfactor scales the applied field (the instability sweep)
    switch = hybrid ? TangentSwitch() : nothing
    hybrid && (modelmech = HessianRegularization(modelmech; δ=1e-6, switch))
    μ0 = 4π * 1e-7; Banorm = 80.0E-9 * Bfactor; αr = 10e-3 / μ0
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
    extra = hybrid ? (xh=uh, xh⁻=uh⁻, tangent_switch=switch) : (xh=uh, xh⁻=uh⁻)
    (; res, jac, U, V, D=Du, nls_kw=(maxiter=20, atol=1e-8, rtol=1e-2), nsteps=20, extra, switch)
end

# Mechanical block alone (the clean S1 measurement, D4 refined 2026-10-30): the EM geometries with the
# neo-Hookean solid only, clamped at "fixedup" and sheared by a prescribed displacement of "topsuf".
function mech_case(model, order, utop; λ=10.0)
    physmodel = NeoHookean3D(λ=λ, μ=1.0)                  # λ = 10: ν = λ/(2(λ+μ)) = 0.45; λ = 1.5: ν = 0.3
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

"""
Mechanical box: the EM box (0.1 × 0.01 × 0.002, slender; Cartesian elements 10:5:1), order `p` (2 by default),
top face sheared by 0.001 along x (shear strain 0.5). With `p = 1` it is the Q1 discretisation S4-A's coarse level sees.
"""
function mech_box(nx::Int, ny::Int, nz::Int, p::Int=2)
    m = CartesianDiscreteModel((0.0, 0.1, 0.0, 0.01, 0.0, 0.002), (nx, ny, nz))
    labels = get_face_labeling(m)
    add_tag_from_tags!(labels, "fixedup", CartesianTags.face0YZ⁺)
    add_tag_from_tags!(labels, "topsuf", CartesianTags.faceXY1⁺)
    mech_case(m, p, [0.001, 0.0, 0.0])
end

"""
Mechanical cube: unit cube of n³ cubic hexahedra (isotropic), order `p` (2 by default), clamped at x = 0 and sheared
at x = 1 by 0.1 along z (shear strain 0.1; at 0.5 and 0.2 the clamped–sheared cube bisected: 97 Newton iterations in 5 steps at n = 10, 7–12 solves at n = 20–34). The h-independence acceptance case of S4-A (design/solvers-s4a-plan.md,
DS4A-3): the same physics as the box without its element anisotropy. Q2 dofs ≈ 3 (2n+1)³: n = 20 → 207k, n = 34 → 985k.
"""
function mech_cube(n::Int, p::Int=2)
    m = CartesianDiscreteModel((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), (n, n, n))
    labels = get_face_labeling(m)
    add_tag_from_tags!(labels, "fixedup", CartesianTags.face0YZ⁺)
    add_tag_from_tags!(labels, "topsuf", CartesianTags.face1YZ⁺)
    mech_case(m, p, [0.0, 0.0, 0.1])
end

# --- S4-B: the mechanical cube as a refined hierarchy (design/solvers-s4b-plan.md §4) ---------------------------------

"Coarse Gmsh tetrahedral mesh of the unit cube with mesh size `h`; closed physical groups fixedup (x = 0) and topsuf (x = 1)."
function cube_tet_mesh(h::Real)
    file = joinpath(MESHDIR, "cube_tet_h$(h).msh")
    isfile(file) && return file
    gmsh = GridapGmsh.gmsh
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cube")
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1); gmsh.model.occ.synchronize()
    ents(d) = [e[2] for e in gmsh.model.getEntities(d)]
    com(t) = gmsh.model.occ.getCenterOfMass(2, t)
    for (name, x0) in (("fixedup", 0.0), ("topsuf", 1.0))
        faces = [t for t in ents(2) if abs(com(t)[1] - x0) < 1e-9]
        gmsh.model.addPhysicalGroup(2, faces, -1, name)
        for d in (1, 0)                                   # the closure: edges and vertices under the same name
            cl = unique([abs(e[2]) for e in gmsh.model.getBoundary([(2, t) for t in faces], false, false, d == 0) if e[1] == d])
            gmsh.model.addPhysicalGroup(d, cl, -1, name)
        end
    end
    gmsh.model.addPhysicalGroup(3, ents(3), -1, "solid")
    gmsh.option.setNumber("Mesh.MeshSizeMax", h); gmsh.option.setNumber("Mesh.MeshSizeMin", h)
    gmsh.model.mesh.generate(3)
    gmsh.write(file); gmsh.finalize()
    file
end

"""
Mechanical cube on a hierarchy of `k` uniform refinements of a coarse mesh (`mech_cube`'s physics and boundary conditions,
order 1): `hcube_tet(h, k)` from the Gmsh tetrahedra of size `h`, `hcube_hex(n, k)` from n³ Cartesian hexahedra. The case
carries the hierarchy (`hier`) and the test-space builder (`space`) the h-multigrid solver needs; the Newton sees the fine
model only.
"""
function hcube(coarse, k::Int; utop=[0.0, 0.0, 0.1], λ=10.0)
    hier = MeshHierarchy(coarse, k + 1)
    case = mech_case(fine_model(hier), 1, utop; λ)
    Du = case.D
    space(m) = TestFESpace(m, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), Du, conformity=:H1)
    (; case..., hier, space)
end
hcube_tet(h::Real, k::Int, λ::Real=10.0) = hcube(GmshDiscreteModel(cube_tet_mesh(h)), k; λ)
function hcube_hex(n::Int, k::Int, λ::Real=10.0)
    m = CartesianDiscreteModel((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), (n, n, n))
    labels = get_face_labeling(m)
    add_tag_from_tags!(labels, "fixedup", CartesianTags.face0YZ⁺)
    add_tag_from_tags!(labels, "topsuf", CartesianTags.face1YZ⁺)
    hcube(m, k; λ)
end

"""
EM plate with constrained electrodes on a hierarchy (S4-B step 5): the 8 × 0.8 × 40 plate as (7n, 2n, 19n) Cartesian
hexahedra refined `k` times, clamped at z = 0, electrodes on both faces y = 0 (φ = 0) and y = 0.8 (φ = φtop) with u_y = 0
on them (closed Cartesian tags inherited by every level; `em_plate_c`'s mid-plane electrode is a vertex tag that the
refinement does not inherit). φtop = 0.2 over the whole thickness: the field of `em_plate_c` (0.1 over half of it).
The case carries the hierarchy and the test-space builders of the two fields (`spaces`).
"""
function hem_plate_c(n::Int, k::Int, φtop::Real=0.2; hybrid::Bool=false)
    m = CartesianDiscreteModel((0.0, 8.0, 0.0, 0.8, 0.0, 40.0), (7n, 2n, 19n))
    labels = get_face_labeling(m)
    add_tag_from_tags!(labels, "fixedup", CartesianTags.faceXY0⁺)
    add_tag_from_tags!(labels, "botsuf", CartesianTags.faceX0Z⁺)
    add_tag_from_tags!(labels, "topsuf", CartesianTags.faceX1Z⁺)
    hier = MeshHierarchy(m, k + 1)
    model = fine_model(hier)
    # hybrid = true (S5): the coupled model wrapped in HessianRegularization with a TangentSwitch — the Newton's
    # HybridTangent projects the total mechanical tangent ∂²Ψ/∂F∂F(F, E), elastic and Maxwell parts, in E
    switch = hybrid ? TangentSwitch() : nothing
    em = ElectroMechModel(IdealDielectric(ε=1.0), NeoHookean3D(λ=10.0, μ=1.0))
    physmodel = hybrid ? HessianRegularization(em; δ=1e-6, switch) : em
    kin = (Kinematics(Mechano, Solid), Kinematics(Electro, Solid))
    Ω = Triangulation(model); dΩ = Measure(Ω, 2)
    Du = BoundaryConditions(BoundaryCondition("fixedup", [0.0, 0.0, 0.0]),
                            BoundaryCondition("topsuf", [0.0, 0.0, 0.0]; mask=(false, true, false)),
                            BoundaryCondition("botsuf", [0.0, 0.0, 0.0]; mask=(false, true, false)))
    Dφ = BoundaryConditions(BoundaryCondition("botsuf", 0.0, Λ -> Λ), BoundaryCondition("topsuf", φtop, Λ -> Λ))
    D = MultiFieldBC([Du, Dφ])
    space_u(mm) = TestFESpace(mm, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), Du, conformity=:H1)
    space_φ(mm) = TestFESpace(mm, ReferenceFE(lagrangian, Float64, 1), Dφ, conformity=:H1)
    Vu, Vφ = space_u(model), space_φ(model)
    V = MultiFieldFESpace([Vu, Vφ])
    U = MultiFieldFESpace([TrialFESpace(Vu, Du, 1.0), TrialFESpace(Vφ, Dφ, 1.0)])
    res(Λ) = ((u, φ), (v, vφ)) -> HyperFEM.residual(physmodel, kin, (u, φ), (v, vφ), dΩ)
    jac(Λ) = ((u, φ), (du, dφ), (v, vφ)) -> HyperFEM.jacobian(physmodel, kin, (u, φ), (du, dφ), (v, vφ), dΩ)
    extra = hybrid ? (; tangent_switch=switch) : (;)
    (; res, jac, U, V, D, nls_kw=(maxiter=20, atol=1e-10, rtol=1e-8), nsteps=5, extra, hier, spaces=(space_u, space_φ), switch)
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

"Case from command-line words: em_plate n | em_plate_c n | hem_plate_c n k [φtop] | hem_plate_ch n k [φtop] (S5: projected mechanical tangent) | em_box nx ny nz | hmre r [rv] | mech_plate n | mech_box nx ny nz | hmre_cell dim h [order] | hmre_cell_c dim h order contrast."
function build_case(words)
    kind = words[1]
    kind == "hmre_cell_c" && return hmre_cell(parse(Int, words[2]), parse(Float64, words[3]), parse(Int, words[4]); contrast=parse(Float64, words[5])), "hmre_cell$(words[2])d_h$(words[3])_p$(words[4])_c$(words[5])"
    kind == "hmre_cell" && return hmre_cell(parse(Int, words[2]), parse(Float64, words[3]), parse(Int, get(words, 4, "1"))), "hmre_cell$(words[2])d_h$(words[3])_p$(get(words, 4, "1"))"
    kind == "mech_plate" && return mech_plate(parse(Int, words[2])), "mech_plate_n$(words[2])"
    kind == "mech_box" && return mech_box(parse.(Int, words[2:4])..., parse(Int, get(words, 5, "2"))), "mech_box_$(join(words[2:4], 'x'))" * (get(words, 5, "2") == "2" ? "" : "_p$(words[5])")
    kind == "hcube_tet" && return hcube_tet(parse(Float64, words[2]), parse(Int, words[3]), parse(Float64, get(words, 4, "10"))), "hcube_tet_h$(words[2])_k$(words[3])" * (length(words) ≥ 4 ? "_lam$(words[4])" : "")
    kind == "hcube_hex" && return hcube_hex(parse(Int, words[2]), parse(Int, words[3]), parse(Float64, get(words, 4, "10"))), "hcube_hex_$(words[2])_k$(words[3])" * (length(words) ≥ 4 ? "_lam$(words[4])" : "")
    kind == "mech_cube" && return mech_cube(parse(Int, words[2]), parse(Int, get(words, 3, "2"))), "mech_cube_$(words[2])_p$(get(words, 3, "2"))"
    kind == "em_plate" && return em_plate(parse(Int, words[2])), "em_plate_n$(words[2])"
    kind == "em_plate_c" && return em_plate_c(parse(Int, words[2])), "em_plate_c_n$(words[2])"
    kind == "hem_plate_c" && return hem_plate_c(parse(Int, words[2]), parse(Int, words[3]), parse(Float64, get(words, 4, "0.2"))), "hem_plate_c_n$(words[2])_k$(words[3])" * (length(words) ≥ 4 ? "_phi$(words[4])" : "")
    kind == "hem_plate_ch" && return hem_plate_c(parse(Int, words[2]), parse(Int, words[3]), parse(Float64, get(words, 4, "0.2")); hybrid=true), "hem_plate_ch_n$(words[2])_k$(words[3])" * (length(words) ≥ 4 ? "_phi$(words[4])" : "")
    kind == "em_box" && return em_box(parse.(Int, words[2:4])...), "em_box_$(join(words[2:4], 'x'))"
    kind == "hmre" && return hmre(parse(Float64, words[2]), parse(Float64, get(words, 3, "2")); Bfactor=parse(Float64, get(words, 4, "1"))), "hmre_r$(words[2])" * (length(words) ≥ 4 ? "_B$(words[4])" : "")
    kind == "hmre_h" && return hmre(parse(Float64, words[2]), parse(Float64, get(words, 3, "2")); hybrid=true, Bfactor=parse(Float64, get(words, 4, "1"))), "hmre_h_r$(words[2])" * (length(words) ≥ 4 ? "_B$(words[4])" : "")
    error("unknown case $kind")
end

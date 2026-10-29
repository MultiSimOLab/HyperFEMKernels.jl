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
    function invU(F; ε=1e-12)
        F_ = get_array(F); S = F_' * F_ + ε * get_array(I2)
        TensorValue(inv(sqrt(S)))
    end
    ∂Ψmag∂F(Fn, N, Ba) = (-αr / μ0) * (Ba ⊗ ((invU(Fn)) * N))
    res(Λ) = (u, v) -> ∫((∇(v)' ⊙ (∂Ψs∂F ∘ (F ∘ (∇(u)'), Nh))))dΩs + Λ * ∫((∂Ψmag∂F ∘ (F ∘ (∇(uh⁻)'), Nh, Bah)) ⊙ (∇(v)'))dΩs
    jac(Λ) = (u, du, v) -> ∫(∇(v)' ⊙ ((∂Ψs∂FF ∘ (F ∘ (∇(u)'), Nh)) ⊙ (∇(du)')))dΩs
    (; res, jac, U, V, D=Du, nls_kw=(maxiter=20, atol=1e-8, rtol=1e-2), nsteps=20, extra=(xh=uh, xh⁻=uh⁻))
end

"Case from command-line words: em_plate n | em_box nx ny nz | hmre r [rv]."
function build_case(words)
    kind = words[1]
    kind == "em_plate" && return em_plate(parse(Int, words[2])), "em_plate_n$(words[2])"
    kind == "em_box" && return em_box(parse.(Int, words[2:4])...), "em_box_$(join(words[2:4], 'x'))"
    kind == "hmre" && return hmre(parse(Float64, words[2]), parse(Float64, get(words, 3, "2"))), "hmre_r$(words[2])"
    error("unknown case $kind")
end

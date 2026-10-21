# =============================================================================
# Patient aorta: pulsatile blood flow (incompressible Navier-Stokes, rigid wall), Gridap vs HyperFEMKernels.
#
# Self-contained extract of hemodynamics/examples/patient_aorta/flow.jl (and of the solver
# `navier_stokes_transient` of hemodynamics/src/Hemodynamics.jl), written with plain Gridap (no Hemodynamics
# module, no HyperFEM boundary conditions) so that the same weak form can be assembled with Gridap and with
# HyperFEMKernels.
#
# Model (units mm, s, g; pressure in Pa):
#   rho (du/dt + (grad u) u) - div(2 mu eps(u)) + grad p = 0,  div u = 0   in the lumen;
#   u = 0 on the wall, pulsatile flow rate at the inlet (half sine in systole, Poisson profile of the inlet and
#   eccentric valve jet), three-element Windkessel (RCR) at the four outlets (implicit Robin coupling) and
#   backflow stabilization.
# Discretization: Taylor-Hood P2-P1 with SUPG + PSPG + LSIC, BDF2 (first step backward Euler), Oseen
# linearization (advecting velocity 2u^n - u^(n-1)): one linear system per step, factorized with LU.
#
# Usage (from the repository root, benchmark environment):
#   julia -t auto --project=benchmark examples/aorta_cfd/aorta_flow.jl [mode=compare|gridap|kernel]
#        [steps=10] [dt=0.005] [jet=on|off] [threads1=on|off] [mesh=aorta_fluido_h1p5_3p0]
#   mode=compare : every step is assembled with Gridap AND with HyperFEMKernels (N threads and, with
#                  threads1=on, 1 thread); prints the relative differences of matrix and vector and the
#                  times of every phase; the solution is advanced with Gridap's system.
#   mode=gridap  : production run with Gridap's assembly (as in flow.jl).
#   mode=kernel  : production run with HyperFEMKernels' assembly.
#   steps=0 runs two cardiac cycles (400 steps with dt = 0.005).
# Output (examples/aorta_cfd/results/<mesh>_<mode>/): history.csv (flow rates per step), timings.csv (seconds
# per phase and step) and, in compare mode, diffs.csv.
#
# Writing note: values that change at every step (the compliance pressures of the Windkessel outlets) enter
# the weak form as cell fields over a vector updated in place, not as Julia numbers: HyperFEMKernels folds
# numbers into the generated code when the operator is built. The BDF2 coefficients change once (after the
# first, backward Euler step) and the kernel operator is rebuilt then with `update`.
# =============================================================================
using Gridap, GridapGmsh, HyperFEMKernels, LinearAlgebra, SparseArrays, Printf
using GridapGmsh: gmsh
using Gridap.FESpaces: assemble_matrix_and_vector, assemble_matrix_and_vector!, collect_cell_matrix_and_vector
using Gridap.Algebra: symbolic_setup, numerical_setup, numerical_setup!

kw = Dict(split(a, "=")[1] => split(a, "=")[2] for a in ARGS if occursin("=", a))
mode = get(kw, "mode", "compare")
mode in ("compare", "gridap", "kernel") || error("mode must be compare, gridap or kernel")
meshname = get(kw, "mesh", "aorta_fluido_h1p5_3p0")
jeton = get(kw, "jet", "on") == "on"
threads1 = get(kw, "threads1", "on") == "on"
dt = parse(Float64, get(kw, "dt", "0.005"))
nsteps_arg = parse(Int, get(kw, "steps", mode == "compare" ? "10" : "0"))
const KG_M3 = 1.0e-6                     # 1 kg/m3 in g/mm3
const ρ = 1050 * KG_M3; const μ = 3.5e-3  # blood: density (g/mm3), viscosity (Pa s)
mfile = joinpath(@__DIR__, meshname * ".msh")
outd = joinpath(@__DIR__, "results", meshname * (jeton ? "" : "_nojet") * "_" * mode); mkpath(outd)

# --- Helpers copied verbatim from hemodynamics/src/Hemodynamics.jl --------------------------------

# GridapGmsh labels with a physical surface only its faces and the nodes classified on it: add to each tag of
# `names` all the edges and vertices of its faces (shared ones keep the LAST tag; put `wall` last).
function close_boundary_tags!(model, names)
  labels = get_face_labeling(model)
  topo = Gridap.Geometry.get_grid_topology(model)
  D = num_cell_dims(model)
  for name in names
    tag = findfirst(==(name), labels.tag_to_name)
    tag === nothing && error("tag $name not found")
    faces = findall(Gridap.Geometry.get_face_mask(labels, name, D - 1))
    for d in 0:D-2
      f_to_d = Gridap.Geometry.get_faces(topo, D - 1, d)
      ents = labels.d_to_dface_to_entity[d+1]
      newent = maximum(maximum(e) for e in labels.d_to_dface_to_entity) + 1
      for f in faces, e in f_to_d[f]
        ents[e] = newent
      end
      push!(labels.tag_to_entities[tag], newent)
    end
  end
  model
end

# 0D model of the rigid aorta with the Windkessel outlets: periodic compliance pressures at t = 0 (initial values
# of the 3D computation) and a reference over one period.
function windkessel_0d(Qin, wk; T, h=1e-4, ncycles=30)
  Z = [w.Z for w in wk]; C = [w.C for w in wk]; R = [w.R for w in wk]
  P0(t, Pc) = (Qin(t) + sum(Pc ./ Z)) / sum(1 ./ Z)
  f(t, Pc) = ((P0(t, Pc) .- Pc) ./ Z .- Pc ./ R) ./ C
  rk4(t, y) = (k1 = f(t, y); k2 = f(t + h / 2, y + h / 2 * k1); k3 = f(t + h / 2, y + h / 2 * k2); k4 = f(t + h, y + h * k3);
               y + h / 6 * (k1 + 2k2 + 2k3 + k4))
  Qm = sum(Qin(k * h) for k in 0:round(Int, T / h)-1) / round(Int, T / h)
  Pc = R ./ (Z .+ R) .* (Qm / sum(1 ./ (Z .+ R)))       # mean state as starting point
  nper = round(Int, T / h)
  for k in 0:ncycles*nper-1; Pc = rk4(k * h, Pc); end
  Pc0 = copy(Pc); ref = Vector{Float64}[]
  for k in 0:nper
    t = k * h; P = P0(t, Pc); k % 50 == 0 && push!(ref, [t; P; (P .- Pc) ./ Z])
    Pc = rk4(t, Pc)
  end
  Pc0, ref
end

# Velocity profile of an inflow opening of arbitrary shape (Poisson problem on the planar opening, normalized to
# unit mean), optionally with an eccentric, tilted valve jet. Returns (g(x), area, inward normal).
function inlet_profile(meshfile::String, tag::String; order=2, jet=nothing)
  gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0); gmsh.open(meshfile)
  pg = [p for (d, p) in gmsh.model.getPhysicalGroups(2) if gmsh.model.getPhysicalName(2, p) == tag]
  isempty(pg) && error("no physical surface named $tag")
  tris = Vector{Int}[]
  for e in gmsh.model.getEntitiesForPhysicalGroup(2, pg[1])
    types, _, nt = gmsh.model.mesh.getElements(2, e)
    for (k, ty) in enumerate(types)
      ty == 2 || error("only linear triangles are supported on $tag")
      append!(tris, [Int.(nt[k][3i-2:3i]) for i in 1:length(nt[k])÷3])
    end
  end
  ntag, xyz, _ = gmsh.model.mesh.getNodes()
  gmsh.finalize()
  X = Dict(Int(ntag[i]) => xyz[3i-2:3i] for i in eachindex(ntag))
  loc = Dict(v => i for (i, v) in enumerate(unique(reduce(vcat, tris))))
  P = zeros(length(loc), 3); for (v, i) in loc; P[i, :] = X[v]; end
  c = vec(sum(P, dims=1)) / size(P, 1)
  F = svd(P .- c'); e1, e2, nrm = F.V[:, 1], F.V[:, 2], F.V[:, 3]
  proj(x) = (x .- c) ⋅ e1, (x .- c) ⋅ e2
  pts = [Point(proj(P[i, :])...) for i in 1:size(P, 1)]
  cells = [[loc[v] for v in t] for t in tris]
  for t in cells                                   # positive orientation in the plane
    a, b, d = pts[t[1]], pts[t[2]], pts[t[3]]
    ((b - a)[1] * (d - a)[2] - (b - a)[2] * (d - a)[1]) < 0 && ((t[2], t[3]) = (t[3], t[2]))
  end
  grid = Gridap.Geometry.UnstructuredGrid(pts, Gridap.Arrays.Table(cells), [Gridap.ReferenceFEs.LagrangianRefFE(Float64, TRI, 1)],
                                          fill(Int8(1), length(cells)))
  m0 = Gridap.Geometry.UnstructuredDiscreteModel(grid)
  topo = Gridap.Geometry.get_grid_topology(m0)
  m2 = Gridap.Geometry.UnstructuredDiscreteModel(grid, topo, Gridap.Geometry.FaceLabeling(topo))   # tags "interior", "boundary"
  V = TestFESpace(m2, ReferenceFE(lagrangian, Float64, order), dirichlet_tags="boundary"); U = TrialFESpace(V, 0.0)
  dΩ = Measure(Triangulation(m2), 2order)
  φ = solve(AffineFEOperator((u, v) -> ∫(∇(u) ⋅ ∇(v))dΩ, v -> ∫(v)dΩ, U, V))
  area = sum(∫(1.0)dΩ)
  req = sqrt(area / π)
  if jet === nothing
    wj = y -> 1.0; tvec = nothing; ta = 1.0
  else
    jt = merge((power=4.0, taper=0.25), jet)
    dj = [jt.dir[1], jt.dir[2], jt.dir[3]]; dj = dj .- (dj ⋅ nrm) * nrm; dj = dj / norm(dj)
    y0 = jt.offset * req .* (dj ⋅ e1, dj ⋅ e2); rj = jt.radius * req; pw = jt.power; ta = jt.taper
    wj = y -> max(0.0, 1.0 - (((y[1] - y0[1])^2 + (y[2] - y0[2])^2) / rj^2)^(pw / 2))
    tvec = dj
  end
  φa(v) = max(v, 0.0)^ta
  φm = sum(∫((φa ∘ φ) * CellField(y -> wj(y), Triangulation(m2)))dΩ) / area
  # inward normal: side of the plane where the mesh nodes close to the opening lie
  sgn = 0.0
  for x in values(X)
    s = (x .- c) ⋅ nrm; r = norm((x .- c) .- s * nrm)
    (1e-6 < abs(s) < 0.5req && r < 0.5req) && (sgn += sign(s))
  end
  n_in = sgn >= 0 ? nrm : -nrm
  nv = VectorValue(n_in...)
  if tvec !== nothing                              # tilted jet: normal component 1, so ∫ g·n_in = area
    θ = jt.tilt * π / 180
    nv = VectorValue((n_in .+ tan(θ) .* tvec)...)
  end
  # φ tabulated at the P2 nodes of the inlet triangles (vertices and edge midpoints): point evaluation in Gridap
  # searches the 2D mesh at every call, far too slow when the boundary data are updated at every time step
  key(y) = (round(y[1], digits=6), round(y[2], digits=6))
  slow(y) = try φ(Point(y...)) catch; 0.0 end
  tab = Dict{Tuple{Float64,Float64},Float64}()
  for t in cells, (i, j) in ((1, 1), (2, 2), (3, 3), (1, 2), (2, 3), (1, 3))
    y = (pts[t[i]] + pts[t[j]]) / 2
    k = key(y); haskey(tab, k) || (tab[k] = φa(slow((y[1], y[2]))) * wj(y))
  end
  rmax = maximum(norm(pp) for pp in pts); hmin = minimum(norm(pts[t[1]] - pts[t[2]]) for t in cells)
  function g(x)
    xv = [x[1], x[2], x[3]]
    abs((xv .- c) ⋅ nrm) > 0.1hmin && return zero(nv)      # off the plane of the opening
    y = proj(xv); norm(y) > 1.001rmax && return zero(nv)
    v = get(tab, key(y), NaN)
    isnan(v) && (v = φa(slow(y)) * wj(y))
    (max(v, 0.0) / φm) * nv
  end
  g, area, n_in
end

ε(∇u) = 0.5 * (∇u + ∇u')

# Element size h_e: edge of the regular tetrahedron with the volume of the cell, h_e = (6 sqrt(2) |K|)^(1/3)
element_size(Ω) = CellField(map(v -> (6sqrt(2) * v)^(1 / 3), collect(get_cell_measure(Ω))), Ω)

# --- Problem data (as in flow.jl) ------------------------------------------------------------------

# Inflow: half sine in systole (period T 1 s, systole Ts 0.34 s, stroke volume 83.3 ml), zero in diastole
T, Ts, Vs = 1.0, 0.34, 83.3e3                      # s, s, mm³
Qmax = π * Vs / (2Ts); Qmean = Vs / T
Qin(t) = (τ = mod(t, T); τ <= Ts ? Qmax * sin(π * τ / Ts) : 0.0)
# Windkessel (SI; converted to mm-s-g: Pa s/m³ * 1e-9 = Pa s/mm³, m³/Pa * 1e9 = mm³/Pa)
SIR, SIC = 1e-9, 1e9
rcr = [(tag="outlet", Z=1.5e7, C=1.05e-8, R=2.5e8),        # descending aorta: Mousavi et al. 2021, Aneurysmal 1
       (tag="B1", Z=1.36e8, C=4.83e-10, R=9.23e8),          # brachiocephalic trunk: Youssefi et al. 2018, subject A
       (tag="B2", Z=2.46e8, C=2.92e-10, R=1.53e9),          # left common carotid
       (tag="B3", Z=1.74e8, C=3.93e-10, R=1.13e9)]          # left subclavian
wk0 = [(tag=w.tag, Z=w.Z * SIR, C=w.C * SIC, R=w.R * SIR) for w in rcr]
Pc0, _ = windkessel_0d(Qin, wk0; T=T)
wk = [(tag=w.tag, Z=w.Z, C=w.C, R=w.R, Pv=0.0, Pc0=Pc0[i]) for (i, w) in enumerate(wk0)]

# Valve jet (ours, see inlet_profile)
jet = jeton ? (offset=parse(Float64, get(kw, "jet_off", "0.3")), radius=parse(Float64, get(kw, "jet_r", "0.8")),
               tilt=parse(Float64, get(kw, "jet_tilt", "15")), dir=parse.(Float64, split(get(kw, "jet_dir", "-0.4525,-0.8736,-0.179"), ","))) : nothing
g, Ain, nin = inlet_profile(mfile, "inlet"; jet=jet)
model = GmshDiscreteModel(mfile)
opens = ["inlet", "outlet", "B1", "B2", "B3"]
Γo = Dict(o => BoundaryTriangulation(model, tags=o) for o in opens)
dΓo = Dict(o => Measure(Γo[o], 4) for o in opens)
nΓo = Dict(o => get_normal_vector(Γo[o]) for o in opens)
flux(uh, o) = sum(∫(uh ⋅ nΓo[o])dΓo[o])
N = nsteps_arg > 0 ? nsteps_arg : round(Int, 2T / dt)

# --- Solver (navier_stokes_transient of Hemodynamics.jl, orderu = 2, orderp = 1, quad_degree = 4) ----------

tsetup = time()
dir_u_tags = ["inlet", "wall"]                     # wall last: the rim of the inlet is a no-slip node
close_boundary_tags!(model, dir_u_tags)
Ω = Triangulation(model)
dΩ = Measure(Ω, 4)
he = element_size(Ω)
zero3 = VectorValue(0.0, 0.0, 0.0)

# FE spaces: U⁺ (new state), U⁻ and U⁻⁻ (previous states). Dirichlet data g(x)/A · Qin(t) at the inlet and 0 on
# the wall: the unit profile is interpolated once and scaled at every step (the wall values are zero).
reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, 2)
reffep = ReferenceFE(lagrangian, Float64, 1)
Vu = TestFESpace(model, reffeu, conformity=:H1, dirichlet_tags=dir_u_tags)
Vp = TestFESpace(model, reffep, conformity=:H1)
V = MultiFieldFESpace([Vu, Vp])
trial() = MultiFieldFESpace([TrialFESpace(Vu, [x -> g(x) / Ain, zero3]), TrialFESpace(Vp)])
U⁺, U⁻, U⁻⁻ = trial(), trial(), trial()
dvals(U) = get_dirichlet_dof_values(U[1])
gD = copy(dvals(U⁺))
foreach(U -> dvals(U) .= Qin(0.0) .* gD, (U⁺, U⁻, U⁻⁻))
nfree = num_free_dofs(U⁺)
xh⁻ = FEFunction(U⁻, zeros(nfree))
xh⁻⁻ = FEFunction(U⁻⁻, zeros(nfree))
un = xh⁻[1]
unm1 = xh⁻⁻[1]

# BDF2 (first step backward Euler): ∂ₜu ≈ c0 u - (c1 uⁿ - c2 uⁿ⁻¹)
bdf2 = Ref(false)
c0() = bdf2[] ? 1.5 / dt : 1.0 / dt
c1() = bdf2[] ? 2.0 / dt : 1.0 / dt
c2() = bdf2[] ? 0.5 / dt : 0.0
hist() = c1() * un - c2() * unm1                 # known part of the time derivative
wadv() = bdf2[] ? 2.0 * un - 1.0 * unm1 : un     # advecting velocity (Oseen)

# Laplacian of uⁿ, elementwise through the discontinuous P1 space (the columns of the gradient are FE
# functions overwritten in place at every step)
Wlap = FESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), conformity=:L2)
eks = [VectorValue(ntuple(i -> i == k ? 1.0 : 0.0, 3)) for k in 1:3]
gk = [interpolate(∇(un) ⋅ eks[k], Wlap) for k in 1:3]
update_lap!() = foreach(k -> Gridap.FESpaces.interpolate!(∇(un) ⋅ eks[k], get_free_dof_values(gk[k]), Wlap), 1:3)
lapn = Operation((a, b, c) -> VectorValue(a, b, c))(∇ ⋅ gk[1], ∇ ⋅ gk[2], ∇ ⋅ gk[3])

# Windkessel outlets (Robin coupling): traction -(Pc + Z A u·n) n, Pc from the flow of the previous step.
# Pc enters the weak form as a cell field over the vector pcv[i], overwritten at every step.
Γw = [BoundaryTriangulation(model, tags=w.tag) for w in wk]
dΓw = [Measure(Γ, 4) for Γ in Γw]
nΓw = [get_normal_vector(Γ) for Γ in Γw]
wkPc = [w.Pc0 for w in wk]
pcv = [fill(wkPc[i], num_cells(Γw[i])) for i in eachindex(wk)]
Pcf = [CellField(pcv[i], Γw[i]) for i in eachindex(wk)]
wkZA = [w.Z * sum(∫(1.0)dΓw[i]) for (i, w) in enumerate(wk)]
fill!(pcv[1], 1.0); @assert sum(∫(Pcf[1])dΓw[1]) ≈ sum(∫(1.0)dΓw[1]) "Pc cell field does not follow its vector"
fill!(pcv[1], wkPc[1])

# Backflow stabilization (Bazilevs et al. 2009; Esmaily-Moghadam et al. 2011): -β ρ ∫ (w·n)₋ (u·v) dΓ
β_bf = 1.0
Γb = [BoundaryTriangulation(model, tags=tag) for tag in ["outlet", "B1", "B2", "B3"]]
dΓb = [Measure(Γ, 5) for Γ in Γb]
nΓb = [get_normal_vector(Γ) for Γ in Γb]
negpart(x) = 0.5 * (x - abs(x))

# Stabilization parameters (semi-discrete), evaluated with uⁿ
τsupg(u, h) = ((2 * norm(u) / h)^2 + 9 * (4μ / (ρ * h^2))^2)^(-1 / 2)
τpspg(u, h) = τsupg(u, h) / ρ
τlsic(u, h) = ρ * norm(u) * h * 0.5
Rvel(c) = (u, ∇u, w) -> ρ * c * u + ρ * (∇u' ⋅ w)

# Bilinear form of the step
function a((u, p), (v, q))
  c = c0(); w = wadv()
  τs = τsupg ∘ (un, he); τp = τpspg ∘ (un, he); τl = τlsic ∘ (un, he)
  Ru = Rvel(c) ∘ (u, ∇(u), w)
  m = ∫(ρ * c * (u ⋅ v))dΩ + ∫(2μ * (ε ∘ (∇(u))) ⊙ (ε ∘ (∇(v))))dΩ + ∫(ρ * ((∇(u)' ⋅ w) ⋅ v))dΩ -
      ∫(p * (∇ ⋅ v))dΩ - ∫(q * (∇ ⋅ u))dΩ +
      ∫(τs * ((∇(v)' ⋅ w) ⋅ Ru))dΩ + ∫(τs * ((∇(v)' ⋅ w) ⋅ ∇(p)))dΩ -
      ∫(τp * (∇(q) ⋅ Ru))dΩ - ∫(τp * (∇(q) ⋅ ∇(p)))dΩ +
      ∫(τl * (∇ ⋅ v) * (∇ ⋅ u))dΩ
  for i in eachindex(dΓb)
    m = m - ∫(β_bf * ρ * (negpart ∘ (w ⋅ nΓb[i])) * (u ⋅ v))dΓb[i]
  end
  for i in eachindex(wk)
    m = m + ∫(wkZA[i] * (u ⋅ nΓw[i]) * (v ⋅ nΓw[i]))dΓw[i]
  end
  m
end
# Linear form of the step (known part of the time derivative, Laplacian of uⁿ, Windkessel pressures)
function l((v, q))
  w = wadv(); f = hist()
  τs = τsupg ∘ (un, he); τp = τpspg ∘ (un, he)
  rk = ρ * f + μ * lapn
  r = ∫(ρ * (f ⋅ v))dΩ + ∫(τs * ((∇(v)' ⋅ w) ⋅ rk))dΩ - ∫(τp * (∇(q) ⋅ rk))dΩ
  for i in eachindex(wk)
    r = r - ∫(Pcf[i] * (v ⋅ nΓw[i]))dΓw[i]
  end
  r
end

# Gridap: matrix and vector with the Dirichlet values lifted (as in Hemodynamics.jl)
assem = SparseMatrixAssembler(U⁺, V)
du = get_trial_fe_basis(U⁺); dv = get_fe_basis(V)
gridap_data() = collect_cell_matrix_and_vector(U⁺, V, a(du, dv), l(dv), zero(U⁺))
# HyperFEMKernels: the step as a (linear) nonlinear operator, r(x) = a(x, ·) - l(·), J = a; with x = 0 (free
# values) and the Dirichlet values of U⁺, Gridap's vector is -r(0) and its matrix J
kres(x, y) = a(x, y) - l(y)
kjac(x, dx, y) = a(dx, y)
x0 = zeros(nfree)
tsetup = time() - tsetup

@printf("Aorta %s: %d tetrahedra, %d free DOFs (u %d, p %d), inlet area %.2f mm², jet %s, %d steps of %.4f s, mode %s, %d threads\n",
        meshname, num_cells(Ω), nfree, num_free_dofs(U⁺[1]), num_free_dofs(U⁺[2]), Ain, jeton ? "on" : "off", N, dt, mode,
        Threads.nthreads())
@printf("  setup (spaces, forms): %.2f s\n", tsetup); flush(stdout)

usek = mode != "gridap"; useg = mode != "kernel"; use1 = mode == "compare" && threads1
op = op1 = nothing
if usek
  tb = @elapsed (op = KernelOperator(kres, kjac, U⁺, V))
  tb1 = use1 ? @elapsed(op1 = KernelOperator(kres, kjac, U⁺, V; nthreads=1)) : 0.0
  nd = count(g -> g.lin === nothing, op.jgroups)
  @printf("  KernelOperator: %.2f s (1 thread: %.2f s); %d residual and %d Jacobian groups (%d with dual numbers); fallback: residual %s, Jacobian %s\n",
          tb, tb1, length(op.rgroups), length(op.jgroups), nd,
          isempty(op.fallback.rtrians) ? "none" : string(length(op.fallback.rtrians)),
          isempty(op.fallback.jtrians) ? "none" : string(length(op.fallback.jtrians)))
  flush(stdout)
end

rk = usek ? Gridap.Algebra.allocate_residual(op, x0) : nothing
Ak = usek ? Gridap.Algebra.allocate_jacobian(op, x0) : nothing
rk1 = use1 ? Gridap.Algebra.allocate_residual(op1, x0) : nothing
Ak1 = use1 ? Gridap.Algebra.allocate_jacobian(op1, x0) : nothing
Ag = bg = nothing; ns = nothing; x = zeros(nfree)
lsolver = LUSolver()
rel(a, b) = norm(a - b) / norm(b)

hio = open(joinpath(outd, "history.csv"), "w"); println(hio, "t,Q_in,Q_desc,Q_B1,Q_B2,Q_B3")
tio = open(joinpath(outd, "timings.csv"), "w"); println(tio, "step,t,bc,asm_gridap,asm_kernel,asm_kernel1,update,lu,solve,post,total")
dio = mode == "compare" ? open(joinpath(outd, "diffs.csv"), "w") : nothing
dio === nothing || println(dio, "step,t,dA,db,dA1,db1")
rows = NamedTuple[]
ok = true
for n in 1:N
  global Ag, bg, ns, op, op1, ok
  t = n * dt
  tstep = time()
  # compliance pressures for this step (flow of the previous step) and Dirichlet data at tⁿ⁺¹
  tbc = @elapsed begin
    for i in eachindex(wk)
      w = wk[i]; Q = sum(∫(un ⋅ nΓw[i])dΓw[i])
      wkPc[i] = (wkPc[i] + dt * Q / w.C + dt * w.Pv / (w.R * w.C)) / (1 + dt / (w.R * w.C))
      fill!(pcv[i], wkPc[i])
    end
    dvals(U⁺) .= Qin(t) .* gD
  end
  tg = tk = tk1 = 0.0
  if useg
    tg = @elapsed if Ag === nothing
      Ag, bg = assemble_matrix_and_vector(assem, gridap_data())
    else
      assemble_matrix_and_vector!(Ag, bg, assem, gridap_data())
    end
  end
  if usek
    tk = @elapsed (Gridap.Algebra.residual!(rk, op, x0); Gridap.Algebra.jacobian!(Ak, op, x0))
  end
  if use1
    tk1 = @elapsed (Gridap.Algebra.residual!(rk1, op1, x0); Gridap.Algebra.jacobian!(Ak1, op1, x0))
  end
  if mode == "compare"
    dA, db = rel(Ak, Ag), rel(-rk, bg)
    dA1, db1 = use1 ? (rel(Ak1, Ag), rel(-rk1, bg)) : (NaN, NaN)
    println(dio, join((n, t, dA, db, dA1, db1), ",")); flush(dio)
    (n <= 3 || n % 10 == 0 || n == N) && @printf("  step %4d  ‖ΔA‖/‖A‖ %.2e  ‖Δb‖/‖b‖ %.2e  (1 thread %.2e %.2e)\n", n, dA, db, dA1, db1)
  end
  A, b = mode == "kernel" ? (Ak, -rk) : (Ag, bg)
  tlu = @elapsed if ns === nothing
    ns = numerical_setup(symbolic_setup(lsolver, A), A)
  else
    numerical_setup!(ns, A)
  end
  tsol = @elapsed Gridap.Algebra.solve!(x, ns, b)
  if !all(isfinite, x)
    @warn "non-finite solution at step $n (t = $t)"; ok = false; break
  end
  tpost = @elapsed begin
    xh⁺ = FEFunction(U⁺, x)
    q = [flux(xh⁺[1], o) for o in opens]
    println(hio, join([t; -q[1]; q[2:end]], ",")); flush(hio)
    # previous states, overwritten in place: xh⁻⁻ ← xh⁻ ← xh⁺
    get_free_dof_values(xh⁻⁻) .= get_free_dof_values(xh⁻)
    dvals(U⁻⁻) .= dvals(U⁻)
    get_free_dof_values(xh⁻) .= x
    dvals(U⁻) .= dvals(U⁺)
    update_lap!()
  end
  # after the backward Euler step: BDF2 coefficients (numbers of the weak form) → rebuild the kernel operators
  tupd = 0.0
  if n == 1
    bdf2[] = true
    tupd = @elapsed begin
      usek && (op = HyperFEMKernels.update(op, kres, kjac))
      use1 && (op1 = HyperFEMKernels.update(op1, kres, kjac))
    end
  end
  ttot = time() - tstep
  r = (step=n, t=t, bc=tbc, asm_gridap=tg, asm_kernel=tk, asm_kernel1=tk1, update=tupd, lu=tlu, solve=tsol, post=tpost, total=ttot)
  push!(rows, r); println(tio, join(values(r), ",")); flush(tio)
  if n % 10 == 0 || n == N || n <= 3
    @printf("  step %4d t %.3f  Qin %8.0f  Qdesc %8.0f | bc %.2f  asm gridap %.3f  kernel %.3f (1 thr %.3f)  update %.2f  LU %.2f  solve %.2f  post %.2f  total %.2f s\n",
            n, t, -q[1], q[2], tbc, tg, tk, tk1, tupd, tlu, tsol, tpost, ttot)
    flush(stdout)
  end
end
close(hio); close(tio); dio === nothing || close(dio)

# Summary: mean over the steps after the two that compile (step 1, and step 2 after `update`)
mean_(v) = isempty(v) ? NaN : sum(v) / length(v)
rs = length(rows) > 2 ? rows[3:end] : rows
avg(k) = mean_([getfield(r, k) for r in rs])
@printf("\nSUMMARY %s mode %s | %d DOFs | mean over steps %d-%d (s/step):\n", meshname, mode, nfree, rs[1].step, rs[end].step)
@printf("  bc %.3f | assembly: gridap %.3f, kernel %d thr %.3f, kernel 1 thr %.3f | LU %.3f | solve %.3f | post %.3f | total %.3f\n",
        avg(:bc), avg(:asm_gridap), Threads.nthreads(), avg(:asm_kernel), avg(:asm_kernel1), avg(:lu), avg(:solve), avg(:post), avg(:total))
if mode == "compare"
  @printf("  assembly speed-up vs Gridap: %.1fx (%d threads), %.1fx (1 thread)\n",
          avg(:asm_gridap) / avg(:asm_kernel), Threads.nthreads(), avg(:asm_gridap) / avg(:asm_kernel1))
  gstep = avg(:bc) + avg(:asm_gridap) + avg(:lu) + avg(:solve) + avg(:post)
  kstep = avg(:bc) + avg(:asm_kernel) + avg(:lu) + avg(:solve) + avg(:post)
  @printf("  step time (estimated from the phases): gridap %.3f s, kernel %.3f s (%.2fx); first steps (compilation): kernel %.2f / %.2f s, update %.2f s\n",
          gstep, kstep, gstep / kstep, rows[1].asm_kernel, length(rows) > 1 ? rows[2].asm_kernel : NaN, rows[1].update)
end
println(ok ? "FIN AORTA" : "FIN AORTA (non-finite solution)")

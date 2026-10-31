# Refinable meshes with the same geometry and physical tags as the HyperFEM examples.
#
#   julia --project=benchmark benchmark/meshes/generate.jl em n      # EM plate, structured hexahedra
#   julia --project=benchmark benchmark/meshes/generate.jl hmre r [rv]   # hMRE Short_magnet, triangles
#   julia --project=benchmark benchmark/meshes/generate.jl turek h   # DFG flow around a cylinder 2D, triangles
#
# em n   : plate 8 × 0.8 × 40 (x, y, z) in two layers y ∈ [0, 0.4] and [0.4, 0.8], like
#          HyperFEM test/models/test_static_EM.msh. Divisions (7n, n + n, 19n): n = 1 gives the
#          original 266 hexahedra; cells grow as n³.
#          Tags: "Volume", "fixedup" (z = 0), "midsuf" (y = 0.4), "topsuf" (y = 0.8).
# hmre r : geometry of test/models/Short_magnet.msh (beam 0.05 × 0.002,
#          magnet, vacuum box and outer box) with all its physical tags. Element sizes ≈ 5e-4 on the
#          beam, 3e-3 on the vacuum box and magnet, 1e-2 on the outer box, divided by r; r = 1
#          reproduces the original element counts closely; cells grow as r². The optional rv (default r)
#          refines the magnet and the vacuum boxes only by rv, to grow the beam (the solid DOFs) without
#          growing the vacuum: `hmre 30 2` for the solver baseline.
#
# turek h: channel of the DFG benchmark "flow around a cylinder" 2D (Schäfer & Turek 1996): rectangle
#          2.2 × 0.41 with a circle of radius 0.05 centred at (0.2, 0.2); element size h on the channel and
#          h/5 on the cylinder. Tags: "fluid", "inlet" (x = 0), "outlet" (x = 2.2), "wall" (y = 0, 0.41),
#          "cylinder" (points included; the corners of the inlet/outlet belong to the wall too).
#          The test mesh of HyperFEM (test/models/turek2d.msh, TurekNavierStokes simulation) is `turek 0.02`.
#
# Output: benchmark/meshes/em_plate_n<n>.msh, hmre_short_r<r>.msh or turek_h<h>.msh (git-ignored).

using GridapGmsh: gmsh

const OUTDIR = @__DIR__

function em_plate(n::Int)
    file = joinpath(OUTDIR, "em_plate_n$(n).msh")
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("em_plate")
    geo = gmsh.model.geo
    p = [geo.addPoint(x...) for x in ((0, 0, 0), (8, 0, 0), (8, 0, 40), (0, 0, 40))]
    l = [geo.addLine(p[i], p[mod1(i + 1, 4)]) for i in 1:4]
    s = geo.addPlaneSurface([geo.addCurveLoop(l)])
    for (c, m) in zip(l, (7n, 19n, 7n, 19n))
        geo.mesh.setTransfiniteCurve(c, m + 1)
    end
    geo.mesh.setTransfiniteSurface(s); geo.mesh.setRecombine(2, s)
    e1 = geo.extrude([(2, s)], 0, 0.4, 0, [n], [], true)
    e2 = geo.extrude([(2, e1[1][2])], 0, 0.4, 0, [n], [], true)
    geo.synchronize()
    ε = 1e-6
    inbox(x0, y0, z0, x1, y1, z1, d) = [t for (_, t) in gmsh.model.getEntitiesInBoundingBox(x0 - ε, y0 - ε, z0 - ε, x1 + ε, y1 + ε, z1 + ε, d)]
    groups = (("midsuf", 1, (0, 0.4, 0, 8, 0.4, 40)), ("topsuf", 2, (0, 0.8, 0, 8, 0.8, 40)), ("fixedup", 3, (0, 0, 0, 8, 0.8, 0)))
    for (name, id, box) in groups, d in 0:2
        gmsh.model.addPhysicalGroup(d, inbox(box..., d), id, name)
    end
    gmsh.model.addPhysicalGroup(3, [e1[2][2], e2[2][2]], 1, "Volume")
    gmsh.model.mesh.generate(3)
    finish(file)
end

function hmre_short(r::Real, rv::Real=r)
    file = joinpath(OUTDIR, rv == r ? "hmre_short_r$(r).msh" : "hmre_short_r$(r)_rv$(rv).msh")
    # point sizes and transfinite counts fitted to the segment counts of Short_magnet.msh (r = 1)
    hb, hv, hm, ho = 2.4e-3 / r, 3e-3 / rv, 8e-3 / rv, 1e-2 / rv
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("hmre_short")
    geo = gmsh.model.geo
    P = Dict{Int,Int}()
    pt(tag, x, y, h) = (P[tag] = geo.addPoint(x, y, 0.0, h, tag))
    # tags as in Short_magnet.msh
    pt(13, 0.0, -0.001, hb); pt(15, 0.0025, -0.001, hb); pt(16, 0.05, -0.001, hb)
    pt(17, 0.05, 0.001, hb); pt(18, 0.0025, 0.001, hb); pt(19, 0.0, 0.001, hb)
    pt(14, 0.0, -0.075, hv); pt(22, 0.075, -0.075, hv); pt(21, 0.075, 0.1125, hv); pt(20, 0.0, 0.1125, hv)
    pt(23, 0.0, -0.1, hm); pt(25, 0.075, -0.1, hm)
    pt(24, 0.0, -0.3, ho); pt(28, 0.35, -0.3, ho); pt(27, 0.35, 0.15, ho); pt(26, 0.0, 0.15, ho)
    L(tag, a, b) = geo.addLine(P[a], P[b], tag)
    L(1, 13, 14); L(2, 13, 15); L(3, 15, 16); L(4, 16, 17); L(5, 17, 18); L(6, 18, 19)
    L(7, 20, 19); L(8, 21, 20); L(9, 22, 21); L(10, 14, 22); L(11, 23, 24); L(12, 23, 25)
    L(13, 25, 22); L(14, 26, 20); L(15, 27, 26); L(16, 28, 27); L(17, 24, 28); L(18, 15, 18)
    L(19, 19, 13); L(20, 14, 23)
    # beam edges (≈ 5e-4 / r) and magnet edges (≈ 3e-3 / r) with fixed divisions
    for (c, m) in ((2, 5), (3, 95), (4, 4), (5, 95), (6, 5), (19, 4), (18, 4))      # beam edges, by r
        geo.mesh.setTransfiniteCurve(c, round(Int, m * r) + 1)
    end
    for (c, m) in ((10, 25), (12, 25), (13, 9), (20, 9))                             # magnet edges, by rv
        geo.mesh.setTransfiniteCurve(c, round(Int, m * rv) + 1)
    end
    S(tag, loop) = geo.addPlaneSurface([geo.addCurveLoop(loop, tag)], tag)
    S(1, [2, 18, 6, 19])                                   # beam, x ≤ 0.0025
    S(2, [3, 4, 5, -18])                                   # beam
    S(4, [12, 13, -10, 20])                                # magnet
    S(5, [1, 10, 9, 8, 7, -6, -5, -4, -3, -2])            # vacuum box around the beam
    S(6, [11, 17, 16, 15, 14, -8, -9, -13, -12])           # outer vacuum
    geo.synchronize()
    phys0 = Dict(5 => ("phi_Dirichlet", [27]), 8 => ("Fixed_Solid", [13, 15]), 9 => ("Fixed_Solid_x", [19]),
                 11 => ("Interface", [13, 15, 16, 17, 18, 19]), 13 => ("Fixed_Air", [14, 22, 23, 24, 25, 26, 27, 28]),
                 16 => ("Fixed_Air_b", [24, 26, 27, 28]), 18 => ("phibot", [24, 28]), 20 => ("phitop", [26, 27]))
    phys1 = Dict(6 => ("Fixed_Solid_x", [19]), 7 => ("Fixed_Solid", [2]), 10 => ("Interface", [2, 3, 4, 5, 6]),
                 12 => ("Fixed_Air", [10, 12, 13, 15, 16, 17]), 14 => ("Fixed_Air_x", [1, 7, 11, 14]),
                 15 => ("Fixed_Air_b", [15, 16, 17]), 17 => ("phibot", [17]), 19 => ("phitop", [15]))
    phys2 = Dict(1 => ("Beam", [1, 2]), 2 => ("Magnet", [4]), 4 => ("Vacuum", [5, 6]), 21 => ("vacuumb", [4, 5, 6]))
    for (d, ph) in ((0, phys0), (1, phys1), (2, phys2)), (id, (name, tags)) in ph
        gmsh.model.addPhysicalGroup(d, tags, id, name)
    end
    gmsh.model.mesh.generate(2)
    finish(file)
end

function turek(h::Real)
    file = joinpath(OUTDIR, "turek_h$(h).msh")
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("turek")
    geo = gmsh.model.geo
    L, H, cx, cy, r = 2.2, 0.41, 0.2, 0.2, 0.05
    hc = h / 5
    p1 = geo.addPoint(0, 0, 0, h); p2 = geo.addPoint(L, 0, 0, h); p3 = geo.addPoint(L, H, 0, h); p4 = geo.addPoint(0, H, 0, h)
    bottom = geo.addLine(p1, p2); outlet = geo.addLine(p2, p3); top = geo.addLine(p3, p4); inlet = geo.addLine(p4, p1)
    c0 = geo.addPoint(cx, cy, 0, hc)
    cp = [geo.addPoint(cx + r * cos(θ), cy + r * sin(θ), 0, hc) for θ in (0, π / 2, π, 3π / 2)]
    arcs = [geo.addCircleArc(cp[i], c0, cp[mod1(i + 1, 4)]) for i in 1:4]
    s = geo.addPlaneSurface([geo.addCurveLoop([bottom, outlet, top, inlet]), geo.addCurveLoop(arcs)])
    geo.synchronize()
    gmsh.model.addPhysicalGroup(2, [s], 1, "fluid")
    gmsh.model.addPhysicalGroup(1, [inlet], 2, "inlet"); gmsh.model.addPhysicalGroup(0, [p1, p4], 2, "inlet")
    gmsh.model.addPhysicalGroup(1, [outlet], 3, "outlet"); gmsh.model.addPhysicalGroup(0, [p2, p3], 3, "outlet")
    gmsh.model.addPhysicalGroup(1, [bottom, top], 4, "wall"); gmsh.model.addPhysicalGroup(0, [p1, p2, p3, p4], 4, "wall")
    gmsh.model.addPhysicalGroup(1, arcs, 5, "cylinder"); gmsh.model.addPhysicalGroup(0, cp, 5, "cylinder")
    gmsh.model.mesh.generate(2)
    finish(file)
end

function finish(file)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    for d in 1:3
        types, tags, _ = gmsh.model.mesh.getElements(d)
        isempty(types) || println("dim $d: ", sum(length, tags), " elements")
    end
    for (d, t) in gmsh.model.getPhysicalGroups()
        if d == maximum(first.(gmsh.model.getPhysicalGroups()))
            n = sum((length(gmsh.model.mesh.getElements(d, e)[2]) > 0 ? length(gmsh.model.mesh.getElements(d, e)[2][1]) : 0)
                    for e in gmsh.model.getEntitiesForPhysicalGroup(d, t))
            println("  ", gmsh.model.getPhysicalName(d, t), ": $n cells")
        end
    end
    println("nodes: ", length(gmsh.model.mesh.getNodes()[1]))
    gmsh.write(file)
    gmsh.finalize()
    println("written $file")
end

# Periodic unit cell of a magneto-active composite (the "periodic cell" study of 2026-09): a square / cube of
# side 1 with a circular / spherical inclusion of radius R at the centre, conforming interface (OCC fragment),
# periodic opposite faces (Gmsh periodic nodes, which GridapGmsh identifies into a torus topology) and the tags
# "matriz", "inclusion", "interfaz", the faces and "corner_dirichlet" (all the corners: one periodic node, where
# u = 0 and φ = 0 remove the rigid translations and the constant of the potential). Geometric order 1 (required
# by GridapGmsh for periodic meshes); the FE order is chosen in Julia. h: element size in the matrix, h/3 on the
# inclusion. The 2D cell of the study (R = 0.175, h = 0.03, h_inc = 0.01) is `cell 2 0.03 0.175`.
function cell(dim::Int, h::Real, R::Real=0.25; hinc::Real=h / 3)
    file = joinpath(OUTDIR, "cell$(dim)d_h$(h)" * (R == 0.25 ? "" : "_R$(R)") * ".msh")
    L = 1.0; tol = 1e-6
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cell")
    occ = gmsh.model.occ
    if dim == 2
        body = occ.addRectangle(0, 0, 0, L, L); incl = occ.addDisk(L / 2, L / 2, 0, R, R)
    else
        body = occ.addBox(0, 0, 0, L, L, L); incl = occ.addSphere(L / 2, L / 2, L / 2, R)
    end
    out, outmap = occ.fragment([(dim, body)], [(dim, incl)])
    occ.synchronize()
    inclusion = Int[t for (_, t) in outmap[2]]
    matriz = Int[t for (d, t) in out if d == dim && !(t in inclusion)]
    inbox(lo, hi, d) = [t for (_, t) in gmsh.model.getEntitiesInBoundingBox((lo .- tol)..., (hi .+ tol)..., d)]
    # faces (dim-1 entities) by bounding box; in 2D the z coordinate is 0
    lo3(v) = dim == 2 ? (v[1], v[2], 0.0) : v
    face(axis, x) = inbox(lo3(ntuple(i -> i == axis ? x : 0.0, dim)), lo3(ntuple(i -> i == axis ? x : L, dim)), dim - 1)
    names = dim == 2 ? ("izquierda", "derecha", "abajo", "arriba") : ("izquierda", "derecha", "abajo", "arriba", "atras", "delante")
    faces = [face(a, x) for a in 1:dim for x in (0.0, L)]
    for (n, f) in zip(names, faces)
        length(f) == 1 || error("face $n: expected 1 entity, found $(length(f))")
    end
    interfaz = Int[abs(t) for (_, t) in gmsh.model.getBoundary([(dim, t) for t in inclusion], true, false, false)]
    corners = vcat([inbox(lo3(c), lo3(c), 0) for c in Iterators.product(ntuple(_ -> (0.0, L), dim)...)]...)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
    gmsh.model.mesh.setSize(gmsh.model.getBoundary([(dim, t) for t in inclusion], false, false, true), hinc)
    trans(a) = [i == j ? 1.0 : (j == 4 && i == a ? L : 0.0) for i in 1:4 for j in 1:4]   # affine translation by L along axis a (row-major)
    for a in 1:dim
        gmsh.model.mesh.setPeriodic(dim - 1, faces[2a], faces[2a-1], trans(a))             # slave x = L ← master x = 0
    end
    gmsh.model.addPhysicalGroup(dim, matriz, -1, "matriz")
    gmsh.model.addPhysicalGroup(dim, inclusion, -1, "inclusion")
    for (n, f) in zip(names, faces)
        gmsh.model.addPhysicalGroup(dim - 1, f, -1, n)
    end
    gmsh.model.addPhysicalGroup(dim - 1, interfaz, -1, "interfaz")
    gmsh.model.addPhysicalGroup(0, corners, -1, "corner_dirichlet")
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.model.mesh.generate(dim)
    for a in 1:dim
        _, nodes, master, _ = gmsh.model.mesh.getPeriodicNodes(dim - 1, faces[2a][1])
        err = maximum((maximum(abs.(gmsh.model.mesh.getNode(n)[1] .- gmsh.model.mesh.getNode(m)[1] .- [i == a ? L : 0.0 for i in 1:3])) for (n, m) in zip(nodes, master)); init=0.0)
        println("periodicity along axis $a: $(length(nodes)) node pairs, max error $err")
    end
    finish(file)
end

kind = ARGS[1]
kind == "cell" ? cell(parse(Int, ARGS[2]), parse(Float64, ARGS[3]), parse(Float64, get(ARGS, 4, "0.25"))) :
kind == "em" ? em_plate(parse(Int, ARGS[2])) :
kind == "hmre" ? hmre_short(parse(Float64, ARGS[2]), parse(Float64, get(ARGS, 3, ARGS[2]))) :
kind == "turek" ? turek(parse(Float64, ARGS[2])) : error("usage: generate.jl em n | hmre r [rv] | turek h | cell dim h [R]")

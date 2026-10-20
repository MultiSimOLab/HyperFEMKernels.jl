# Refinable meshes with the same geometry and physical tags as the HyperFEM examples.
#
#   julia --project=benchmark benchmark/meshes/generate.jl em n      # EM plate, structured hexahedra
#   julia --project=benchmark benchmark/meshes/generate.jl hmre r    # hMRE Short_magnet, triangles
#
# em n   : plate 8 × 0.8 × 40 (x, y, z) in two layers y ∈ [0, 0.4] and [0.4, 0.8], like
#          HyperFEM test/models/test_static_EM.msh. Divisions (7n, n + n, 19n): n = 1 gives the
#          original 266 hexahedra; cells grow as n³.
#          Tags: "Volume", "fixedup" (z = 0), "midsuf" (y = 0.4), "topsuf" (y = 0.8).
# hmre r : geometry of Adv_Mater_Ultra_Soft_Magnet data/models/Short_magnet.msh (beam 0.05 × 0.002,
#          magnet, vacuum box and outer box) with all its physical tags. Element sizes ≈ 5e-4 on the
#          beam, 3e-3 on the vacuum box and magnet, 1e-2 on the outer box, divided by r; r = 1
#          reproduces the original element counts closely; cells grow as r².
#
# Output: benchmark/meshes/em_plate_n<n>.msh or hmre_short_r<r>.msh (git-ignored).

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

function hmre_short(r::Real)
    file = joinpath(OUTDIR, "hmre_short_r$(r).msh")
    # point sizes and transfinite counts fitted to the segment counts of Short_magnet.msh (r = 1)
    hb, hv, hm, ho = 2.4e-3 / r, 3e-3 / r, 8e-3 / r, 1e-2 / r
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
    for (c, m) in ((2, 5), (3, 95), (4, 4), (5, 95), (6, 5), (19, 4), (18, 4), (10, 25), (12, 25), (13, 9), (20, 9))
        geo.mesh.setTransfiniteCurve(c, round(Int, m * r) + 1)
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

kind = ARGS[1]
kind == "em" ? em_plate(parse(Int, ARGS[2])) :
kind == "hmre" ? hmre_short(parse(Float64, ARGS[2])) : error("usage: generate.jl em n | hmre r")

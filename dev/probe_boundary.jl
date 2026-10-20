# What Gridap 0.19.9 builds for a boundary integral: glue, quadrature on the faces, normals,
# and the lazy tree of ∫(v*u)dΓ and ∫((n⋅∇(u))*v)dΓ with probes.
using HyperFEMKernels, Gridap, Gridap.Geometry, Gridap.CellData, Gridap.Arrays, Gridap.FESpaces, Gridap.ReferenceFEs
const HK = HyperFEMKernels
model = CartesianDiscreteModel((0, 1, 0, 1), (3, 2))
Ω = Triangulation(model)
Γ = BoundaryTriangulation(model; tags=[6, 8])
dΓ = Measure(Γ, 2)
V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 2), dirichlet_tags=[1]); U = TrialFESpace(V, 0.0)
println("Γ: ", typeof(Γ).name.name, " ncells=", num_cells(Γ), " D=", num_cell_dims(Γ))
g = Γ.glue
println("glue fields: ", fieldnames(typeof(g)))
println("face_to_cell: ", g.face_to_cell)
println("face_to_lface: ", g.face_to_lface)
println("face_to_ftype: ", g.face_to_ftype)
println("cell_to_ctype: ", typeof(g.cell_to_ctype))
q = Gridap.CellData.get_cell_quadrature(dΓ)
println("quad points (face ref) uniform? ", HK.uniform_value(q.cell_point, "pts"))
println("weights: ", HK.uniform_value(q.cell_weight, "w"))
# points mapped to the cell reference domain
pts = get_cell_points(dΓ)
cell_pts = get_data(pts)
println("cell_points type: ", typeof(cell_pts))
println("cell point 1: ", cell_pts[1]); println("cell point 2: ", cell_pts[2])
n = get_normal_vector(Γ)
println("normal type: ", typeof(get_data(n)))
println("normal values cell 1: ", get_data(n)[1])
nx = n(pts); println("n at points: ", typeof(nx).name.name, " ", nx[1], " ", nx[2])
# the lazy tree with probes
pu = HK.probes(U, :unknown); pv = HK.probes(V, :test)
dc = ∫(pv * pu)dΓ
arr = dc.dict[Γ]
println("integral maps: ", typeof(first(arr.maps)), " nargs=", length(arr.args))
for (i, a) in enumerate(arr.args); println("  arg $i: ", typeof(a).name.name, " ", eltype(a)); end
HK.show_tree(stdout, HK.to_node(arr.args[1]))
dc2 = ∫((n ⋅ ∇(pu)) * pv)dΓ
arr2 = dc2.dict[Γ]
try
    HK.show_tree(stdout, HK.to_node(arr2.args[1]))
catch e; println("tree with normal: ", e); end
# ∇(pu) on Γ: what array type does Gridap produce (reference gradient at the cell points?)
gu = ∇(pu)
x = gu(pts)
println("∇u on Γ: ", typeof(x).name.name, " eltype ", eltype(x))
println("Jt on Γ (args[3]): ", typeof(arr.args[3]), " ", arr.args[3][1])

glue = Geometry.get_glue(Γ, Val(2))
println("glue D=2: ", typeof(glue).name.name, " fields ", fieldnames(typeof(glue)))
println("tface_to_mface: ", glue.tface_to_mface)
fmap = glue.tface_to_mface_map
println("mface_map type: ", typeof(fmap).name.name, " ", typeof(fmap[1]))
cpts = lazy_map(evaluate, fmap, get_data(pts))
println("cell-ref points per face: ", typeof(cpts).name.name); for i in 1:5; println("  face $i (lface $(g.face_to_lface[i])): ", cpts[i]); end
jf = lazy_map(∇, fmap); println("face map jacobian: ", typeof(jf[1]), " ", evaluate(jf[1], get_data(pts)[1]))
println("ref normals per lface: ", Gridap.Geometry.get_facet_normal(Γ) |> typeof |> x->x.name.name)

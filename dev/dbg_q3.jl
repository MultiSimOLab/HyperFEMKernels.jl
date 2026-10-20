using HyperFEMKernels, Gridap, Gridap.FESpaces, Gridap.CellData, Gridap.Arrays
const HK = HyperFEMKernels
model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (2, 2, 2))
Ω = Triangulation(model); dΩ = Measure(Ω, 6)
V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 3))
xq = HK.uniform_value(get_data(get_cell_points(dΩ)), "p")
fields = HK.uniform_value(get_data(get_fe_basis(V)), "b")
probe_pts = [Point(ntuple(i -> 0.1 + 0.07 * (k + i) % 0.6, 3)) for k in 1:7]
pts = vcat(collect(xq), probe_pts)
vals = evaluate(fields, pts)
nd = size(vals, 2); nc = 3
comp = [findfirst(c -> any(q -> abs(vals[q, j][c]) > 0, axes(vals, 1)), 1:nc) for j in 1:nd]
println("comp counts: ", [count(==(c), comp) for c in 1:3])
keys = [[vals[q, j][comp[j]] for q in axes(vals, 1)] for j in 1:nd]
reps = Vector{Float64}[]; node = zeros(Int, nd)
for j in 1:nd
    n = findfirst(r -> isapprox(r, keys[j]; atol=1e-10, rtol=0.0), reps)
    n === nothing && (push!(reps, keys[j]); n = length(reps)); node[j] = n
end
println("nn = ", length(reps), " ; nodes with ≠3 dofs: ", count(n -> count(==(n), node) != 3, 1:length(reps)))
# max distance between closest different keys
using LinearAlgebra
d = [norm(keys[j] - keys[k], Inf) for j in 1:nd for k in j+1:nd]
println("min key distance: ", minimum(d))

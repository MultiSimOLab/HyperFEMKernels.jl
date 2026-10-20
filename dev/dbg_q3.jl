using HyperFEMKernels, Gridap, Gridap.FESpaces, Gridap.CellData, Gridap.Arrays
const HK = HyperFEMKernels
model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (2, 2, 2))
for order in (2, 3), T in (Float64, VectorValue{3,Float64})
    Ω = Triangulation(model); dΩ = Measure(Ω, 2order)
    V = TestFESpace(model, ReferenceFE(lagrangian, T, order))
    xq = HK.uniform_value(get_data(get_cell_points(dΩ)), "p")
    fields = HK.uniform_value(get_data(get_fe_basis(V)), "b")
    vals = evaluate(fields, xq)
    nd = size(vals, 2); nc = HK.ncomps(eltype(vals))
    keys = Set()
    for j in 1:nd
        c = nc == 1 ? 1 : findfirst(c -> any(q -> abs(vals[q, j][c]) > 0, axes(vals, 1)), 1:nc)
        push!(keys, [nc == 1 ? vals[q, j] : vals[q, j][c] for q in axes(vals, 1)])
    end
    println("order $order $T: nd=$nd nc=$nc distinct keys=$(length(keys)) expected nodes=$(nd ÷ nc)")
    try; HK.FieldTable(V, xq, 3); println("  FieldTable ok"); catch e; println("  ", e); end
end

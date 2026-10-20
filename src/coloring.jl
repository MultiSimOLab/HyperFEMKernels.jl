# Cell colouring for race-free threaded assembly: two cells of the same colour never share a
# free (test) dof, so they can write to the residual and to the matrix concurrently.

function color_cells(fields::Vector{FieldLayout}, ncells::Int, nrows::Int)
    masks = zeros(UInt128, nrows)
    color = zeros(Int, ncells)
    for cell in 1:ncells
        used = UInt128(0)
        for fl in fields, j in 1:fl.nd
            r = fl.test_ids[j, cell]
            r > 0 && (used |= masks[fl.test_off+r])
        end
        c = trailing_ones(used)
        c < 128 || error("HyperFEMKernels: more than 128 colours needed")
        color[cell] = c + 1
        bit = UInt128(1) << c
        for fl in fields, j in 1:fl.nd
            r = fl.test_ids[j, cell]
            r > 0 && (masks[fl.test_off+r] |= bit)
        end
    end
    ncol = maximum(color; init=0)
    groups = [Int32[] for _ in 1:ncol]
    for cell in 1:ncells
        push!(groups[color[cell]], cell)
    end
    groups
end

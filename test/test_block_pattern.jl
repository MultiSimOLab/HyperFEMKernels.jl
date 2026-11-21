# Sparsity pattern by field blocks: the Jacobian reserves a (test field, trial field) block only on the cells
# (and facets) of the integrals that couple the two fields, which is the pattern Gridap allocates. Every case
# is checked against Gridap: same pattern (colptr and rowval) and same matrix.

"Gridap's assembled Jacobian at `x` (pattern and values)."
gridap_jacobian(jac, U, V, x) = (uh = FEFunction(U, x); assemble_matrix((du, dv) -> jac(uh, du, dv), U, V))
same_structure(A, B) = size(A) == size(B) && A.colptr == B.colptr && A.rowval == B.rowval
"Pattern with every (test, trial) pair of every cell (the pattern before the blocks were taken into account)."
full_pattern(op) = HK.sparsity(op.fields, op.ndl, num_cells(get_triangulation(HK.field_spaces(op.U)[1])),
                               size(HK.sparsity_pattern(op))...; skel=HK.jacobian_skeletons(op.skel, op.jgroups))[1]

"Kernel and Gridap give the same pattern and the same Jacobian; returns the operator."
function check_blocks(res, jac, U, V; x=randx(U))
    op = KernelOperator(res, jac, U, V)
    A = jacobian(op, x)
    Ag = gridap_jacobian(jac, U, V, x)
    @test same_structure(A, Ag)
    @test reldiff(A, Ag) < TOL
    c = check(res, jac, U, V, x; op)
    @test c.res < TOL && c.jac < TOL
    op
end

function stokes_spaces(; order_u=1, order_p=1, simplex=false, n=(5, 4))
    model = cartesian_model(2, n; simplex)
    Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, order_u), dirichlet_tags="boundary")
    Vp = TestFESpace(model, ReferenceFE(lagrangian, Float64, order_p))
    V = MultiFieldFESpace([Vu, Vp])
    U = MultiFieldFESpace([TrialFESpace(Vu, VectorValue(0.0, 0.0)), TrialFESpace(Vp)])
    Ω = Triangulation(model)
    (; model, U, V, Ω, dΩ=Measure(Ω, 2max(order_u, order_p)))
end

@testset "Stokes: no pressure–pressure block without a term (P1–P1, Taylor–Hood), present with PSPG" begin
    f = VectorValue(1.0, 2.0)
    for (ou, op_, simplex) in ((1, 1, false), (1, 1, true), (2, 1, false), (2, 1, true))
        s = stokes_spaces(; order_u=ou, order_p=op_, simplex)
        dΩ = s.dΩ
        res((u, p), (v, q)) = ∫(∇(v) ⊙ ∇(u) - (∇ ⋅ v) * p + q * (∇ ⋅ u) - v ⋅ f)dΩ
        jac(_, (du, dp), (v, q)) = ∫(∇(v) ⊙ ∇(du) - (∇ ⋅ v) * dp + q * (∇ ⋅ du))dΩ
        op = check_blocks(res, jac, s.U, s.V)
        @test nnz(HK.sparsity_pattern(op)) < nnz(full_pattern(op))
        # the pressure–pressure block is empty
        nu = num_free_dofs(s.U[1])
        @test nnz(HK.sparsity_pattern(op)[nu+1:end, nu+1:end]) == 0
        # PSPG-like term: the block is reserved
        τ = 0.1
        resp((u, p), (v, q)) = res((u, p), (v, q)) + ∫(τ * (∇(q) ⋅ ∇(p)))dΩ
        jacp(x, (du, dp), (v, q)) = jac(x, (du, dp), (v, q)) + ∫(τ * (∇(q) ⋅ ∇(dp)))dΩ
        opp = check_blocks(resp, jacp, s.U, s.V)
        @test nnz(HK.sparsity_pattern(opp)[nu+1:end, nu+1:end]) > 0
    end
end

@testset "a term multiplied by a zero number keeps its block (structure, not values)" begin
    s = stokes_spaces()
    dΩ = s.dΩ
    α = 0.0
    res((u, p), (v, q)) = ∫(∇(v) ⊙ ∇(u) - (∇ ⋅ v) * p + q * (∇ ⋅ u) + α * (q * p))dΩ
    jac(_, (du, dp), (v, q)) = ∫(∇(v) ⊙ ∇(du) - (∇ ⋅ v) * dp + q * (∇ ⋅ du) + α * (q * dp))dΩ
    op = KernelOperator(res, jac, s.U, s.V)
    nu = num_free_dofs(s.U[1])
    @test nnz(HK.sparsity_pattern(op)[nu+1:end, nu+1:end]) > 0
    @test op.pattern.spec.cells[1][3] == trues(2, 2)
end

@testset "coupling on part of the cells, coupling on the boundary only" begin
    model = cartesian_model(2, (6, 4))
    cc = collect(get_cell_coordinates(Triangulation(model)))
    Ω = Triangulation(model); dΩ = Measure(Ω, 2)
    Ωa = Triangulation(model, [sum(x[1] for x in c) / length(c) < 0.5 for c in cc]); dΩa = Measure(Ωa, 2)
    Γ = BoundaryTriangulation(model, tags="boundary"); dΓ = Measure(Γ, 2)
    V1 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); dirichlet_tags=[1])
    V2 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 2))
    V = MultiFieldFESpace([V1, V2]); U = MultiFieldFESpace([TrialFESpace(V1, 0.0), TrialFESpace(V2)])
    # field 1 ↔ field 2 only on Ωa
    res((u, w), (v, z)) = ∫(∇(v) ⋅ ∇(u) + z * w - v)dΩ + ∫(3.0 * (v * w) + z * u)dΩa
    jac(_, (du, dw), (v, z)) = ∫(∇(v) ⋅ ∇(du) + z * dw)dΩ + ∫(3.0 * (v * dw) + z * du)dΩa
    op = check_blocks(res, jac, U, V)
    @test nnz(HK.sparsity_pattern(op)) < nnz(full_pattern(op))
    # one-way coupling on the boundary: rows of field 2, columns of field 1
    resb((u, w), (v, z)) = ∫(∇(v) ⋅ ∇(u) + z * w - v)dΩ + ∫(z * u)dΓ
    jacb(_, (du, dw), (v, z)) = ∫(∇(v) ⋅ ∇(du) + z * dw)dΩ + ∫(z * du)dΓ
    opb = check_blocks(resb, jacb, U, V)
    n1 = num_free_dofs(U[1])
    @test nnz(HK.sparsity_pattern(opb)[1:n1, n1+1:end]) == 0
    @test nnz(HK.sparsity_pattern(opb)[n1+1:end, 1:n1]) > 0
    # nonlinear coupling through the unknown: the block follows the trial leaves of the Jacobian
    resn((u, w), (v, z)) = ∫(∇(v) ⋅ ∇(u) + z * w + z * (u * u))dΩ
    jacn((u, w), (du, dw), (v, z)) = ∫(∇(v) ⋅ ∇(du) + z * dw + 2.0 * (z * (u * du)))dΩ
    check_blocks(resn, jacn, U, V; x=randx(U; scale=0.5))
end

@testset "skeleton: blocks of each side and across the facet from the jump and mean terms" begin
    model = cartesian_model(2, (5, 4))
    Ω = Triangulation(model); dΩ = Measure(Ω, 4)
    Λ = SkeletonTriangulation(model); dΛ = Measure(Λ, 4)
    V1 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 2); conformity=:L2)
    V2 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity=:L2)
    V = MultiFieldFESpace([V1, V2]); U = MultiFieldFESpace([TrialFESpace(V1), TrialFESpace(V2)])
    γ = 10.0
    # field 1 coupled across facets; field 2 only in the cells
    res((u, w), (v, z)) = ∫(∇(v) ⋅ ∇(u) + z * w + z * u - v)dΩ + ∫(γ * jump(v) * jump(u))dΛ
    jac(_, (du, dw), (v, z)) = ∫(∇(v) ⋅ ∇(du) + z * dw + z * du)dΩ + ∫(γ * jump(v) * jump(du))dΛ
    op = check_blocks(res, jac, U, V)
    @test nnz(HK.sparsity_pattern(op)) < nnz(full_pattern(op))
    # one side only, and a mixed term across the facet
    res2((u, w), (v, z)) = res((u, w), (v, z)) + ∫(v.plus * u.plus + mean(z) * jump(u))dΛ
    jac2(x, (du, dw), (v, z)) = jac(x, (du, dw), (v, z)) + ∫(v.plus * du.plus + mean(z) * jump(du))dΛ
    check_blocks(res2, jac2, U, V)
    # a skeleton term with every volume term (one field) kept by the volume blocks
    s = ip_poisson(; order=2)
    check_blocks(s.res, s.jac, s.p.U, s.p.V)
end

@testset "fallback: the part assembled by Gridap keeps Gridap's pattern" begin
    s = product_gradient_problem(; order=1)
    x = randx(s.p.U)
    op = fallback_op(KernelOperator, s.res, s.jac, s.p.U, s.p.V)
    @test same_structure(jacobian(op, x), gridap_jacobian(s.jac, s.p.U, s.p.V, x))
    @test reldiff(jacobian(op, x), gridap_jacobian(s.jac, s.p.U, s.p.V, x)) < TOL
end

@testset "update: the pattern is kept when the new forms fit in it, rebuilt when a block is added" begin
    s = stokes_spaces()
    dΩ = s.dΩ
    mk(τ) = (((u, p), (v, q)) -> ∫(∇(v) ⊙ ∇(u) - (∇ ⋅ v) * p + q * (∇ ⋅ u) + τ * (∇(q) ⋅ ∇(p)))dΩ,
             (_, (du, dp), (v, q)) -> ∫(∇(v) ⊙ ∇(du) - (∇ ⋅ v) * dp + q * (∇ ⋅ du) + τ * (∇(q) ⋅ ∇(dp)))dΩ)
    r0 = ((u, p), (v, q)) -> ∫(∇(v) ⊙ ∇(u) - (∇ ⋅ v) * p + q * (∇ ⋅ u))dΩ
    j0 = (_, (du, dp), (v, q)) -> ∫(∇(v) ⊙ ∇(du) - (∇ ⋅ v) * dp + q * (∇ ⋅ du))dΩ
    r1, j1 = mk(0.1)
    x = randx(s.U)
    op1 = KernelOperator(r1, j1, s.U, s.V)
    A1 = allocate_jacobian(op1, x)
    # a block lost: same pattern, the driver's matrix still serves
    op2 = update(op1, r0, j0)
    @test op2.pattern === op1.pattern
    jacobian!(A1, op2, x)
    @test reldiff(A1, gridap_jacobian(j0, s.U, s.V, x)) < TOL
    # a block added: a new pattern, and the old matrix is refused with a clear error
    opa = KernelOperator(r0, j0, s.U, s.V)
    A0 = allocate_jacobian(opa, x)
    opb = update(opa, r1, j1)
    @test opb.pattern !== opa.pattern
    @test_throws ErrorException jacobian!(A0, opb, x)
    @test same_structure(jacobian(opb, x), gridap_jacobian(j1, s.U, s.V, x))
    @test reldiff(jacobian(opb, x), gridap_jacobian(j1, s.U, s.V, x)) < TOL
    # same structure, other values: fast path, same pattern
    r2, j2 = mk(0.3)
    op3 = update(op1, r2, j2)
    @test op3.pattern === op1.pattern
    @test reldiff(jacobian(op3, x), gridap_jacobian(j2, s.U, s.V, x)) < TOL
end

"Two scalar P1 fields on the unit square without Dirichlet conditions (every dof free)."
function two_fields(; n=(6, 5))
    model = CartesianDiscreteModel((0, 1, 0, 1), n)
    Ω = Triangulation(model); dΩ = Measure(Ω, 2)
    V1 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1)); V2 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1))
    (; model, Ω, dΩ, V=MultiFieldFESpace([V1, V2]), U=MultiFieldFESpace([TrialFESpace(V1), TrialFESpace(V2)]))
end

@testset "update exchanging a block for another of the same size: the old matrix is refused, not filled wrongly" begin
    s = two_fields(); dΩ = s.dΩ
    r1 = ((u, w), (v, z)) -> ∫(∇(v) ⋅ ∇(u) + z * w + v * w)dΩ
    j1 = (_, (du, dw), (v, z)) -> ∫(∇(v) ⋅ ∇(du) + z * dw + v * dw)dΩ
    r2 = ((u, w), (v, z)) -> ∫(∇(v) ⋅ ∇(u) + z * w + z * u)dΩ
    j2 = (_, (du, dw), (v, z)) -> ∫(∇(v) ⋅ ∇(du) + z * dw + z * du)dΩ
    x = randx(s.U)
    op1 = KernelOperator(r1, j1, s.U, s.V)
    A = allocate_jacobian(op1, x)
    op2 = update(op1, r2, j2)
    @test op2.pattern !== op1.pattern
    @test nnz(HK.sparsity_pattern(op2)) == nnz(A) && !same_structure(HK.sparsity_pattern(op2), A)
    @test_throws ErrorException jacobian!(A, op2, x)
    @test reldiff(jacobian(op2, x), gridap_jacobian(j2, s.U, s.V, x)) < TOL
    # a copy of the pattern made by the user is checked once and then accepted
    B = copy(HK.sparsity_pattern(op2))
    jacobian!(B, op2, x); jacobian!(B, op2, x)
    @test reldiff(B, gridap_jacobian(j2, s.U, s.V, x)) < TOL
    @test length(op2.pattern.checked) >= 1
    # a matrix sharing the row indices of a checked one but with other column pointers is compared, not trusted
    C = SparseMatrixCSC(size(B)..., copy(SparseArrays.getcolptr(B)), rowvals(B), zeros(nnz(B)))
    C.colptr[2] += 1
    @test_throws ErrorException jacobian!(C, op2, x)
    # the list of checked matrices does not grow with dead matrices
    for _ in 1:5
        allocate_jacobian(op2, x)
    end
    GC.gc(); allocate_jacobian(op2, x)
    @test length(op2.pattern.checked) <= 4
end

@testset "Jacobian with a Gridap (fallback) part: every block on every cell, as before; the fallback may gain a block in update" begin
    s = two_fields(); dΩ = s.dΩ
    Ω1 = Triangulation(s.model, 1:12); dΩ1 = Measure(Ω1, 2)
    r = ((u, w), (v, z)) -> ∫(∇(v) ⋅ ∇(u) + z * w)dΩ
    j1 = ((u, w), (du, dw), (v, z)) -> ∫(∇(v) ⋅ ∇(du) + z * dw)dΩ + ∫(v * (∇(du) ⋅ ∇(u * u)))dΩ1
    j2 = ((u, w), (du, dw), (v, z)) -> ∫(∇(v) ⋅ ∇(du) + z * dw)dΩ + ∫(v * (∇(du) ⋅ ∇(u * u)) + z * (∇(du) ⋅ ∇(u * u)))dΩ1
    x = randx(s.U; scale=0.5)
    op1 = fallback_op(KernelOperator, r, j1, s.U, s.V)
    @test !isempty(op1.fallback.jtrians) && op1.pattern.spec === nothing
    @test nnz(HK.sparsity_pattern(op1)) == nnz(full_pattern(op1))
    A = allocate_jacobian(op1, x)
    op2 = fallback_op(update, op1, r, j2)                  # fast path: same signature
    jacobian!(A, op2, x)
    @test reldiff(A, gridap_jacobian(j2, s.U, s.V, x)) < TOL
    j3 = ((u, w), (du, dw), (v, z)) -> ∫(∇(v) ⋅ ∇(du) + z * dw + 2.0 * (v * dw))dΩ + ∫(z * (∇(du) ⋅ ∇(u * u)))dΩ1
    op3 = fallback_op(update, op1, r, j3)                  # full path
    @test reldiff(jacobian(op3, x), gridap_jacobian(j3, s.U, s.V, x)) < TOL
end

@testset "update adding a triangulation and then changing its form keeps the pattern" begin
    s = two_fields(); dΩ = s.dΩ
    cc = collect(get_cell_coordinates(s.Ω))
    Ωa = Triangulation(s.model, [sum(p[1] for p in c) / length(c) < 0.5 for c in cc]); dΩa = Measure(Ωa, 2)
    r1 = ((u, w), (v, z)) -> ∫(∇(v) ⋅ ∇(u) + z * w)dΩ
    j1 = (_, (du, dw), (v, z)) -> ∫(∇(v) ⋅ ∇(du) + z * dw)dΩ
    r2 = ((u, w), (v, z)) -> ∫(∇(v) ⋅ ∇(u) + z * w)dΩ + ∫(v * w)dΩa
    j2 = (_, (du, dw), (v, z)) -> ∫(∇(v) ⋅ ∇(du) + z * dw)dΩ + ∫(v * dw)dΩa
    r3 = ((u, w), (v, z)) -> ∫(∇(v) ⋅ ∇(u) + z * w)dΩ + ∫(v * w + 3.0 * (v * w))dΩa
    j3 = (_, (du, dw), (v, z)) -> ∫(∇(v) ⋅ ∇(du) + z * dw)dΩ + ∫(v * dw + 3.0 * (v * dw))dΩa
    x = randx(s.U)
    op1 = KernelOperator(r1, j1, s.U, s.V)
    op2 = update(op1, r2, j2)
    @test op2.pattern !== op1.pattern
    op3 = update(op2, r3, j3)
    op4 = update(op3, r2, j2)
    @test op3.pattern === op2.pattern && op4.pattern === op3.pattern
    @test same_structure(jacobian(op3, x), gridap_jacobian(j3, s.U, s.V, x))
    @test reldiff(jacobian(op3, x), gridap_jacobian(j3, s.U, s.V, x)) < TOL
end

@testset "sums that are not split, corner cells of a boundary coupling, H(div)–L2 Darcy" begin
    s = two_fields(); dΩ = s.dΩ
    c = 2.0
    # (v + z)(du + dw): one part, every block; with a factor that is not a test or trial function
    r = ((u, w), (v, z)) -> ∫((v + z) * (u + w) + ∇(v) ⋅ ∇(u))dΩ
    j = (_, (du, dw), (v, z)) -> ∫((v + z) * (du + dw) + ∇(v) ⋅ ∇(du))dΩ
    check_blocks(r, j, s.U, s.V)
    rc = ((u, w), (v, z)) -> ∫(∇(v) ⋅ ∇(u) + z * w + (c * v) * (u + w))dΩ
    jc = (_, (du, dw), (v, z)) -> ∫(∇(v) ⋅ ∇(du) + z * dw + (c * v) * (du + dw))dΩ
    check_blocks(rc, jc, s.U, s.V)
    # coupling only on two sides (corner cells have two faces), Dirichlet on both fields
    model = CartesianDiscreteModel((0, 1, 0, 1), (5, 5))
    labels = get_face_labeling(model)
    Ω = Triangulation(model); dΩq = Measure(Ω, 2)
    Γ = BoundaryTriangulation(model, tags=[5, 7]); dΓ = Measure(Γ, 2)   # y = 0 and x = 0
    V1 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); dirichlet_tags=[2, 6])
    V2 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 2); dirichlet_tags=[8])
    V = MultiFieldFESpace([V1, V2]); U = MultiFieldFESpace([TrialFESpace(V1, 1.0), TrialFESpace(V2, 0.5)])
    rb = ((u, w), (v, z)) -> ∫(∇(v) ⋅ ∇(u) + ∇(z) ⋅ ∇(w))dΩq + ∫(z * u + 2.0 * (v * w))dΓ
    jb = (_, (du, dw), (v, z)) -> ∫(∇(v) ⋅ ∇(du) + ∇(z) ⋅ ∇(dw))dΩq + ∫(z * du + 2.0 * (v * dw))dΓ
    opb = check_blocks(rb, jb, U, V)
    @test nnz(HK.sparsity_pattern(opb)) < nnz(full_pattern(opb))
    # Darcy: flux in RT (signs), discontinuous pressure, no pressure–pressure block
    md = CartesianDiscreteModel((0, 1, 0, 1), (4, 3); map=x -> VectorValue(x[1] + 0.1 * x[2]^2, x[2] + 0.05 * sin(3x[1])))
    md = Gridap.Geometry.UnstructuredDiscreteModel(md)
    Ωd = Triangulation(md); dΩd = Measure(Ωd, 4)
    Vq = TestFESpace(md, ReferenceFE(raviart_thomas, Float64, 1), conformity=:Hdiv, dirichlet_tags=[5, 6])
    Vp = TestFESpace(md, ReferenceFE(lagrangian, Float64, 1), conformity=:L2)
    Vd = MultiFieldFESpace([Vq, Vp]); Ud = MultiFieldFESpace([TrialFESpace(Vq, x -> VectorValue(0.1 * x[1], 0.2 * x[1])), TrialFESpace(Vp)])
    rd = ((q, p), (v, w)) -> ∫(2.0 * (v ⋅ q) - (∇ ⋅ v) * p + w * (∇ ⋅ q) - w)dΩd
    jd = (_, (dq, dp), (v, w)) -> ∫(2.0 * (v ⋅ dq) - (∇ ⋅ v) * dp + w * (∇ ⋅ dq))dΩd
    opd = check_blocks(rd, jd, Ud, Vd)
    nq = num_free_dofs(Ud[1])
    @test nnz(HK.sparsity_pattern(opd)[nq+1:end, nq+1:end]) == 0
end

@testset "skeleton blocks in update; matrix-free operator with the pattern built later" begin
    model = cartesian_model(2, (4, 4))
    Ω = Triangulation(model); dΩ = Measure(Ω, 4)
    Λ = SkeletonTriangulation(model); dΛ = Measure(Λ, 4)
    V1 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity=:L2)
    V2 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity=:L2)
    V = MultiFieldFESpace([V1, V2]); U = MultiFieldFESpace([TrialFESpace(V1), TrialFESpace(V2)])
    base_r = ((u, w), (v, z)) -> ∫(∇(v) ⋅ ∇(u) + z * w)dΩ + ∫(10.0 * jump(v) * jump(u))dΛ
    base_j = (_, (du, dw), (v, z)) -> ∫(∇(v) ⋅ ∇(du) + z * dw)dΩ + ∫(10.0 * jump(v) * jump(du))dΛ
    more_r = ((u, w), (v, z)) -> ∫(∇(v) ⋅ ∇(u) + z * w)dΩ + ∫(10.0 * jump(v) * jump(u) + z.plus * w.minus)dΛ
    more_j = (_, (du, dw), (v, z)) -> ∫(∇(v) ⋅ ∇(du) + z * dw)dΩ + ∫(10.0 * jump(v) * jump(du) + z.plus * dw.minus)dΛ
    x = randx(U)
    op1 = KernelOperator(more_r, more_j, U, V)
    A = allocate_jacobian(op1, x)
    op2 = update(op1, base_r, base_j)                      # loses a block across the facets: pattern kept
    @test op2.pattern === op1.pattern
    jacobian!(A, op2, x)
    @test reldiff(A, gridap_jacobian(base_j, U, V, x)) < TOL
    op3 = update(KernelOperator(base_r, base_j, U, V), more_r, more_j)   # gains it: new pattern
    @test same_structure(jacobian(op3, x), gridap_jacobian(more_j, U, V, x))
    @test reldiff(jacobian(op3, x), gridap_jacobian(more_j, U, V, x)) < TOL
    # matrix-free: the pattern is built on first use, also after an update
    mf1 = KernelOperator(base_r, base_j, U, V; jacobian=:matrixfree)
    mf2 = update(mf1, more_r, more_j)
    @test mf2.pattern.A0 === nothing
    @test same_structure(HK.sparsity_pattern(mf2), gridap_jacobian(more_j, U, V, x))
    B = copy(HK.sparsity_pattern(mf2)); jacobian!(B, mf2, x)
    @test reldiff(B, gridap_jacobian(more_j, U, V, x)) < TOL
end

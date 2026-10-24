# Cross-code benchmark, HyperFEM implementation with the Gridap and kernel backends.
# Problem and protocol: ../README.md. From the repository root, benchmark environment:
#
#   julia -t auto --project=benchmark benchmark/crosscode/julia/bench.jl p=2 n=8 bench=both backends=gridap,kernel
#   julia -t 1    --project=benchmark benchmark/crosscode/julia/bench.jl p=2 n=8 bench=assembly backends=kernel
#
# One JSON line per backend is printed and appended to `out`.
using HyperFEM, HyperFEMKernels, Gridap
using Gridap.FESpaces, Gridap.Algebra
using LinearAlgebra, SparseArrays, Printf

const L, LAM, MU, A0 = 4.0, 10.0, 1.0, 0.05

opts = Dict("p" => "2", "n" => "8", "bench" => "both", "backends" => "gridap,kernel", "reps" => "5",
            "steps" => "5", "delta" => "0.5",
            "out" => joinpath(@__DIR__, "..", "..", "results", "crosscode", "julia.jsonl"))
for a in ARGS
    k, v = split(a, "="; limit=2)
    haskey(opts, k) || error("unknown option $k (known: $(join(keys(opts), ", ")))")
    opts[k] = v
end
const p, n, reps, steps = parse.(Int, (opts["p"], opts["n"], opts["reps"], opts["steps"]))
const δ = parse(Float64, opts["delta"])
const bench = opts["bench"]
const backends = Symbol.(split(opts["backends"], ","))

wtime(f) = (t = time_ns(); r = f(); ((time_ns() - t) / 1e9, r))
best(f, reps) = minimum(first(wtime(f)) for _ in 1:reps)

# ---- setup: mesh, space, weak form --------------------------------------------------------------
t_space, (model, V, dΩ) = wtime() do
    model = CartesianDiscreteModel((0.0, L, 0.0, 1.0, 0.0, 1.0), (4n, n, n))
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "left", HyperFEM.CartesianTags.face0YZ⁺)    # faces with their edges and vertices
    add_tag_from_tags!(labels, "right", HyperFEM.CartesianTags.face1YZ⁺)
    dΩ = Measure(Triangulation(model), 2p)
    V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, p), dirichlet_tags=["left", "right"])
    model, V, dΩ
end

physmodel = NeoHookean3D(λ=LAM, μ=MU)       # Ψ = μ/2 (tr FᵀF − 3) − μ ln J + λ/2 (J − 1)²
km = Kinematics(Mechano, Solid)
res(u, v) = HyperFEM.residual(physmodel, km, u, v, dΩ)
jac(u, du, v) = HyperFEM.jacobian(physmodel, km, u, du, v, dΩ)
Ψ, _, _ = physmodel()
Fk, _, _ = get_Kinematics(km; Λ=1.0)
energy(uh) = sum(∫(Ψ ∘ (Fk ∘ ∇(uh)'))dΩ)

build_operator(b, U) =
    b === :gridap ? get_algebraic_operator(FEOperator(res, jac, U, V)) :
    b === :kernel ? KernelOperator(res, jac, U, V) : error("unknown backend $b")

json(x::AbstractString) = "\"" * x * "\""
json(x::Symbol) = json(string(x))
json(x::Real) = isfinite(x) ? repr(x) : "null"
json(x::AbstractVector) = "[" * join(json.(x), ", ") * "]"
json(d::Vector{<:Pair}) = "{" * join(("\"$(k)\": $(json(v))" for (k, v) in d), ", ") * "}"

for b in backends
    rec = Pair{String,Any}["code" => "hyperfem-$b", "version" => string(pkgversion(HyperFEMKernels)),
                           "p" => p, "n" => n, "ndofs" => num_free_dofs(V) + num_dirichlet_dofs(V),
                           "nprocs" => 1, "nthreads" => (b === :kernel ? Threads.nthreads() : 1),
                           "t_setup_space" => t_space]

    # ---- benchmark 1: assembly at the fixed state u0 ---------------------------------------------
    if bench in ("assembly", "both")
        u0(x) = VectorValue(A0 * x[1] * x[2], A0 * x[1] * x[3], A0 * x[1] * x[2] * x[3])
        U = TrialFESpace(V, [u0, u0])
        uh = interpolate_everywhere(u0, U)
        x = copy(get_free_dof_values(uh))
        t_build, op = wtime(() -> build_operator(b, U))
        t_alloc, (bv, A) = wtime(() -> (allocate_residual(op, x), allocate_jacobian(op, x)))
        t_res1, _ = wtime(() -> residual!(bv, op, x))
        t_jac1, _ = wtime(() -> jacobian!(A, op, x))
        t_res = best(() -> residual!(bv, op, x), reps)
        t_jac = best(() -> jacobian!(A, op, x), reps)
        residual!(bv, op, x)
        jacobian!(A, op, x)
        append!(rec, ["t_build" => t_build, "t_alloc" => t_alloc, "t_res_first" => t_res1, "t_jac_first" => t_jac1,
                      "t_res" => t_res, "t_jac" => t_jac, "energy" => energy(uh), "res_norm" => norm(bv),
                      "jac_fro" => norm(A)])     # free DOFs only: no Dirichlet rows to discount
    end

    # ---- benchmark 2: Newton with Dirichlet load steps ---------------------------------------------
    if bench in ("newton", "both")
        g0(x) = VectorValue(0.0, 0.0, 0.0)
        U = TrialFESpace(V, [g0, g0])
        op = build_operator(b, U)
        x = zeros(num_free_dofs(U))
        bv, A = allocate_residual(op, x), allocate_jacobian(op, x)
        t_asm = t_sol = 0.0
        iters = Int[]
        t_total = @elapsed for k in 1:steps
            Λ = k / steps
            TrialFESpace!(U, [g0, x -> VectorValue(0.0, 0.0, -δ * Λ)])   # new Dirichlet values, in place
            it, r0 = 0, -1.0
            while true
                t_asm += @elapsed residual!(bv, op, x)
                r = norm(bv)
                r0 < 0 && (r0 = r)
                (r <= max(1e-8 * r0, 1e-11) || it == 25) && break
                t_asm += @elapsed jacobian!(A, op, x)
                t_sol += @elapsed (dx = lu(A) \ bv)
                x .-= dx
                it += 1
            end
            push!(iters, it)
        end
        append!(rec, ["newton_iters" => iters, "t_newton_total" => t_total, "t_newton_assembly" => t_asm,
                      "t_newton_solve" => t_sol, "linear_solver" => "UMFPACK (SparseArrays lu)",
                      "energy_final" => energy(FEFunction(U, x))])
    end

    line = json(rec)
    println(line)
    flush(stdout)
    mkpath(dirname(opts["out"]))
    open(io -> println(io, line), opts["out"], "a")
end

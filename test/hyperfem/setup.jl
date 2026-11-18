# Environment for the HyperFEM-dependent tests: the HyperFEM.jl package must be cloned next to this
# package (CI does it; locally it already is in ~/Desktop/repos). Meshes are in test/models.
using Pkg, TOML
Pkg.activate(@__DIR__)
root = normpath(joinpath(@__DIR__, "..", ".."))
hyperfem = joinpath(root, "..", "HyperFEM.jl")
Pkg.develop(path=root)
Pkg.develop(path=hyperfem)
# GridapSolvers: not used by the tests of this folder (they use HyperFEM's NewtonRaphson) but by HyperFEM's
# Stokes and Turek simulation tests (block solvers), which runtests.jl includes.
# HyperFEM's simulation tests, included by runtests.jl, load packages of HyperFEM's test environment (e.g. JSON
# for the piston reference curve): they are taken from its test/Project.toml, so that a new test dependency of
# HyperFEM does not break this environment.
hyperfem_test_deps = keys(TOML.parsefile(joinpath(hyperfem, "test", "Project.toml"))["deps"])
own_deps = ["Gridap", "GridapGmsh", "GridapSolvers", "Test", "LinearAlgebra", "Random", "SparseArrays", "TimerOutputs"]
Pkg.add(sort(union(own_deps, setdiff(hyperfem_test_deps, ["HyperFEM", "HyperFEMKernels"]))))
Pkg.instantiate()
Pkg.status()

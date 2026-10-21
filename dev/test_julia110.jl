# Run the package tests with Julia 1.10 (the oldest version in CI) through juliaup, from the watcher:
#   dev/test_julia110.jl [names of test/test_<name>.jl; none = Pkg.test()]
# Installs the 1.10 channel if missing. Uses a scratch environment so the benchmark Manifest is untouched.
root = normpath(joinpath(@__DIR__, ".."))
try run(`juliaup add 1.10`) catch; end
run(`julia +1.10 --version`)
if isempty(ARGS)
    run(Cmd(`julia +1.10 --project=. -e "using Pkg; Pkg.instantiate(); Pkg.test()"`; dir=root))
else
    env = joinpath(root, "dev", "env110")
    mkpath(env)
    cmd = """using Pkg; Pkg.activate("$env"); Pkg.develop(path="$root"); Pkg.add(["Test", "Random", "Gridap", "LinearAlgebra", "SparseArrays"]); Pkg.instantiate()"""
    run(Cmd(`julia +1.10 -e $cmd`; dir=root))
    run(Cmd(`julia +1.10 -t auto --project=$env dev/test_subset.jl $ARGS`; dir=root))
end
println("FIN 1.10")

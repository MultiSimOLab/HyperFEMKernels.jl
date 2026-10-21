# Run a script with Julia 1.10 in the dev/env110 environment (see test_julia110.jl):  dev/run110.jl dev/<script>.jl [args]
root = normpath(joinpath(@__DIR__, ".."))
run(Cmd(`julia +1.10 -t auto --project=$(joinpath(root, "dev", "env110")) $ARGS`; dir=root))

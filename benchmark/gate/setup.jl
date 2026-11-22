# Environment of the performance gate for a checkout of HyperFEMKernels.jl:
#   julia benchmark/gate/setup.jl [PKG_ROOT] [ENV_DIR]
# writes ENV_DIR/Project.toml — benchmark/gate/Project.toml with HyperFEMKernels taken from PKG_ROOT (absolute path;
# default: the checkout of this file) and HyperFEM from the General registry (Gridap as HyperFEM's compat allows) —
# then resolves, instantiates and precompiles it. An environment whose Project.toml is unchanged keeps its
# Manifest. No tracked file is modified. gate.sh passes ENV_DIR in its work folder (outside the repository).
using Pkg

function gate_env(pkg::AbstractString, env::AbstractString)
    pkg = abspath(pkg)
    isfile(joinpath(pkg, "Project.toml")) && occursin("name = \"HyperFEMKernels\"", read(joinpath(pkg, "Project.toml"), String)) ||
        error("setup.jl: $pkg is not a checkout of HyperFEMKernels.jl")
    tmpl = read(joinpath(@__DIR__, "Project.toml"), String)
    path = replace(rstrip(pkg, '/'), "\\" => "\\\\", "\"" => "\\\"")
    spec = replace(tmpl, r"HyperFEMKernels = \{path = \"[^\"]*\"\}" => "HyperFEMKernels = {path = \"$path\"}")
    spec == tmpl && !occursin(path, spec) && error("setup.jl: [sources] entry of HyperFEMKernels not found in the template")
    mkpath(env)
    proj, man = joinpath(env, "Project.toml"), joinpath(env, "Manifest.toml")
    if !(isfile(proj) && read(proj, String) == spec)
        rm(man; force=true)
        write(proj, spec)
    end
    Pkg.activate(env)
    try
        Pkg.instantiate()
    catch err                   # the measured checkout changed its dependencies: resolve again
        @warn "setup.jl: instantiate failed, resolving again" exception = err
        rm(man; force=true)
        Pkg.instantiate()
    end
    Pkg.precompile()
    Pkg.status()
end

gate_env(get(ARGS, 1, joinpath(@__DIR__, "..", "..")), get(ARGS, 2, joinpath(@__DIR__, "env", "head")))

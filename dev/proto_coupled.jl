# Prototype of the proposed HyperFEM fix for legacy Dirichlet couplings (see docs/regresion_DirichletCoupling_HyperFEM.pdf):
# a `Coupled` prescription applied BY MASK after Gridap's TrialFESpace!, evaluated only at the Dirichlet nodes of its tag
# with a persistent point-search cache. Checked here on the vacuum space of examples/hmre_vacuum_2d against Gridap's own
# interpolation of the same analytic field, without touching HyperFEM.
#   julia -t auto --project=benchmark dev/proto_coupled.jl
using Gridap, GridapGmsh, Gridap.FESpaces, Gridap.CellData, Gridap.Arrays, LinearAlgebra, Printf, Test

# ---- code proposed for HyperFEM (BoundaryConditions.jl / FESpaces.jl) ---------------------------------------------
"""
    Coupled(g)

Prescribed field `g(Λ)` (an `FEFunction`, `CellField` or `Interpolable`) that may be defined only on the faces of
its tag, e.g. a field of an `FESpace` built on an interface. Gridap receives a placeholder for the tag; the field
is evaluated solely at the Dirichlet nodes of the tag and written by mask after Gridap's interpolation, keeping the
point-search cache between calls. `g(Λ) === nothing` leaves the values untouched (coupling not set yet).
"""
mutable struct Coupled{G}
    g::G
    mask::Union{Nothing,Vector{Bool}}          # Dirichlet dofs of the tag (set at first use for the space)
    coords::Vector                             # coordinates of the nodes of those dofs (one per node)
    ndofs::Int                                 # dofs per node (components)
    cache::Any                                 # Gridap point-search cache
    trian::Any                                 # triangulation the cache was built for
end
Coupled(g) = Coupled(g, nothing, [], 0, nothing, nothing)
(p::Coupled)(Λ) = x -> x                      # placeholder for Gridap (overwritten by the mask); same as the legacy code

_field(f) = f
_field(f::Interpolable) = f.uh

"Mask and node coordinates of the Dirichlet dofs of `tag` in `space` (a TrialFESpace)."
function _coupling_nodes!(p::Coupled, space, tag::Int)
    mask = space.space.dirichlet_dof_tag .== tag
    U0 = TrialFESpace(space.space, [x -> x for _ in 1:num_dirichlet_tags(space.space)])  # dirichlet values = coordinates
    dim = num_point_dims(get_triangulation(space))
    vals = get_dirichlet_dof_values(U0)[mask]
    p.ndofs = length(vals) == 0 ? dim : dim    # vector-valued fields: dim components per node
    p.coords = [VectorValue(vals[i:i+dim-1]...) for i in 1:dim:length(vals)]
    p.mask = mask
    p
end

"Write `p.g(Λ)` at the Dirichlet nodes of its tag into `space` (after Gridap's TrialFESpace!)."
function apply_coupling!(space, p::Coupled, tag::Int, Λ)
    f = p.g(Λ)
    f === nothing && return space
    p.mask === nothing && _coupling_nodes!(p, space, tag)
    fld = _field(f)
    trian = get_triangulation(fld)
    if p.cache === nothing || p.trian !== trian
        p.cache = return_cache(fld, p.coords[1]); p.trian = trian
    end
    dv = get_dirichlet_dof_values(space)
    idx = findall(p.mask)
    k = 0
    for x in p.coords
        v = evaluate!(p.cache, fld, x)
        for c in 1:p.ndofs
            dv[idx[k+c]] = v[c]
        end
        k += p.ndofs
    end
    space
end
# In HyperFEM, TrialFESpace!(space, bc::BoundaryConditions, Λ) would do:
#   TrialFESpace!(space, get_space_functions(bc, Λ))
#   for (i, c) in enumerate(bc.conditions); c.value isa Coupled && apply_coupling!(space, c.value, i, Λ); end
# and _legacy_prescription(DirichletCoupling, _) = Coupled(LegacyCoupling(nothing)), with
#   (c::LegacyCoupling)(Λ) = isnothing(c.Interpolable) ? nothing : c.Interpolable(Λ)
# ---------------------------------------------------------------------------------------------------------------

mesh = normpath(joinpath(@__DIR__, "..", "test", "models", "Short_magnet.msh"))
geomodel = GmshDiscreteModel(mesh)
Ωvacuum = Triangulation(geomodel, tags=["vacuumb"])
Γair_int = BoundaryTriangulation(Ωvacuum, tags="Interface")
reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
tags = ["Fixed_Air", "Fixed_Air_x", "Interface"]
Vv = TestFESpace(Ωvacuum, reffeu; dirichlet_tags=tags, dirichlet_masks=[(true, true), (true, false), (true, true)])
zero2 = VectorValue(0.0, 0.0)
g(x) = VectorValue(1e-3 * x[2] + 2e-4, -5e-4 * x[1])                 # analytic interface displacement

# the coupled field lives ONLY on the interface (as uhair_int in the example)
Uair_int = FESpace(Γair_int, reffeu)
uh_int = interpolate_everywhere(g, Uair_int)

# reference: Gridap interpolating the analytic function on the tag
Uref = TrialFESpace(Vv, [zero2, zero2, g])
# proposed path: placeholder + coupling by mask
p = Coupled(Λ -> Interpolable(uh_int))
U = TrialFESpace(Vv, [zero2, zero2, p(1.0)])
t1 = @elapsed apply_coupling!(U, p, 3, 1.0)
t2 = @elapsed apply_coupling!(U, p, 3, 1.0)
a = @allocated apply_coupling!(U, p, 3, 1.0)
err = norm(get_dirichlet_dof_values(U) - get_dirichlet_dof_values(Uref)) / norm(get_dirichlet_dof_values(Uref))
@printf("vacuum space: %d free, %d Dirichlet dofs, %d interface nodes | first apply %.3f s, next %.4f s (%d bytes) | rel. diff vs Gridap %.2e\n",
        num_free_dofs(U), num_dirichlet_dofs(U), length(p.coords), t1, t2, a, err)
@test err < 1e-13
# the field changes in place (staggered loop): the same coupling follows it without rebuilding anything
get_free_dof_values(uh_int) .*= 2.0
apply_coupling!(U, p, 3, 1.0)
Uref2 = TrialFESpace(Vv, [zero2, zero2, x -> 2.0 * g(x)])
@test norm(get_dirichlet_dof_values(U) - get_dirichlet_dof_values(Uref2)) / norm(get_dirichlet_dof_values(Uref2)) < 1e-13
# the other tags are untouched
@test all(get_dirichlet_dof_values(U)[.!p.mask] .== 0.0)
println("FIN PROTO COUPLED")

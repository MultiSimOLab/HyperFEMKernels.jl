# Cross-code benchmark, FEniCSx implementation. Problem and protocol: ../README.md.
#
#   python bench.py --p 2 --n 8 --bench both
#   mpirun -n 6 python bench.py --p 2 --n 13 --bench assembly
#
# One JSON line per run is printed and appended to --out.
import argparse
import json
import os
import time

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.fem.petsc as fpetsc
from dolfinx import fem, mesh

L, LAM, MU, A0 = 4.0, 10.0, 1.0, 0.05

ap = argparse.ArgumentParser()
ap.add_argument("--p", type=int, default=2)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--bench", choices=["assembly", "newton", "both"], default="both")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--delta", type=float, default=0.5)
ap.add_argument("--jit-flags", default="-O3 -march=native",
                help='C compiler flags for the FFCx kernels (DOLFINx default: "-O2")')
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "results", "crosscode", "fenicsx.jsonl"))
args = ap.parse_args()

comm = MPI.COMM_WORLD
rank, nprocs = comm.rank, comm.size


def wtime(f):
    """Wall time of f(), maximum over the MPI ranks."""
    comm.barrier()
    t = time.perf_counter()
    r = f()
    comm.barrier()
    return comm.allreduce(time.perf_counter() - t, op=MPI.MAX), r


def best(f, reps):
    return min(wtime(f)[0] for _ in range(reps))


# ---- setup: mesh, space, forms -----------------------------------------------------------------
def build_space():
    msh = mesh.create_box(comm, [np.array([0.0, 0.0, 0.0]), np.array([L, 1.0, 1.0])],
                          [4 * args.n, args.n, args.n], cell_type=mesh.CellType.hexahedron)
    V = fem.functionspace(msh, ("Lagrange", args.p, (3,)))
    return msh, V


t_space, (msh, V) = wtime(build_space)
ndofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

u = fem.Function(V)
v, du = ufl.TestFunction(V), ufl.TrialFunction(V)
F = ufl.Identity(3) + ufl.grad(u)
J = ufl.det(F)
psi = MU / 2 * (ufl.tr(F.T * F) - 3) - MU * ufl.ln(J) + LAM / 2 * (J - 1) ** 2   # HyperFEM NeoHookean3D
dx = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree": 2 * args.p})
Pi = psi * dx
R = ufl.derivative(Pi, u, v)
K = ufl.derivative(R, u, du)

jit = {"cffi_extra_compile_args": args.jit_flags.split()}
t_jit, (Pi_f, R_f, K_f) = wtime(lambda: tuple(fem.form(f, jit_options=jit) for f in (Pi, R, K)))

fdim = msh.topology.dim - 1
left = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
right = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], L))
dofs_l = fem.locate_dofs_topological(V, fdim, left)
dofs_r = fem.locate_dofs_topological(V, fdim, right)
g_l, g_r = fem.Function(V), fem.Function(V)
bcs = [fem.dirichletbc(g_l, dofs_l), fem.dirichletbc(g_r, dofs_r)]


def bc_dofs_owned():
    idx = []
    for bc in bcs:
        d, n_owned = bc.dof_indices()
        idx.append(d[:n_owned])
    return np.unique(np.concatenate(idx))


bc_owned = bc_dofs_owned()
n_bc = comm.allreduce(len(bc_owned), op=MPI.SUM)

t_alloc, (A, b) = wtime(lambda: (fpetsc.create_matrix(K_f), fpetsc.create_vector(V)))


def assemble_residual():
    with b.localForm() as loc:
        loc.set(0.0)
    fpetsc.assemble_vector(b, R_f)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)


def zero_bc_rows():
    b.array_w[bc_owned] = 0.0          # owned entries of the unrolled vector


def assemble_jacobian():
    A.zeroEntries()
    fpetsc.assemble_matrix(A, K_f, bcs=bcs)   # BC rows and columns zeroed, 1 on the diagonal
    A.assemble()


rec = dict(code="fenicsx", version=dolfinx.__version__, p=args.p, n=args.n, ndofs=ndofs,
           nprocs=nprocs, nthreads=1, jit_flags=args.jit_flags, t_setup_space=t_space, t_jit=t_jit,
           t_alloc=t_alloc)

# ---- benchmark 1: assembly at the fixed state u0 -------------------------------------------------
if args.bench in ("assembly", "both"):
    u0 = lambda x: np.vstack([A0 * x[0] * x[1], A0 * x[0] * x[2], A0 * x[0] * x[1] * x[2]])
    u.interpolate(u0)
    u.x.scatter_forward()
    t_res1, _ = wtime(assemble_residual)
    t_jac1, _ = wtime(assemble_jacobian)
    t_res = best(assemble_residual, args.reps)
    t_jac = best(assemble_jacobian, args.reps)
    assemble_residual()
    zero_bc_rows()
    res_norm = b.norm()
    assemble_jacobian()
    jac_fro = np.sqrt(A.norm(PETSc.NormType.FROBENIUS) ** 2 - n_bc)
    energy = comm.allreduce(fem.assemble_scalar(Pi_f), op=MPI.SUM)
    rec.update(t_res_first=t_res1, t_jac_first=t_jac1, t_res=t_res, t_jac=t_jac,
               energy=energy, res_norm=res_norm, jac_fro=jac_fro)

# ---- benchmark 2: Newton with Dirichlet load steps ------------------------------------------------
if args.bench in ("newton", "both"):
    u.x.array[:] = 0.0
    ksp = PETSc.KSP().create(comm)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    dU = fpetsc.create_vector(V)
    t_asm = t_sol = 0.0
    iters = []
    comm.barrier()
    t0 = time.perf_counter()
    for k in range(1, args.steps + 1):
        lam = k / args.steps
        g_r.interpolate(lambda x: np.vstack([0 * x[0], 0 * x[0], -args.delta * lam + 0 * x[0]]))
        for bc in bcs:                       # impose the new Dirichlet values on the state
            bc.set(u.x.array)
        u.x.scatter_forward()
        it, r0 = 0, None
        while True:
            t, _ = wtime(assemble_residual)
            t_asm += t
            zero_bc_rows()
            r = b.norm()
            r0 = r if r0 is None else r0
            if r <= max(1e-8 * r0, 1e-11) or it == 25:
                break
            t, _ = wtime(assemble_jacobian)
            t_asm += t
            ksp.setOperators(A)
            t, _ = wtime(lambda: ksp.solve(b, dU))
            t_sol += t
            u.x.petsc_vec.axpy(-1.0, dU)
            u.x.scatter_forward()
            it += 1
        iters.append(it)
    comm.barrier()
    t_total = comm.allreduce(time.perf_counter() - t0, op=MPI.MAX)
    energy_final = comm.allreduce(fem.assemble_scalar(Pi_f), op=MPI.SUM)
    rec.update(newton_iters=iters, t_newton_total=t_total, t_newton_assembly=t_asm,
               t_newton_solve=t_sol, linear_solver="MUMPS (PETSc)", energy_final=energy_final)

if rank == 0:
    line = json.dumps(rec)
    print(line, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as f:
        f.write(line + "\n")

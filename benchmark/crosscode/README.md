# Cross-code benchmark: HyperFEM (Gridap and kernel backends), FEniCSx and MFEM

The same nonlinear problem, discretisation and quadrature in the three codes, so that assembly times
can be compared and the results checked against each other to round-off.

| Directory | Code | Parallelism |
|---|---|---|
| `julia/` | HyperFEM's weak form (`NeoHookean3D`), assembled by Gridap (`backend gridap`) and by HyperFEMKernels (`backend kernel`) | Gridap serial; kernel threads |
| `fenicsx/` | DOLFINx, energy written in UFL and differentiated symbolically | MPI |
| `mfem/` | MFEM `ParNonlinearForm` + `HyperelasticNLFIntegrator` with the same model written as a `HyperelasticModel` | MPI |

## Problem

- **Domain and mesh.** Beam Ω = [0, 4] × [0, 1] × [0, 1], structured mesh of 4n × n × n hexahedra.
- **Discretisation.** Vector Lagrange elements Q1 or Q2 (`p = 1, 2`). Tensor Gauss quadrature of
  degree 2p, i.e. (p + 1)³ points per cell, in all codes.
- **Material.** HyperFEM's `NeoHookean3D` with λ = 10, μ = 1:
  Ψ(F) = μ/2 (tr FᵀF − 3) − μ ln J + λ/2 (J − 1)²,
  P = μF + (−μ/J + λ(J − 1)) J F⁻ᵀ. (HyperFEM's regularisation of ln J below J = 0.01 is never
  reached here.)
- **Dirichlet boundaries.** The faces x = 0 and x = 4, all components.

Sizes with the same number of DOFs for Q1 and Q2 (Q2 with n equals Q1 with 2n):

| Q2 `n` | Q1 `n` | DOFs |
|---|---|---|
| 4 | 8 | 8 019 |
| 8 | 16 | 56 355 |
| 13 | 26 | 229 635 |
| 22 | 44 | 1 075 275 |

## What is measured

**Benchmark 1, assembly** (`bench=assembly`). At the fixed state

u₀(X) = 0.05 (X₁X₂, X₁X₃, X₁X₂X₃),

which is trilinear and therefore interpolated exactly by Q1 and Q2 in every code (whatever the
node placement), so all codes assemble at the same state:

- time of the residual and of the Jacobian, best of `reps` repetitions into preallocated
  structures (sparsity pattern not included), and the first call (which includes compilation in
  Julia and in MFEM's first sparse-pattern build);
- check values, which must agree between codes to round-off: the energy ∫Ψ dΩ, the Euclidean norm
  of the residual on the non-Dirichlet DOFs, and the Frobenius norm of the Jacobian restricted to
  the non-Dirichlet DOFs (FEniCSx and MFEM keep the Dirichlet rows with 1 on the diagonal; those
  ones are discounted).

**Benchmark 2, Newton** (`bench=newton`). Starting from u = 0, the face x = 4 is moved to
u = (0, 0, −δΛ), δ = 0.5, in `steps` = 5 equal load steps (x = 0 clamped). At each step the new
Dirichlet values are imposed on the state and full Newton iterations are run, K du = −R, until
‖R‖ ≤ max(10⁻⁸ ‖R₀‖, 10⁻¹¹), with R₀ the residual at the start of the step. The Newton
iterations per step and the final energy must agree between codes. Assembly and linear-solve times
are reported separately, because the linear solvers differ:

| Code | Linear solver |
|---|---|
| HyperFEM | UMFPACK (`lu` of SparseArrays) |
| FEniCSx | MUMPS through PETSc |
| MFEM | MUMPS if MFEM was built with it, otherwise hypre GMRES + BoomerAMG (systems options) to 10⁻¹² |

The comparison that matters here is the assembly; the linear solver is the subject of the solver
block of the plan (PETSc in all codes then).

## Running

Each script prints one JSON line per run and appends it to `benchmark/results/crosscode/<code>.jsonl`
(git-ignored). Then, from the repository root:

```
python3 benchmark/crosscode/compare.py
```

prints, for each size, tables with the times, the ratio to the fastest run and the relative
difference of the check values to the first code.

### HyperFEM (Julia)

Benchmark environment of this repository (`julia benchmark/setup.jl` once; it uses the HyperFEM
checkout next to this repository, branch `kernel-backend`):

```
julia -t auto --project=benchmark benchmark/crosscode/julia/bench.jl p=2 n=8 bench=both backends=gridap,kernel
julia -t 1    --project=benchmark benchmark/crosscode/julia/bench.jl p=2 n=8 bench=assembly backends=kernel
```

Options: `p`, `n`, `bench` (`assembly`, `newton`, `both`), `backends` (`gridap`, `kernel`, or both),
`reps`, `steps`, `delta`, `out`. The kernel backend uses all Julia threads; Gridap is serial.

### FEniCSx

With conda-forge (`fenics-dolfinx` 0.11, `mpich`, `petsc4py`):

```
conda create -n fenicsx -c conda-forge fenics-dolfinx=0.11 mpich petsc4py
conda activate fenicsx
export OMP_NUM_THREADS=1
python benchmark/crosscode/fenicsx/bench.py --p 2 --n 8 --bench both
mpirun -n 6 python benchmark/crosscode/fenicsx/bench.py --p 2 --n 8 --bench assembly
```

The first run of a given `p` includes the JIT compilation of the forms (`t_jit`); later runs reuse
the cache.

### MFEM

Needs an MFEM built with MPI (conda-forge `mfem`, Homebrew `mfem`, spack, or a source build). The
`Makefile` reads MFEM's `config.mk`:

```
cd benchmark/crosscode/mfem
make MFEM_DIR=<MFEM install prefix>          # or: make CONFIG_MK=<mfem build>/config/config.mk
./bench -p 2 -n 8 -b both
mpirun -n 6 ./bench -p 2 -n 8 -b assembly
```

With the conda-forge package the recorded library paths of `config.mk` point to the build machine;
override them: `make MFEM_DIR=$CONDA_PREFIX MFEM_EXT_LIBS="-L$CONDA_PREFIX/lib -lHYPRE -lmetis"`.

### All sizes

`run_all.sh` runs the four codes over the sizes of the table above for one number of threads /
processes (edit the variables at the top).

## Fairness notes

- **Same work.** Same mesh, element, quadrature points, state and boundary DOFs; the check values
  confirm that every code computes the same residual, Jacobian and Newton path.
- **Parallelism.** Julia uses threads in one process; FEniCSx and MFEM use MPI processes. Compare
  1 thread with 1 process, and N threads with N processes. With MPI set `OMP_NUM_THREADS=1`, so that
  BLAS threads do not compete with the processes. On machines with performance and efficiency cores
  (Apple), measure also at the number of performance cores.
- **What each Jacobian time contains.** HyperFEM: element matrices and assembly into the
  `SparseMatrixCSC` of the free DOFs. FEniCSx: element matrices (FFCx kernels), assembly into the
  PETSc matrix and Dirichlet rows/columns. MFEM: element matrices, assembly of the local sparse
  matrix, the parallel triple product PᵀAP into a `HypreParMatrix` and the Dirichlet elimination —
  what `ParNonlinearForm::GetGradient` does for every user.
- **Compiler flags.** FEniCSx compiles its kernels with `--jit-flags` (default here
  `-O3 -march=native`, about 10 % faster than DOLFINx's default `-O2`; if the compiler rejects
  `-march=native`, use `-mcpu=native`). Julia compiles for the host CPU. MFEM's numbers depend on
  how the library was built: binary packages (conda-forge, Homebrew) use generic flags; for the best
  MFEM figures build it from source with `-O3 -march=native` (or `-mcpu=native`).
- **Compilation.** The best-of-`reps` times exclude it; `t_res_first`, `t_jac_first` (Julia, MFEM)
  and `t_jit` (FEniCSx) show it.
- **MFEM's Jacobian kernel** is the generic `AssembleH` written here (as MFEM's own models do, with
  dense 3 × 3 blocks per pair of components); MFEM has no partial assembly for hyperelasticity.

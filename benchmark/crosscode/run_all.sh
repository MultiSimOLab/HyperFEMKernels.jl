#!/usr/bin/env bash
# Runs the cross-code benchmark over the sizes of README.md for one level of parallelism.
# From the repository root:  bash benchmark/crosscode/run_all.sh
# Edit the variables below; a code whose command is empty is skipped.
set -euo pipefail

NPAR=1                                   # Julia threads = MPI processes
BENCH=assembly                           # assembly | newton | both (newton only up to ~230k DOFs with direct solvers)
SIZES_Q2="4 8 13 22"
SIZES_Q1="8 16 26 44"
JULIA="julia -t $NPAR --project=benchmark"
FENICSX_PY="python3"                     # python of the FEniCSx environment ("" to skip)
MFEM_BENCH="benchmark/crosscode/mfem/bench"   # built executable ("" to skip)
MPIRUN="mpirun -n $NPAR"

export OMP_NUM_THREADS=1
mkdir -p benchmark/results/crosscode
run() { echo ">> $*"; "$@"; }

for p in 1 2; do
    sizes=$([ "$p" = 1 ] && echo "$SIZES_Q1" || echo "$SIZES_Q2")
    for n in $sizes; do
        backends=kernel
        [ "$NPAR" = 1 ] && backends=gridap,kernel          # Gridap is serial: run it once
        run $JULIA benchmark/crosscode/julia/bench.jl p=$p n=$n bench=$BENCH backends=$backends
        [ -n "$FENICSX_PY" ] && run $MPIRUN $FENICSX_PY benchmark/crosscode/fenicsx/bench.py --p $p --n $n --bench $BENCH
        [ -n "$MFEM_BENCH" ] && run $MPIRUN $MFEM_BENCH -p $p -n $n -b $BENCH -o benchmark/results/crosscode/mfem.jsonl
    done
done
python3 benchmark/crosscode/compare.py

#!/usr/bin/env bash
# Runs the cross-code benchmark over the sizes of README.md for one level of parallelism.
# From the repository root:  bash benchmark/crosscode/run_all.sh
# Edit the variables below; a code whose command is empty is skipped. FEniCSx and MFEM may come
# with different MPI libraries (e.g. separate conda environments): each has its own mpirun.
set -euo pipefail

NPAR=1                                   # Julia threads = MPI processes
BENCH=assembly                           # assembly | newton | both (newton only up to ~230k DOFs with direct solvers)
SIZES_Q2="4 8 13 22"
SIZES_Q1="8 16 26 44"
JULIA="julia -t $NPAR --project=benchmark"
FENICSX_ENV="${FENICSX_ENV:-}"           # prefix of the FEniCSx environment ("" to skip)
MFEM_ENV="${MFEM_ENV:-}"                 # prefix of the MFEM environment (its mpirun and libraries; "" to skip)
MFEM_BENCH="benchmark/crosscode/mfem/bench"

export OMP_NUM_THREADS=1
R=benchmark/results/crosscode
mkdir -p $R
run() { echo ">> $*"; "$@"; }
libpath() { [ "$(uname)" = Darwin ] && echo DYLD_LIBRARY_PATH="$1/lib" || echo LD_LIBRARY_PATH="$1/lib"; }

for p in 1 2; do
    sizes=$([ "$p" = 1 ] && echo "$SIZES_Q1" || echo "$SIZES_Q2")
    for n in $sizes; do
        backends=kernel
        [ "$NPAR" = 1 ] && backends=gridap,kernel          # Gridap is serial: run it once
        run $JULIA benchmark/crosscode/julia/bench.jl p=$p n=$n bench=$BENCH backends=$backends
        [ -n "$FENICSX_ENV" ] && run "$FENICSX_ENV/bin/mpirun" -n $NPAR "$FENICSX_ENV/bin/python" \
            benchmark/crosscode/fenicsx/bench.py --p $p --n $n --bench $BENCH
        [ -n "$MFEM_ENV" ] && run env "$(libpath "$MFEM_ENV")" "$MFEM_ENV/bin/mpirun" -n $NPAR $MFEM_BENCH \
            -p $p -n $n -b $BENCH -o $R/mfem.jsonl
    done
done
python3 benchmark/crosscode/compare.py

#!/bin/bash
# GAMG parameter sweep on the mechanical cases (S1): block size, strength threshold, coarsening, smoother sweeps, tolerance.
#   bash benchmark/solvers/sweep_s1.sh "mech_plate 4" "mech_box 32 7 7"
cd "$(dirname "$0")/../.." || exit 1
T=${T:-6}
run() { echo "=== $1 compiled $2 $3  $(date +%H:%M:%S)"; julia -t $T --project=benchmark/solvers benchmark/solvers/krylov.jl $1 compiled $2 $3 2>&1 | grep --line-buffered -v "^Info\|Warning\|└\|┌" | grep --line-buffered "backend=\|solve!\|per iteration"; }
for c in "$@"; do
  [ -f benchmark/results/STOP ] && rm -f benchmark/results/STOP && echo "stopped" && exit 0
  run "$c" cg_gamg "bs=3"
  run "$c" cg_gamg "bs=3 opts=-pc_gamg_threshold,0.01"
  run "$c" cg_gamg "bs=3 opts=-pc_gamg_threshold,0.05"
  run "$c" cg_gamg "bs=3 opts=-pc_gamg_aggressive_coarsening,0"
  run "$c" cg_gamg "bs=3 opts=-mg_levels_ksp_max_it,4"
  run "$c" cg_gamg "bs=3 opts=-pc_gamg_agg_nsmooths,0"
  run "$c" cg_gamg "bs=3 rtol=1e-6"
  run "$c" cg_gamg_nonull "bs=3"
done

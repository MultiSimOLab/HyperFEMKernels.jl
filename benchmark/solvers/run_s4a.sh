#!/bin/bash
# S4-A (design/solvers-s4a-plan.md, step 5): p-multigrid against the GAMG references of step 0, through krylov.jl.
#   bash benchmark/solvers/run_s4a.sh cube N CFG...    # mech_cube N (Q2) with the given configurations
#   bash benchmark/solvers/run_s4a.sh box CFG...       # mech_box 52 11 11 (Q2)
# Stop between two runs with:  touch benchmark/results/STOP
cd "$(dirname "$0")/../.." || exit 1
MODE=$1; shift
T=${T:-6}
case "$MODE" in
  cube) N=$1; shift; CASE="mech_cube $N" ;;
  box)  CASE="mech_box 52 11 11" ;;
  boxl) CASE="mech_box 90 18 18" ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
for cfg in "$@"; do
  [ -f benchmark/results/STOP ] && rm -f benchmark/results/STOP && echo "stopped" && exit 0
  echo "=== $CASE compiled $cfg threads=$T  $(date +%H:%M:%S)"
  julia -t $T --project=benchmark/solvers benchmark/solvers/krylov.jl $CASE compiled $cfg bs=3 2>&1 | grep --line-buffered -v "^Info\|Warning\|└\|┌"
done

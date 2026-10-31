#!/bin/bash
# S2 matrix (design/solvers-s2-plan.md): block preconditioners on the monolithic EM cases through the driver.
#   bash benchmark/solvers/run_s2.sh em-small          # 57k Q2 / 75k P1: mumps and the fieldsplit configurations
#   bash benchmark/solvers/run_s2.sh em-medium         # 218k Q2 / 246k P1
#   bash benchmark/solvers/run_s2.sh em-large CFG      # 974k Q2 / 1.11M P1 with one configuration
#   bash benchmark/solvers/run_s2.sh hmre-small        # hMRE vacuum 2D (robustness; LU is cheap there)
# Stop between two runs with:  touch benchmark/results/STOP
cd "$(dirname "$0")/../.." || exit 1
MODE=$1; shift
T=${T:-6}
FS=("mumps" "fs_schur_upper_a11" "fs_schur_full_a11" "fs_schur_upper_selfp" "fs_additive" "fs_multiplicative" "fs_schur_upper_a11_lag3")
case "$MODE" in
  em-small)   CASES=("em_plate 4" "em_box 32 7 7"); CONFIGS=("${FS[@]}") ;;
  em-medium)  CASES=("em_plate 6" "em_box 52 11 11"); CONFIGS=("${FS[@]}") ;;
  em-large)   CASES=("em_plate 10" "em_box 90 18 18"); CONFIGS=("$*") ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
for c in "${CASES[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    [ -f benchmark/results/STOP ] && rm -f benchmark/results/STOP && echo "stopped" && exit 0
    echo "=== $c compiled $cfg threads=$T  $(date +%H:%M:%S)"
    julia -t $T --project=benchmark/solvers benchmark/solvers/krylov.jl $c compiled $cfg 2>&1 | grep --line-buffered -v "^Info\|Warning\|└\|┌"
  done
done

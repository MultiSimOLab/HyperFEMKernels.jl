#!/bin/bash
# S1 matrix (design/solvers.md): Krylov + AMG vs LU through the driver, one process per run, backend :compiled.
#   bash benchmark/solvers/run_s1.sh mech-small       # mechanical block, 43k Q2 / 56k P1: lu, mumps, CG+GAMG (rigid modes, without, lagged), BoomerAMG
#   bash benchmark/solvers/run_s1.sh mech-medium      # mechanical block, 164k Q2 / 185k P1 (lu ≈ 15 min each)
#   bash benchmark/solvers/run_s1.sh mech-large CFG [kw...]   # mechanical block, 730k Q2 / 830k P1 with one configuration
#   bash benchmark/solvers/run_s1.sh em-small         # monolithic EM, 57k / 75k: lu, mumps, GMRES + AMG on the negated form
#   bash benchmark/solvers/run_s1.sh em-medium        # monolithic EM, 218k / 246k (no lu: ≈ 30 min each, see em-medium-lu)
#   bash benchmark/solvers/run_s1.sh em-medium-lu
#   bash benchmark/solvers/run_s1.sh em-large CFG
# Stop between two runs with:  touch benchmark/results/STOP
cd "$(dirname "$0")/../.." || exit 1
MODE=$1; shift
T=${T:-6}
MECH=("mumps" "cg_gamg bs=3" "cg_gamg bs=3 opts=-pc_gamg_agg_nsmooths,0" "cg_gamg_lag3 bs=3 opts=-pc_gamg_agg_nsmooths,0" "cg_gamg_nonull bs=3 opts=-pc_gamg_agg_nsmooths,0")
EM=("mumps" "gmres_gamg_neg" "gmres_boomer_neg" "gmres_gamg_neg_lag3")
case "$MODE" in
  mech-small)   CASES=("mech_box 32 7 7" "mech_plate 4"); CONFIGS=("lu" "${MECH[@]}") ;;
  mech-medium)  CASES=("mech_box 52 11 11" "mech_plate 6"); CONFIGS=("lu" "${MECH[@]}") ;;
  mech-large)   CASES=("mech_box 90 18 18" "mech_plate 10"); CONFIGS=("$*") ;;
  em-small)     CASES=("em_box 32 7 7" "em_plate 4"); CONFIGS=("lu" "${EM[@]}") ;;
  em-medium)    CASES=("em_box 52 11 11" "em_plate 6"); CONFIGS=("${EM[@]}") ;;
  em-medium-lu) CASES=("em_box 52 11 11" "em_plate 6"); CONFIGS=("lu") ;;
  em-large)     CASES=("em_box 90 18 18" "em_plate 10"); CONFIGS=("$*") ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
for c in "${CASES[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    [ -f benchmark/results/STOP ] && rm -f benchmark/results/STOP && echo "stopped" && exit 0
    echo "=== $c compiled $cfg threads=$T  $(date +%H:%M:%S)"
    julia -t $T --project=benchmark/solvers benchmark/solvers/krylov.jl $c compiled $cfg 2>&1 | grep --line-buffered -v "^Info\|Warning\|└\|┌"
  done
done

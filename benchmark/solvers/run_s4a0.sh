#!/bin/bash
# S4-A Step 0 (design/solvers-s4a-plan.md §5): the GAMG numbers the p-multigrid is measured against, through krylov.jl.
#   bash benchmark/solvers/run_s4a0.sh q3            # Q3 operator check (s4a_step0.jl) on the cube and the box, small
#   bash benchmark/solvers/run_s4a0.sh box-q1        # CG + GAMG on the Q1 discretisation of the box (the coarse level): 20k, 100k
#   bash benchmark/solvers/run_s4a0.sh cube-small    # cube Q2 n=10 (28k): lu, mumps, cg_gamg — the acceptance case, smallest size
#   bash benchmark/solvers/run_s4a0.sh cube-medium   # cube Q2 n=20 (207k): mumps, cg_gamg  (lu ≈ 30 min: cube-medium-lu)
#   bash benchmark/solvers/run_s4a0.sh cube-medium-lu
#   bash benchmark/solvers/run_s4a0.sh cube-large    # cube Q2 n=34 (985k): cg_gamg
#   bash benchmark/solvers/run_s4a0.sh box-large     # box Q2 90x18x18 (730k): cg_gamg (the assembled Q2 baseline S1 did not run)
# Stop between two runs with:  touch benchmark/results/STOP
cd "$(dirname "$0")/../.." || exit 1
MODE=$1; shift
T=${T:-6}
GAMG="cg_gamg bs=3"
case "$MODE" in
  q3)             for c in "mech_cube 8" "mech_box 26 6 6"; do
                    echo "=== s4a_step0 $c threads=$T  $(date +%H:%M:%S)"
                    julia -t $T --project=benchmark/solvers benchmark/solvers/s4a_step0.jl $c 2>&1 | grep --line-buffered -v "^Info\|Warning\|└\|┌"
                  done; exit 0 ;;
  box-q1)         CASES=("mech_box 52 11 11 1" "mech_box 90 18 18 1"); CONFIGS=("$GAMG") ;;
  cube-small)     CASES=("mech_cube 10"); CONFIGS=("mumps" "$GAMG") ;;
  cube-medium)    CASES=("mech_cube 20"); CONFIGS=("mumps" "$GAMG") ;;
  cube-medium-lu) CASES=("mech_cube 20"); CONFIGS=("lu") ;;
  cube-large)     CASES=("mech_cube 34"); CONFIGS=("$GAMG") ;;
  box-large)      CASES=("mech_box 90 18 18"); CONFIGS=("$GAMG") ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
for c in "${CASES[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    [ -f benchmark/results/STOP ] && rm -f benchmark/results/STOP && echo "stopped" && exit 0
    echo "=== $c compiled $cfg threads=$T  $(date +%H:%M:%S)"
    julia -t $T --project=benchmark/solvers benchmark/solvers/krylov.jl $c compiled $cfg 2>&1 | grep --line-buffered -v "^Info\|Warning\|└\|┌"
  done
done

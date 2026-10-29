#!/bin/bash
# Baseline matrix, one process per run. From the repository root:
#   bash benchmark/solvers/run_all.sh iter 1 6 18      # per-iteration costs: 3 cases × 3 sizes × 2 backends × threads
#   bash benchmark/solvers/run_all.sh driver 6         # full solve! through the driver (R0), sizes ≤ lu_max
#   bash benchmark/solvers/run_all.sh small 1 18        # per-iteration costs, 50k and 200k only (thread scaling)
#   bash benchmark/solvers/run_all.sh large 6           # the 1M cases with --heap-size-hint=24G (true memory need)
# Stop between two runs with:  touch benchmark/results/STOP
cd "$(dirname "$0")/../.." || exit 1
MODE=$1; shift
THREADS=${@:-"6"}
CASES=("em_plate 4" "em_box 32 7 7" "hmre 7" "em_plate 6" "em_box 52 11 11" "hmre 14" "em_plate 10" "em_box 90 18 18" "hmre 30")
[ "$MODE" = small ] && CASES=("em_plate 4" "em_box 32 7 7" "hmre 7" "em_plate 6" "em_box 52 11 11" "hmre 14")
[ "$MODE" = large ] && CASES=("em_plate 10" "em_box 90 18 18" "hmre 30") && JOPT="--heap-size-hint=24G"
[ "$MODE" = driver ] && CASES=("em_plate 4" "em_box 32 7 7" "hmre 7" "em_plate 6" "em_box 52 11 11" "hmre 14") && EXTRA="300000 driver"
for t in $THREADS; do
  for c in "${CASES[@]}"; do
    for b in compiled lazy; do
      [ -f benchmark/results/STOP ] && rm -f benchmark/results/STOP && echo "stopped" && exit 0
      echo "=== $c $b threads=$t  $(date +%H:%M:%S)"
      julia -t $t $JOPT --project=benchmark/solvers benchmark/solvers/baseline.jl $c $b $EXTRA 2>&1 | grep -v "^Info"
    done
  done
done

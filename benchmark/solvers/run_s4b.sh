#!/bin/bash
# S4-B (design/solvers-s4b-plan.md, step 3): h-multigrid against CG + GAMG on the same fine mesh, through krylov.jl.
#   bash benchmark/solvers/run_s4b.sh tet H K CFG...    # hcube_tet H K (Gmsh tetrahedra of size H refined K times)
#   bash benchmark/solvers/run_s4b.sh hex N K CFG...    # hcube_hex N K (n³ Cartesian hexahedra refined K times)
# Stop between two runs with:  touch benchmark/results/STOP
cd "$(dirname "$0")/../.." || exit 1
MODE=$1; shift
T=${T:-6}
case "$MODE" in
  tet) H=$1; K=$2; shift 2; CASE="hcube_tet $H $K" ;;
  hex) N=$1; K=$2; shift 2; CASE="hcube_hex $N $K" ;;
  tetl) H=$1; K=$2; L=$3; shift 3; CASE="hcube_tet $H $K $L" ;;     # λ of the neo-Hookean law (10: ν = 0.45; 1.5: ν = 0.3)
  hexl) N=$1; K=$2; L=$3; shift 3; CASE="hcube_hex $N $K $L" ;;
  em) N=$1; K=$2; shift 2; CASE="hem_plate_c $N $K" ;;                # step 5: EM plate (u, φ), fs_additive vs fshmg_additive
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
for cfg in "$@"; do
  [ -f benchmark/results/STOP ] && rm -f benchmark/results/STOP && echo "stopped" && exit 0
  echo "=== $CASE compiled $cfg threads=$T  $(date +%H:%M:%S)"
  julia -t $T --project=benchmark/solvers benchmark/solvers/krylov.jl $CASE compiled $cfg bs=3 2>&1 | grep --line-buffered -v "^Info\|Warning\|└\|┌"
done

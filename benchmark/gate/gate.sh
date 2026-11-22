#!/usr/bin/env bash
# Performance gate of HyperFEMKernels.jl (benchmark/gate/README.md). Run by hand on the Mac, with the machine quiet,
# after the implementation and the independent review of a branch, before asking to push.
#
#   benchmark/gate/gate.sh                 this checkout (its HEAD) against origin/main: every case, 1 and 4 threads
#   benchmark/gate/gate.sh --quick         a subset (HyperFEM simulations, EM, pattern build), to iterate
#   benchmark/gate/gate.sh --cases a,b --threads 1 --pairs 6    repeat some cases (as an INCONCLUSIVE report says)
#   benchmark/gate/gate.sh --aa            origin/main against itself: self-check of the gate (must PASS)
# Options: --threads 1,4[,18]  --groups g1,g2  --rounds N (7)  --pairs N (stage 2, 2)  --force (do not wait for a quiet
#          machine)  --no-fetch  --workdir DIR  --base REF (origin/main)  --inject F (acceptance test: the branch
#          processes run every kernel call F slower, e.g. 0.10, so the gate must FAIL)
# Output: report.md (table for the PR), samples.csv, summary.toml and logs in WORKDIR (default
# <folder containing _work>/_work/<branch>/gate), a macOS notification at the end.
# Exit code: 0 PASS, 1 FAIL, 3 INCONCLUSIVE (repeat the cases the report lists), 2 could not run.
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../.." && pwd)
julia=${JULIA:-julia}
base_ref=origin/main; threads=1,4; cases=""; groups=""; quick=""; force=""; rounds=7; pairs=2; fetch=1; workdir=""
aa=0; inject=0
notify() { command -v osascript >/dev/null 2>&1 && osascript -e "display notification \"$1\" with title \"HyperFEMKernels gate\"" >/dev/null 2>&1 || true; }
die() { echo "gate.sh: $*" >&2; notify "could not run: $*"; exit 2; }
# a stop (Ctrl-C, kill, the watcher's stop) kills every process started by the gate (TERM, then KILL)
descendants() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do echo "$c"; descendants "$c"; done; }
stopped() {
  echo "gate.sh: stopped" >&2
  local pids; pids=$(descendants $$)
  [ -n "$pids" ] && { kill -TERM $pids 2>/dev/null; sleep 3; kill -KILL $pids 2>/dev/null; }
  notify "stopped"; exit 2
}
trap stopped INT TERM HUP
while [ $# -gt 0 ]; do
  case "$1" in
    --aa) aa=1 ;;
    --base) base_ref=$2; shift ;;
    --threads) threads=$2; shift ;;
    --cases) cases=$2; shift ;;
    --groups) groups=$2; shift ;;
    --quick) quick=--quick ;;
    --force) force=--force ;;
    --rounds) rounds=$2; shift ;;
    --pairs) pairs=$2; shift ;;
    --no-fetch) fetch=0 ;;
    --workdir) workdir=$2; shift ;;
    --inject) inject=$2; shift ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown argument $1" ;;
  esac
  shift
done
up() { local d=$root; while [ "$d" != / ]; do d=$(dirname "$d"); [ -d "$d/$1" ] && { echo "$d/$1"; return; }; done; }
branch=$(git -C "$root" rev-parse --abbrev-ref HEAD) || die "not a git checkout: $root"
[ "$branch" = HEAD ] && branch=detached-$(git -C "$root" rev-parse --short HEAD)
label=$([ $aa = 1 ] && echo main-aa || echo "$branch")
if [ -z "$workdir" ]; then w=$(up _work); workdir=${w:+$w/$label/gate}; workdir=${workdir:-$root/benchmark/results/gate}; fi
mkdir -p "$workdir" || die "cannot create $workdir"
W=$(cd "$workdir" && pwd)
stamp=$(date +%Y%m%d-%H%M%S); OUT=$W/$stamp; mkdir -p "$OUT/logs"
[ $fetch = 1 ] && { git -C "$root" fetch -q origin || die "git fetch failed"; }
# origin/main in a worktree of its own (the working checkout is never touched)
if [ -d "$W/main" ]; then
  [ -z "$(git -C "$W/main" status --porcelain --untracked-files=no)" ] || die "$W/main has local changes"
  git -C "$W/main" checkout -q --detach "$base_ref" || die "cannot check out $base_ref in $W/main"
else
  git -C "$root" worktree add -q --detach "$W/main" "$base_ref" || die "cannot add a worktree of $base_ref"
fi
main_commit=$(git -C "$W/main" rev-parse --short HEAD)
head_commit=$(git -C "$root" rev-parse --short HEAD)
dirty=$([ -n "$(git -C "$root" status --porcelain --untracked-files=no)" ] && echo " (with uncommitted changes)" || echo "")
pkg_branch=$([ $aa = 1 ] && echo "$W/main" || echo "$root")
echo "gate.sh: $([ $aa = 1 ] && echo "A/A: main $main_commit against itself" || echo "$branch $head_commit$dirty against $base_ref $main_commit"); output $OUT"
# environments: HyperFEMKernels from each checkout, the rest resolved once (setup.jl keeps a Manifest whose Project
# did not change)
for e in main branch; do
  pkg=$([ $e = main ] && echo "$W/main" || echo "$pkg_branch")
  "$julia" --startup-file=no "$here/setup.jl" "$pkg" "$W/env/$e" > "$OUT/logs/setup_$e.log" 2>&1 & wait $! ||
    die "environment $e failed (see $OUT/logs/setup_$e.log)"
done
machine=$( (sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m) | tr ';' ' ')
jver=$("$julia" --startup-file=no -e 'print(VERSION)')
start=$(date +%s)
"$julia" --startup-file=no "$here/driver.jl" --main-env "$W/env/main" --branch-env "$W/env/branch" --out "$OUT" \
  --threads "$threads" --rounds "$rounds" --pairs "$pairs" --inject "$inject" $quick $force \
  ${cases:+--cases "$cases"} ${groups:+--groups "$groups"} \
  --meta "branch=$([ $aa = 1 ] && echo "main $main_commit (A/A)" || echo "$branch $head_commit$dirty");main=$base_ref $main_commit;machine=$machine;julia=$jver" \
  2> >(tee "$OUT/logs/driver.log" >&2) &
wait $!
status=$?
mins=$(( ($(date +%s) - start) / 60 ))
case $status in
  0) v=PASS ;; 1) v=FAIL ;; 3) v=INCONCLUSIVE ;; *) v="could not run (exit $status)"; status=2 ;;
esac
ln -sfn "$OUT" "$W/latest"
[ -f "$OUT/report.md" ] && echo "gate.sh: $v in $mins min — report $OUT/report.md" || echo "gate.sh: $v in $mins min — see $OUT/logs/driver.log"
notify "$label: $v ($mins min)"
exit $status

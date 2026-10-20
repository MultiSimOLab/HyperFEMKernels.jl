#!/bin/bash
# Vigilante (same protocol as ~/Desktop/aneurisma/watch.command): leave this window open.
# Claude requests a run by writing .run_request; its first line is a script path under
# this folder plus its arguments, run as  julia -t auto --project=benchmark <line>.
# Claude stops a run by writing .run_stop. Output goes to run.log (copy in logs/).
# Close the window or Ctrl+C to finish.
cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.juliaup/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
command -v julia >/dev/null 2>&1 || { echo "ERROR: julia not found"; exit 1; }
command -v caffeinate >/dev/null 2>&1 && caffeinate -i -w $$ &
mkdir -p logs
rm -f .run_request .run_stop
echo "idle $(date +%s)" > .run_status
echo "Watching $(pwd) with $(julia --version). Waiting for requests..."
while true; do
  if [ -f .run_request ]; then
    args=$(head -n 1 .run_request | tr -cd 'a-zA-Z0-9 _./=-')
    rm -f .run_request .run_stop
    script=${args%% *}
    case "$script" in
      *..*|/*) echo "rejected: $args"; echo "== rejected: $args ==" > run.log; continue ;;
    esac
    echo "[$(date +%H:%M:%S)] launching: julia -t auto --project=benchmark $args"
    echo "== start: $(date) | args: $args ==" > run.log
    julia -t auto --project=benchmark $args >> run.log 2>&1 &
    pid=$!
    echo "running $pid $(date +%s)" > .run_status
    while kill -0 "$pid" 2>/dev/null; do
      if [ -f .run_stop ]; then
        kill "$pid" 2>/dev/null
        rm -f .run_stop
        echo "== stopped on request ==" >> run.log
      fi
      sleep 1
    done
    wait "$pid"
    echo "== exit code: $? | end: $(date) ==" >> run.log
    cp run.log "logs/run_$(date +%Y%m%d_%H%M%S).log"
    echo "idle $(date +%s)" > .run_status
    echo "[$(date +%H:%M:%S)] done"
  else
    echo "idle $(date +%s)" > .run_status
  fi
  sleep 1
done

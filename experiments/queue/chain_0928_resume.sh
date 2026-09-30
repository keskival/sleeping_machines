#!/usr/bin/env bash
# Resume the 2026-09-28 priority chain after e71a; watchdog trips stop the chain.
set -uo pipefail
cd /workspace/experiments/queue
for q in e68 e69 e71b e64b e70 e67 e93b; do
  if WAIT=1 ./run_safe.sh "$q.txt" >> "runner_${q}.out" 2>&1; then
    continue
  else
    rc=$?
    echo "$(date +%T) queue $q stopped with exit $rc" >> "runner_${q}.out"
    [ "$rc" -ne 2 ] || exit "$rc"
  fi
done

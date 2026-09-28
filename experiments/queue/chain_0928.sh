#!/usr/bin/env bash
# priority chain (2026-09-28): gap-closing and fairness first; each queue waits for the global lock
set -uo pipefail
cd /workspace/experiments/queue
for q in e71a e68 e69 e71b e64b e70 e67 e93b; do
  if WAIT=1 ./run_safe.sh "$q.txt" >> "runner_${q}.out" 2>&1; then
    continue
  else
    rc=$?
    echo "$(date +%T) queue $q stopped with exit $rc" >> "runner_${q}.out"
    # Exit 2 is reserved for the memory/host-availability watchdog.
    # Ordinary model failures stay recorded but do not suppress later queues.
    [ "$rc" -ne 2 ] || exit "$rc"
  fi
done

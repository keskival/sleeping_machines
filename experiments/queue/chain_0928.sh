#!/usr/bin/env bash
# priority chain (2026-09-28): gap-closing and fairness first; each queue waits for the global lock
cd /workspace/experiments/queue
for q in e71a e68 e69 e71b e64b e70 e67 e93b; do WAIT=1 ./run_safe.sh $q.txt >> runner_$q.out 2>&1; done

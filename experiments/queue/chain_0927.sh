#!/usr/bin/env bash
# priority chain: frontier tests first; each queue waits for the global lock
cd /workspace/experiments/queue
for q in e53 e52 e36g e49 e45 e46 e46q e50; do WAIT=1 ./run_safe.sh $q.txt >> runner_$q.out 2>&1; done

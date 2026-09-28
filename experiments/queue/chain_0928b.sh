#!/usr/bin/env bash
# after the priority chain (pid 89273) finishes: the remaining 27-September queues
cd /workspace/experiments/queue
while kill -0 89273 2>/dev/null; do sleep 60; done
for q in e45 e46 e46q e50; do WAIT=1 ./run_safe.sh $q.txt >> runner_$q.out 2>&1; done

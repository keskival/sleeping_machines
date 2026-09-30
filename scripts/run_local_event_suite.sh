#!/usr/bin/env bash
# Serial full-model suite. Every argument is a uniquely named one-job queue.
set -euo pipefail
cd /workspace
export MEM_CAP_KB=${MEM_CAP_KB:-4000000}
export MEM_CAP_RSS_KB=${MEM_CAP_RSS_KB:-2500000}
export MIN_AVAIL_MB=${MIN_AVAIL_MB:-8192}
export JOB_TIMEOUT_S=${JOB_TIMEOUT_S:-864000}
if [ "$#" -eq 0 ]; then
    echo "usage: scripts/run_local_event_suite.sh queue1.txt [queue2.txt ...]" >&2
    exit 2
fi
for queue in "$@"; do
    if [ ! -f "$queue" ]; then
        echo "Missing queue: $queue" >&2
        exit 2
    fi
    count=$(awk 'NF && $1 !~ /^#/ {n++} END {print n+0}' "$queue")
    if [ "$count" != 1 ]; then
        echo "Each queue must contain exactly one job: $queue" >&2
        exit 2
    fi
done
failures=0
for queue in "$@"; do
    echo "$(date -u +%FT%TZ) suite starting $queue"
    if WAIT=1 bash experiments/queue/run_safe.sh "$queue"; then
        echo "$(date -u +%FT%TZ) suite completed $queue"
    else
        failures=$((failures+1))
        echo "$(date -u +%FT%TZ) suite failed $queue; log retained" >&2
    fi
done
echo "$(date -u +%FT%TZ) suite finished with $failures failed jobs"
test "$failures" -eq 0

#!/usr/bin/env bash
# Run experiment jobs strictly one at a time, with a memory watchdog.
#
#   experiments/run_queue.sh QUEUE_FILE [PYTHON]
#
# QUEUE_FILE has one job per line: "<name> <arguments for python>". Lines starting
# with # are ignored. The file is re-read before every job, so jobs can be appended
# while it runs. A job whose log ends in "EXIT <code>" is skipped (failures too: they
# need a look, not a retry loop), so the queue can be
# restarted. If a job trips the watchdog the whole queue stops: this host has no
# swap, and running out of memory has hung it before.
set -u
QUEUE=$1
PY=${2:-python3}
LOGS=$(dirname "$QUEUE")/logs
MIN_AVAIL_MB=${MIN_AVAIL_MB:-6000}
MEM_CAP_KB=${MEM_CAP_KB:-3500000}
mkdir -p "$LOGS"
exec 9>"$QUEUE.lock"
flock -n 9 || { echo "another runner holds $QUEUE.lock"; exit 1; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."

while :; do
  job=$(grep -v '^\s*#' "$QUEUE" | grep -v '^\s*$' | while read -r name args; do
          tail -1 "$LOGS/$name.log" 2>/dev/null | grep -q "^EXIT " || { echo "$name $args"; break; }
        done)
  [ -z "$job" ] && { echo "queue empty $(date +%T)"; exit 0; }
  read -r name args <<< "$job"
  log=$LOGS/$name.log
  echo "$(date +%T) start $name"
  echo "# $(date -Is) $PY $args" > "$log"
  ( ulimit -v "$MEM_CAP_KB"; exec nice -n 19 $PY $args ) >> "$log" 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "$avail" -lt "$MIN_AVAIL_MB" ]; then
      kill -9 $pid
      echo "WATCHDOG: killed at MemAvailable=${avail}MB" | tee -a "$log"
      exit 2
    fi
    sleep 1
  done
  wait $pid; rc=$?
  echo "EXIT $rc" >> "$log"
  echo "$(date +%T) done $name (exit $rc)"
done

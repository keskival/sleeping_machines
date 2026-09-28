#!/usr/bin/env bash
# Safe experiment runner: runs the commands in a queue file ONE AT A TIME.
#   usage: queue/run_safe.sh queue/<name>.txt   (lines: "<name> <script.py> <args...>", run from repo root; # comments ok)
# Guards (the host hung before resource limits were added):
#   * global lock: only one runner (hence one job) at a time across all queues
#   * BLAS/torch threads pinned to 1 per job
#   * per-process address-space cap, whole-job RSS cap, and host available-memory floor
#   * watchdog stops this queue if a job exceeds either memory guard
#   * successful jobs are skipped on restart; give changed configurations new job names
set -u
Q=${1:?queue file}
LOCK=/tmp/experiments-runner.lock
QUEUE_DIR=$(dirname "$Q")
QUEUE_NAME=$(basename "${Q%.txt}")
RUNNER_LOG="$QUEUE_DIR/runner_${QUEUE_NAME}.out"
MEM_CAP_KB=${MEM_CAP_KB:-6000000}
MEM_CAP_RSS_KB=${MEM_CAP_RSS_KB:-3500000}
MIN_AVAIL_MB=${MIN_AVAIL_MB:-6000}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_NUM_THREADS=1
exec 9>"$LOCK"
flock $([ -n "${WAIT:-}" ] && echo "-w 86400" || echo -n) 9 || { echo "another runner holds $LOCK; refusing to run in parallel" >&2; exit 1; }
job_pid=""
stop_job() {
  local pid=${job_pid:-}
  [ -n "$pid" ] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in {1..10}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  job_pid=""
}
on_signal() {
  local code=$1
  stop_job
  exit "$code"
}
trap 'on_signal 130' INT
trap 'on_signal 143' TERM HUP
mkdir -p "$(dirname "$0")/logs"
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  name=${line%% *}; cmd="/workspace/.venv-docker/bin/python ${line#* }"
  if [ -f "$RUNNER_LOG" ] && grep -Fq "done $name (exit 0)" "$RUNNER_LOG"; then
    echo "$(date +%T) skip $name: prior successful completion in $RUNNER_LOG"
    continue
  fi
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  if [ "$avail" -lt "$MIN_AVAIL_MB" ]; then
    echo "$(date +%T) STOP before $name: MemAvailable=${avail}MB below ${MIN_AVAIL_MB}MB"
    exit 2
  fi
  echo "$(date +%T) start $name"
  ( ulimit -v "$MEM_CAP_KB"; exec setsid nice -n 19 bash -c "cd /workspace && $cmd" ) > "$(dirname "$0")/logs/$name.log" 2>&1 &
  pid=$!
  job_pid=$pid
  last_heartbeat=$SECONDS
  while kill -0 $pid 2>/dev/null; do
    avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    rss=$(ps -eo pgid=,rss= | awk -v group="$pid" '$1 == group {total += $2} END {print total + 0}')
    if [ "$rss" -gt "$MEM_CAP_RSS_KB" ]; then
      echo "$(date +%T) STOP $name: process group RSS=${rss}KB above ${MEM_CAP_RSS_KB}KB"
      stop_job
      exit 2
    fi
    if [ "$avail" -lt "$MIN_AVAIL_MB" ]; then
      echo "$(date +%T) STOP $name: MemAvailable=${avail}MB below ${MIN_AVAIL_MB}MB"
      stop_job
      exit 2
    fi
    if (( SECONDS - last_heartbeat >= 60 )); then
      echo "$(date +%T) alive $name: process group RSS=${rss}KB, MemAvailable=${avail}MB"
      last_heartbeat=$SECONDS
    fi
    sleep 2
  done
  wait "$pid"; rc=$?; job_pid=""; echo "$(date +%T) done $name (exit $rc)"
  [ "$rc" -eq 0 ] || exit "$rc"
done < "$Q"

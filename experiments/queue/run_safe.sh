#!/usr/bin/env bash
# Safe experiment runner: runs the commands in a queue file ONE AT A TIME.
#   usage: queue/run_safe.sh queue/<name>.txt   (lines: "<name> <script.py> <args...>", run from repo root; # comments ok)
# Watchdog measures anonymous memory (memory.stat anon), not memory.current, which includes reclaimable page cache.
# Guards (the host hung twice from ~12 concurrent torch jobs in a 3-CPU / 10 GB no-swap container):
#   * global lock: only one runner (hence one job) at a time across all queues
#   * BLAS/torch threads pinned to 1 per job
#   * watchdog kills the job if the container's cgroup memory passes MEM_FRAC (default 70%)
set -u
Q=${1:?queue file}
LOCK=/tmp/experiments-runner.lock
MEM_FRAC=${MEM_FRAC:-70}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_NUM_THREADS=1
exec 9>"$LOCK"
flock $([ -n "${WAIT:-}" ] && echo "-w 86400" || echo -n) 9 || { echo "another runner holds $LOCK; refusing to run in parallel" >&2; exit 1; }
max=$(cat /sys/fs/cgroup/memory.max); [ "$max" = max ] && max=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) * 1024 ))
limit=$(( max * MEM_FRAC / 100 ))
mkdir -p "$(dirname "$0")/logs"
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  name=${line%% *}; cmd="/workspace/.venv-docker/bin/python ${line#* }"
  echo "$(date +%T) start $name"
  nice -n 10 bash -c "cd /workspace && $cmd" > "$(dirname "$0")/logs/$name.log" 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    if [ "$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)" -gt "$limit" ]; then
      echo "$(date +%T) KILL $name: memory above ${MEM_FRAC}%"; pkill -TERM -P $pid; kill -TERM $pid; sleep 5; pkill -KILL -P $pid; kill -KILL $pid
    fi
    sleep 2
  done
  wait $pid; rc=$?; echo "$(date +%T) done $name (exit $rc)"
done < "$Q"

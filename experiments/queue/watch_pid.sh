#!/usr/bin/env bash
# memory watchdog for one orphaned job (its runner was stopped): kill it if anonymous memory passes 70%
P=$1; limit=$(( $(cat /sys/fs/cgroup/memory.max) * 70 / 100 ))
while kill -0 $P 2>/dev/null; do
  if [ "$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)" -gt "$limit" ]; then echo "$(date +%T) KILL $P: memory above 70%"; kill -TERM $P; sleep 5; kill -KILL $P; fi
  sleep 2
done; echo "$(date +%T) $P exited"

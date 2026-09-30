#!/usr/bin/env bash
# Resume the unfinished reference in a one-job queue, then reload the AWS plan.
set -euo pipefail
cd /workspace
export WAIT=1 MIN_AVAIL_MB=8192 MEM_CAP_KB=6000000 MEM_CAP_RSS_KB=3500000
export JOB_TIMEOUT_S=259200 PYTHONUNBUFFERED=1
export AFTER_JOB_HOOK=/workspace/scripts/publish_aws_safe_result.py
date -u
uname -a
lscpu
free -h
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi
# This reference uses the existing CPU configuration, measured at ~2.2 GiB RSS
# for the smaller Transformer. The watchdog limits RSS and reserves 8 GiB.
./experiments/queue/run_safe.sh experiments/queue/aws_e64_tf_D90M_baseline_20260929.txt
unset AFTER_JOB_HOOK
exec /workspace/.venv-docker/bin/python -u scripts/run_aws_non_shd.py

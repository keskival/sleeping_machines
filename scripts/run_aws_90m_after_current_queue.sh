#!/usr/bin/env bash
# Restart the non-SHD controller after its currently loaded manifest drains.
# The restarted controller reloads the committed manifest, including appended 90M baselines.
set -euo pipefail
while tmux has-session -t aws-non-shd 2>/dev/null; do
  sleep 30
done
exec /workspace/.venv-docker/bin/python -u /workspace/scripts/run_aws_non_shd.py >> /tmp/aws-non-shd-controller.log 2>&1

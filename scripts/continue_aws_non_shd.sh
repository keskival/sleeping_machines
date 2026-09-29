#!/usr/bin/env bash
# The initial controller loaded its manifest before the data downloads finished.
# Start the expanded plan only after that controller reports normal completion.
set -eu
cd /workspace
while kill -0 "${1:?initial controller PID}" 2>/dev/null; do sleep 30; done
if ! tail -n 1 /tmp/aws-non-shd-controller.log | grep -Fq 'All scheduled non-SHD jobs attempted'; then
  echo 'Initial controller stopped early; refusing automatic continuation.'
  exit 1
fi
exec .venv-docker/bin/python -u scripts/run_aws_non_shd.py

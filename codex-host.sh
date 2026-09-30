#!/usr/bin/env bash
# Run the host-installed Codex without Docker or Codex's internal sandbox.
# Usage: ./codex-host.sh [codex arguments...]
set -euo pipefail

if ! command -v codex >/dev/null 2>&1; then
  echo "Codex is not installed or not in PATH on this host." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
exec codex --dangerously-bypass-approvals-and-sandbox "$@"

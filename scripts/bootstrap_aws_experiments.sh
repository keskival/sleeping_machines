#!/usr/bin/env bash
# Provision the Sleeping Machines experiment host on Ubuntu 24.04 x86_64.
# Run from the checkout at /workspace; this installs packages and text8 data,
# but it never starts an experiment.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap_aws_experiments.sh [--cpu|--gpu]

  --cpu  Install the CPU-only PyTorch wheel (default; c7i experiment host).
  --gpu  Install the current PyTorch GPU wheel. Requires an NVIDIA GPU and a
         working driver (use an AWS GPU AMI with a current driver, such as the
         AWS Deep Learning Base OSS Nvidia Driver GPU AMI for Ubuntu 24.04).
EOF
}

MODE=cpu
case "${1:---cpu}" in
  --cpu) MODE=cpu ;;
  --gpu) MODE=gpu ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then usage >&2; exit 2; fi

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ "$ROOT" != /workspace ]]; then
  echo "Expected the checkout at /workspace because run_safe.sh uses absolute /workspace paths; found $ROOT" >&2
  exit 2
fi
if [[ ! -f "$ROOT/requirements.txt" || ! -f "$ROOT/experiments/queue/run_safe.sh" ]]; then
  echo "Run this script from the Sleeping Machines checkout." >&2
  exit 2
fi
if [[ ! -w "$ROOT" ]]; then
  echo "The current user cannot write to $ROOT. Make the checkout user-owned, then rerun." >&2
  exit 2
fi

if [[ ! -r /etc/os-release ]]; then
  echo "Cannot identify the OS; Ubuntu 24.04 LTS is required." >&2
  exit 2
fi
# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != ubuntu || "${VERSION_ID:-}" != 24.04 ]]; then
  echo "Expected Ubuntu 24.04 LTS; found ${PRETTY_NAME:-unknown}." >&2
  exit 2
fi
if [[ "$(dpkg --print-architecture)" != amd64 ]]; then
  echo "Expected x86_64/amd64 for the recommended c7i/g6e/g7e hosts." >&2
  exit 2
fi

if (( EUID == 0 )); then
  APT=()
else
  command -v sudo >/dev/null || { echo "sudo is required to install system packages." >&2; exit 2; }
  APT=(sudo)
fi

if [[ "$MODE" == gpu ]]; then
  command -v nvidia-smi >/dev/null || {
    echo "No nvidia-smi found. Choose an AWS GPU instance and install a supported NVIDIA driver first." >&2
    exit 2
  }
  DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
  if [[ -z "$DRIVER_VERSION" ]] || ! dpkg --compare-versions "$DRIVER_VERSION" ge 580.65.06; then
    echo "NVIDIA driver $DRIVER_VERSION is too old for the current Blackwell-capable PyTorch wheel; use a current AWS GPU AMI." >&2
    exit 2
  fi
fi

echo "Installing OS tools (mode: $MODE)."
"${APT[@]}" apt-get update
DEBIAN_FRONTEND=noninteractive "${APT[@]}" apt-get install -y --no-install-recommends \
  build-essential ca-certificates coreutils curl git gawk grep htop libgomp1 \
  procps psmisc python3-dev python3-venv tmux unzip util-linux

VENV="$ROOT/.venv-docker"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip setuptools wheel
"$PY" -m pip install -r "$ROOT/requirements.txt" h5py

# The report builder and queued PyTorch baselines run from this environment.
# Select the CPU wheel unless the user explicitly chose a GPU host.
TORCH_MODE="$("$PY" -c 'import torch; print("gpu" if torch.version.cuda else "cpu")' 2>/dev/null || echo missing)"
if [[ "$MODE" == gpu ]]; then
  if [[ "$TORCH_MODE" != gpu ]]; then
    "$PY" -m pip uninstall -y torch >/dev/null 2>&1 || true
    # Current stable PyTorch on PyPI includes its CUDA runtime. The host driver
    # is supplied by the AWS GPU AMI; no separate toolkit install is needed.
    "$PY" -m pip install --upgrade torch
  fi
else
  if [[ "$TORCH_MODE" != cpu ]]; then
    "$PY" -m pip uninstall -y torch >/dev/null 2>&1 || true
    "$PY" -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
  fi
fi

# e62_charlm.py expects this exact 100 MB text8 file. Keep an existing copy;
# otherwise fetch the published archive and verify its ZIP CRC and payload size.
DATA_DIR="$ROOT/data/text8"
DATA_FILE="$DATA_DIR/text8"
mkdir -p "$DATA_DIR"
if [[ -f "$DATA_FILE" ]]; then
  SIZE="$(stat -c '%s' "$DATA_FILE")"
  if [[ "$SIZE" != 100000000 ]]; then
    echo "Existing $DATA_FILE is $SIZE bytes; expected 100000000. Refusing to overwrite it." >&2
    exit 2
  fi
else
  TMP_ZIP="$(mktemp "$DATA_DIR/.text8.XXXXXX.zip")"
  TMP_DATA="$(mktemp "$DATA_DIR/.text8.XXXXXX")"
  cleanup() { rm -f "$TMP_ZIP" "$TMP_DATA"; }
  trap cleanup EXIT
  curl --fail --location --retry 3 --output "$TMP_ZIP" https://mattmahoney.net/dc/text8.zip
  unzip -t "$TMP_ZIP"
  unzip -p "$TMP_ZIP" text8 > "$TMP_DATA"
  SIZE="$(stat -c '%s' "$TMP_DATA")"
  if [[ "$SIZE" != 100000000 ]]; then
    echo "Downloaded text8 payload is $SIZE bytes; expected 100000000." >&2
    exit 2
  fi
  mv "$TMP_DATA" "$DATA_FILE"
  rm -f "$TMP_ZIP"
  trap - EXIT
fi

echo "Checking the installed experiment stack."
if [[ "$MODE" == gpu ]]; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  "$PY" - <<'PY'
import torch
assert torch.cuda.is_available(), "PyTorch cannot see the NVIDIA GPU"
print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}; GPU {torch.cuda.get_device_name(0)}")
PY
else
  "$PY" - <<'PY'
import torch
assert not torch.cuda.is_available(), "CPU setup unexpectedly reports a usable CUDA device"
print(f"PyTorch {torch.__version__}; CPU-only wheel; numpy {__import__('numpy').__version__}")
PY
fi
"$PY" -c 'import h5py, matplotlib, reportlab; print("h5py", h5py.__version__, "matplotlib", matplotlib.__version__, "reportlab", reportlab.Version)'
echo "Ready at $ROOT. Safe runner: ./experiments/queue/run_safe.sh <queue.txt>"
echo "No experiment was started. Current runner limits remain in experiments/queue/run_safe.sh."

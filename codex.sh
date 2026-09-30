#!/usr/bin/env bash
# Run Codex inside the project's Docker container. Docker provides isolation;
# Codex runs without an additional filesystem sandbox or approval prompts.
#
# Usage:
#   ./codex.sh                         default container; reattach if running
#   ./codex.sh speech                   named container (same working tree)
#   ./codex.sh -- "continue the work"   initial prompt for a new container
#   ./codex.sh --configure-only         update an existing container's defaults
#   ./codex.sh --list                   list this user's Codex containers
#
# Options: IMAGE_NAME, CONTAINER_NAME, NO_REBUILD=1, CLAUDE_VERSION (shared image),
# DEV_MEMORY (default 10g), DEV_CPUS (default 3).
# See CODEX_CONTAINER.md for existing app sessions and container reuse.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR}"
USER_NAME="${USER:-$(id -un 2>/dev/null || echo "uid$(id -u)")}"
USER_NAME="${USER_NAME//[^a-zA-Z0-9_.-]/-}"

if ! command -v docker &>/dev/null; then
    echo "Docker is not installed or not in PATH." >&2
    exit 1
fi

CONFIGURE_ONLY=0
if [[ "${1:-}" == "--configure-only" ]]; then
    CONFIGURE_ONLY=1
    shift
fi

FEATURE=""
if [[ "${1:-}" == "--list" ]]; then
    docker ps --all --filter "name=^codex-${USER_NAME}-" \
        --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
    exit 0
elif [[ "${1:-}" == "--" ]]; then
    shift
elif [[ -n "${1:-}" && "${1}" != -* ]]; then
    FEATURE="$1"
    shift
    if [[ ! "${FEATURE}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
        echo "Invalid session name: ${FEATURE}" >&2
        echo 'Pass prompts with: ./codex.sh -- "your prompt"' >&2
        exit 2
    fi
fi
if (( CONFIGURE_ONLY && $# )); then
    echo '--configure-only accepts an optional session name, but no Codex arguments.' >&2
    exit 2
fi

IMAGE_NAME="${IMAGE_NAME:-codex-sandbox}"
FOLDER_NAME="$(basename "${WORKSPACE_DIR}")"
DEFAULT_NAME="codex-${USER_NAME}-${FOLDER_NAME//[^a-zA-Z0-9_.-]/-}"
[[ -n "${FEATURE}" ]] && DEFAULT_NAME+="-${FEATURE}"
CONTAINER_NAME="${CONTAINER_NAME:-${DEFAULT_NAME}}"

configure_container() {
    # Execute from the host through Docker, outside any running Codex tool
    # sandbox. Only the container's Codex config is changed, with a backup.
    docker exec --interactive "${CONTAINER_NAME}" \
        python3 - /home/node/.codex/config.toml \
        < "${SCRIPT_DIR}/scripts/configure_codex_container.py"
}

if docker container inspect "${CONTAINER_NAME}" &>/dev/null; then
    STATUS="$(docker container inspect -f '{{.State.Status}}' "${CONTAINER_NAME}")"
    if [[ "${STATUS}" != "running" ]]; then
        if (( CONFIGURE_ONLY )); then
            echo "Container ${CONTAINER_NAME} is stopped; start it before applying its config." >&2
            exit 1
        fi
        docker start "${CONTAINER_NAME}" >/dev/null
    fi
    configure_container
    if (( CONFIGURE_ONLY )); then
        echo 'Container defaults configured. Restart the Codex client to load them.'
        echo 'An existing app thread may also need Full Access selected in Permissions.'
        exit 0
    fi

    # Attaching cannot change the original process command. Check that an old
    # container actually launched Codex with the requested access setting.
    LAUNCH_MODE="$(docker container inspect -f \
        '{{index .Config.Labels "sleeping-machines.codex-sandbox"}} {{range .Config.Cmd}}{{if or (eq . "--yolo") (eq . "--dangerously-bypass-approvals-and-sandbox")}}disabled{{end}}{{end}}' \
        "${CONTAINER_NAME}")"
    if [[ "${LAUNCH_MODE}" != *disabled* ]]; then
        echo "Configured ${CONTAINER_NAME}, but its original launch command still needs updating." >&2
        echo 'See CODEX_CONTAINER.md; docker attach cannot change launch flags.' >&2
        exit 1
    fi
    if (( $# )); then
        echo 'Reattaching the existing process; new Codex arguments cannot apply on attach.' >&2
    fi
    echo "Attaching to ${CONTAINER_NAME}; running sessions retain their current permissions."
    exec docker attach "${CONTAINER_NAME}"
fi

if (( CONFIGURE_ONLY )); then
    echo "Container ${CONTAINER_NAME} does not exist. Use ./codex.sh --list to find it." >&2
    exit 1
fi

if [[ "${NO_REBUILD:-0}" != "1" ]]; then
    # dev.sh and codex.sh share this image and its Claude installation.
    source "${SCRIPT_DIR}/scripts/resolve-claude-version.sh"
    CLAUDE_VERSION="$(resolve_claude_version "${CLAUDE_VERSION:-latest}")"
    echo "Building ${IMAGE_NAME}"
    docker build --build-arg CLAUDE_VERSION="${CLAUDE_VERSION}" \
        --tag "${IMAGE_NAME}" "${SCRIPT_DIR}"
fi

echo "Starting ${CONTAINER_NAME}: Docker isolation, Codex sandbox disabled."
exec docker run \
    --interactive \
    --tty \
    --name "${CONTAINER_NAME}" \
    --label sleeping-machines.codex-sandbox=disabled \
    --user "$(id -u):$(id -g)" \
    --network host \
    --memory "${DEV_MEMORY:-10g}" \
    --memory-swap "${DEV_MEMORY:-10g}" \
    --cpus "${DEV_CPUS:-3}" \
    --env "HOME=/home/node" \
    --env "CODEX_HOME=/home/node/.codex" \
    --env "TERM=${TERM:-xterm-256color}" \
    --env "DEV_FEATURE=${FEATURE}" \
    --volume "${WORKSPACE_DIR}:/workspace:rw" \
    --workdir /workspace \
    --entrypoint /bin/bash \
    "${IMAGE_NAME}" \
    -e -c 'python3 - /home/node/.codex/config.toml < /workspace/scripts/configure_codex_container.py
exec codex --dangerously-bypass-approvals-and-sandbox "$@"' \
    codex-container "$@"

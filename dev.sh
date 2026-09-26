#!/usr/bin/env bash
# dev.sh – build (if needed) and run Claude Code in a sandboxed Docker container
#
# Usage:
#   ./dev.sh                          # default session (no feature)
#   ./dev.sh billing-split            # session for feature "billing-split"
#   ./dev.sh billing-split "fix CI"   # ...with a prompt passed to Claude Code
#   ./dev.sh --help                   # show Claude Code help
#   ./dev.sh --list                   # list this user's sandbox containers
#
# FEATURE NAME (first argument)
#   Each feature name gets its OWN container, so several features can run in
#   parallel in separate terminals, and re-running with the SAME name reattaches
#   to that feature's session (history, auth and in-flight work intact).
#   Container name: claude-<user>-<workspace-folder>[-<feature>]
#   Omit it and you get the same unsuffixed container as before.
#
#   ⚠️ CHANGED: the first argument is now the FEATURE NAME, not a prompt.
#   `./dev.sh "fix the tests"` used to send that string to Claude Code; it now
#   fails with a hint, because a silent reinterpretation would have created a
#   container named after a sentence. Pass a prompt AFTER the feature name, or
#   use `./dev.sh -- "fix the tests"` for the default session.
#
#   All features share ONE checkout (/workspace is this repo, live-mounted).
#   Parallel sessions therefore edit the same working tree — separate containers
#   isolate the AGENTS, not the files. For file-level isolation, run this script
#   from a `git worktree` of the repo; the folder basename keeps the names apart.
#
# Environment variables (all optional):
#   ANTHROPIC_API_KEY   – ONLY set if you're using an API-key billing account.
#                         The SolidMaint team uses Claude Code Pro subscriptions
#                         (browser SSO on first run), not API keys — leave this
#                         unset and Claude Code will prompt you to log in.
#   CLAUDE_VERSION      – pin a specific @anthropic-ai/claude-code version
#   CLAUDE_CODE_EFFORT  – reasoning effort: low | medium | high | xhigh | max.
#                         Defaults to LOW (Tero 2026-07-29, after the parallel
#                         burn-down exhausted a weekly token budget). Passed as
#                         the `--effort` FLAG, not CLAUDE_CODE_EFFORT_LEVEL: both
#                         exist, but only the flag is provably honoured — an
#                         invalid flag value warns, an invalid env value is
#                         accepted in silence. Set here as well as in the
#                         Dockerfile so it is visible at the call site AND still
#                         applies to a bare `docker run` of the image.
#                         Unlike the --env vars below, this is an ARGUMENT, so a
#                         change takes effect on the NEXT run — no container
#                         recreation needed. Raise it for one session with:
#                             CLAUDE_CODE_EFFORT=high ./dev.sh
#                         or per invocation (later flags win): ./dev.sh --effort high
#   CLAUDE_CODE_MAX_SUBAGENTS_PER_SESSION
#                       – lifetime cap on subagents ONE session may spawn.
#                         Defaults to 400 (set in the Dockerfile); Claude Code's
#                         own default is 200, which a long session on this repo
#                         does hit — and once hit, every parallel task quietly
#                         becomes serial. Raise it here for a single session
#                         without rebuilding. It is not a concurrency limit, so
#                         a bigger number costs no CPU or memory.
#                         ⚠️ Only applies to a NEWLY CREATED container: this
#                         script REUSES a container by name and `docker attach`
#                         cannot change env. See the note at the run block.
#   DEV_MEMORY          – hard memory cap for the container (default 10g; swap
#                         is capped to the same value, so it gets none). This
#                         host has no swap: without a cap, a runaway experiment
#                         froze the WHOLE HOST three times. With it, the kernel
#                         OOM-kills inside the container instead.
#   DEV_CPUS            – CPU cap (default 3 of 4, leaving one for the host).
#   DEV_GPUS            – GPUs to expose via --gpus (default "all"; needs the
#                         NVIDIA Container Toolkit on the host). DEV_GPUS= disables.
#                         ⚠️ Like the env vars, these three only apply to a NEWLY
#                         CREATED container, not a reattached one.
#   IMAGE_NAME          – Docker image tag  (default: claude-code-sandbox)
#   NO_REBUILD          – set to "1" to skip the docker build step
#   CONTAINER_NAME      – override the derived container name entirely
#                         (default: claude-<user>-<workspace-folder>[-<feature>];
#                         the user part keeps sessions from colliding on shared
#                         hosts like DaVinci, where several devs run this script
#                         against same-named checkouts). Set explicitly, it wins
#                         over the feature name — so don't combine the two.

set -euo pipefail

# ── Arguments: [feature-name] [claude-code args…] ────────────────────────────
# The first argument is the feature name IF it does not begin with a dash, so
# `./dev.sh --help` and `./dev.sh -p …` still go straight to Claude Code. A bare
# `--` means "no feature, everything after is for Claude Code".
FEATURE=""
if [[ "${1:-}" == "--" ]]; then
    shift
elif [[ "${1:-}" == "--list" ]]; then
    USER_PREFIX="claude-${USER:-$(id -un)}-"
    echo "Sandbox containers for ${USER_PREFIX}*:"
    docker ps --all --filter "name=^${USER_PREFIX}" \
        --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
    exit 0
elif [[ -n "${1:-}" && "${1}" != -* ]]; then
    FEATURE="$1"
    shift
    # A feature name becomes part of a Docker container name, and it is also how
    # you find your session again tomorrow — so it must be a slug, not a
    # sentence. Rejecting loudly is deliberate: quietly slugifying a prompt
    # would strand the session under a name nobody would guess.
    if [[ ! "${FEATURE}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
        echo "❌  \"${FEATURE}\" is not a usable feature name." >&2
        echo "    Use letters, digits, dot, dash or underscore, e.g. billing-split." >&2
        echo "    Did you mean to send a prompt? The feature name comes first:" >&2
        echo "        ./dev.sh <feature> \"${FEATURE}\"    # feature session" >&2
        echo "        ./dev.sh -- \"${FEATURE}\"           # default session" >&2
        exit 2
    fi
fi

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_NAME="${IMAGE_NAME:-claude-code-sandbox}"
CLAUDE_VERSION="${CLAUDE_VERSION:-latest}"
# Resolve the `latest` dist-tag to a concrete version BEFORE docker build: with
# a constant "latest" the install layer's cache key never changes and Docker
# silently reuses a months-old CLI. See scripts/resolve-claude-version.sh.
# shellcheck source=scripts/resolve-claude-version.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/resolve-claude-version.sh"
CLAUDE_VERSION="$(resolve_claude_version "${CLAUDE_VERSION}")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR}"   # mount the repo root as /workspace

# ── Sanity checks ─────────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo "❌  Docker is not installed or not in PATH." >&2
    exit 1
fi

# SolidMaint team standard: log in to Claude Code Pro on first run
# (browser SSO). The ANTHROPIC_API_KEY env var is only used by people
# who specifically have an API-key billing account — most devs don't,
# so an empty value is the expected default and not a warning-worthy
# state. The earlier "⚠️ ANTHROPIC_API_KEY is not set" message
# misled new devs into thinking something was wrong; removed.
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ℹ️  ANTHROPIC_API_KEY is set — using API-key billing for Claude Code." >&2
fi

# ── Build ─────────────────────────────────────────────────────────────────────
if [[ "${NO_REBUILD:-0}" != "1" ]]; then
    echo "🔨  Building image ${IMAGE_NAME} (Claude Code ${CLAUDE_VERSION}) …"
    docker build \
        --build-arg CLAUDE_VERSION="${CLAUDE_VERSION}" \
        --tag "${IMAGE_NAME}" \
        "${SCRIPT_DIR}"
fi

# ── Derive a stable container name from the user + workspace folder ───────────
# The user name is part of the name because DaVinci (and any other shared dev
# host) runs several devs' sandboxes side by side, and the workspace folder
# basename is usually identical across their checkouts — without it, one dev's
# `./dev.sh` attaches to (or is blocked by) another dev's container.
# The feature name is the third part, and it is what makes parallel sessions
# possible: one container per feature, reattached by re-running with the same
# name. Without a feature the name is unchanged from before, so existing
# containers keep working.
FOLDER_NAME="$(basename "${WORKSPACE_DIR}")"
USER_NAME="${USER:-$(id -un 2>/dev/null || echo "uid$(id -u)")}"
# Sanitise: replace characters Docker doesn't allow in container names
DEFAULT_NAME="claude-${USER_NAME//[^a-zA-Z0-9_.-]/-}-${FOLDER_NAME//[^a-zA-Z0-9_.-]/-}"
[[ -n "${FEATURE}" ]] && DEFAULT_NAME+="-${FEATURE}"
CONTAINER_NAME="${CONTAINER_NAME:-${DEFAULT_NAME}}"
# ── Run ───────────────────────────────────────────────────────────────────────
echo "🚀  Starting Claude Code in ${WORKSPACE_DIR}"
echo "    Feature   : ${FEATURE:-(none)}"
echo "    Container : ${CONTAINER_NAME}"
echo "    Image     : ${IMAGE_NAME}"
echo ""

# Key flags explained:
#   --interactive --tty     needed for Claude Code's interactive TUI
#   --user $(id -u):$(id -g) files written inside the container are owned by YOU
#   -v workspace            your code is live-mounted; no copies needed
#   --network host          lets Claude Code reach the Anthropic API
#   --privileged            REQUIRED for the Android emulator (KVM access).
#                           Tero confirmed 2026-07-28.
#   --cap-drop ALL          present, but see the warning below
#
# ── WHAT THIS SANDBOX DOES AND DOES NOT GIVE YOU (read this) ─────────────────
# This comment block used to claim four protections that are NOT in the command
# below: `-v .claude` (absent — auth persists because the container is REUSED by
# name, not because anything is mounted), `--security-opt` (absent),
# `--read-only /` (absent), and `--tmpfs /tmp` (absent). It also described
# `--cap-drop ALL` as giving "minimal attack surface", which is false here:
# `--privileged` on the next line grants every capability straight back and
# relaxes the other isolation, so the cap-drop is decorative. Anyone reading the
# old comments would believe they were far better isolated than they are.
#
# The privilege is legitimate — the Android emulator needs /dev/kvm — so the
# flag stays. What is fixed is the description, because a wrong security comment
# is worse than none: it is what makes someone comfortable running this against
# a repo full of production credentials.
#
# So, honestly: this container isolates your FILESYSTEM (only this repo is
# mounted) but it is NOT a security boundary against a privileged escape, and
# with --network host it shares your machine's network stack — it can reach
# anything you can, including localhost services and any VPN you are on.
#
# The documentation repo has a hardened equivalent (documentation/dev.sh) that
# drops caps for real, adds no-new-privileges, --read-only and tmpfs scratch, and
# uses bridge networking — because prose needs no emulator. Prefer that one for
# any work that does not require building or running the apps.

# ⚠️ ENV VARS ONLY TAKE EFFECT ON A FRESHLY CREATED CONTAINER.
# The branch below reuses an existing container by name, and neither
# `docker start` nor `docker attach` can change its environment — the values were
# fixed when it was created. So adding or changing any `--env` in the run block
# (e.g. CLAUDE_CODE_MAX_SUBAGENTS_PER_SESSION) does nothing until the old
# container is gone:
#
#     docker rm -f "${CONTAINER_NAME}"   # then ./dev.sh [feature]
#
# (`./dev.sh --list` prints every sandbox container of yours and its status.)
#
# Same applies to a Dockerfile `ENV`: rebuilding the image is not enough, because
# the running container was built from the previous one.
if docker inspect "${CONTAINER_NAME}" &>/dev/null; then
  # Container exists – restart it if stopped, then attach
  STATUS="$(docker inspect -f '{{.State.Status}}' "${CONTAINER_NAME}")"
  if [[ "${STATUS}" != "running" ]]; then
    echo "♻️  Restarting existing container ${CONTAINER_NAME} …"
    docker start "${CONTAINER_NAME}"
  else
    echo "✅  Container ${CONTAINER_NAME} is already running, attaching …"
  fi
  exec docker attach "${CONTAINER_NAME}"
else
  # No container yet – create it fresh
  DEV_GPUS=${DEV_GPUS-all}
  exec docker run \
    --interactive \
    --tty \
    --name "${CONTAINER_NAME}" \
    --user "$(id -u):$(id -g)" \
    --network host \
    --cap-drop ALL \
    --privileged \
    --memory "${DEV_MEMORY:-10g}" \
    --memory-swap "${DEV_MEMORY:-10g}" \
    --cpus "${DEV_CPUS:-3}" \
    ${DEV_GPUS:+--gpus "${DEV_GPUS}"} \
    --env "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}" \
    --env "HOME=/home/node" \
    --env "TERM=${TERM:-xterm-256color}" \
    --env "CLAUDE_CODE_DISABLE_MOUSE_CLICKS=1" \
    --env "CLAUDE_CODE_DISABLE_MOUSE=1" \
    --env "DEV_FEATURE=${FEATURE}" \
    --env "CLAUDE_CODE_MAX_SUBAGENTS_PER_SESSION=${CLAUDE_CODE_MAX_SUBAGENTS_PER_SESSION:-400}" \
    --volume "${WORKSPACE_DIR}:/workspace" \
    --workdir /workspace \
    "${IMAGE_NAME}" \
    --effort "${CLAUDE_CODE_EFFORT:-low}" \
    ${@:+"$@"}
fi


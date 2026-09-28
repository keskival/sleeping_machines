#!/usr/bin/env bash
#
# resolve-claude-version.sh — turn a Claude Code dist-tag into a concrete version
# before `docker build` sees it.
#
# ── WHY THIS EXISTS ──────────────────────────────────────────────────────────
# The sandbox Dockerfile installs Claude Code with
#
#     ARG CLAUDE_VERSION=latest
#     RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_VERSION}
#
# which READS as "always install the newest version" and does not behave that
# way. Docker caches a RUN layer on the literal command text, and with
# CLAUDE_VERSION=latest that text is byte-identical on every build forever. npm
# moving the `latest` tag is invisible to Docker, so the layer built on the day
# the image was first created is reused indefinitely: the container shipped
# 2.1.273 for a week after 2.1.280 was published. Nothing warns, because from
# Docker's point of view the build did exactly what it was told.
#
# The build arg is therefore the only thing that can carry the freshness signal.
# Resolving `latest` to the concrete version here makes the arg — and with it
# the layer's cache key — change precisely when a new version ships: cached when
# nothing moved, rebuilt the moment it did. It also makes the image
# reproducible, since the tag it was built from is recorded rather than implied.
#
# An explicit pin (CLAUDE_VERSION=2.1.280) is passed through untouched — pinning
# is the one case where reusing the cached layer is exactly right.
#
# Failure is LOUD but never blocking: a dev on a plane still gets a build, with
# a warning that it may be a stale cached layer. Hard-failing here would trade a
# possibly-stale CLI for no sandbox at all.

# Echoes the version to use for --build-arg CLAUDE_VERSION.
# Usage:  CLAUDE_VERSION="$(resolve_claude_version "${CLAUDE_VERSION}")"
resolve_claude_version() {
    local requested="${1:-latest}"

    # Only dist-tags need resolving; an explicit version is already concrete.
    case "${requested}" in
        latest|stable) ;;
        *) echo "${requested}"; return 0 ;;
    esac

    if ! command -v npm &>/dev/null; then
        echo "⚠️  npm not found — cannot resolve '${requested}' to a version." >&2
        echo "   Docker will reuse the cached Claude Code layer, which may be" >&2
        echo "   an OLDER version than '${requested}'. Install npm, or pass an" >&2
        echo "   explicit CLAUDE_VERSION=<x.y.z>, to guarantee a fresh CLI." >&2
        echo "${requested}"
        return 0
    fi

    local resolved
    # cd to HOME so a checked-in .npmrc/bunfig.toml in the repo cannot redirect
    # this lookup to another registry; the registry is pinned for the same reason.
    resolved="$(cd "${HOME}" && npm view "@anthropic-ai/claude-code@${requested}" version \
        --registry https://registry.npmjs.org/ 2>/dev/null | tr -d '[:space:]')"

    # Accept only a plain semver — npm errors and any other output must not be
    # interpolated into a build arg.
    if [[ ! "${resolved}" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.-]+)*$ ]]; then
        echo "⚠️  Could not resolve '@anthropic-ai/claude-code@${requested}' from npm." >&2
        echo "   Docker will reuse the cached Claude Code layer, which may be" >&2
        echo "   an OLDER version than '${requested}'. Check your network, or" >&2
        echo "   pass an explicit CLAUDE_VERSION=<x.y.z>." >&2
        echo "${requested}"
        return 0
    fi

    echo "${resolved}"
}

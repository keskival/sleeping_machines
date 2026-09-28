# Claude Code – safe local sandbox with Android/React Native toolchain
# Usage: ./dev.sh [claude args...]
#
# Build args:
#   NODE_VERSION   – Node.js major version  (default: 24)
#   CLAUDE_VERSION – @anthropic-ai/claude-code version to pin  (default: latest)

# Base: Debian trixie (Debian 13). Bumped from bookworm (Debian 12) to pull a
# current git — bookworm freezes git at 1:2.39.5, whose rebase machinery is
# unreliable on linux/arm64 (backlog A-ci-5: `git rebase` fails on arm64 even
# for zero-conflict replays). trixie ships git 1:2.47.3, a self-consistent
# native build on both amd64 and arm64/v8 (official node:*-trixie-slim is
# multi-arch), so the fix does not rely on cross-suite package pinning (a
# trixie git on a bookworm glibc would not even run — glibc ABI mismatch).
ARG NODE_VERSION=24
# SHA-pinned for supply-chain provenance (backlog A-ci-2). Digest resolved
# 2026-07-24 for tag 24-trixie-slim (Node 24 is the current LTS). Docker
# Official Images sign via Docker Content Trust (Notary v1), not cosign/sigstore,
# so there is no cosign OIDC identity or key to `cosign verify` against here.
# TODO(A-ci-2): revisit if the Docker Official Images program publishes
# sigstore signatures. NOTE: the digest is tied to the NODE_VERSION=24 default;
# if NODE_VERSION is overridden the pin no longer applies (build resolves the
# tag live) — re-pin here when the default Node major is bumped.
FROM node:${NODE_VERSION}-trixie-slim@sha256:ae91dcc111a68c9d2d81ff2a17bda61be126426176fde6fe7d08ab13b7f50573

ARG CLAUDE_VERSION=latest
LABEL org.opencontainers.image.title="claude-code-sandbox"
LABEL org.opencontainers.image.description="Safe local environment for Claude Code with Android SDK"

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Essentials
    bash curl wget git ca-certificates gnupg \
    # Build toolchain
    build-essential pkg-config \
    # Archive / file utils
    unzip zip tar xz-utils \
    # Text / search utils
    ripgrep jq less tree \
    # Process / network utils
    procps psmisc postgresql-client lsof iproute2 \
    # Python (many tools need it)
    python3 python3-pip python3-venv \
    lsb-release \
    sudo \
    gosu \
    # Editor for interactive git commits etc.
    nano vim \
    && rm -rf /var/lib/apt/lists/*

# ── Tailscale (for in-sandbox access to GB10 / DGX-Spark inference) ───────────
# Userspace networking only (sandbox runs unprivileged). The actual `up`
# happens at container-run time with a preauthorized ephemeral key in
# `TAILSCALE_AUTHKEY`; this layer just provisions the binaries.
RUN curl -fsSL https://pkgs.tailscale.com/stable/debian/trixie.noarmor.gpg \
       -o /usr/share/keyrings/tailscale-archive-keyring.gpg \
    && curl -fsSL https://pkgs.tailscale.com/stable/debian/trixie.tailscale-keyring.list \
       -o /etc/apt/sources.list.d/tailscale.list \
    && apt-get update && apt-get install -y --no-install-recommends \
       tailscale \
    && rm -rf /var/lib/apt/lists/*

# ── Docker CLI + daemon (Docker-in-Docker) ─────────────────────────────────────
RUN install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg \
       | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo \
       "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
       https://download.docker.com/linux/debian $(lsb_release -cs) stable" \
       > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y --no-install-recommends \
       docker-ce \
       docker-ce-cli \
       containerd.io \
       docker-buildx-plugin \
       docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

# ── GitHub CLI (handle issues / PRs / workflow runs from inside the sandbox) ──
RUN install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
       | dd of=/etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo \
       "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] \
       https://cli.github.com/packages stable main" \
       > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

# ── gitleaks (secret scanner required by .githooks/pre-commit) ────────────────
# The commit gate runs `gitleaks protect --staged` and hard-fails when the
# binary is absent, so a sandbox without it cannot commit at all — the only
# way out was --no-verify, which defeats the gate. Version tracks
# .github/workflows/gitleaks.yml so local and CI scan with the same engine;
# bump both together (the version is also quoted in the hook's help text).
ARG GITLEAKS_VERSION=8.30.1
# SHA-256 of the upstream release tarballs, resolved 2026-07-27 from
# gitleaks_${GITLEAKS_VERSION}_checksums.txt. Pinned as literals rather than
# re-fetching checksums.txt at build time (as the CI job does — it has the
# actions/checkout provenance chain we lack here): the checksum file ships
# from the same host as the tarball, so fetching it live proves nothing about
# a re-uploaded release. Both arches are listed because the base image is
# multi-arch (amd64 + arm64/v8) — see the NODE_VERSION digest note above.
ARG GITLEAKS_SHA256_AMD64=551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb
ARG GITLEAKS_SHA256_ARM64=e4a487ee7ccd7d3a7f7ec08657610aa3606637dab924210b3aee62570fb4b080
# No `set -o pipefail` here: /bin/sh is dash, which lacks it. The one pipeline
# below (`echo | sha256sum -c -`) already exits on the checksum's status, and
# `set -e` covers the rest.
RUN set -eux; \
    case "$(dpkg --print-architecture)" in \
      amd64) GL_ARCH=x64;   GL_SHA="${GITLEAKS_SHA256_AMD64}" ;; \
      arm64) GL_ARCH=arm64; GL_SHA="${GITLEAKS_SHA256_ARM64}" ;; \
      *) echo "gitleaks: unsupported arch $(dpkg --print-architecture)" >&2; exit 1 ;; \
    esac; \
    TARBALL="gitleaks_${GITLEAKS_VERSION}_linux_${GL_ARCH}.tar.gz"; \
    curl -fsSL -o "/tmp/${TARBALL}" \
      "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/${TARBALL}"; \
    echo "${GL_SHA}  /tmp/${TARBALL}" | sha256sum -c -; \
    tar -xzf "/tmp/${TARBALL}" -C /tmp gitleaks; \
    install -m 0755 /tmp/gitleaks /usr/local/bin/gitleaks; \
    rm -f "/tmp/${TARBALL}" /tmp/gitleaks; \
    gitleaks version

# ── Python deps for the repo's LLM helper scripts ─────────────────────────────
# `anthropic` only. It is the import behind scripts/translate-locale-gaps.py,
# which is the remediation the pre-commit hook itself prints when check:i18n
# finds locale gaps — without it the suggested fix dies on ModuleNotFoundError.
# Debian trixie marks the system interpreter externally-managed (PEP 668);
# --break-system-packages is correct for a single-purpose sandbox image and
# keeps `python3` the one interpreter on PATH (a venv would shadow it).
# Deliberately NOT installed: torch / datasets / llmcompressor / pdfplumber,
# imported by the model-quantisation and PDF-scraping scripts under scripts/.
# Those are multi-GB and unrelated to committing; install them ad hoc.
RUN pip3 install --no-cache-dir --break-system-packages anthropic \
    && python3 -c "import anthropic; print('anthropic', anthropic.__version__)"

# The edge-agent's Python test suite (packages/edge-agent/tests, 250 tests).
# Without these, `python3 -m pytest` is simply absent and the agent's tests get
# skipped silently — which is how a `disk_full` detector measuring the read-only
# HAOS rootfs shipped and alerted CRITICAL on every device for two days
# (2026-07-30). `httpx` + `psutil` are the agent's own runtime deps, needed for
# the tests to import it at all; the rest is the `[dev]` extra from
# packages/edge-agent/pyproject.toml.
RUN pip3 install --no-cache-dir --break-system-packages \
      pytest pytest-asyncio pytest-httpx httpx psutil \
    && python3 -m pytest --version

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jdk-headless \
    && rm -rf /var/lib/apt/lists/* \
    # OpenJDK installs to an arch-specific dir; symlink to a stable path.
    && ln -s "/usr/lib/jvm/java-21-openjdk-$(dpkg --print-architecture)" /usr/lib/jvm/java-21-openjdk
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk

ENV ANDROID_HOME=/opt/android-sdk
ENV ANDROID_SDK_ROOT=/opt/android-sdk
ENV PATH="${ANDROID_HOME}/cmdline-tools/latest/bin:${ANDROID_HOME}/platform-tools:${ANDROID_HOME}/build-tools/36.0.0:${PATH}"

# Grab the current revision from developer.android.com/studio#command-line-tools
ARG ANDROID_CMDLINE_TOOLS_VERSION=13114758
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p "${ANDROID_HOME}/cmdline-tools" \
    && curl -fsSL -o /tmp/clt.zip \
        "https://dl.google.com/android/repository/commandlinetools-linux-${ANDROID_CMDLINE_TOOLS_VERSION}_latest.zip" \
    && unzip -q /tmp/clt.zip -d "${ANDROID_HOME}/cmdline-tools" \
    && mv "${ANDROID_HOME}/cmdline-tools/cmdline-tools" "${ANDROID_HOME}/cmdline-tools/latest" \
    && rm /tmp/clt.zip \
    && yes | sdkmanager --licenses \
    && sdkmanager --install "platform-tools" "build-tools;36.0.0" "platforms;android-36"

# Install Google's cmdline-tools (sdkmanager) — needed for installing
# platforms, build-tools, NDK, and system images
RUN mkdir -p ${ANDROID_HOME}/cmdline-tools \
    && cd /tmp \
    && curl -fsSL https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip -o cmdline-tools.zip \
    && unzip -q cmdline-tools.zip \
    && mv cmdline-tools ${ANDROID_HOME}/cmdline-tools/latest \
    && rm cmdline-tools.zip

# Accept all SDK licenses non-interactively
RUN yes | ${ANDROID_HOME}/cmdline-tools/latest/bin/sdkmanager --licenses > /dev/null 2>&1 || true

# Install specific SDK components needed for Expo/React Native Android builds.
# CMake 3.22.1 is the version AGP pins for native builds (react-native-worklets,
# reanimated, etc.) — building without it falls back to system cmake which mismatches
# the prefab/header expectations and breaks linking. ninja-build feeds CMake's
# Ninja generator; without it, native builds fall back to slower Make.
RUN ${ANDROID_HOME}/cmdline-tools/latest/bin/sdkmanager --install \
    "platforms;android-36" \
    "platforms;android-31" \
    "build-tools;36.0.0" \
    "build-tools;35.0.1" \
    "build-tools;35.0.0" \
    "ndk;27.1.12297006" \
    "cmake;3.22.1" \
    # No system-image: this container builds/signs APKs (gradle assemble); the
    # emulator runs on the host. An x86_64 image is also wrong-arch on arm64.
    && chmod -R a+r ${ANDROID_HOME}

# Standalone cmake + ninja-build for any tooling that invokes them outside the
# Android SDK CMake (e.g. shell scripts, dev REPL). The Android-SDK cmake at
# ${ANDROID_HOME}/cmake/3.22.1 stays the one Gradle uses.
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# ── User setup ────────────────────────────────────────────────────────────────
ARG UID=1000
ARG GID=1000
ARG USERNAME=node

RUN echo "node ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd \
    && usermod -aG docker "node"

# ── Claude Code (npm – works reliably in non-interactive Docker builds) ───────
#
# CLAUDE_VERSION must arrive here ALREADY RESOLVED to a concrete version. A
# literal "latest" makes this RUN layer's cache key constant, so Docker reuses
# the layer forever and the image keeps shipping whatever version was current
# the day it was first built — the Dockerfile reads as "always newest" and does
# the opposite. The build scripts resolve the tag via
# scripts/resolve-claude-version.sh before calling docker build; a raw
# `docker build` with the default ARG below gets that stale-cache behaviour.
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_VERSION} \
    && claude --version

# Give the default user a writable npm prefix so global installs inside the
# container don't require root (handy if Claude installs extra tooling).
ENV NPM_CONFIG_PREFIX=/home/node/.npm-global
ENV PATH="/home/node/.npm-global/bin:${PATH}"

# How many subagents one Claude Code session may spawn in total.
#
# The default is 200, and a long working session on this repo hits it: the work
# fans out across HQ / api / crew-native / db surfaces, and once the ceiling is
# reached EVERY further parallel task silently degrades to running serially in
# the main loop. That is not a soft slowdown — it is the difference between
# eight issues being worked at once and one. Tero hit exactly this on
# 2026-07-29 ("Use subagents and parallelize!" against a run that could no
# longer spawn any).
#
# It is a lifetime counter per session, not a concurrency limit: raising it does
# not increase how many agents run at the same time (that is capped separately
# at ~min(16, cores-2)), so it costs nothing in CPU or memory. It only stops the
# session from running out of budget partway through.
#
# dev.sh forwards a host-side override of the same name, so bumping it further
# for one session needs no rebuild.
ENV CLAUDE_CODE_MAX_SUBAGENTS_PER_SESSION=400

RUN npm install -g @openai/codex
# ── Bun runtime (used by the API and as package manager) ─────────────────────
RUN npm install -g bun@1.3.14

RUN npm install -g @google/gemini-cli

# ── Maestro (mobile E2E testing framework) ───────────────────────────────────
# Installed into the node user's home so it persists across sessions
USER node
RUN curl -fsSL "https://get.maestro.mobile.dev" | bash
ENV PATH="/home/node/.maestro/bin:${PATH}"
USER root

# ── Make $HOME writable by ANY host uid ───────────────────────────────────────
# dev.sh runs the container with `--user <host-uid>:<host-gid>` so files written
# into the mounted /workspace are owned by the host user. That user is arbitrary:
# on Linux dev hosts (DaVinci) it is usually uid 1000, which happens to equal the
# image's `node` user, so $HOME (/home/node, created mode 0700 owned by 1000) is
# writable and everything works. On macOS the host uid is 501, which does NOT own
# /home/node and cannot write it — so Claude Code can neither persist its login
# (~/.claude/.credentials.json → you are "logged out" the instant you log in) nor
# write session transcripts (~/.claude/projects → "Transcript writes are failing,
# EACCES"). Both symptoms are the same unwritable-$HOME cause.
#
# Opening the home dir to any uid fixes it for every host uid without hardcoding
# one. This is a throwaway single-user dev sandbox (already --privileged, see
# dev.sh), not a multi-tenant host, so a world-writable home is an acceptable
# trade for "works on every developer's machine".
RUN chmod -R a+rwX /home/node

# ── Workspace ─────────────────────────────────────────────────────────────────
# Mounted at runtime by dev.sh → no files are baked into the image.
WORKDIR /workspace

# ── Security posture ──────────────────────────────────────────────────────────
# Drop to non-root by default. dev.sh passes --user to align with host UID.
USER node

# Default to LOW reasoning effort. Tero 2026-07-29, after the parallel burn-down
# exhausted a weekly token budget: "The costs are too high, can we set the effort
# level lower?"
#
# This is the `--effort` FLAG and deliberately not the `CLAUDE_CODE_EFFORT_LEVEL`
# environment variable. Both exist in the CLI, but only the flag is *provably*
# honoured: `claude --effort bogus` warns "Unknown --effort value … using the
# default effort", while `CLAUDE_CODE_EFFORT_LEVEL=bogus` is accepted in silence,
# so there is no evidence the env var is read. Do not "simplify" this to an ENV —
# a setting that silently does nothing is the defect class this repo keeps paying
# for (see tasks/backlog.md P2-c14/c15/c16: three real gates that existed and were
# unreachable). If you can prove the env var works, swap it and cite the proof.
#
# Override per session — later flags win: ./dev.sh --effort high
ENTRYPOINT ["claude", "--dangerously-skip-permissions", "--effort", "low"]

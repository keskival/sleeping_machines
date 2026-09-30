# Development container for Sleeping Machines.
# Launch with ./codex.sh or ./dev.sh.
ARG NODE_VERSION=24
FROM node:${NODE_VERSION}-trixie-slim@sha256:ae91dcc111a68c9d2d81ff2a17bda61be126426176fde6fe7d08ab13b7f50573

ARG CLAUDE_VERSION=latest
LABEL org.opencontainers.image.title="sleeping-machines-development"
LABEL org.opencontainers.image.description="Sleeping Machines research with Codex and Claude Code"

# Research dependencies, Git access and guarded experiment jobs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates curl wget git openssh-client gh \
    build-essential pkg-config cmake ninja-build \
    python3 python3-pip python3-venv \
    unzip zip tar xz-utils ripgrep jq less \
    procps psmisc lsof util-linux tmux sudo nano \
    && rm -rf /var/lib/apt/lists/*

# CPU research environment; GPU hosts use the AWS bootstrap script.
COPY requirements.txt /tmp/research-requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages \
        --index-url https://download.pytorch.org/whl/cpu torch \
    && pip3 install --no-cache-dir --break-system-packages \
        -r /tmp/research-requirements.txt h5py \
    && rm /tmp/research-requirements.txt

# The launchers resolve CLAUDE_VERSION before building to refresh its cache.
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_VERSION} @openai/codex \
    && claude --version && codex --version

ENV NPM_CONFIG_PREFIX=/home/node/.npm-global
ENV PATH="/home/node/.npm-global/bin:${PATH}"
ENV CLAUDE_CODE_MAX_SUBAGENTS_PER_SESSION=400
ENV CODEX_HOME=/home/node/.codex

COPY scripts/configure_codex_container.py /usr/local/lib/configure_codex_container.py
RUN python3 /usr/local/lib/configure_codex_container.py /home/node/.codex/config.toml

# Host UID mapping keeps workspace files owned by the launching user.
RUN echo "node ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/node \
    && chmod 0440 /etc/sudoers.d/node \
    && mkdir -p /home/node/.npm-global \
    && chmod -R a+rwX /home/node

WORKDIR /workspace
USER node
ENTRYPOINT ["claude", "--dangerously-skip-permissions", "--effort", "low"]

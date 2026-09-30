# Codex inside Docker

`codex.sh` uses Docker for isolation and disables Codex's additional command
sandbox. The workspace volume, including `.git`, is mounted read-write.
New containers retain the existing limits: **10 GiB RAM**, no additional swap,
and **3 CPUs**, adjustable with `DEV_MEMORY` and `DEV_CPUS`. Experiment jobs
still use `experiments/queue/run_safe.sh`, its lock and RSS watchdog.
The image contains the project's scientific Python dependencies, Codex, Claude
Code, Git/SSH, GitHub CLI and the tools used by the guarded experiment runner.
The launchers use Docker's default capabilities. GPU experiment hosts use the
AWS bootstrap script to install the appropriate PyTorch wheel.

## Apply to an existing container

Run from the **host checkout**:

```bash
./codex.sh --configure-only
```

For a named container, append the session name, or set `CONTAINER_NAME`:

```bash
./codex.sh --list
./codex.sh --configure-only speech
```

This updates `/home/node/.codex/config.toml` inside the container, backs up its
previous contents and preserves unrelated settings, authentication and
installed tools. It does not rebuild or remove the container. The container
must be running. Restart the Codex client to load the defaults. A running or
resumed app thread can retain its own explicit permission profile; select
**Full Access** in the client's Permissions control for that thread.

## Start a new container

```bash
./codex.sh
```

New launches configure the container's user defaults and pass
`--dangerously-bypass-approvals-and-sandbox` to Codex explicitly. The startup
configuration also applies with `NO_REBUILD=1`. The Dockerfile seeds the same
defaults for clients that start an app-server daemon without going through
the CLI launcher.

Re-running `codex.sh` reattaches to an existing process. Docker attach cannot
change its original launch flags, model arguments, image or resource limits.
The launcher checks for the bypass flag or its own startup label before
attaching. For an older container with a different command, use
`--configure-only` and restart its Codex client, or use a new session name to
create a fresh container while preserving the original container.
Image contents and Docker capability changes apply only to newly created
containers; updating Codex's config alone does not replace an existing image.

## Why `.git` was read-only

Codex's `workspace-write` / `:workspace` policy protects `.git` recursively even
when its parent workspace is writable. A `codex-linux-sandbox` tool process can
therefore see a read-only `.git` mount inside a writable Docker volume. Changing
Docker privileges or running `sudo` does not change that tool policy.

The CLI's `--yolo` alias already disabled its inner sandbox in fresh launches.
It does not set permission defaults for a separately started app-server daemon
or replace an explicit permission profile on an existing app thread. Current
session permissions cannot be changed by editing a launcher or image; the
client must apply the new setting. Managed administrative requirements, if
present, remain authoritative.

References: [OpenAI sandbox and approvals](https://learn.chatgpt.com/docs/agent-approvals-security),
[permission profiles](https://learn.chatgpt.com/docs/permissions),
[configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).

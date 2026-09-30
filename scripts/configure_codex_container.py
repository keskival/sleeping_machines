#!/usr/bin/env python3
"""Set container Codex defaults, preserving unrelated config and login files.

Invoked by Dockerfile or codex.sh on the host, outside a running tool sandbox.
The config path is explicit so invoking this file alone cannot modify host
Codex settings. Both legacy settings and permission-profile configs are supported.
"""
from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import sys
import tempfile
import tomllib


def configure(path: Path) -> None:
    original = path.read_text() if path.exists() else ""
    config = tomllib.loads(original)
    # Select one settings family. Explicit legacy settings take precedence in
    # Codex; never add a conflicting default_permissions alongside them.
    profiles = ("default_permissions" in config or "permissions" in config) and not (
        "sandbox_mode" in config or "sandbox_workspace_write" in config
    )
    settings = {"approval_policy": "never"}
    settings["default_permissions" if profiles else "sandbox_mode"] = (
        ":danger-full-access" if profiles else "danger-full-access"
    )
    replaced = {"approval_policy", "sandbox_mode", "default_permissions"}

    # Parse complete root statements, including multiline strings and inline
    # tables. Preserve the text of unrelated settings and all named sections.
    lines = original.splitlines(keepends=True)
    kept = []
    i = 0
    while i < len(lines):
        start = i
        while True:
            i += 1
            statement = "".join(lines[start:i])
            try:
                parsed = tomllib.loads(statement)
                break
            except tomllib.TOMLDecodeError:
                if i == len(lines):
                    raise ValueError("Cannot isolate a root config statement; config was not changed")
        if statement.lstrip().startswith("["):
            kept.append("".join(lines[start:]))
            break
        if not replaced.intersection(parsed):
            kept.append(statement)

    header = "".join(f'{key} = "{value}"\n' for key, value in settings.items())
    updated = header + "".join(kept)
    expected = {key: value for key, value in config.items() if key not in replaced}
    expected.update(settings)
    if tomllib.loads(updated) != expected:
        raise ValueError("Config edit would change unrelated settings; config was not changed")
    if tomllib.loads(original) == expected:
        print(f"Codex container defaults already configured: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
    if original:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = path.with_name(path.name + ".before-docker-access-" + stamp)
        shutil.copyfile(path, backup)
        backup.chmod(0o600)
        print(f"Previous Codex config saved to {backup}")
    fd, temporary = tempfile.mkstemp(prefix=".codex-config-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(updated)
        os.chmod(temporary, mode)
        if (path.read_text() if path.exists() else "") != original:
            raise RuntimeError("Codex config changed concurrently; retry configuration")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f"Configured {path}: no inner sandbox; approval_policy=never")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: configure_codex_container.py /path/to/container/config.toml")
    configure(Path(sys.argv[1]))

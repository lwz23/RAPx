#!/usr/bin/env python3
"""Exec a command with a JSON environment inside a systemd user unit."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 4:
        print("usage: run_in_user_memcg.py ENV.json CWD COMMAND...", file=sys.stderr)
        return 2
    env = json.loads(Path(sys.argv[1]).read_text())
    if not isinstance(env, dict):
        print("env json must be an object", file=sys.stderr)
        return 2
    cwd = sys.argv[2]
    command = sys.argv[3:]
    os.chdir(cwd)
    os.execvpe(command[0], command, {str(k): str(v) for k, v in env.items()})
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

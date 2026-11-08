#!/usr/bin/env python3
"""Wake the managing session when leftover RAP exits, stalls, or finishes.

Stdout is exactly one line: DONE, FAILED, or STALE.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main() -> int:
    if len(sys.argv) < 3:
        print("FAILED")
        return 2
    pid = int(sys.argv[1])
    heartbeat = Path(sys.argv[2])
    stale_seconds = int(sys.argv[3]) if len(sys.argv) > 3 else 1800
    log = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    started = time.time()
    while pid_alive(pid):
        if heartbeat.exists():
            age = time.time() - heartbeat.stat().st_mtime
            if age > stale_seconds:
                print("STALE")
                return 0
        elif time.time() - started > stale_seconds:
            print("STALE")
            return 0
        time.sleep(20)
    tail = ""
    if log and log.exists():
        try:
            tail = log.read_text(errors="replace")[-4000:]
        except OSError:
            tail = ""
    print("FAILED" if "Traceback (most recent call last)" in tail else "DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Try CodeQL database create on RAP no_artifact checkouts."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

CODEQL = Path(
    "/Users/liaowenzhao/Documents/RAP重构/lwz23-valid-issue-scan/codeql-poc/codeql/codeql"
)
SRC_ROOT = Path("/Users/liaowenzhao/Documents/RAP重构/lwz23-valid-issue-src")
OUT = Path(
    "/Users/liaowenzhao/Documents/RAP重构/lwz23-valid-issue-scan/codeql-poc/no-artifact-probe"
)
REPOS = [
    "NVSL/Corundum",
    "asg017/sqlite-loadable-rs",
    "archseer/enigma",
    "Indra-db/Flecs-Rust",
    "DioxusLabs/blitz",
    "Pagefind/pagefind",
]
TIMEOUT = 300


def decode(blob) -> str:
    if blob is None:
        return ""
    if isinstance(blob, bytes):
        return blob.decode("utf-8", "replace")
    return str(blob)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CodeQL no_artifact extraction probe",
        f"started {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        f"timeout_s {TIMEOUT}",
        "",
    ]
    for repo in REPOS:
        safe = repo.replace("/", "_")
        src = SRC_ROOT / repo
        db = OUT / f"db-{safe}"
        log_path = OUT / f"{safe}.log"
        print(f"===== {repo} =====", flush=True)
        lines.append(f"## {repo}")
        if not (src / "Cargo.toml").is_file():
            msg = f"SKIP no Cargo.toml at {src}"
            print(msg, flush=True)
            lines.extend([msg, ""])
            continue
        if db.exists():
            shutil.rmtree(db)
        cmd = [
            str(CODEQL),
            "database",
            "create",
            str(db),
            "--language=rust",
            "--build-mode=none",
            "--source-root",
            str(src),
            "--overwrite",
        ]
        t0 = time.time()
        with log_path.open("w") as log:
            log.write(" ".join(cmd) + "\n")
            log.flush()
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                rc = proc.wait(timeout=TIMEOUT)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, 15)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, 9)
                    except (ProcessLookupError, PermissionError):
                        pass
                rc = 124
        elapsed = time.time() - t0
        ok = (db / "db-rust").is_dir() and rc == 0
        if ok:
            verdict = "EXTRACT_OK"
        elif rc == 124:
            verdict = "TIMEOUT"
        else:
            verdict = f"EXTRACT_FAIL rc={rc}"
        text = log_path.read_text(errors="replace")
        tail = "\n".join(text.strip().splitlines()[-12:])
        print(f"{verdict} elapsed={elapsed:.1f}s", flush=True)
        print(tail, flush=True)
        lines.append(f"- {verdict} elapsed={elapsed:.1f}s")
        lines.append("```")
        lines.append(tail)
        lines.append("```")
        lines.append("")
        if db.exists():
            shutil.rmtree(db, ignore_errors=True)
    lines.append(f"done {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    (OUT / "probe.md").write_text("\n".join(lines) + "\n")
    print("WROTE", OUT / "probe.md", flush=True)


if __name__ == "__main__":
    main()

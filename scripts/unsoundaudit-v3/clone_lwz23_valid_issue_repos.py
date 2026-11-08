#!/usr/bin/env python3
"""Clone the frozen 125 scan repositories and check out the pre-report revision."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


MIN_FREE_BYTES = 6 * 1024 * 1024 * 1024
MAX_GIT_DIR_BYTES = 400 * 1024 * 1024
CLONE_TIMEOUT_S = 180
CHECKOUT_TIMEOUT_S = 180


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return usage.free


def dir_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def run(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: int,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


GIT_LFS_CFG = [
    "-c",
    "filter.lfs.smudge=cat",
    "-c",
    "filter.lfs.clean=cat",
    "-c",
    "filter.lfs.process=",
    "-c",
    "filter.lfs.required=false",
]


def git_env() -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "never"
    env["GIT_HTTP_LOW_SPEED_LIMIT"] = "1000"
    env["GIT_HTTP_LOW_SPEED_TIME"] = "30"
    return env


def git_cmd(*args: str) -> list[str]:
    return ["git", *GIT_LFS_CFG, *args]


def already_done(journal: Path, repository: str) -> bool:
    if not journal.exists():
        return False
    last = None
    for line in journal.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("repository") == repository:
            last = row
    return bool(last and last.get("status") in {"checked_out", "cloned_no_checkout"})


def resolve_revision(repo_dir: Path, created_at: str) -> dict[str, Any]:
    show = run(
        git_cmd("rev-parse", "--abbrev-ref", "origin/HEAD"),
        cwd=repo_dir,
        timeout=30,
        env=git_env(),
    )
    default_ref = (show.stdout or "").strip() or "origin/HEAD"
    before = created_at.replace("T", " ").replace("Z", " UTC")
    listed = run(
        git_cmd("rev-list", "-n", "1", f"--before={before}", default_ref),
        cwd=repo_dir,
        timeout=60,
        env=git_env(),
    )
    sha = (listed.stdout or "").strip()
    fallback = False
    if listed.returncode != 0 or not sha:
        head = run(
            git_cmd("rev-parse", default_ref),
            cwd=repo_dir,
            timeout=30,
            env=git_env(),
        )
        sha = (head.stdout or "").strip()
        fallback = True
    return {
        "default_ref": default_ref,
        "sha": sha,
        "fallback_to_head": fallback,
        "rev_list_returncode": listed.returncode,
        "rev_list_stderr": (listed.stderr or "")[-500:],
    }


def clone_one(entry: dict[str, Any], src_root: Path, journal: Path) -> dict[str, Any]:
    repo = entry["repository"]
    dest = src_root / repo
    started = time.time()
    row: dict[str, Any] = {
        "repository": repo,
        "clone_url": entry["clone_url"],
        "dest": str(dest),
        "primary_issue_url": entry["primary_issue_url"],
        "primary_created_at": entry["primary_created_at"],
        "started_unix": started,
    }
    free = free_bytes(src_root)
    row["free_bytes_before"] = free
    if free < MIN_FREE_BYTES:
        row["status"] = "stopped_low_disk"
        return row

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (dest / ".git").exists():
        row["clone_skipped"] = True
    else:
        if dest.exists():
            shutil.rmtree(dest)
        cloned = run(
            git_cmd(
                "clone",
                "--filter=blob:none",
                "--no-tags",
                "--single-branch",
                "--no-checkout",
                entry["clone_url"],
                str(dest),
            ),
            timeout=CLONE_TIMEOUT_S,
            env=git_env(),
        )
        row["clone_returncode"] = cloned.returncode
        row["clone_stderr"] = (cloned.stderr or "")[-800:]
        if cloned.returncode != 0:
            row["status"] = "clone_failed"
            if dest.exists() and not (dest / ".git").exists():
                shutil.rmtree(dest, ignore_errors=True)
            return row

    git_size = dir_bytes(dest / ".git")
    row["git_dir_bytes"] = git_size
    if git_size > MAX_GIT_DIR_BYTES:
        row["status"] = "cloned_no_checkout"
        row["reason"] = f".git exceeded {MAX_GIT_DIR_BYTES} bytes"
        return row

    resolved = resolve_revision(dest, entry["primary_created_at"])
    row["revision"] = resolved
    sha = resolved.get("sha") or ""
    if not sha:
        row["status"] = "revision_unresolved"
        return row

    checked = run(
        git_cmd("checkout", "--detach", "--force", sha),
        cwd=dest,
        timeout=CHECKOUT_TIMEOUT_S,
        env=git_env(),
    )
    row["checkout_returncode"] = checked.returncode
    row["checkout_stderr"] = (checked.stderr or "")[-800:]
    row["worktree_bytes"] = dir_bytes(dest)
    if checked.returncode != 0:
        row["status"] = "checkout_failed"
        return row
    row["status"] = "checked_out"
    row["elapsed_seconds"] = round(time.time() - started, 3)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("docs/unsoundaudit-v3/lwz23-valid-issue-roster-v1/clone_plan.json"),
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("/Users/liaowenzhao/Documents/RAP重构/lwz23-valid-issue-src"),
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    plan = load_json(args.plan)
    src_root = args.src_root
    src_root.mkdir(parents=True, exist_ok=True)
    journal = src_root / "clone_journal.jsonl"
    summary_path = src_root / "clone_summary.json"

    repos = list(plan["repositories"])
    if args.limit:
        repos = repos[: args.limit]

    counts: dict[str, int] = {}
    stopped = False
    for entry in repos:
        if already_done(journal, entry["repository"]):
            counts["already_done"] = counts.get("already_done", 0) + 1
            continue
        try:
            row = clone_one(entry, src_root, journal)
        except subprocess.TimeoutExpired:
            row = {
                "repository": entry["repository"],
                "status": "timeout",
                "clone_url": entry["clone_url"],
            }
        except Exception as exc:  # noqa: BLE001
            row = {
                "repository": entry["repository"],
                "status": "error",
                "error": str(exc),
            }
        row["free_bytes_after"] = free_bytes(src_root)
        append_jsonl(journal, row)
        counts[row.get("status") or "unknown"] = counts.get(row.get("status") or "unknown", 0) + 1
        print(f"{row.get('status'):22s} {entry['repository']}", flush=True)
        if row.get("status") == "stopped_low_disk":
            stopped = True
            break

    summary = {
        "src_root": str(src_root),
        "counts": counts,
        "stopped_low_disk": stopped,
        "free_bytes": free_bytes(src_root),
        "plan_scan_repositories": plan["scan_repositories"],
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if not stopped else 2


if __name__ == "__main__":
    sys.exit(main())

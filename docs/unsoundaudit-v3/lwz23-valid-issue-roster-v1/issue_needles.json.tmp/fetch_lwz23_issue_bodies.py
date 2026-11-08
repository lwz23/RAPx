#!/usr/bin/env python3
"""Fetch GitHub issue bodies for the frozen 148-issue roster and cache them locally.

Uses `gh` (authenticated). Does not use the anonymized report as an oracle.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


WEAK = {
    "slice",
    "crates",
    "stack",
    "pointer",
    "possible",
    "report",
    "access",
    "starlight",
    "length",
    "offset",
    "table",
    "free",
    "name",
    "path",
    "from",
    "get",
}

GENERIC_TITLE_TOKENS = WEAK | {
    "maybe",
    "multiple",
    "unsound",
    "unsoundness",
    "problems",
    "problem",
    "potential",
    "possible",
    "undefined",
    "behavior",
    "function",
    "functions",
    "implementation",
    "issue",
    "issues",
    "public",
    "soundness",
    "allows",
    "due",
    "unsafe",
    "pointer",
    "operations",
    "usage",
    "causes",
    "leading",
    "memory",
    "corruption",
    "missing",
    "check",
    "checks",
    "lack",
    "bounds",
    "invalid",
    "input",
    "handling",
    "method",
    "other",
}

KNOWN_FALSE_NEEDLES = {
    "starlight",
    "find_symbol",
    "retrowin32",
}

FENCE_RE = re.compile(r"```(?:rust|rs|ignore|no_run)?\n(.*?)```", re.DOTALL | re.IGNORECASE)
PUB_FN_RE = re.compile(r"\b(?:pub(?:\s*\([^)]*\))?\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)")
PATH_RE = re.compile(r"(?:^|[\s`'\"(])((?:src|crates|core|lib|include|ffi)/[\w./-]+\.rs)")
BLOB_PATH_RE = re.compile(
    r"github\.com/[^/]+/[^/]+/(?:blob|tree)/[^/]+/((?:src|crates|core|lib|include|ffi)[\w./-]*\.rs)",
    re.I,
)
BACKTICK_RE = re.compile(r"`([^`]+)`")
QUALIFIED_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+")
TYPE_RE = re.compile(r"\b([A-Z][A-Za-z0-9]{3,})\b")
IDENT_RE = re.compile(r"\b([a-z_][a-z0-9_]{4,})\b")
CALLEE_NAMES = (
    "from_utf8_unchecked",
    "from_raw_parts",
    "from_raw_parts_mut",
    "get_unchecked",
    "get_unchecked_mut",
    "from_ptr",
    "assume_init",
    "memcpy",
    "transmute",
    "zeroed",
    "copy_nonoverlapping",
    "from_utf8_unchecked_mut",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def parse_issue_url(url: str) -> tuple[str, str]:
    # https://github.com/owner/repo/issues/N
    parts = url.rstrip("/").split("/")
    if "github.com" not in url or "issues" not in parts:
        raise ValueError(f"unrecognized issue url: {url}")
    idx = parts.index("issues")
    repo = "/".join(parts[idx - 2 : idx])
    number = parts[idx + 1]
    return repo, number


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def gh_issue(repo: str, number: str) -> dict[str, Any]:
    env = dict(os.environ)
    env.update(
        {
            "GH_PAGER": "cat",
            "GH_PROMPT_DISABLED": "1",
            "NO_COLOR": "1",
            "CLICOLOR": "0",
            "CLICOLOR_FORCE": "0",
            "GH_FORCE_TTY": "0",
        }
    )
    proc = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{repo}/issues/{number}",
            "--jq",
            "{title:.title, body:.body, url:.html_url, number:.number, state:.state}",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    raw = ANSI_RE.sub("", proc.stdout or "").strip()
    if not raw:
        raise RuntimeError(
            f"gh empty stdout rc={proc.returncode} stderr={(proc.stderr or '').strip()[:400]!r}"
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"gh non-json rc={proc.returncode} stdout={raw[:200]!r} stderr={(proc.stderr or '')[:200]!r}"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(payload.get("message") if isinstance(payload, dict) else raw[:400])
    return {
        "title": payload.get("title") or "",
        "body": payload.get("body") or "",
        "url": payload.get("url") or "",
        "number": payload.get("number"),
        "state": payload.get("state") or "",
    }


def add_unique(dst: list[str], value: str) -> None:
    item = value.strip()
    if not item:
        return
    if item not in dst:
        dst.append(item)


def extract_needles(title: str, body: str, obligation: str) -> dict[str, Any]:
    fences = [m.group(1) for m in FENCE_RE.finditer(body or "")]
    fence_blob = "\n".join(fences)
    sources = {
        "body": body or "",
        "fence": fence_blob,
        "obligation": obligation or "",
        "title": title or "",
    }
    qualified: list[str] = []
    paths: list[str] = []
    fns: list[str] = []
    callees: list[str] = []
    types: list[str] = []

    def harvest(text: str, *, from_title: bool) -> None:
        for match in PATH_RE.finditer(text):
            add_unique(paths, match.group(1).replace("\\", "/").lower())
        for match in BLOB_PATH_RE.finditer(text):
            add_unique(paths, match.group(1).replace("\\", "/").lower())
        for match in re.finditer(r"(?:src|crates)/[\w./-]+\.rs", text):
            add_unique(paths, match.group(0).replace("\\", "/").lower())
        for match in PUB_FN_RE.finditer(text):
            add_unique(fns, match.group(1))
        for match in QUALIFIED_RE.finditer(text):
            add_unique(qualified, match.group(0))
        for match in BACKTICK_RE.finditer(text):
            inner = match.group(1).split("(")[0].split("<")[0].strip()
            if inner.endswith(".rs"):
                add_unique(paths, inner.replace("\\", "/").lower())
            elif "::" in inner:
                add_unique(qualified, inner)
            elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", inner):
                if inner[0].isupper():
                    add_unique(types, inner)
                elif not from_title or inner.lower() not in GENERIC_TITLE_TOKENS:
                    add_unique(fns, inner)
        for match in TYPE_RE.finditer(text):
            name = match.group(1)
            if name.lower() not in GENERIC_TITLE_TOKENS:
                add_unique(types, name)
        blob_l = text.lower()
        for name in CALLEE_NAMES:
            if name in blob_l:
                add_unique(callees, name)
        if not from_title:
            for match in IDENT_RE.finditer(blob_l):
                name = match.group(1)
                if name not in GENERIC_TITLE_TOKENS and name not in KNOWN_FALSE_NEEDLES:
                    if name in CALLEE_NAMES:
                        add_unique(callees, name)

    harvest(fence_blob, from_title=False)
    harvest(body or "", from_title=False)
    harvest(obligation or "", from_title=False)
    harvest(title or "", from_title=True)

    strong = bool(paths or fns or [q for q in qualified if q.count("::") >= 1] or callees)
    return {
        "qualified": qualified,
        "paths": paths,
        "fns": fns,
        "callees": callees,
        "types": types,
        "has_fence": bool(fences),
        "strong": strong,
        "source_priority": "body" if (body or "").strip() else ("obligation" if obligation else "title"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()
    roster = args.repo_root / "docs/unsoundaudit-v3/lwz23-valid-issue-roster-v1"
    issues_path = roster / "issues.csv"
    out_jsonl = roster / "issue_bodies.jsonl"
    out_needles = roster / "issue_needles.json"
    obligations: dict[str, str] = {}
    instances = args.repo_root / "docs/unsoundaudit-v3/lwz23-valid-issue-roster-inputs/bug_instances.jsonl"
    for instance in load_jsonl(instances):
        url = str(instance.get("issue_url") or "")
        text = instance.get("repair_obligation_zh") or ""
        if url and text:
            obligations[url] = (obligations.get(url, "") + " " + text).strip()

    existing = {row.get("issue_url"): row for row in load_jsonl(out_jsonl)}
    issues = list(csv.DictReader(issues_path.open()))
    if len(issues) != 148:
        print(f"expected 148 issues, got {len(issues)}", file=sys.stderr)
        return 2

    wrote = 0
    failed: list[str] = []
    for issue in issues:
        url = issue["issue_url"]
        if url in existing and (existing[url].get("body") is not None):
            continue
        repo, number = parse_issue_url(url)
        try:
            payload = gh_issue(repo, number)
        except Exception as exc:  # noqa: BLE001 — keep going, record failure
            failed.append(f"{url}: {exc}")
            time.sleep(max(args.sleep, 1.0))
            continue
        row = {
            "repository": issue["repository"],
            "issue_number": issue["issue_number"],
            "issue_url": url,
            "title": payload.get("title") or issue["title"],
            "body": payload.get("body") or "",
            "state": payload.get("state") or issue.get("state"),
            "fetched_via": "gh issue view",
        }
        existing[url] = row
        wrote += 1
        # rewrite jsonl from existing to stay resume-safe
        with out_jsonl.open("w", encoding="utf-8") as handle:
            for item in issues:
                cached = existing.get(item["issue_url"])
                if cached:
                    handle.write(json.dumps(cached, ensure_ascii=False) + "\n")
        time.sleep(args.sleep)

    needles: dict[str, Any] = {}
    missing_body = 0
    for issue in issues:
        url = issue["issue_url"]
        cached = existing.get(url) or {}
        body = cached.get("body") or ""
        if not str(body).strip():
            missing_body += 1
        needles[url] = {
            "repository": issue["repository"],
            "issue_number": issue["issue_number"],
            "title": cached.get("title") or issue["title"],
            "body_chars": len(body),
            "needles": extract_needles(
                cached.get("title") or issue["title"],
                body,
                obligations.get(url, ""),
            ),
        }
    out_needles.write_text(json.dumps(needles, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "issues": len(issues),
                "cached": sum(1 for i in issues if i["issue_url"] in existing),
                "fetched_this_run": wrote,
                "missing_body": missing_body,
                "failed": failed,
                "bodies": str(out_jsonl),
                "needles": str(out_needles),
            },
            indent=2,
        )
    )
    return 1 if failed and missing_body else 0


if __name__ == "__main__":
    raise SystemExit(main())

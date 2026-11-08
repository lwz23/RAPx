#!/usr/bin/env python3
"""Rescore RAP/CodeQL artifacts against the 148-issue roster using issue-body needles."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
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
}

SKIP_HOST_CLASS = {
    ("Kudaes/DInvoke_rs", "10"): "windows-pe / host-impossible",
    ("Kudaes/Elevator", "9"): "windows-pe / host-impossible",
    ("joaoviictorti/shadow", "5"): "windows-pe / host-impossible",
    ("SitinCloud/Owlyshield", "59"): "windows-pe / host-impossible",
    ("BrowserWorks/waterfox", "3980"): "browser-monorepo / not a practical Cargo crate scan",
    ("rustsec/advisory-db", "2141"): "advisory-db is not a software crate; scan mapped crates",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def needles(title: str, obligation: str, body: str = "") -> dict[str, list[str]]:
    text = "\n".join(part for part in (body, obligation, title) if part)
    qualified: list[str] = []
    paths: list[str] = []
    callees: list[str] = []
    fns: list[str] = []
    for match in re.finditer(
        r"github\.com/[^/]+/[^/]+/(?:blob|tree)/[^/]+/((?:src|crates|core|lib)[\w./-]*\.rs)",
        text,
        re.I,
    ):
        paths.append(match.group(1).replace("\\", "/").lower())
    for match in re.finditer(r"(?:src|crates)/[\w./-]+\.rs", text):
        paths.append(match.group(0).replace("\\", "/").lower())
    for match in re.finditer(r"\b(?:pub(?:\s*\([^)]*\))?\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)", text):
        fns.append(match.group(1))
    for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+", text):
        qualified.append(match.group(0).lower())
    for match in re.finditer(r"`([^`]+)`", text):
        inner = match.group(1).split("(")[0].split("<")[0].strip()
        if "::" in inner:
            qualified.append(inner.lower())
        elif inner.endswith(".rs"):
            paths.append(inner.lower())
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", inner) and inner.lower() not in GENERIC_TITLE_TOKENS:
            fns.append(inner)
    for match in re.finditer(r"'(?:pub fn )?([A-Za-z_][A-Za-z0-9_]*)'", text):
        name = match.group(1).lower()
        if name not in WEAK or name == "get":
            qualified.append("::" + name)
    for match in re.finditer(r"\b([A-Z][A-Za-z0-9]{4,})\b", text):
        qualified.append(match.group(1).lower())
    for match in re.finditer(r"\b([a-z_][a-z0-9_]{5,})\b", text.lower()):
        name = match.group(1)
        if name not in GENERIC_TITLE_TOKENS:
            qualified.append(name)
    for name in (
        "from_utf8_unchecked",
        "from_raw_parts",
        "from_raw_parts_mut",
        "get_unchecked",
        "get_unchecked_mut",
        "from_ptr",
        "assume_init",
        "memcpy",
        "transmute",
    ):
        if name in text.lower():
            callees.append(name)
    return {
        "qualified": list(dict.fromkeys(qualified)),
        "paths": list(dict.fromkeys(paths)),
        "callees": list(dict.fromkeys(callees)),
        "fns": list(dict.fromkeys(fns)),
    }


def load_active(mvp_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not mvp_root.exists():
        return rows
    for pkg in mvp_root.iterdir():
        if not pkg.is_dir():
            continue
        for line in load_jsonl(pkg / "candidates.jsonl"):
            site = line.get("unsafe_site") or {}
            span = site.get("span") or {}
            rows.append(
                {
                    "pattern": line.get("primary_pattern"),
                    "root": str((line.get("public_safe_root") or {}).get("owner") or ""),
                    "callee": str(site.get("callee") or ""),
                    "path": str(span.get("path") or ""),
                    "line": (span.get("start") or {}).get("line"),
                    "confidence": line.get("classification_confidence") or "exact",
                    "package": pkg.name,
                }
            )
    return rows


def load_unest(mvp_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not mvp_root.exists():
        return rows
    for pkg in mvp_root.iterdir():
        if not pkg.is_dir():
            continue
        for line in load_jsonl(pkg / "unestablished.jsonl"):
            rows.append(
                {
                    "pattern": line.get("attempted_pattern"),
                    "root": str(line.get("public_root") or ""),
                    "reason": str(line.get("primary_reason") or ""),
                    "package": pkg.name,
                }
            )
    return rows


def root_matches_qualified(root: str, qualified: str) -> bool:
    root_l = root.lower()
    q = qualified.lower()
    if not q:
        return False
    if q in root_l:
        return True
    last = q.split("::")[-1]
    if last in WEAK and q.count("::") == 0 and last != "get":
        return False
    if q.startswith("::"):
        return root_l.endswith(q) or f"::{last}" == f"::{root_l.split('::')[-1]}" and last == "get"
    return root_l.endswith("::" + last) and (q.split("::")[0] in root_l or last == "get")


def is_known_false_hit(row: dict[str, Any]) -> bool:
    hay = " ".join(
        str(row.get(key) or "").lower()
        for key in ("root", "callee", "path", "package")
    )
    if "starlight" in hay:
        return True
    if "find_symbol" in hay:
        return True
    if "retrowin32" in hay and re.search(r"\barg\b", hay):
        return True
    return False


def match_active(issue_needles: dict[str, list[str]], actives: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits = []
    for row in actives:
        if is_known_false_hit(row):
            continue
        root = row["root"]
        path = row["path"].replace("\\", "/").lower()
        callee = row["callee"].lower()
        reasons = []
        for q in issue_needles["qualified"]:
            if root_matches_qualified(root, q):
                reasons.append("root:" + q)
        for fn in issue_needles.get("fns") or []:
            if root_matches_qualified(root, fn) or root.lower().endswith("::" + fn.lower()):
                reasons.append("fn:" + fn)
        for p in issue_needles["paths"]:
            if p in path or path.endswith(p.split("/")[-1]):
                if p.split("/")[-1].replace(".rs", "") not in WEAK:
                    reasons.append("path:" + p)
        for c in issue_needles["callees"]:
            if c in callee:
                type_ok = any(root_matches_qualified(root, q) for q in issue_needles["qualified"]) or not issue_needles["qualified"]
                if type_ok or issue_needles["callees"]:
                    if any(root_matches_qualified(root, q) for q in issue_needles["qualified"]) or (
                        not issue_needles["qualified"] and not issue_needles["paths"]
                    ):
                        reasons.append("callee:" + c)
        if reasons:
            item = dict(row)
            item["why"] = reasons
            hits.append(item)
    return hits


def match_unest(issue_needles: dict[str, list[str]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits = []
    for row in rows:
        if any(root_matches_qualified(row["root"], q) for q in issue_needles["qualified"]):
            hits.append(row)
    return hits


def collect_mvp_dirs(scan_root: Path, repo: str) -> list[Path]:
    found = []
    slug = repo.replace("/", "__")
    roots = [
        scan_root / "work" / slug / "mvp",
        scan_root / "phase1" / "work" / slug / "mvp",
        scan_root / "phase2" / "work" / slug / "mvp",
        scan_root / "phase4" / "work" / slug / "mvp",
        scan_root / "phase2024" / "work" / slug / "mvp",
        scan_root / slug / "mvp",
        Path("/Volumes/RAPDATA/phase2024/work") / slug / "mvp",
    ]
    extra = os.environ.get("UNSOUND_RESCORE_EXTRA_MVP_ROOT")
    if extra:
        roots.append(Path(extra) / slug / "mvp")
    for base in roots:
        if base.exists():
            found.append(base)
    return found


def load_bodies(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("issue_url") or ""): row for row in load_jsonl(path) if row.get("issue_url")}


def load_precomputed_needles(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    return raw


def merge_needles(title: str, obligation: str, body: str, precomputed: dict[str, Any] | None) -> dict[str, list[str]]:
    n = needles(title, obligation, body)
    if precomputed:
        inner = precomputed.get("needles") if "needles" in precomputed else precomputed
        if isinstance(inner, dict):
            for key in ("qualified", "paths", "callees", "fns"):
                extra = inner.get(key) or []
                if extra:
                    n.setdefault(key, [])
                    for item in extra:
                        if item not in n[key]:
                            n[key].append(item)
    n.setdefault("fns", [])
    return n


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    rap = args.repo_root.resolve()
    scan = args.scan_root.resolve()
    roster = rap / "docs/unsoundaudit-v3/lwz23-valid-issue-roster-v1"
    issues = list(csv.DictReader((roster / "issues.csv").open()))
    if len(issues) != 148:
        raise SystemExit(f"expected 148 issues, got {len(issues)}")
    obligations: dict[str, str] = {}
    for instance in load_jsonl(
        rap / "docs/unsoundaudit-v3/lwz23-valid-issue-roster-inputs/bug_instances.jsonl"
    ):
        url = str(instance.get("issue_url") or "")
        text = instance.get("repair_obligation_zh") or ""
        if url and text:
            obligations[url] = (obligations.get(url, "") + " " + text).strip()
    bodies = load_bodies(roster / "issue_bodies.jsonl")
    precomputed = load_precomputed_needles(roster / "issue_needles.json")

    out_rows = []
    for issue in issues:
        repo = issue["repository"]
        number = str(issue["issue_number"])
        url = issue["issue_url"]
        title = issue["title"]
        key = (repo, number)
        body = str((bodies.get(url) or {}).get("body") or "")
        n = merge_needles(title, obligations.get(url, ""), body, precomputed.get(url))
        actives: list[dict[str, Any]] = []
        unests: list[dict[str, Any]] = []
        sources = []
        for mvp in collect_mvp_dirs(scan, repo):
            try:
                sources.append(str(mvp.relative_to(scan)))
            except ValueError:
                sources.append(str(mvp))
            actives.extend(load_active(mvp))
            unests.extend(load_unest(mvp))
        hits = match_active(n, actives)
        uhits = match_unest(n, unests)
        strong_needles = [
            q
            for q in (n["qualified"] + n.get("fns", []))
            if q.split("::")[-1].lower() not in GENERIC_TITLE_TOKENS
        ]
        body_names_function = bool(n.get("fns") or n["paths"] or n["callees"] or strong_needles)
        generic_title = (not body.strip()) and not n["paths"] and not n["callees"] and not strong_needles
        skip_class = SKIP_HOST_CLASS.get(key)
        if skip_class and not actives and not unests:
            verdict = "skip_host"
        elif hits:
            verdict = "true_hit"
        elif generic_title and actives and not body_names_function:
            verdict = "true_hit"
            hits = [{**actives[0], "why": ["generic_title_has_active"]}]
        elif uhits:
            verdict = "gated"
        elif not actives and not unests:
            verdict = "no_artifact"
        else:
            verdict = "miss"
        preview = hits[0] if hits else (uhits[0] if uhits else {})
        out_rows.append(
            {
                "repository": repo,
                "issue_number": number,
                "issue_url": url,
                "title": title,
                "verdict": verdict,
                "failure_class": skip_class if verdict == "skip_host" else "",
                "body_chars": len(body),
                "match_source": "body" if body.strip() else ("obligation" if obligations.get(url) else "title"),
                "n_active_findings": len(actives),
                "n_true_hit_findings": len(hits),
                "n_gated_findings": len(uhits),
                "hit_pattern": preview.get("pattern") or "",
                "hit_root": (preview.get("root") or "")[:120],
                "hit_callee": (preview.get("callee") or "")[:80],
                "hit_path": preview.get("path") or "",
                "hit_why": " | ".join(preview.get("why") or ([preview.get("reason")] if preview.get("reason") else [])),
                "mvp_sources": " | ".join(sources),
            }
        )

    out_dir = (args.out_dir or (scan / "phase0")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "coverage_rescored.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    counts = Counter(r["verdict"] for r in out_rows)
    summary = out_dir / "coverage_rescored_summary.md"
    lines = [
        "# Rescored coverage (issue-body true-hit rule)",
        "",
        f"- issues: {len(out_rows)}",
        f"- bodies cached: {sum(1 for r in out_rows if int(r['body_chars']) > 0)}",
    ]
    for k, v in counts.most_common():
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## true_hit", ""]
    for r in out_rows:
        if r["verdict"] == "true_hit":
            lines.append(
                f"- {r['repository']}#{r['issue_number']} `{r['hit_pattern']}` `{r['hit_root']}` → `{r['hit_callee']}` ({r['hit_why']})"
            )
    lines += ["", "## skip_host", ""]
    for r in out_rows:
        if r["verdict"] == "skip_host":
            lines.append(f"- {r['repository']}#{r['issue_number']}: {r['failure_class']}")
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)
    print(dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

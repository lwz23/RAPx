#!/usr/bin/env python3
"""Ground truth for lwz23 148 issues: buggy source in the issue body, located in checkout.

A RAP finding is a true hit only when it names the reported function and/or
the reported source path (and unsafe API when the function name is weak).
Repo-name tokens and bare get_unchecked are not enough.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

UNSAFE_APIS = (
    "from_raw_parts_mut",
    "from_raw_parts",
    "from_utf8_unchecked_mut",
    "from_utf8_unchecked",
    "get_unchecked_mut",
    "get_unchecked",
    "transmute_copy",
    "copy_nonoverlapping",
    "read_unaligned",
    "assume_init",
    "from_raw",
    "from_ptr",
    "new_unchecked",
    "write_bytes",
    "set_len",
    "transmute",
)

WEAK_FNS = {
    "new",
    "get",
    "set",
    "add",
    "next",
    "write",
    "read",
    "drop",
    "from",
    "len",
    "ptr",
    "name",
    "path",
    "clone",
    "default",
    "fmt",
    "main",
    "into",
    "free",
    "init",
    "load",
    "store",
    "call",
    "run",
    "and",
    "or",
    "as_ptr",
    "as_mut",
}

SKIP_DIR_NAMES = {".git", "target", "node_modules", ".github"}

FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+\-]*)?[ \t]*\r?\n(.*?)```", re.S)
BLOB_RE = re.compile(
    r"github\.com/[^/\s]+/[^/\s]+/blob/([0-9a-fA-F]{7,40})/([\w./\-]+\.rs)(?:#L(\d+))?",
    re.I,
)
PATH_RE = re.compile(
    r"(?:(?<=[\s`\"'(])|^)((?:src|crates|packages|lib|core|pagefind(?:_[a-z0-9]+)?|wasm|runtime)[\w./\-]*\.rs)",
    re.I,
)
FN_RE = re.compile(
    r"\b(?:pub(?:\s*\([^)]*\))?\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)"
)
TITLE_IN_RE = re.compile(
    r"\b(?:in|of)\s+(?:the\s+)?(?:function\s+|method\s+|api\s+)?"
    r"(?:`|'|\"|:)?([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*)",
    re.I,
)
TYPE_METHOD_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*)::([A-Za-z_][A-Za-z0-9_]*)")


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def extract_fences(body: str) -> list[str]:
    if not body:
        return []
    return [block.strip("\n") for block in FENCE_RE.findall(body) if block.strip()]


def extract_blobs(text: str) -> list[dict[str, Any]]:
    rows = []
    for match in BLOB_RE.finditer(text or ""):
        rows.append(
            {
                "sha": match.group(1),
                "path": match.group(2).replace("\\", "/"),
                "line": int(match.group(3)) if match.group(3) else None,
            }
        )
    return rows


def extract_paths(text: str) -> list[str]:
    paths = [m.group(1).replace("\\", "/") for m in PATH_RE.finditer(text or "")]
    for blob in extract_blobs(text or ""):
        paths.append(blob["path"])
    return _unique(paths)


def extract_fns_from_text(text: str) -> list[str]:
    return _unique(FN_RE.findall(text or ""))


def extract_title_fns(title: str) -> list[str]:
    fns: list[str] = []
    for match in TYPE_METHOD_RE.finditer(title or ""):
        fns.append(match.group(2))
    for match in TITLE_IN_RE.finditer(title or ""):
        token = match.group(1)
        fns.append(token.split("::")[-1])
    for match in re.finditer(r"[`']([A-Za-z_][A-Za-z0-9_]*)[`']", title or ""):
        fns.append(match.group(1))
    return _unique(fns)


def extract_callees(text: str) -> list[str]:
    blob = (text or "").lower()
    found = [name for name in UNSAFE_APIS if name in blob]
    return _unique(found)


def extract_issue(title: str, body: str) -> dict[str, Any]:
    fences = extract_fences(body or "")
    fence_blob = "\n".join(fences)
    blobs = extract_blobs(body or "")
    paths = extract_paths(body or "")
    fns = extract_fns_from_text(fence_blob)
    if not fns:
        fns = extract_fns_from_text(body or "")
    title_fns = extract_title_fns(title or "")
    for name in title_fns:
        if name not in fns:
            fns.append(name)
    type_methods = [
        {"type": ty, "method": method}
        for ty, method in TYPE_METHOD_RE.findall((title or "") + "\n" + (body or "")[:4000])
    ]
    seen_tm: set[str] = set()
    uniq_tm = []
    for item in type_methods:
        key = item["type"] + "::" + item["method"]
        if key in seen_tm:
            continue
        seen_tm.add(key)
        uniq_tm.append(item)
    callees = extract_callees(fence_blob) or extract_callees(body or "")
    strong_fns = [name for name in fns if name.lower() not in WEAK_FNS]
    matchable = bool(strong_fns or paths or (fns and callees) or blobs or uniq_tm)
    return {
        "fences": fences,
        "blobs": blobs,
        "paths": paths,
        "fns": fns,
        "title_fns": title_fns,
        "type_methods": uniq_tm,
        "callees": callees,
        "strong_fns": strong_fns,
        "has_fence": bool(fences),
        "matchable": matchable,
    }


def _fn_line(text: str, name: str) -> int | None:
    match = re.search(r"\b(?:pub(?:\s*\([^)]*\))?\s+)?(?:unsafe\s+)?fn\s+" + re.escape(name) + r"\b", text)
    if not match:
        return None
    return text[: match.start()].count("\n") + 1


def locate_in_checkout(src: Path, extracted: dict[str, Any]) -> list[dict[str, Any]]:
    located: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(path: Path, fn: str | None, line: int | None, how: str) -> None:
        try:
            rel = str(path.resolve().relative_to(src.resolve())).replace("\\", "/")
        except (ValueError, OSError):
            rel = str(path)
        key = f"{rel}|{fn or ''}|{line or ''}"
        if key in seen:
            return
        seen.add(key)
        located.append(
            {
                "rel": rel,
                "fn": fn,
                "line": line,
                "exists": path.is_file(),
                "how": how,
            }
        )

    if not src.exists():
        return located

    for blob in extracted.get("blobs") or []:
        candidate = src / blob["path"]
        if candidate.is_file():
            fn = None
            line = blob.get("line")
            text = candidate.read_text(errors="replace")
            for name in extracted.get("fns") or []:
                found = _fn_line(text, name)
                if found:
                    fn = name
                    line = line or found
                    break
            add(candidate, fn, line, "blob")

    for rel in extracted.get("paths") or []:
        candidate = src / rel
        if not candidate.is_file():
            matches = [
                path
                for path in src.rglob(Path(rel).name)
                if path.is_file()
                and not any(part in SKIP_DIR_NAMES for part in path.parts)
                and path.as_posix().endswith(rel.replace("\\", "/"))
            ]
            candidate = matches[0] if matches else candidate
        if candidate.is_file():
            text = candidate.read_text(errors="replace")
            fn = None
            line = None
            for name in extracted.get("fns") or []:
                found = _fn_line(text, name)
                if found:
                    fn = name
                    line = found
                    break
            add(candidate, fn, line, "path")

    hint_files: list[Path] = []
    for row in located:
        path = src / row["rel"]
        if path.is_file():
            hint_files.append(path)
    search_fns = [
        name
        for name in (extracted.get("strong_fns") or extracted.get("fns") or [])
        if name.lower() not in WEAK_FNS
    ][:6]
    search_roots = hint_files or []
    rust_files: list[Path] | None = None

    def all_rust() -> list[Path]:
        nonlocal rust_files
        if rust_files is None:
            rust_files = [
                path
                for path in src.rglob("*.rs")
                if path.is_file() and not any(part in SKIP_DIR_NAMES for part in path.parts)
            ]
        return rust_files

    if not search_roots and search_fns:
        search_roots = all_rust()[:400]
    for name in search_fns:
        for path in search_roots:
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            line = _fn_line(text, name)
            if line:
                add(path, name, line, "fn-search")
                break
    for item in (extracted.get("type_methods") or [])[:8]:
        ty = str(item.get("type") or "")
        method = str(item.get("method") or "")
        if not ty or not method:
            continue
        for path in all_rust()[:800]:
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            if ty not in text:
                continue
            line = _fn_line(text, method)
            if line:
                add(path, method, line, "type-method")
                break
    return located


def _norm_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./").lower()


def _root_fn(root: str) -> str:
    return (root or "").split("::")[-1].split("<")[0].strip()


def match_finding(gt: dict[str, Any], finding: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return whether a RAP/CodeQL finding names the issue's reported source."""
    reasons: list[str] = []
    root = str(finding.get("root") or "")
    path = _norm_path(str(finding.get("path") or ""))
    callee = str(finding.get("callee") or "").lower()
    line = finding.get("line")
    root_fn = _root_fn(root)
    fns = [name for name in (gt.get("fns") or []) if name]
    strong_fns = [name for name in fns if name.lower() not in WEAK_FNS]
    paths = [_norm_path(p) for p in (gt.get("paths") or [])]
    for row in gt.get("located") or []:
        if row.get("rel"):
            paths.append(_norm_path(row["rel"]))
    paths = _unique(paths)
    callees = [c.lower() for c in (gt.get("callees") or [])]

    fn_hit = root_fn in fns if root_fn else False
    strong_fn_hit = root_fn in strong_fns if root_fn else False
    path_hit = False
    for reported in paths:
        if not reported:
            continue
        if path == reported or path.endswith("/" + reported) or path.endswith(reported):
            if Path(reported).name.replace(".rs", "") not in WEAK_FNS or "/" in reported.strip("/"):
                path_hit = True
                break
    callee_hit = any(c in callee for c in callees) if callees and callee else False
    line_hit = False
    if isinstance(line, int):
        for row in gt.get("located") or []:
            loc_line = row.get("line")
            loc_path = _norm_path(str(row.get("rel") or ""))
            if not loc_line or not loc_path:
                continue
            if abs(line - int(loc_line)) <= 25 and (path.endswith(loc_path) or loc_path.endswith(path.split("/")[-1])):
                line_hit = True
                break

    if strong_fn_hit:
        reasons.append("fn:" + root_fn)
    elif fn_hit and path_hit:
        reasons.append("weak-fn+path:" + root_fn)
    if path_hit:
        reasons.append("path")
    if callee_hit and (strong_fn_hit or path_hit):
        reasons.append("callee")
    if line_hit and (fn_hit or path_hit):
        reasons.append("line")

    ok = bool(
        strong_fn_hit
        or (fn_hit and path_hit)
        or (path_hit and callee_hit)
        or (line_hit and fn_hit)
    )
    return ok, reasons


def build_record(
    issue: dict[str, Any],
    body_row: dict[str, Any],
    src_root: Path,
) -> dict[str, Any]:
    title = issue.get("title") or body_row.get("title") or ""
    body = body_row.get("body") or ""
    extracted = extract_issue(title, body)
    repo = issue["repository"]
    src = src_root / repo
    located = locate_in_checkout(src, extracted)
    extracted["located"] = located
    extracted["located_n"] = sum(1 for row in located if row.get("exists"))
    return {
        "repository": repo,
        "issue_number": str(issue.get("issue_number")),
        "issue_url": issue.get("issue_url"),
        "title": title,
        "src_exists": src.is_dir(),
        **extracted,
    }


def _receipt_is_analyzed(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    schema = data.get("schema_version") or ""
    if schema == "rap-unit-skip-v1" or data.get("status") == "skipped":
        return False
    return schema.startswith("rap-unit")


def package_scan_status(work_dir: Path, package: str) -> str:
    """Whether RAP actually analyzed `package` under a scanner work dir.

    `scanned` requires a real analysis receipt or a non-empty candidates.jsonl.
    Skip receipts (`rap-unit-skip-v1`, including isolate `manifest_outside_project`)
    do not count. An empty mvp/ folder or control-only logs means the bug crate
    was not analyzed.
    """
    receipts = list((work_dir / "receipts" / package).glob("*.json"))
    cand = work_dir / "mvp" / package / "candidates.jsonl"
    has_cand = cand.exists() and cand.stat().st_size > 0
    if any(_receipt_is_analyzed(path) for path in receipts) or has_cand:
        return "scanned"
    if receipts:
        return "not_scanned"
    scan_logs = list((work_dir / "logs").glob(package + ".scan*.log"))
    if scan_logs:
        return "scan_failed"
    control_logs = list((work_dir / "logs").glob(package + ".control*.log"))
    if control_logs:
        return "control_failed"
    mvp_dir = work_dir / "mvp" / package
    if mvp_dir.is_dir():
        return "not_scanned"
    return "absent"


def score_actives(gt: dict[str, Any], actives: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits = []
    for row in actives:
        ok, reasons = match_finding(gt, row)
        if ok:
            item = dict(row)
            item["why"] = reasons
            hits.append(item)
    return hits


def load_bodies(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row.get("issue_url") or "")] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    roster = args.repo_root / "docs/unsoundaudit-v3/lwz23-valid-issue-roster-v1"
    issues = list(csv.DictReader((roster / "issues.csv").open()))
    bodies = load_bodies(roster / "issue_bodies.jsonl")
    out_dir = args.out_dir or roster
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for issue in issues:
        records.append(build_record(issue, bodies.get(issue["issue_url"], {}), args.src_root))
    jsonl_path = out_dir / "ground_truth.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = len(records)
    summary = {
        "issues": n,
        "has_fence": sum(1 for r in records if r["has_fence"]),
        "has_fn": sum(1 for r in records if r["fns"]),
        "has_strong_fn": sum(1 for r in records if r["strong_fns"]),
        "has_path": sum(1 for r in records if r["paths"]),
        "has_callee": sum(1 for r in records if r["callees"]),
        "src_exists": sum(1 for r in records if r["src_exists"]),
        "located": sum(1 for r in records if r["located_n"] > 0),
        "matchable": sum(1 for r in records if r["matchable"] or r["located_n"] > 0),
        "unmatchable": [
            f"{r['repository']}#{r['issue_number']} {r['title'][:80]}"
            for r in records
            if not r["matchable"] and r["located_n"] == 0
        ],
    }
    (out_dir / "ground_truth_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    md = [
        "# lwz23 ground truth (issue-body source)",
        "",
        f"- issues: {summary['issues']}",
        f"- fenced code: {summary['has_fence']}",
        f"- extracted fn: {summary['has_fn']} (strong names {summary['has_strong_fn']})",
        f"- source path: {summary['has_path']}",
        f"- unsafe API: {summary['has_callee']}",
        f"- checkout present: {summary['src_exists']}",
        f"- located in checkout: {summary['located']}",
        f"- matchable later: {summary['matchable']}",
        "",
        "A later RAP finding counts as the reported bug only via `match_finding`:",
        "function name from the issue (strong), or weak function plus path/callee, or path plus callee.",
        "",
    ]
    if summary["unmatchable"]:
        md += ["## no extractable source", ""]
        md.extend(f"- {row}" for row in summary["unmatchable"])
    (out_dir / "ground_truth_summary.md").write_text("\n".join(md) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "unmatchable"}, ensure_ascii=False))
    print("wrote", jsonl_path)
    return 0


def crate_has_no_std(manifest: Path) -> bool:
    """True if the crate entrypoint enables no_std (host cannot use libstd)."""
    for name in ("src/lib.rs", "src/main.rs", "lib.rs", "main.rs"):
        path = manifest.parent / name
        if not path.exists():
            continue
        try:
            head = path.read_text(errors="replace")[:2500]
        except OSError:
            continue
        if "#![no_std]" in head:
            return True
    return False


def classify_located_crate(
    *,
    status: str,
    no_std: bool,
    located_rel: str = "",
    windows_only: bool = False,
    missing_path_dep: bool = False,
) -> tuple[str, str]:
    """Tag one GT-located crate. skip_analysis only if THAT crate was scanned.

    Does not look at sibling n_active. Does not treat the substring "native"
    in a source path as a C library.
    """
    rel = (located_rel or "").replace(chr(92), "/").lower()
    if windows_only or "d3d12" in rel:
        return "skip_host", "windows_cfg"
    if no_std:
        return "skip_host", "missing_std_or_core"
    if missing_path_dep:
        return "skip_repo", "missing_path_dep"
    if status == "scanned":
        return "skip_analysis", "located_crate_scanned"
    if status in {"control_failed", "scan_failed"}:
        return "skip_repo", "last_run_" + status
    if status in {"absent", "not_scanned", ""}:
        return "lift_now", "located_crate_never_built"
    return "skip_repo", "unclassified:" + status


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""CodeQL custom P1–P6 recall campaign over the frozen 148/125 universe.

SSD-only work root. Resume from journal. Score findings against issue bodies.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import rescore_lwz23_coverage as rescore  # noqa: E402

QUERIES = [
    ("P1PublicParam.ql", "pattern1", "unsoundaudit/p1-public-param"),
    ("P2PublicField.ql", "pattern2", "unsoundaudit/p2-public-field"),
    ("P3InternalInvalid.ql", "pattern3", "unsoundaudit/p3-internal-invalid"),
    ("P4InternalBounds.ql", "pattern4", "unsoundaudit/p4-internal-bounds"),
    ("P5GenericNonEmpty.ql", "pattern5", "unsoundaudit/p5-generic-nonempty"),
    ("P6ExternalBehavior.ql", "pattern6", "unsoundaudit/p6-external-behavior"),
]
RULE_TO_PATTERN = {rule: pattern for _, pattern, rule in QUERIES}

SKIP_SCAN = {
    "BrowserWorks/waterfox": "browser-monorepo / not a practical Cargo crate scan",
    "rustsec/advisory-db": "advisory-db is not a software crate; scan mapped crates",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def write_heartbeat(path: Path, **fields: Any) -> None:
    path.write_text(json.dumps({"ts": now(), **fields}, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def latest_by_key(path: Path, key: str) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get(key):
            latest[str(rec[key])] = rec
    return latest


def run_timeout(cmd: list[str], log_path: Path, timeout_s: int, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n# {now()} {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            return int(proc.wait(timeout=timeout_s))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, 15)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, 9)
                except (ProcessLookupError, PermissionError):
                    pass
            log.write(f"\n# TIMEOUT {timeout_s}s\n")
            return 124


def classify_extract_fail(log_path: Path) -> str:
    text = log_path.read_text(encoding="utf-8", errors="replace")[-8000:] if log_path.exists() else ""
    low = text.lower()
    if re.search(r"windows only|can only be built for the `windows`|requires windows", low):
        return "windows-pe / host-impossible"
    if "json-target" in low or "bare metal" in low:
        return "json-target / bare-metal"
    if "linker" in low and ("not found" in low or "cannot find" in low):
        return "missing native lib / linker"
    return "extract_fail"


def parse_sarif(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return []
    rows: list[dict[str, Any]] = []
    for run in data.get("runs") or []:
        for result in run.get("results") or []:
            rule = str(result.get("ruleId") or "")
            pattern = RULE_TO_PATTERN.get(rule, "")
            message = str((result.get("message") or {}).get("text") or "")
            path_str = ""
            locs = result.get("locations") or []
            if locs:
                phys = (locs[0] or {}).get("physicalLocation") or {}
                path_str = str((phys.get("artifactLocation") or {}).get("uri") or "")
            roots = [message]
            for flow in result.get("codeFlows") or []:
                for thread in flow.get("threadFlows") or []:
                    locs2 = thread.get("locations") or []
                    if locs2:
                        msg = str(((locs2[0].get("location") or {}).get("message") or {}).get("text") or "")
                        if msg:
                            roots.append(msg)
            rows.append(
                {
                    "pattern": pattern,
                    "root": " | ".join(roots)[:400],
                    "callee": message[:200],
                    "path": path_str.replace("\\", "/"),
                    "package": "",
                    "confidence": "heuristic",
                }
            )
    return rows


def parse_csv_findings(path: Path, pattern: str) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            blob = " ".join(str(v) for v in row.values() if v)
            file_path = ""
            match = re.search(r"file=(\S+)", blob)
            if match:
                file_path = match.group(1).rstrip(".,;")
            if not file_path:
                for value in row.values():
                    text = str(value or "")
                    if ".rs" in text:
                        file_path = text
                        break
            rows.append(
                {
                    "pattern": pattern,
                    "root": blob[:400],
                    "callee": blob[:200],
                    "path": file_path.replace("\\", "/"),
                    "package": "",
                    "confidence": "heuristic",
                }
            )
    return rows


def mapped_software_repo(issue: dict[str, str]) -> str:
    repo = issue["repository"]
    if repo != "rustsec/advisory-db":
        return repo
    keys = issue.get("canonical_instance_keys") or ""
    match = re.search(r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)#", keys)
    return match.group(1) if match else repo


def score_issue(
    issue: dict[str, str],
    body: str,
    obligation: str,
    precomputed: dict[str, Any] | None,
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    needles = rescore.merge_needles(issue["title"], obligation, body, precomputed)
    hits = rescore.match_active(needles, findings)
    key = (issue["repository"], str(issue["issue_number"]))
    skip_class = rescore.SKIP_HOST_CLASS.get(key) or SKIP_SCAN.get(issue["repository"])
    body_names = bool(needles.get("fns") or needles["paths"] or needles["callees"])
    strong = [
        q
        for q in (needles["qualified"] + needles.get("fns", []))
        if q.split("::")[-1].lower() not in rescore.GENERIC_TITLE_TOKENS
    ]
    generic_title = (not body.strip()) and not needles["paths"] and not needles["callees"] and not strong
    if skip_class and not findings:
        verdict = "skip_host"
    elif hits:
        verdict = "true_hit"
    elif generic_title and findings and not body_names:
        verdict = "true_hit"
        hits = [{**findings[0], "why": ["generic_title_has_active"]}]
    elif findings:
        verdict = "miss"
    else:
        verdict = "no_artifact"
    preview = hits[0] if hits else {}
    return {
        "repository": issue["repository"],
        "issue_number": str(issue["issue_number"]),
        "issue_url": issue.get("issue_url", ""),
        "title": issue["title"],
        "verdict": verdict,
        "failure_class": skip_class if verdict == "skip_host" else "",
        "body_chars": len(body),
        "n_active_findings": len(findings),
        "n_true_hit_findings": len(hits),
        "hit_pattern": preview.get("pattern") or "",
        "hit_root": str(preview.get("root") or "")[:120],
        "hit_callee": str(preview.get("callee") or "")[:80],
        "hit_path": preview.get("path") or "",
        "hit_why": " | ".join(preview.get("why") or []),
        "engine": "codeql",
    }


def load_priority_repos(mac_coverage: Path | None, all_repos: list[str]) -> list[str]:
    if mac_coverage is None or not mac_coverage.exists():
        return all_repos
    buckets = {"no_artifact": [], "miss": [], "other": []}
    seen = set()
    with mac_coverage.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            repo = row["repository"]
            if repo in seen or repo not in all_repos:
                continue
            seen.add(repo)
            verdict = row.get("verdict") or "other"
            if verdict in buckets:
                buckets[verdict].append(repo)
            else:
                buckets["other"].append(repo)
    ordered = buckets["no_artifact"] + buckets["miss"] + buckets["other"]
    for repo in all_repos:
        if repo not in seen:
            ordered.append(repo)
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codeql", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--mac-coverage", type=Path)
    parser.add_argument("--extract-timeout", type=int, default=900)
    parser.add_argument("--query-timeout", type=int, default=360)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-repos", type=Path)
    args = parser.parse_args()

    rap = args.repo_root.resolve()
    src_root = args.src_root.resolve()
    work = args.work_root.resolve()
    pack = rap / "codeql/unsoundaudit-poc"
    roster = rap / "docs/unsoundaudit-v3/lwz23-valid-issue-roster-v1"
    codeql = args.codeql.resolve(strict=True)
    work.mkdir(parents=True, exist_ok=True)
    journal = work / "scan_journal.jsonl"
    heartbeat = work / "heartbeat.txt"
    ledger_path = work / "coverage_rescored.csv"
    summary_path = work / "coverage_rescored_summary.md"

    issues = list(csv.DictReader((roster / "issues.csv").open()))
    if len(issues) != 148:
        raise SystemExit(f"expected 148 issues, got {len(issues)}")
    repos = [row["repository"] for row in csv.DictReader((roster / "repositories.csv").open())]
    if args.only_repos:
        wanted = {
            line.strip()
            for line in args.only_repos.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        repos = [name for name in repos if name in wanted]
    repos = load_priority_repos(args.mac_coverage, repos)
    if args.limit:
        repos = repos[: args.limit]

    obligations: dict[str, str] = {}
    for instance in rescore.load_jsonl(
        rap / "docs/unsoundaudit-v3/lwz23-valid-issue-roster-inputs/bug_instances.jsonl"
    ):
        url = str(instance.get("issue_url") or "")
        text = instance.get("repair_obligation_zh") or ""
        if url and text:
            obligations[url] = (obligations.get(url, "") + " " + text).strip()
    bodies = rescore.load_bodies(roster / "issue_bodies.jsonl")
    precomputed = rescore.load_precomputed_needles(roster / "issue_needles.json")

    issues_by_scan_repo: dict[str, list[dict[str, str]]] = defaultdict(list)
    for issue in issues:
        issues_by_scan_repo[mapped_software_repo(issue)].append(issue)

    env = os.environ.copy()
    env["CARGO_HOME"] = str(work / "cargo-home")
    env["CARGO_TARGET_DIR"] = str(work / "cargo-target")
    env["TMPDIR"] = str(work / "tmp")
    env["TMP"] = env["TMPDIR"]
    env["TEMP"] = env["TMPDIR"]
    env["XDG_CACHE_HOME"] = str(work / "cache")
    env["CODEQL_ENABLE_EXPERIMENTAL_FEATURES"] = "true"
    (work / "tmp").mkdir(parents=True, exist_ok=True)
    (work / "cargo-home").mkdir(parents=True, exist_ok=True)
    (work / "cache").mkdir(parents=True, exist_ok=True)

    print(f"+ pack install {pack}", flush=True)
    subprocess.run([str(codeql), "pack", "install", str(pack)], check=False, env=env)

    done = latest_by_key(journal, "repo")
    findings_by_repo: dict[str, list[dict[str, Any]]] = {}
    status_by_repo: dict[str, str] = {}

    for idx, repo in enumerate(repos, 1):
        previous = done.get(repo)
        if previous and previous.get("status") not in {None, "running"}:
            status_by_repo[repo] = str(previous.get("status"))
            findings_by_repo[repo] = previous.get("findings") or []
            print(f"skip already done {repo} {previous.get('status')}", flush=True)
            continue
        write_heartbeat(heartbeat, pid=os.getpid(), repo=repo, index=idx, total=len(repos), phase="start")
        rec: dict[str, Any] = {"ts": now(), "repo": repo, "index": idx, "status": "running"}
        skip_reason = SKIP_SCAN.get(repo)
        src = src_root / repo
        if skip_reason:
            rec["status"] = "skip_host"
            rec["failure_class"] = skip_reason
            rec["findings"] = []
            append_jsonl(journal, rec)
            status_by_repo[repo] = "skip_host"
            findings_by_repo[repo] = []
            print(f"[{idx}/{len(repos)}] {repo} skip_host {skip_reason}", flush=True)
            continue
        if not src.exists():
            rec["status"] = "missing_source"
            rec["findings"] = []
            append_jsonl(journal, rec)
            status_by_repo[repo] = "missing_source"
            findings_by_repo[repo] = []
            print(f"[{idx}/{len(repos)}] {repo} missing_source", flush=True)
            continue
        cargo_tomls = list(src.rglob("Cargo.toml"))
        if not cargo_tomls:
            rec["status"] = "skip_no_cargo"
            rec["findings"] = []
            append_jsonl(journal, rec)
            status_by_repo[repo] = "skip_no_cargo"
            findings_by_repo[repo] = []
            print(f"[{idx}/{len(repos)}] {repo} skip_no_cargo", flush=True)
            continue

        slug = repo.replace("/", "__")
        db = work / "db" / slug
        logs = work / "logs" / slug
        logs.mkdir(parents=True, exist_ok=True)
        db.parent.mkdir(parents=True, exist_ok=True)
        (work / "extractor-target" / slug).mkdir(parents=True, exist_ok=True)
        if db.exists():
            shutil.rmtree(db, ignore_errors=True)
        t0 = time.time()
        extract_log = logs / "extract.log"
        rc = run_timeout(
            [
                str(codeql),
                "database",
                "create",
                str(db),
                "--language=rust",
                "--build-mode=none",
                "--source-root",
                str(src),
                "--overwrite",
                f"--extractor-option=rust.cargo_target_dir={work / 'extractor-target' / slug}",
            ],
            extract_log,
            args.extract_timeout,
            env,
        )
        rec["extract_s"] = round(time.time() - t0, 1)
        rec["extract_rc"] = rc
        extracted = (db / "db-rust").is_dir() and rc == 0
        findings: list[dict[str, Any]] = []
        if rc == 124:
            rec["status"] = "extract_timeout"
        elif not extracted:
            rec["status"] = "extract_fail"
            rec["failure_class"] = classify_extract_fail(extract_log)
        else:
            rec["status"] = "extracted"
        if extracted:
            write_heartbeat(heartbeat, pid=os.getpid(), repo=repo, phase="query", index=idx, total=len(repos))
            sarif = logs / "results.sarif"
            q0 = time.time()
            qrc = run_timeout(
                [
                    str(codeql),
                    "database",
                    "analyze",
                    str(db),
                    str(pack),
                    "--format=sarifv2.1.0",
                    f"--output={sarif}",
                    "--threads=0",
                    "--ram=8192",
                    "--sarif-add-query-help=never",
                ],
                logs / "analyze.log",
                args.query_timeout * 6,
                env,
            )
            rec["analyze_s"] = round(time.time() - q0, 1)
            rec["analyze_rc"] = qrc
            if qrc == 0:
                findings = parse_sarif(sarif)
                rec["status"] = "scanned"
            else:
                rec["status"] = "query_fail"
                for query_name, pattern, _rule in QUERIES:
                    bqrs = logs / f"{pattern}.bqrs"
                    csv_path = logs / f"{pattern}.csv"
                    qrc2 = run_timeout(
                        [
                            str(codeql),
                            "query",
                            "run",
                            str(pack / query_name),
                            "--database",
                            str(db),
                            "--output",
                            str(bqrs),
                            "--threads=0",
                            "--ram=8192",
                        ],
                        logs / f"{pattern}.log",
                        args.query_timeout,
                        env,
                    )
                    if qrc2 == 0 and bqrs.is_file():
                        subprocess.run(
                            [
                                str(codeql),
                                "bqrs",
                                "decode",
                                str(bqrs),
                                "--format=csv",
                                f"--output={csv_path}",
                            ],
                            check=False,
                            env=env,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        findings.extend(parse_csv_findings(csv_path, pattern))
                if findings:
                    rec["status"] = "scanned"
        rec["n_findings"] = len(findings)
        rec["findings"] = findings[:200]
        append_jsonl(journal, rec)
        status_by_repo[repo] = rec["status"]
        findings_by_repo[repo] = findings
        print(
            f"[{idx}/{len(repos)}] {repo} {rec['status']} extract={rec.get('extract_s')}s findings={len(findings)}",
            flush=True,
        )
        shutil.rmtree(db, ignore_errors=True)
        shutil.rmtree(work / "extractor-target" / slug, ignore_errors=True)
        write_heartbeat(
            heartbeat,
            pid=os.getpid(),
            repo=repo,
            phase="done",
            status=rec["status"],
            index=idx,
            total=len(repos),
        )

    # Rebuild findings from full journal (resume + this run).
    for rec in latest_by_key(journal, "repo").values():
        findings_by_repo[rec["repo"]] = rec.get("findings") or []
        status_by_repo[rec["repo"]] = str(rec.get("status") or "")

    out_rows = []
    for issue in issues:
        scan_repo = mapped_software_repo(issue)
        findings = findings_by_repo.get(scan_repo, [])
        body = str((bodies.get(issue["issue_url"]) or {}).get("body") or "")
        row = score_issue(
            issue,
            body,
            obligations.get(issue["issue_url"], ""),
            precomputed.get(issue["issue_url"]),
            findings,
        )
        row["scan_repo"] = scan_repo
        row["scan_status"] = status_by_repo.get(scan_repo, "not_scanned")
        if row["verdict"] == "no_artifact" and status_by_repo.get(scan_repo) == "skip_host":
            row["verdict"] = "skip_host"
            row["failure_class"] = SKIP_SCAN.get(scan_repo) or row.get("failure_class") or "skip_host"
        out_rows.append(row)

    with ledger_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    counts = Counter(r["verdict"] for r in out_rows)
    repo_counts = Counter(status_by_repo.values())
    lines = [
        "# CodeQL lwz23 148/125 recall",
        "",
        f"- issues: {len(out_rows)}",
        f"- bodies cached: {sum(1 for r in out_rows if int(r['body_chars']) > 0)}",
        f"- mac title-only RAP baseline: true_hit 32 / 148",
        "",
        "## issue verdicts",
        "",
    ]
    for key, value in counts.most_common():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## repo status", ""]
    for key, value in repo_counts.most_common():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## true_hit", ""]
    for row in out_rows:
        if row["verdict"] == "true_hit":
            lines.append(
                f"- {row['repository']}#{row['issue_number']} `{row['hit_pattern']}` `{row['hit_root']}` ({row['hit_why']})"
            )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(ledger_path)
    print(dict(counts))
    print(dict(repo_counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

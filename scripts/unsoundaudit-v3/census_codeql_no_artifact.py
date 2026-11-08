#!/usr/bin/env python3
"""Sample CodeQL extraction + P1–P6 queries on RAP no_artifact repos.

Estimates how much frozen-ledger recall could rise if RAP-failed crates
are scanned with CodeQL instead. Writes journal/heartbeat under RAPDATA.
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
    ("P1PublicParam.ql", "pattern1"),
    ("P2PublicField.ql", "pattern2"),
    ("P3InternalInvalid.ql", "pattern3"),
    ("P4InternalBounds.ql", "pattern4"),
    ("P5GenericNonEmpty.ql", "pattern5"),
    ("P6ExternalBehavior.ql", "pattern6"),
]

# Frozen 24-repo sample: 8 small / 8 medium / 8 large, plus multi-issue repos.
SAMPLE_REPOS = [
    # small (<50 .rs)
    "changlan/kytan",
    "m4b/dryad",
    "Noeda/rllama",
    "asg017/sqlite-loadable-rs",
    "RedisJSON/RedisJSON",
    "rust-bio/rust-htslib",
    "pipedown/noise",
    "kahing/catfs",
    # medium (50-200)
    "Pagefind/pagefind",
    "archseer/enigma",
    "DioxusLabs/blitz",
    "NVSL/Corundum",
    "NetSys/NetBricks",
    "MegEngine/MegFlow",
    "georust/gdal",
    "microsoft/aici",
    # large (>200)
    "Indra-db/Flecs-Rust",
    "zhihu/rucene",
    "CosmWasm/cosmwasm",
    "posit-dev/ark",
    "tursodatabase/libsql",
    "tensorbase/tensorbase",
    "Phala-Network/phala-blockchain",
    "arceos-org/arceos",
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def write_heartbeat(path: Path, **fields: Any) -> None:
    payload = {"ts": now(), **fields}
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n")


def run_timeout(
    cmd: list[str],
    log_path: Path,
    timeout_s: int,
    env: dict[str, str],
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
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


def parse_findings(csv_path: Path, pattern: str) -> list[dict[str, Any]]:
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return []
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            blob = " ".join(str(v) for v in row.values() if v)
            path = ""
            match = re.search(r"file=(\S+)", blob)
            if match:
                path = match.group(1).rstrip(".,;")
            rows.append(
                {
                    "pattern": pattern,
                    "root": str(row.get("source") or row.get("col0") or ""),
                    "callee": str(row.get("sink") or ""),
                    "path": path,
                    "blob": blob[:500],
                    "confidence": "heuristic",
                    "package": "",
                }
            )
    return rows


def score_issue(
    issue: dict[str, str],
    obligation: str,
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    needles = rescore.needles(issue["title"], obligation)
    hits = rescore.match_active(needles, findings)
    strong = [
        q
        for q in needles["qualified"]
        if q.split("::")[-1] not in rescore.GENERIC_TITLE_TOKENS
    ]
    generic_title = not needles["paths"] and not needles["callees"] and not strong
    if hits:
        verdict = "strict_hit"
        why = hits[0].get("why") or []
    elif generic_title and findings:
        verdict = "generic_hit"
        why = ["generic_title_has_active"]
        hits = [{**findings[0], "why": why}]
    elif findings:
        verdict = "extract_miss"
        why = []
    else:
        verdict = "no_finding"
        why = []
    preview = hits[0] if hits else {}
    return {
        "repository": issue["repository"],
        "issue_number": issue["issue_number"],
        "issue_url": issue.get("issue_url", ""),
        "title": issue["title"],
        "verdict": verdict,
        "n_findings": len(findings),
        "n_hits": len(hits),
        "hit_pattern": preview.get("pattern") or "",
        "hit_root": str(preview.get("root") or "")[:120],
        "hit_callee": str(preview.get("callee") or "")[:80],
        "hit_path": preview.get("path") or "",
        "hit_why": " | ".join(why),
    }


def load_sample_issues(coverage: Path, repos: list[str]) -> list[dict[str, str]]:
    wanted = set(repos)
    out = []
    with coverage.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["verdict"] == "no_artifact" and row["repository"] in wanted:
                out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--codeql",
        default="/Users/liaowenzhao/Documents/RAP重构/lwz23-valid-issue-scan/codeql-poc/codeql/codeql",
    )
    parser.add_argument(
        "--work-root",
        default="/Volumes/RAPDATA/codeql-census",
    )
    parser.add_argument(
        "--src-root",
        default="/Users/liaowenzhao/Documents/RAP重构/lwz23-valid-issue-src",
    )
    parser.add_argument(
        "--pack",
        default=str(
            Path("/Users/liaowenzhao/Documents/RAP重构/RAPx-unsoundaudit-v2/codeql/unsoundaudit-poc")
        ),
    )
    parser.add_argument("--extract-timeout", type=int, default=300)
    parser.add_argument("--query-timeout", type=int, default=240)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--repos",
        nargs="*",
        default=None,
        help="Override the frozen 24-repo sample with an explicit list.",
    )
    args = parser.parse_args()

    work = Path(args.work_root)
    work.mkdir(parents=True, exist_ok=True)
    src_root = Path(args.src_root)
    pack = Path(args.pack)
    codeql = Path(args.codeql)
    coverage = Path(
        "/Users/liaowenzhao/Documents/RAP重构/lwz23-valid-issue-scan/phase0/coverage_rescored.csv"
    )
    rap = Path("/Users/liaowenzhao/Documents/RAP重构/RAPx-unsoundaudit-v2")
    journal = work / "census_journal.jsonl"
    heartbeat = work / "heartbeat.txt"
    report_json = work / "census_report.json"
    report_md = work / "census_report.md"
    sample_path = work / "sample.json"

    env = os.environ.copy()
    env["CARGO_HOME"] = "/Volumes/RAPDATA/cargo-home"
    env["CARGO_TARGET_DIR"] = str(work / "cargo-target")
    tmp = work / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(tmp)
    env["TMP"] = str(tmp)
    env["TEMP"] = str(tmp)
    env["XDG_CACHE_HOME"] = str(work / "cache")
    env.setdefault("CODEQL_ENABLE_EXPERIMENTAL_FEATURES", "true")

    repos = list(args.repos) if args.repos else list(SAMPLE_REPOS)
    if args.limit:
        repos = repos[: args.limit]
    issues = load_sample_issues(coverage, repos)
    sample_path.write_text(
        json.dumps(
            {
                "n_repos": len(repos),
                "n_issues": len(issues),
                "repos": repos,
                "issues": [
                    {
                        "repository": i["repository"],
                        "issue_number": i["issue_number"],
                        "title": i["title"],
                    }
                    for i in issues
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    obligations: dict[str, str] = {}
    for instance in rescore.load_jsonl(
        rap / "docs/unsoundaudit-v3/lwz23-valid-issue-roster-inputs/bug_instances.jsonl"
    ):
        url = str(instance.get("issue_url") or "")
        text = instance.get("repair_obligation_zh") or ""
        if url and text:
            obligations[url] = (obligations.get(url, "") + " " + text).strip()

    print(f"+ pack install {pack}", flush=True)
    subprocess.run([str(codeql), "pack", "install", str(pack)], check=False, env=env)

    done_repos = set()
    if journal.exists():
        for line in journal.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("repo") and rec.get("status") not in {"running"}:
                done_repos.add(rec["repo"])

    issues_by_repo: dict[str, list[dict[str, str]]] = defaultdict(list)
    for issue in issues:
        issues_by_repo[issue["repository"]].append(issue)

    for idx, repo in enumerate(repos, 1):
        if repo in done_repos:
            print(f"skip already done {repo}", flush=True)
            continue
        write_heartbeat(
            heartbeat,
            repo=repo,
            index=idx,
            total=len(repos),
            phase="start",
        )
        src = src_root / repo
        slug = repo.replace("/", "__")
        db = work / "db" / slug
        logs = work / "logs" / slug
        logs.mkdir(parents=True, exist_ok=True)
        rec: dict[str, Any] = {
            "ts": now(),
            "repo": repo,
            "index": idx,
            "n_rs": None,
            "status": "running",
        }
        if not (src / "Cargo.toml").is_file():
            rec["status"] = "skip_no_cargo"
            journal.open("a").write(json.dumps(rec, ensure_ascii=False) + "\n")
            continue
        rec["n_rs"] = sum(1 for _ in src.rglob("*.rs"))
        db.parent.mkdir(parents=True, exist_ok=True)
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
                f"--extractor-option=rust.cargo_target_dir={work / 'extractor-target'}",
            ],
            extract_log,
            args.extract_timeout,
            env,
        )
        rec["extract_s"] = round(time.time() - t0, 1)
        rec["extract_rc"] = rc
        extracted = (db / "db-rust").is_dir() and rc == 0
        if rc == 124:
            rec["status"] = "extract_timeout"
        elif not extracted:
            rec["status"] = "extract_fail"
        else:
            rec["status"] = "extracted"
        findings: list[dict[str, Any]] = []
        if extracted:
            write_heartbeat(heartbeat, repo=repo, phase="query", index=idx, total=len(repos))
            query_counts = {}
            for query_name, pattern in QUERIES:
                qlog = logs / f"{pattern}.log"
                bqrs = logs / f"{pattern}.bqrs"
                csv_path = logs / f"{pattern}.csv"
                q0 = time.time()
                qrc = run_timeout(
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
                    qlog,
                    args.query_timeout,
                    env,
                )
                rec[f"{pattern}_s"] = round(time.time() - q0, 1)
                rec[f"{pattern}_rc"] = qrc
                if qrc == 0 and bqrs.is_file():
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
                    found = parse_findings(csv_path, pattern)
                    query_counts[pattern] = len(found)
                    findings.extend(found)
                else:
                    query_counts[pattern] = None
            rec["query_counts"] = query_counts
            rec["n_findings"] = len(findings)
            rec["status"] = "scanned"
        rec["issue_scores"] = [
            score_issue(issue, obligations.get(issue.get("issue_url", ""), ""), findings)
            for issue in issues_by_repo.get(repo, [])
        ]
        journal.open("a").write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(
            f"[{idx}/{len(repos)}] {repo} {rec['status']} extract={rec.get('extract_s')}s findings={rec.get('n_findings', 0)}",
            flush=True,
        )
        shutil.rmtree(db, ignore_errors=True)
        write_heartbeat(heartbeat, repo=repo, phase="done", status=rec["status"], index=idx)

    records = []
    if journal.exists():
        for line in journal.read_text().splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    latest: dict[str, dict[str, Any]] = {}
    for rec in records:
        latest[rec["repo"]] = rec
    records = [latest[r] for r in repos if r in latest]

    extract_ok = [r for r in records if r.get("status") in {"extracted", "scanned"}]
    scanned = [r for r in records if r.get("status") == "scanned"]
    issue_scores = [s for r in records for s in r.get("issue_scores") or []]
    verdicts = Counter(s["verdict"] for s in issue_scores)
    n_sample_issues = len(issue_scores)
    n_strict = verdicts.get("strict_hit", 0)
    n_generic = verdicts.get("generic_hit", 0)
    n_frozen = n_strict + n_generic
    extract_rate = (len(extract_ok) / len(records)) if records else 0.0
    frozen_rate = (n_frozen / n_sample_issues) if n_sample_issues else 0.0
    strict_rate = (n_strict / n_sample_issues) if n_sample_issues else 0.0

    # Universe: 76 no_artifact issues; 60 of them sit in cargo-present unique repos.
    cargo_universe_issues = 60
    no_artifact_all = 76
    current_true = 32
    denom = 148
    est_cargo_frozen = round(frozen_rate * cargo_universe_issues, 1)
    est_cargo_strict = round(strict_rate * cargo_universe_issues, 1)
    # Non-cargo no_artifact stay no_artifact in this estimate.
    report = {
        "ts": now(),
        "n_sample_repos": len(records),
        "n_sample_issues": n_sample_issues,
        "extract_ok_repos": len(extract_ok),
        "extract_rate": extract_rate,
        "status_repos": dict(Counter(r.get("status") for r in records)),
        "issue_verdicts": dict(verdicts),
        "strict_hits": n_strict,
        "generic_hits": n_generic,
        "frozen_hits": n_frozen,
        "strict_rate": strict_rate,
        "frozen_rate": frozen_rate,
        "universe": {
            "denominator": denom,
            "current_true_hit": current_true,
            "no_artifact": no_artifact_all,
            "no_artifact_with_cargo": cargo_universe_issues,
        },
        "extrapolated_added_frozen": est_cargo_frozen,
        "extrapolated_added_strict": est_cargo_strict,
        "extrapolated_true_hit_frozen": current_true + est_cargo_frozen,
        "extrapolated_true_hit_strict": current_true + est_cargo_strict,
        "records": [
            {
                "repo": r.get("repo"),
                "status": r.get("status"),
                "extract_s": r.get("extract_s"),
                "n_findings": r.get("n_findings"),
                "issues": r.get("issue_scores"),
            }
            for r in records
        ],
    }
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# CodeQL no_artifact sampling census",
        "",
        f"- time: {now()}",
        f"- sample repos: {len(records)} / planned {len(repos)}",
        f"- sample issues: {n_sample_issues}",
        f"- extract ok: {len(extract_ok)} ({extract_rate:.0%})",
        f"- repo status: {dict(Counter(r.get('status') for r in records))}",
        f"- issue verdicts: {dict(verdicts)}",
        "",
        "## Recall estimate (denominator 148, current true_hit 32)",
        "",
        "Frozen rule = RAP ledger rule (needle match **or** generic title + any finding).",
        "Strict rule = needle match only (path / API / callee).",
        "",
        f"- sample frozen hits: {n_frozen}/{n_sample_issues} ({frozen_rate:.0%})",
        f"- sample strict hits: {n_strict}/{n_sample_issues} ({strict_rate:.0%})",
        f"- extrapolate to 60 cargo no_artifact issues: frozen +{est_cargo_frozen}, strict +{est_cargo_strict}",
        f"- implied true_hit if CodeQL covers those 60: frozen {current_true + est_cargo_frozen:.0f}/148, strict {current_true + est_cargo_strict:.0f}/148",
        f"- 16 no_artifact issues without local Cargo.toml are **not** in this lift",
        f"- 26 RAP `miss` and 11 `skip_host` are **not** in this lift",
        "",
        "## Per issue",
        "",
    ]
    for score in issue_scores:
        lines.append(
            f"- `{score['verdict']}` {score['repository']}#{score['issue_number']} "
            f"{score['title'][:70]} → {score['hit_pattern']} {score['hit_path']} {score['hit_why']}"
        )
    report_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "records"}, indent=2), flush=True)
    print(f"wrote {report_md}", flush=True)
    write_heartbeat(heartbeat, phase="census_complete", n=len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

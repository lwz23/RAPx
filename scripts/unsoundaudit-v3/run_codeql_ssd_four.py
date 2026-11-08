#!/usr/bin/env python3
"""Extract 4 RAP-failed crates on internal SSD, query P1–P6, score issues.

Does not point CodeQL cargo/scratch at USB. Reports go to RAPDATA.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import census_codeql_no_artifact as census  # noqa: E402
import rescore_lwz23_coverage as rescore  # noqa: E402

REPOS = [
    "NVSL/Corundum",
    "asg017/sqlite-loadable-rs",
    "archseer/enigma",
    "Indra-db/Flecs-Rust",
]
CODEQL = Path(
    "/Users/liaowenzhao/Documents/RAP重构/lwz23-valid-issue-scan/codeql-poc/codeql/codeql"
)
SRC_ROOT = Path("/Users/liaowenzhao/Documents/RAP重构/lwz23-valid-issue-src")
PACK = Path(
    "/Users/liaowenzhao/Documents/RAP重构/RAPx-unsoundaudit-v2/codeql/unsoundaudit-poc"
)
DB_ROOT = Path(
    "/Users/liaowenzhao/Documents/RAP重构/lwz23-valid-issue-scan/codeql-poc/ssd-four"
)
REPORT_ROOT = Path("/Volumes/RAPDATA/codeql-ssd-four")
COVERAGE = Path(
    "/Users/liaowenzhao/Documents/RAP重构/lwz23-valid-issue-scan/phase0/coverage_rescored.csv"
)
EXISTING_CORUNDUM = Path(
    "/Users/liaowenzhao/Documents/RAP重构/lwz23-valid-issue-scan/codeql-poc/verify-corundum-db"
)
EXTRACT_TIMEOUT = 400
QUERY_TIMEOUT = 300


def host_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("CARGO_HOME", "CARGO_TARGET_DIR", "TMPDIR", "TMP", "TEMP"):
        env.pop(key, None)
    return env


def main() -> int:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    DB_ROOT.mkdir(parents=True, exist_ok=True)
    journal = REPORT_ROOT / "journal.jsonl"
    heartbeat = REPORT_ROOT / "heartbeat.txt"
    env = host_env()
    subprocess.run([str(CODEQL), "pack", "install", str(PACK)], check=False, env=env)

    issues = census.load_sample_issues(COVERAGE, REPOS)
    issues_by_repo: dict[str, list] = defaultdict(list)
    for issue in issues:
        issues_by_repo[issue["repository"]].append(issue)

    obligations: dict[str, str] = {}
    for instance in rescore.load_jsonl(
        Path("/Users/liaowenzhao/Documents/RAP重构/RAPx-unsoundaudit-v2")
        / "docs/unsoundaudit-v3/lwz23-valid-issue-roster-inputs/bug_instances.jsonl"
    ):
        url = str(instance.get("issue_url") or "")
        text = instance.get("repair_obligation_zh") or ""
        if url and text:
            obligations[url] = (obligations.get(url, "") + " " + text).strip()

    done = set()
    if journal.exists():
        for line in journal.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("status") == "scanned":
                done.add(rec["repo"])

    for idx, repo in enumerate(REPOS, 1):
        if repo in done:
            print(f"skip scanned {repo}", flush=True)
            continue
        slug = repo.replace("/", "__")
        src = SRC_ROOT / repo
        db = DB_ROOT / f"{slug}-db"
        logs = REPORT_ROOT / "logs" / slug
        logs.mkdir(parents=True, exist_ok=True)
        census.write_heartbeat(heartbeat, repo=repo, phase="extract", index=idx, total=4)
        rec = {"ts": census.now(), "repo": repo, "index": idx, "status": "running"}
        if repo == "NVSL/Corundum" and EXISTING_CORUNDUM.joinpath("db-rust").is_dir() and not db.joinpath("db-rust").is_dir():
            if db.exists():
                shutil.rmtree(db)
            shutil.copytree(EXISTING_CORUNDUM, db)
            rec["extract_s"] = 0
            rec["extract_rc"] = 0
            rec["extract_reused"] = str(EXISTING_CORUNDUM)
            print(f"reuse Corundum db {db}", flush=True)
        elif db.joinpath("db-rust").is_dir():
            rec["extract_s"] = 0
            rec["extract_rc"] = 0
            rec["extract_reused"] = str(db)
            print(f"reuse existing db {db}", flush=True)
        else:
            db.parent.mkdir(parents=True, exist_ok=True)
            if db.exists():
                shutil.rmtree(db)
            t0 = census.time.time()
            rc = census.run_timeout(
                [
                    str(CODEQL),
                    "database",
                    "create",
                    str(db),
                    "--language=rust",
                    "--build-mode=none",
                    "--source-root",
                    str(src),
                    "--overwrite",
                ],
                logs / "extract.log",
                EXTRACT_TIMEOUT,
                env,
            )
            rec["extract_s"] = round(census.time.time() - t0, 1)
            rec["extract_rc"] = rc
        extracted = db.joinpath("db-rust").is_dir() and rec.get("extract_rc", 1) == 0
        if not extracted:
            rec["status"] = "extract_timeout" if rec.get("extract_rc") == 124 else "extract_fail"
            rec["issue_scores"] = [
                census.score_issue(i, obligations.get(i.get("issue_url", ""), ""), [])
                for i in issues_by_repo[repo]
            ]
            journal.open("a").write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{idx}/4] {repo} {rec['status']}", flush=True)
            continue

        census.write_heartbeat(heartbeat, repo=repo, phase="query", index=idx)
        findings = []
        query_counts = {}
        for query_name, pattern in census.QUERIES:
            bqrs = logs / f"{pattern}.bqrs"
            csv_path = logs / f"{pattern}.csv"
            q0 = census.time.time()
            qrc = census.run_timeout(
                [
                    str(CODEQL),
                    "query",
                    "run",
                    str(PACK / query_name),
                    "--database",
                    str(db),
                    "--output",
                    str(bqrs),
                    "--threads=0",
                    "--ram=8192",
                ],
                logs / f"{pattern}.log",
                QUERY_TIMEOUT,
                env,
            )
            rec[f"{pattern}_s"] = round(census.time.time() - q0, 1)
            rec[f"{pattern}_rc"] = qrc
            if qrc == 0 and bqrs.is_file():
                subprocess.run(
                    [
                        str(CODEQL),
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
                found = census.parse_findings(csv_path, pattern)
                query_counts[pattern] = len(found)
                findings.extend(found)
            else:
                query_counts[pattern] = None
        rec["query_counts"] = query_counts
        rec["n_findings"] = len(findings)
        rec["status"] = "scanned"
        rec["issue_scores"] = [
            census.score_issue(i, obligations.get(i.get("issue_url", ""), ""), findings)
            for i in issues_by_repo[repo]
        ]
        usb_db = REPORT_ROOT / "db" / slug
        usb_db.parent.mkdir(parents=True, exist_ok=True)
        if usb_db.exists():
            shutil.rmtree(usb_db, ignore_errors=True)
        try:
            shutil.copytree(db, usb_db)
            rec["usb_db"] = str(usb_db)
        except OSError as exc:
            rec["usb_db_error"] = str(exc)
        journal.open("a").write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(
            f"[{idx}/4] {repo} scanned findings={len(findings)} extract={rec.get('extract_s')}s",
            flush=True,
        )

    records = []
    if journal.exists():
        latest = {}
        for line in journal.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                latest[rec["repo"]] = rec
        records = [latest[r] for r in REPOS if r in latest]
    scores = [s for r in records for s in r.get("issue_scores") or []]
    verdicts = Counter(s["verdict"] for s in scores)
    report = {
        "ts": census.now(),
        "repos": [{k: r.get(k) for k in ("repo", "status", "extract_s", "n_findings", "query_counts") if k in r} | {"issues": r.get("issue_scores")} for r in records],
        "issue_verdicts": dict(verdicts),
        "strict_hits": verdicts.get("strict_hit", 0),
        "generic_hits": verdicts.get("generic_hit", 0),
        "n_issues": len(scores),
        "extract_ok": sum(1 for r in records if r.get("status") == "scanned"),
    }
    (REPORT_ROOT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# CodeQL SSD four-repo recall probe",
        "",
        f"- time: {census.now()}",
        f"- extract ok: {report['extract_ok']}/4",
        f"- issues: {report['n_issues']}",
        f"- verdicts: {dict(verdicts)}",
        f"- strict hits: {report['strict_hits']}",
        f"- generic hits: {report['generic_hits']}",
        "",
        "## Per issue",
        "",
    ]
    for score in scores:
        lines.append(
            f"- `{score['verdict']}` {score['repository']}#{score['issue_number']} "
            f"{score['title'][:80]} → {score['hit_pattern']} {score.get('hit_path','')} {score.get('hit_why','')}"
        )
    (REPORT_ROOT / "report.md").write_text("\n".join(lines) + "\n")
    census.write_heartbeat(heartbeat, phase="complete", n=len(records))
    print(json.dumps({k: report[k] for k in report if k != "repos"}, indent=2), flush=True)
    print("wrote", REPORT_ROOT / "report.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

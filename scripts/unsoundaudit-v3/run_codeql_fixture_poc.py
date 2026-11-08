#!/usr/bin/env python3
"""Run the CodeQL P1–P6 PoC against UnsoundAudit v2 fixtures."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

PATTERN_BY_QUERY = {
    "unsoundaudit/p1-public-param": "pattern1",
    "unsoundaudit/p2-public-field": "pattern2",
    "unsoundaudit/p3-internal-invalid": "pattern3",
    "unsoundaudit/p4-internal-bounds": "pattern4",
    "unsoundaudit/p5-generic-nonempty": "pattern5",
    "unsoundaudit/p6-external-behavior": "pattern6",
}

QUERIES = [
    "P1PublicParam.ql",
    "P2PublicField.ql",
    "P3InternalInvalid.ql",
    "P4InternalBounds.ql",
    "P5GenericNonEmpty.ql",
    "P6ExternalBehavior.ql",
]


def run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True, text=True)


def find_codeql(explicit: Optional[str]) -> Path:
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
        raise SystemExit(f"codeql not found at {explicit}")
    which = shutil.which("codeql")
    if which:
        return Path(which)
    fallback = Path(
        "/Users/liaowenzhao/Documents/RAP重构/lwz23-valid-issue-scan/codeql-poc/codeql/codeql"
    )
    if fallback.is_file():
        return fallback
    raise SystemExit("codeql CLI not found; pass --codeql")


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def prepare_workspace(fixtures_dir: Path, workspace_dir: Path) -> List[str]:
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True)
    members = []
    for child in sorted(fixtures_dir.iterdir()):
        if not child.is_dir() or not (child / "Cargo.toml").is_file():
            continue
        members.append(child.name)
        shutil.copytree(child, workspace_dir / child.name, symlinks=False)
    cargo = ["[workspace]", 'resolver = "2"', "members = ["]
    cargo.extend(f'    "{name}",' for name in members)
    cargo.append("]")
    cargo.append("")
    (workspace_dir / "Cargo.toml").write_text("\n".join(cargo))
    return members


def decode_bqrs_csv(codeql: Path, bqrs: Path, csv_path: Path) -> List[dict]:
    run(
        [
            str(codeql),
            "bqrs",
            "decode",
            str(bqrs),
            "--format=csv",
            f"--output={csv_path}",
        ]
    )
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return []
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fixture_id_from_path(path: str, members: List[str]) -> Optional[str]:
    normalized = path.replace("\\", "/")
    for name in members:
        token = f"/{name}/"
        if token in normalized or normalized.endswith(f"/{name}") or f"/{name}/src/" in normalized:
            return name
    return None


def empty_counts() -> Dict[str, int]:
    return {f"pattern{i}": 0 for i in range(1, 7)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codeql", default=None)
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
    )
    parser.add_argument(
        "--work-root",
        default="/Users/liaowenzhao/Documents/RAP重构/lwz23-valid-issue-scan/codeql-poc",
    )
    parser.add_argument("--reuse-db", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    work_root = Path(args.work_root)
    pack_dir = repo_root / "codeql" / "unsoundaudit-poc"
    fixtures_dir = repo_root / "tests" / "unsoundaudit-v2" / "fixtures"
    manifest_path = repo_root / "tests" / "unsoundaudit-v2" / "fixture_manifest.json"
    workspace_dir = work_root / "fixture-workspace"
    db_dir = work_root / "fixture-db"
    results_dir = work_root / "fixture-results"
    results_dir.mkdir(parents=True, exist_ok=True)

    codeql = find_codeql(args.codeql)
    manifest = load_manifest(manifest_path)
    members = prepare_workspace(fixtures_dir, workspace_dir)
    print(f"workspace members: {len(members)}", flush=True)

    env = os.environ.copy()
    env.setdefault("CODEQL_ENABLE_EXPERIMENTAL_FEATURES", "true")

    run([str(codeql), "pack", "install", str(pack_dir)], env=env)

    if args.reuse_db and (db_dir / "db-rust").exists():
        print(f"reusing database {db_dir}", flush=True)
    else:
        if db_dir.exists():
            shutil.rmtree(db_dir)
        run(
            [
                str(codeql),
                "database",
                "create",
                str(db_dir),
                "--language=rust",
                "--build-mode=none",
                "--source-root",
                str(workspace_dir),
                "--overwrite",
            ],
            env=env,
        )

    counts_by_fixture = {name: empty_counts() for name in members}
    raw_hits = []
    for query in QUERIES:
        query_path = pack_dir / query
        stem = query_path.stem
        bqrs = results_dir / f"{stem}.bqrs"
        csv_path = results_dir / f"{stem}.csv"
        try:
            run(
                [
                    str(codeql),
                    "query",
                    "run",
                    str(query_path),
                    "--database",
                    str(db_dir),
                    "--output",
                    str(bqrs),
                    "--threads=0",
                    "--ram=4096",
                ],
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            (results_dir / f"{stem}.error.txt").write_text(str(exc))
            print(f"QUERY_FAILED {query}", flush=True)
            continue
        rows = decode_bqrs_csv(codeql, bqrs, csv_path)
        pattern = None
        # Map via filename if metadata is not in CSV.
        filename_map = {
            "P1PublicParam": "pattern1",
            "P2PublicField": "pattern2",
            "P3InternalInvalid": "pattern3",
            "P4InternalBounds": "pattern4",
            "P5GenericNonEmpty": "pattern5",
            "P6ExternalBehavior": "pattern6",
        }
        pattern = filename_map[stem]
        for row in rows:
            blob = " ".join(str(v) for v in row.values())
            fid = fixture_id_from_path(blob, members)
            if fid is None:
                # try every column
                for value in row.values():
                    fid = fixture_id_from_path(str(value), members)
                    if fid:
                        break
            raw_hits.append({"query": stem, "pattern": pattern, "fixture": fid, "row": row})
            if fid:
                counts_by_fixture[fid][pattern] += 1

    cases = []
    summary = {
        "positive_ok": 0,
        "positive_fail": 0,
        "negative_ok": 0,
        "negative_fail": 0,
        "noise_ok": 0,
        "noise_fail": 0,
        "query_compile_or_run_errors": [
            p.name for p in results_dir.glob("*.error.txt")
        ],
    }
    for case in manifest["cases"]:
        case_id = case["case_id"]
        expected = case["expected_pattern_counts"]
        actual = counts_by_fixture.get(case_id, empty_counts())
        expected_any = {k: (1 if v else 0) for k, v in expected.items()}
        actual_any = {k: (1 if v else 0) for k, v in actual.items()}
        # PoC scoring: presence of the expected primary, not RAP's exact count=1.
        role = case["role"]
        primary = case.get("expected_primary_pattern")
        if role == "positive":
            ok = actual_any.get(primary, 0) == 1
        else:
            ok = sum(actual_any.values()) == 0
        summary_key = f"{role}_{'ok' if ok else 'fail'}"
        summary[summary_key] += 1
        cases.append(
            {
                "case_id": case_id,
                "role": role,
                "expected_primary_pattern": primary,
                "expected_pattern_counts": expected,
                "actual_pattern_counts": actual,
                "actual_presence": actual_any,
                "ok": ok,
            }
        )

    report = {
        "codeql": str(codeql),
        "database": str(db_dir),
        "workspace": str(workspace_dir),
        "members": members,
        "summary": summary,
        "cases": cases,
        "raw_hit_count": len(raw_hits),
    }
    out = results_dir / "fixture_poc_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    md = results_dir / "fixture_poc_report.md"
    lines = [
        "# CodeQL fixture PoC",
        "",
        f"- codeql: `{codeql}`",
        f"- database: `{db_dir}`",
        f"- raw hits: {len(raw_hits)}",
        "",
        "## Summary",
        "",
        f"- positive ok/fail: {summary['positive_ok']}/{summary['positive_fail']}",
        f"- negative ok/fail: {summary['negative_ok']}/{summary['negative_fail']}",
        f"- noise ok/fail: {summary['noise_ok']}/{summary['noise_fail']}",
        "",
        "| case | role | expected | actual presence | ok |",
        "|---|---|---|---|---|",
    ]
    for case in cases:
        actual = ",".join(
            f"{k}={v}" for k, v in case["actual_presence"].items() if v
        ) or "none"
        lines.append(
            f"| `{case['case_id']}` | {case['role']} | {case['expected_primary_pattern'] or 'none'} | {actual} | {case['ok']} |"
        )
    md.write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    print(f"wrote {md}", flush=True)
    return 0 if not summary["query_compile_or_run_errors"] else 2


if __name__ == "__main__":
    sys.exit(main())

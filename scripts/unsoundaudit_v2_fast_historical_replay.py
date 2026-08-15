#!/usr/bin/env python3
"""One-pass replay of ten frozen historical source revisions with legacy RAPx."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


PILOT_PATH = Path(__file__).with_name("unsoundaudit_v2_real_project_pilot.py")
PILOT_SPEC = importlib.util.spec_from_file_location("real_project_pilot", PILOT_PATH)
if PILOT_SPEC is None or PILOT_SPEC.loader is None:
    raise RuntimeError(f"cannot load pilot support module from {PILOT_PATH}")
pilot = importlib.util.module_from_spec(PILOT_SPEC)
PILOT_SPEC.loader.exec_module(pilot)


CASES: tuple[dict[str, Any], ...] = (
    {"case": 1, "expected_pattern": "P1", "repository": "servo/rust-smallvec", "repo_issue": "servo/rust-smallvec #343", "checkout": "0f3aacb99ccfe0cbbf0bb83ecf736f67339ce8ae^"},
    {"case": 2, "expected_pattern": "P1", "repository": "GitoxideLabs/gitoxide", "repo_issue": "GitoxideLabs/gitoxide #1460", "checkout": "a807dd1ffb05efd177700d065095249e6c4b3c68"},
    {"case": 3, "expected_pattern": "P2", "repository": "rust-lang/libc", "repo_issue": "rust-lang/libc #3560", "checkout": "6faa521f32fc11db9fc43a248a64463ce288b48d^"},
    {"case": 4, "expected_pattern": "P3", "repository": "jeromefroe/lru-rs", "repo_issue": "jeromefroe/lru-rs #73", "checkout": "a96f71a84d108f9b7506908b8e1c5da9524bc4da^"},
    {"case": 5, "expected_pattern": "P3", "repository": "hyperium/http", "repo_issue": "hyperium/http #639", "checkout": "bbe2a8f34f0f0aadcd124f5f2841bdae454bc837^"},
    {"case": 6, "expected_pattern": "P4", "repository": "fitzgen/bumpalo", "repo_issue": "fitzgen/bumpalo #164", "checkout": "71d99b89835a36a497e1f3ff54b625de5012e3ed^"},
    {"case": 7, "expected_pattern": "P4", "repository": "rust-random/rand", "repo_issue": "rust-random/rand #779", "checkout": "9828cdf0201f816e4e4bff07834d849a10be80e9"},
    {"case": 8, "expected_pattern": "P5.1", "repository": "rust-random/rand", "repo_issue": "rust-random/rand #1158", "checkout": "90b89cdbb2655ac278cee43b5c343102a89616b1^"},
    {"case": 9, "expected_pattern": "P6", "repository": "nix-rust/nix", "repo_issue": "nix-rust/nix #1819", "checkout": "8e91b28b64bdd18fc6fa61af8e89947ad7fb97bf^"},
    {"case": 10, "expected_pattern": "P6", "repository": "diesel-rs/diesel", "repo_issue": "diesel-rs/diesel #4223", "checkout": "49aa163f1a9c793da6d591e10481fbfb6c044b51^"},
)
REVISION = re.compile(r"[0-9a-f]{40}\^?\Z")


def validate_revision_expression(value: str) -> str:
    if not isinstance(value, str) or not REVISION.fullmatch(value):
        raise ValueError("historical checkout must be an exact lowercase commit with optional parent")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _result_status(row: dict[str, Any]) -> str:
    if row["control"] != "ok":
        return row.get("failure", "control_fail")
    if row["rap"] != "ok":
        return row.get("failure", "rap_fail")
    counts = row["pattern_counts"]
    return "no_finding" if sum(counts.values()) == 0 else "finding"


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# RAPx 快速历史源码回放 v1",
        "",
        "该表是历史源码诊断，不是 recall、precision 或生态 coverage 结论。",
        "",
        "| case | expected_pattern | repo_issue | detached_revision | control | rap | p1 | p2 | p3 | p4 | p5 | p6 | result_dir | outcome |",
        "|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in sorted(rows, key=lambda value: value["case"]):
        counts = row.get("pattern_counts", {})
        lines.append(
            "| {case} | {expected_pattern} | {repo_issue} | `{detached_revision}` | "
            "{control} | {rap} | {p1} | {p2} | {p3} | {p4} | {p5} | {p6} | "
            "{result_dir} | {outcome} |".format(
                **row,
                p1=counts.get("pattern1", 0),
                p2=counts.get("pattern2", 0),
                p3=counts.get("pattern3", 0),
                p4=counts.get("pattern4", 0),
                p5=counts.get("pattern5", 0),
                p6=counts.get("pattern6", 0),
                outcome=_result_status(row),
            )
        )
    return "\n".join(lines) + "\n"


def _base_row(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "rapx-fast-historical-replay-case-v1",
        "case": case["case"],
        "expected_pattern": case["expected_pattern"],
        "repository": case["repository"],
        "repo_issue": case["repo_issue"],
        "checkout": case["checkout"],
        "detached_revision": "source_unavailable",
        "control": "fail",
        "rap": "skipped_control_fail",
        "pattern_counts": {f"pattern{number}": 0 for number in range(1, 7)},
        "finding_count": 0,
        "result_dir": f"cases/case-{case['case']:02d}",
    }


def _journal(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in sorted(rows, key=lambda value: value["case"]):
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    path.with_name("results.md").write_text(markdown_table(rows), encoding="utf-8")


def _run_case(
    case: dict[str, Any],
    result_root: Path,
    cargo: Path,
    scan_runner: Any,
) -> dict[str, Any]:
    row = _base_row(case)
    case_root = result_root / row["result_dir"]
    repository = case_root / "source"
    logs = case_root / "logs"
    logs.mkdir(parents=True)
    clone = pilot._run_logged(
        ["git", "clone", "--no-recurse-submodules", f"https://github.com/{case['repository']}.git", str(repository)],
        case_root,
        logs / "clone.log",
    )
    row["clone_exit"] = clone.returncode
    if clone.returncode != 0:
        row["failure"] = "source_unavailable"
        return row
    try:
        expression = validate_revision_expression(case["checkout"])
        detached = pilot._git_output(repository, "rev-parse", "--verify", f"{expression}^{{commit}}")
    except ValueError as error:
        row.update({"failure": "source_unavailable", "error": str(error)})
        return row
    checkout = pilot._run_logged(
        ["git", "checkout", "--detach", detached], repository, logs / "checkout.log"
    )
    row.update({"checkout_exit": checkout.returncode, "detached_revision": detached})
    if checkout.returncode != 0:
        row["failure"] = "source_unavailable"
        return row
    lockfile = repository / "Cargo.lock"
    if not lockfile.is_file():
        row["failure"] = "lockfile_missing"
        return row

    with tempfile.TemporaryDirectory(prefix=f"rapx-history-control-home-{case['case']:02d}-") as cargo_home:
        with tempfile.TemporaryDirectory(prefix=f"rapx-history-control-target-{case['case']:02d}-") as cargo_target:
            environment = dict(os.environ)
            environment.update({"CARGO_HOME": cargo_home, "CARGO_TARGET_DIR": cargo_target, "CARGO_BUILD_JOBS": "1"})
            metadata_argv = [str(cargo), "metadata", "--locked", "--no-deps", "--format-version", "1"]
            metadata = pilot._run_logged(metadata_argv, repository, logs / "metadata.log", environment)
            if metadata.returncode != 0:
                row.update({"failure": "metadata_failed", "metadata_exit": metadata.returncode})
                return row
            try:
                target = pilot._select_cargo_target(json.loads(metadata.stdout), repository)
            except (json.JSONDecodeError, ValueError) as error:
                row.update({"failure": "metadata_invalid", "error": str(error)})
                return row
            if target is None:
                row["failure"] = "target_missing"
                return row
            control_argv = [str(cargo), "check", "--locked", "--jobs", "1", "--package", target["package"]]
            control_argv.extend(["--lib"] if target["kind"] == "lib" else ["--bin", target["name"]])
            control = pilot._run_logged(control_argv, repository, logs / "control.log", environment)
    row.update({"control_exit": control.returncode, "target": target})
    if control.returncode != 0:
        row["failure"] = "control_failed"
        return row
    if pilot._git_output(repository, "status", "--porcelain"):
        row["failure"] = "control_mutated_source"
        return row
    row["control"] = "ok"

    admission = {
        "repository": case["repository"],
        "repository_url": f"https://github.com/{case['repository']}",
        "repo_id": case["case"],
        "stratum": "historical",
        "clone_path": str(repository),
        "source_commit": detached,
        "source_tree": pilot._git_output(repository, "rev-parse", "HEAD^{tree}"),
        "cargo_lock_sha256": pilot.sha256_file(lockfile),
        "target": target,
    }
    scan_row, receipt = scan_runner.run(admission, "rapx-perf", 1)
    row["rap_exit"] = scan_row["exit_code"]
    row["scan_log"] = scan_row["log"]
    if receipt is None:
        row.update({"rap": "fail", "failure": scan_row["status"]})
        return row
    row.update(
        {
            "rap": "ok",
            "finding_count": receipt["finding_count"],
            "pattern_counts": receipt["pattern_counts"],
            "receipt": scan_row["receipt"],
            "receipt_sha256": scan_row["receipt_sha256"],
            "findings": [
                {
                    "finding_id": finding.get("finding_id"),
                    "primary_pattern": finding.get("primary_pattern"),
                    "source": finding.get("source"),
                    "sink": finding.get("sink"),
                }
                for finding in receipt["findings"]
            ],
        }
    )
    return row


def run_replay(profile_contract: Path, bin_dir: Path, result_root: Path) -> list[dict[str, Any]]:
    if result_root.exists():
        raise ValueError("result root already exists; historical replay is append-only")
    result_root.mkdir(parents=True)
    preflight = result_root / "preflight.txt"
    if pilot.verify_profiles(profile_contract, bin_dir, preflight) != "ok":
        raise ValueError("legacy RAP profile preflight failed")
    cargo, _, _ = pilot._required_environment()
    profiles = pilot._profile_map(pilot.load_profile_contract(profile_contract), bin_dir)
    scan_runner = pilot.RealScanRunner(result_root, profiles, result_root / "scan-cargo-home")
    rows: list[dict[str, Any]] = []
    results_path = result_root / "results.jsonl"
    for case in CASES:
        row = _run_case(case, result_root, cargo, scan_runner)
        rows.append(row)
        _journal(results_path, rows)
    environment = {
        "schema": "rapx-fast-historical-replay-environment-v1",
        "rap_source_commit": pilot.LEGACY_SOURCE_COMMIT,
        "rustc_commit": pilot.LEGACY_RUSTC_COMMIT,
        "cargo_commit": pilot.LEGACY_CARGO_COMMIT,
        "profile": "rapx-perf",
        "profile_executable_sha256": profiles["rapx-perf"]["sha256"],
        "profile_rapx_sha256": profiles["rapx-perf"]["rapx_sha256"],
        "case_count": len(rows),
    }
    _write_json(result_root / "environment.json", environment)
    return rows


def freeze_evidence(result_root: Path, output_dir: Path) -> dict[str, Any]:
    result_root = result_root.resolve(strict=True)
    rows = [
        json.loads(line)
        for line in (result_root / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if [row.get("case") for row in rows] != list(range(1, 11)):
        raise ValueError("historical replay evidence requires the exact ten ordered cases")
    output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("results.jsonl", "results.md", "environment.json"):
        (output_dir / name).write_bytes((result_root / name).read_bytes())
    artifact_hashes: dict[str, str] = {}
    case_log_roots = sorted((result_root / "cases").glob("case-*/logs"))
    artifact_roots = (*case_log_roots, result_root / "scan-logs", result_root / "raw-receipts")
    for root in artifact_roots:
        if not root.is_dir():
            continue
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
            artifact_hashes[path.relative_to(result_root).as_posix()] = pilot.sha256_file(path)
    failures: dict[str, int] = {}
    for row in rows:
        failure = row.get("failure")
        if isinstance(failure, str):
            failures[failure] = failures.get(failure, 0) + 1
    manifest = {
        "schema": "rapx-fast-historical-replay-evidence-v1",
        "case_count": len(rows),
        "control_ok": sum(row.get("control") == "ok" for row in rows),
        "rap_ok": sum(row.get("rap") == "ok" for row in rows),
        "finding_count": sum(row.get("finding_count", 0) for row in rows),
        "failure_counts": dict(sorted(failures.items())),
        "evidence_files": {
            name: pilot.sha256_file(output_dir / name)
            for name in ("environment.json", "results.jsonl", "results.md")
        },
        "local_artifact_hashes": artifact_hashes,
        "raw_logs_or_receipts_committed": False,
        "binaries_or_caches_committed": False,
        "interpretation": "inconclusive_input_or_execution_gate",
    }
    _write_json(output_dir / "evidence_manifest.json", manifest)
    forbidden = ("/Users/", "/private/tmp/", "/tmp/", "github_pat_", "ghp_")
    for path in output_dir.iterdir():
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            if any(token in text for token in forbidden):
                raise ValueError(f"shareable historical replay evidence is not sanitized: {path.name}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-contract", type=Path, required=True)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    run_replay(args.profile_contract, args.bin_dir, args.result_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

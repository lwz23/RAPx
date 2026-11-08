#!/usr/bin/env python3
"""Run original or challenge fixtures against the compiler-aligned v3 analyzer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from validate_receipt import ContractError, read_json, validate_receipt


PATTERN_KEYS = tuple(f"pattern{number}" for number in range(1, 7))
PATTERN_NAMES = {f"pattern{number}": f"P{number}" for number in range(1, 7)}
SUITES = {
    "original": {
        "root": "tests/unsoundaudit-v2/fixtures",
        "manifest": "tests/unsoundaudit-v2/fixture_manifest.json",
        "count": 27,
        "counts": {
            "pattern1": 3,
            "pattern2": 2,
            "pattern5": 2,
            "pattern6": 2,
            "pattern4": 1,
            "pattern3": 2,
        },
    },
    "challenge": {
        "root": "tests/unsoundaudit-v2/challenges",
        "manifest": "tests/unsoundaudit-v2/challenge_manifest.json",
        "count": 27,
        "counts": {
            "pattern1": 13,
            "pattern2": 0,
            "pattern5": 0,
            "pattern6": 0,
            "pattern4": 0,
            "pattern3": 0,
        },
    },
}


class RunnerError(ValueError):
    """Raised when a fixture run violates the closed v3 gate."""


def fail(message: str) -> None:
    raise RunnerError(message)


def command_commit(path: Path) -> str:
    output = subprocess.check_output([str(path), "-Vv"], text=True)
    for line in output.splitlines():
        if line.startswith("commit-hash:"):
            return line.split()[1]
    fail(f"{path} -Vv has no commit-hash")


def pin_exact_toolchain(cargo: Path) -> Path:
    rustc = cargo.with_name("rustc")
    if not rustc.is_file():
        fail(f"rustc sibling missing: {rustc}")
    sysroot = Path(
        subprocess.check_output([str(rustc), "--print", "sysroot"], text=True).strip()
    )
    lib = str(sysroot / "lib")
    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = lib if not old_ld else f"{lib}:{old_ld}"
    os.environ["UNSOUND_SCANNER_EXACT_CARGO_PATH"] = str(cargo)
    os.environ["UNSOUND_SCANNER_EXACT_CARGO_COMMIT"] = command_commit(cargo)
    os.environ["RUSTC"] = str(rustc)
    os.environ["UNSOUND_SCANNER_EXPECTED_RUSTC_PATH"] = str(rustc)
    os.environ["UNSOUND_SCANNER_EXPECTED_RUSTC_COMMIT"] = command_commit(rustc)
    os.environ["RUST_SYSROOT"] = str(sysroot)
    return rustc


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_finding(finding: dict[str, Any]) -> dict[str, Any]:
    propagation = finding["propagation"]
    return {
        "finding_id": finding["finding_id"],
        "primary_pattern": finding["primary_pattern"],
        "mechanism_id": finding["mechanism_id"],
        "classification_confidence": finding["classification_confidence"],
        "public_safe_root": {
            "owner": finding["public_safe_root"]["owner"],
            "span": finding["public_safe_root"]["span"],
        },
        "source": finding["source"],
        "first_contract_failure": finding["first_contract_failure"],
        "unsafe_site": finding["unsafe_site"],
        "exposure": finding["exposure"],
        "contract": {
            "precision": finding["contract"]["precision"],
            "model_id": finding["contract"]["model_id"],
            "relevant_roles": finding["contract"]["relevant_roles"],
            "obligation": finding["contract"]["obligation"],
        },
        "propagation": {
            "depth": propagation["depth"],
            "boundaries": propagation["boundaries"],
        },
    }


def normalized_case(case_id: str, receipt: dict[str, Any]) -> dict[str, Any]:
    if receipt.get("analysis_status") != "complete":
        fail(f"{case_id}: fixture receipt is partial")
    if receipt.get("coverage_failures") != []:
        fail(f"{case_id}: complete fixture has coverage failures")
    observed = receipt.get("observed_counts")
    if not isinstance(observed, dict) or observed.get("semantics") != "exact":
        fail(f"{case_id}: fixture counts are not exact")
    counts = {key: observed.get(key) for key in PATTERN_KEYS}
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values()):
        fail(f"{case_id}: fixture pattern counts are invalid")
    findings = receipt.get("findings")
    if not isinstance(findings, list) or receipt.get("finding_count") != len(findings):
        fail(f"{case_id}: fixture finding count is invalid")
    projected = sorted((semantic_finding(finding) for finding in findings), key=lambda row: row["finding_id"])
    return {
        "case_id": case_id,
        "package_name": receipt["package_name"],
        "crate_name": receipt["crate_name"],
        "rustc_commit": receipt["rustc_commit"],
        "adapter_id": receipt["toolchain_adapter"]["adapter_id"],
        "analysis_status": "complete",
        "unsafe_site_census": receipt["unsafe_site_census"],
        "pattern_counts": counts,
        "finding_count": len(findings),
        "semantic_findings": projected,
    }


def normalized_suite(suite: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    if suite not in SUITES:
        fail(f"unknown suite: {suite}")
    ordered = sorted(cases, key=lambda case: case["case_id"])
    if len({case["case_id"] for case in ordered}) != len(ordered):
        fail("suite contains duplicate case ids")
    aggregate = {key: 0 for key in PATTERN_KEYS}
    for case in ordered:
        for key in PATTERN_KEYS:
            aggregate[key] += case["pattern_counts"][key]
    return {
        "schema_version": "rap-fixture-suite-v3",
        "suite": suite,
        "case_count": len(ordered),
        "aggregate_pattern_counts": aggregate,
        "cases": ordered,
    }


def validate_semantic_oracle(case_id: str, case: dict[str, Any], oracle: dict[str, Any]) -> None:
    expected = oracle.get("expected_pattern_counts")
    if not isinstance(expected, dict) or set(expected) != set(PATTERN_KEYS):
        fail(f"{case_id}: oracle pattern counts are invalid")
    if case["pattern_counts"] != expected:
        fail(
            f"{case_id}: pattern counts differ: "
            f"expected={expected}, actual={case['pattern_counts']}"
        )
    expected_primary = oracle.get("expected_primary_pattern")
    actual_primary = [finding["primary_pattern"] for finding in case["semantic_findings"]]
    if expected_primary is None:
        if actual_primary:
            fail(f"{case_id}: expected no finding, got {actual_primary}")
    else:
        expected_name = PATTERN_NAMES.get(expected_primary)
        if expected_name is None or actual_primary != [expected_name]:
            fail(
                f"{case_id}: primary pattern differs: "
                f"expected={expected_name}, actual={actual_primary}"
            )


def verified_cases(repo_root: Path, suite: str) -> tuple[Path, list[str]]:
    contract = SUITES[suite]
    case_root = (repo_root / contract["root"]).resolve(strict=True)
    manifest_path = (repo_root / contract["manifest"]).resolve(strict=True)
    manifest = read_json(manifest_path)
    rows = manifest.get("cases") if isinstance(manifest, dict) else None
    if not isinstance(rows, list) or len(rows) != contract["count"]:
        fail(f"{suite}: manifest case count drifted")
    if manifest.get("aggregate_expected_pattern_counts") != contract["counts"]:
        fail(f"{suite}: manifest aggregate drifted")
    case_ids: list[str] = []
    aggregate = {key: 0 for key in PATTERN_KEYS}
    for row in rows:
        if not isinstance(row, dict):
            fail(f"{suite}: manifest case row is not an object")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", case_id) is None:
            fail(f"{suite}: invalid case id")
        fixture = (case_root / case_id).resolve(strict=True)
        if fixture.parent != case_root or fixture.is_symlink():
            fail(f"{case_id}: fixture escapes the suite root")
        files = row.get("files")
        if not isinstance(files, dict):
            fail(f"{case_id}: file manifest is invalid")
        for role in ("manifest", "source", "lock", "oracle"):
            descriptor = files.get(role)
            if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
                fail(f"{case_id}: {role} descriptor is invalid")
            path = (manifest_path.parent / descriptor["path"]).resolve(strict=True)
            if not path.is_file() or path.is_symlink() or sha256(path) != descriptor["sha256"]:
                fail(f"{case_id}: {role} hash differs")
        expected = row.get("expected_pattern_counts")
        if not isinstance(expected, dict) or set(expected) != set(PATTERN_KEYS):
            fail(f"{case_id}: expected pattern counts are invalid")
        for key in PATTERN_KEYS:
            aggregate[key] += expected[key]
        case_ids.append(case_id)
    if len(set(case_ids)) != len(case_ids) or aggregate != contract["counts"]:
        fail(f"{suite}: manifest identities or aggregate differ")
    return case_root, sorted(case_ids)


def run_command(command: list[str], cwd: Path, environment: dict[str, str], log: Path) -> None:
    with log.open("wb") as output:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        fail(f"command exited {result.returncode}; log={log}")


def temporary_target(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix, dir=os.environ.get("TMPDIR") or "/tmp"))


def run_case(
    case_id: str,
    fixture: Path,
    cargo: Path,
    cargo_rapx: Path,
    schema: dict[str, Any],
    output_root: Path,
    runs: int,
) -> list[dict[str, Any]]:
    logs = output_root / "logs"
    receipts = output_root / "receipts"
    control_target = temporary_target(f"rapx-v3-control-{case_id}.")
    try:
        environment = os.environ.copy()
        environment.update({"CARGO_BUILD_JOBS": "1", "CARGO_TARGET_DIR": str(control_target)})
        run_command(
            [str(cargo), "check", "--lib", "--locked", "--jobs", "1"],
            fixture,
            environment,
            logs / f"{case_id}.control.log",
        )
    finally:
        shutil.rmtree(control_target, ignore_errors=True)

    raw_runs: list[bytes] = []
    normalized_runs: list[dict[str, Any]] = []
    for run in range(1, runs + 1):
        target = temporary_target(f"rapx-v3-scan-{case_id}-run{run}.")
        receipt_dir = receipts / f"run{run}" / case_id
        receipt_dir.mkdir(parents=True)
        try:
            environment = os.environ.copy()
            environment.update({
                "CARGO_BUILD_JOBS": "1",
                "CARGO_TARGET_DIR": str(target),
                "UNSOUND_SCANNER_PROJECT_ROOT": str(fixture),
                "UNSOUND_SCANNER_RAP_JSON_DIR": str(receipt_dir),
            })
            run_command(
                [
                    str(cargo_rapx),
                    "rapx",
                    "-unsoundaudit",
                    "--",
                    "--lib",
                    "--locked",
                    "--jobs",
                    "1",
                ],
                fixture,
                environment,
                logs / f"{case_id}.scan{run}.log",
            )
            paths = sorted(receipt_dir.glob("*.json"))
            if len(paths) != 1:
                fail(f"{case_id}: run {run} produced {len(paths)} receipts")
            raw = paths[0].read_bytes()
            receipt = json.loads(raw)
            validate_receipt(receipt, schema)
            normalized = normalized_case(case_id, receipt)
            validate_semantic_oracle(case_id, normalized, read_json(fixture / "fixture.json"))
            raw_runs.append(raw)
            normalized_runs.append(normalized)
        finally:
            shutil.rmtree(target, ignore_errors=True)
    if any(raw != raw_runs[0] for raw in raw_runs[1:]):
        fail(f"{case_id}: raw v3 receipts are nondeterministic")
    if any(case != normalized_runs[0] for case in normalized_runs[1:]):
        fail(f"{case_id}: semantic projections are nondeterministic")
    return normalized_runs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=sorted(SUITES), required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--cargo", type=Path, required=True)
    parser.add_argument("--rap-bin-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--case", action="append", dest="selected_cases")
    args = parser.parse_args()
    try:
        repo_root = args.repo_root.expanduser().resolve(strict=True)
        schema_path = args.schema.expanduser().resolve(strict=True)
        cargo = args.cargo.expanduser().resolve(strict=True)
        pin_exact_toolchain(cargo)
        bin_dir = args.rap_bin_dir.expanduser().resolve(strict=True)
        cargo_rapx = (bin_dir / "cargo-rapx").resolve(strict=True)
        (bin_dir / "rapx").resolve(strict=True)
        if args.runs < 1:
            fail("runs must be positive")
        schema = read_json(schema_path)
        case_root, all_cases = verified_cases(repo_root, args.suite)
        selected = set(args.selected_cases or all_cases)
        unknown = sorted(selected - set(all_cases))
        if unknown:
            fail(f"unknown selected cases: {unknown}")
        case_ids = [case_id for case_id in all_cases if case_id in selected]
        output_root = args.output_root.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=False)
        (output_root / "logs").mkdir()
        (output_root / "receipts").mkdir()

        per_run = [[] for _ in range(args.runs)]
        for case_id in case_ids:
            runs = run_case(
                case_id,
                (case_root / case_id).resolve(strict=True),
                cargo,
                cargo_rapx,
                schema,
                output_root,
                args.runs,
            )
            for index, case in enumerate(runs):
                per_run[index].append(case)
        normalized_runs = [normalized_suite(args.suite, cases) for cases in per_run]
        if any(run != normalized_runs[0] for run in normalized_runs[1:]):
            fail("suite semantic projections are nondeterministic")
        if not args.selected_cases:
            expected = SUITES[args.suite]
            if normalized_runs[0]["case_count"] != expected["count"]:
                fail("full suite case count differs")
            if normalized_runs[0]["aggregate_pattern_counts"] != expected["counts"]:
                fail("full suite aggregate differs")
        for index, normalized in enumerate(normalized_runs, 1):
            (output_root / f"{args.suite}.run{index}.normalized.json").write_bytes(
                canonical_json(normalized)
            )
        (output_root / f"{args.suite}.normalized.json").write_bytes(
            canonical_json(normalized_runs[0])
        )
    except (ContractError, RunnerError, OSError, json.JSONDecodeError) as error:
        print(f"v3 fixture suite error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the frozen UnsoundAudit v2 challenge cases serially and fail closed."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from run_fixture_suite import (
    baseline_toolchain,
    canonical_existing,
    fresh_target,
    git_tracked,
    normalized_case,
    run_command,
    verify_case_contract,
    verify_environment,
    verify_receipt_toolchain,
)
from validate_receipt import ContractError, read_json, validate_oracle, validate_receipt


CHALLENGE_SCHEMA = "unsoundaudit-v2-challenge-manifest-v1"
CHALLENGE_COUNT = 27
CHALLENGE_COUNTS = {
    "pattern1": 13,
    "pattern2": 0,
    "pattern3": 0,
    "pattern4": 0,
    "pattern5": 0,
    "pattern6": 0,
}
CHALLENGE_MANIFEST_NAME = "challenge_manifest.json"
CHALLENGE_ROOT_NAME = "challenges"
CHALLENGE_ORACLE_KEYS = {
    "case_id",
    "expected_finding_count",
    "expected_pattern_counts",
    "expected_primary_pattern",
    "expected_rule_id",
    "family_support",
    "fix_fact",
    "forbidden_extra_patterns",
    "limitations",
    "manifest_relative_path",
    "obligation",
    "package_name",
    "propagation_depth",
    "role",
    "schema_version",
    "sink",
    "source",
    "source_relative_path",
    "expected_finding_spans",
}


def fail(message: str) -> None:
    raise ContractError(message)


def verify_challenge_contract(
    repo_root: Path,
    challenge_root: Path,
    manifest_path: Path,
    manifest: dict,
) -> list[str]:
    repo_root = repo_root.resolve(strict=True)
    challenge_root = challenge_root.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    expected_parent = repo_root / "tests/unsoundaudit-v2"
    if challenge_root != expected_parent / CHALLENGE_ROOT_NAME:
        fail("challenge root must be the independent tests/unsoundaudit-v2/challenges tree")
    if manifest_path != expected_parent / CHALLENGE_MANIFEST_NAME:
        fail("challenge manifest must be the independent challenge_manifest.json")
    if isinstance(manifest, dict) and "legacy_provenance_manifest" in manifest:
        fail("challenge manifest must not contain legacy provenance")
    case_ids = verify_case_contract(
        repo_root,
        challenge_root,
        manifest_path,
        manifest,
        expected_schema=CHALLENGE_SCHEMA,
        expected_count=CHALLENGE_COUNT,
        expected_counts=CHALLENGE_COUNTS,
        require_legacy_provenance=False,
    )
    for case_id in case_ids:
        oracle = read_json(challenge_root / case_id / "fixture.json")
        if not isinstance(oracle, dict) or set(oracle) != CHALLENGE_ORACLE_KEYS:
            fail(f"{case_id}: challenge oracle keys drifted")
        if oracle["schema_version"] != "unsoundaudit-v2-challenge-v1":
            fail(f"{case_id}: challenge oracle schema drifted")
        validate_challenge_assertions(case_id, {"findings": []}, oracle, shape_only=True)
    return case_ids


def normalized_suite(cases: list[dict]) -> dict:
    ordered = sorted(cases, key=lambda case: case["case_id"])
    aggregate = {f"pattern{number}": 0 for number in range(1, 7)}
    for case in ordered:
        for pattern, count in case["pattern_counts"].items():
            aggregate[pattern] += count
    return {
        "schema_version": "rap-challenge-suite-v1",
        "challenge_count": len(ordered),
        "aggregate_pattern_counts": aggregate,
        "cases": ordered,
    }


def validate_challenge_assertions(
    case_id: str,
    receipt: dict,
    oracle: dict,
    *,
    shape_only: bool = False,
) -> None:
    expected_spans = oracle.get("expected_finding_spans", [])
    if not isinstance(expected_spans, list) or any(
        not isinstance(assertion, dict)
        or set(assertion) != {"finding_field", "path"}
        or assertion["finding_field"] not in {"source", "first_contract_failure", "sink"}
        or not isinstance(assertion["path"], str)
        or not assertion["path"]
        for assertion in expected_spans
    ):
        fail(f"{case_id}: expected_finding_spans is invalid")
    fields = [assertion["finding_field"] for assertion in expected_spans] if isinstance(expected_spans, list) else []
    if len(fields) != len(set(fields)):
        fail(f"{case_id}: expected_finding_spans contains duplicate finding fields")
    if shape_only:
        return
    if expected_spans and len(receipt["findings"]) != 1:
        fail(f"{case_id}: span assertions require exactly one finding")
    for assertion in expected_spans:
        actual = receipt["findings"][0][assertion["finding_field"]]["span"]["path"]
        if actual != assertion["path"]:
            fail(
                f"{case_id}: {assertion['finding_field']} span differs: "
                f"expected={assertion['path']}, actual={actual}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--challenge-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--cargo", type=Path, required=True)
    parser.add_argument("--rap-bin-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case", action="append", dest="selected_cases")
    args = parser.parse_args()
    try:
        repo_root = canonical_existing(args.repo_root, "repository root")
        challenge_root = canonical_existing(args.challenge_root, "challenge root")
        manifest_path = canonical_existing(args.manifest, "challenge manifest")
        schema_path = canonical_existing(args.schema, "receipt schema")
        cargo = canonical_existing(args.cargo, "Cargo")
        rap_bin_dir = canonical_existing(args.rap_bin_dir, "RAP binary directory")
        cargo_rapx, _ = verify_environment(repo_root, cargo, rap_bin_dir)
        manifest = read_json(manifest_path)
        schema = read_json(schema_path)
        _, expected_rustc_commit = baseline_toolchain(repo_root)
        all_case_ids = verify_challenge_contract(repo_root, challenge_root, manifest_path, manifest)
        if args.selected_cases:
            unknown = sorted(set(args.selected_cases) - set(all_case_ids))
            if unknown:
                fail(f"unknown selected cases: {', '.join(unknown)}")
            selected = set(args.selected_cases)
            case_ids = [case_id for case_id in all_case_ids if case_id in selected]
        else:
            case_ids = all_case_ids

        output_root = args.output_root.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=False)
        receipt_root = output_root / "receipts"
        log_root = output_root / "logs"
        receipt_root.mkdir()
        log_root.mkdir()

        cases = []
        for case_id in case_ids:
            challenge = (challenge_root / case_id).resolve(strict=True)
            control_target = fresh_target(f"rapx-v2-challenge-control-{case_id}.")
            scan_target = fresh_target(f"rapx-v2-challenge-scan-{case_id}.")
            receipt_dir = receipt_root / case_id
            receipt_dir.mkdir()
            if any(receipt_dir.iterdir()):
                fail(f"{case_id}: receipt directory was not empty before scan")

            control_environment = os.environ.copy()
            control_environment["CARGO_BUILD_JOBS"] = "1"
            control_environment["CARGO_TARGET_DIR"] = str(control_target)
            run_command(
                [str(cargo), "check", "--lib", "--locked", "--jobs", "1"],
                challenge,
                control_environment,
                log_root / f"{case_id}.control.log",
            )

            scan_environment = os.environ.copy()
            scan_environment["UNSOUND_SCANNER_RAP_JSON_DIR"] = str(receipt_dir.resolve())
            scan_environment["UNSOUND_SCANNER_PROJECT_ROOT"] = str(challenge)
            scan_environment["CARGO_BUILD_JOBS"] = "1"
            scan_environment["CARGO_TARGET_DIR"] = str(scan_target)
            run_command(
                [str(cargo_rapx), "rapx", "-unsoundaudit", "--", "--lib", "--locked", "--jobs", "1"],
                challenge,
                scan_environment,
                log_root / f"{case_id}.scan.log",
            )
            receipts = sorted(receipt_dir.glob("*.json"))
            if len(receipts) != 1:
                fail(f"{case_id}: expected one unit receipt, found {len(receipts)}")
            receipt = validate_receipt(read_json(receipts[0]), schema)
            verify_receipt_toolchain(receipt, expected_rustc_commit)
            oracle = read_json(challenge / "fixture.json")
            validate_oracle(receipt, oracle)
            validate_challenge_assertions(case_id, receipt, oracle)
            cases.append(normalized_case(case_id, receipt))

        normalized = normalized_suite(cases)
        if not args.selected_cases:
            if normalized["challenge_count"] != CHALLENGE_COUNT:
                fail(
                    f"full challenge suite produced {normalized['challenge_count']} cases, "
                    f"expected {CHALLENGE_COUNT}"
                )
            if normalized["aggregate_pattern_counts"] != CHALLENGE_COUNTS:
                fail(
                    "full challenge aggregate differs from oracle: "
                    f"{normalized['aggregate_pattern_counts']}"
                )
        (output_root / "challenge_results.normalized.json").write_text(
            json.dumps(normalized, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (ContractError, OSError) as error:
        print(f"challenge suite error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

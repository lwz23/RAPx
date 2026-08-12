#!/usr/bin/env python3
"""Normalize validated unit receipts into the deterministic fixture-suite artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from run_fixture_suite import (
    BASELINE_MANIFEST_RELATIVE,
    baseline_toolchain,
    canonical_existing,
    normalized_case,
    verify_contract,
    verify_receipt_toolchain,
)
from validate_receipt import ContractError, read_json, validate_oracle, validate_receipt


def discover_repo_root(manifest_path: Path) -> Path:
    for candidate in manifest_path.parents:
        if (candidate / BASELINE_MANIFEST_RELATIVE).is_file() and (candidate / ".git").exists():
            return candidate.resolve(strict=True)
    raise ContractError("could not derive repository root from the fixture manifest")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest_path = canonical_existing(args.manifest, "fixture manifest")
        fixture_root = canonical_existing(args.fixture_root, "fixture root")
        receipt_root = canonical_existing(args.receipt_root, "receipt root")
        schema_path = canonical_existing(args.schema, "receipt schema")
        repo_root = discover_repo_root(manifest_path)
        schema = read_json(schema_path)
        manifest = read_json(manifest_path)
        _, expected_rustc_commit = baseline_toolchain(repo_root)
        case_ids = verify_contract(repo_root, fixture_root, manifest_path, manifest)
        receipt_ids = sorted(path.name for path in receipt_root.iterdir() if path.is_dir())
        if receipt_ids != case_ids:
            raise ContractError("receipt directories differ from the verified manifest")
        cases = []
        aggregate = {f"pattern{number}": 0 for number in range(1, 7)}
        for case_id in case_ids:
            fixture_dir = fixture_root / case_id
            oracle = read_json(fixture_dir / "fixture.json")
            receipt_files = sorted((receipt_root / case_id).glob("*.json"))
            if len(receipt_files) != 1:
                raise ContractError(f"{case_id}: expected one receipt, found {len(receipt_files)}")
            receipt = validate_receipt(read_json(receipt_files[0]), schema)
            verify_receipt_toolchain(receipt, expected_rustc_commit)
            validate_oracle(receipt, oracle)
            case = normalized_case(case_id, receipt)
            cases.append(case)
            for pattern, count in case["pattern_counts"].items():
                aggregate[pattern] += count
        if aggregate != manifest.get("aggregate_expected_pattern_counts"):
            raise ContractError("aggregate pattern counts differ from the fixture manifest")
        normalized = {
            "schema_version": "rap-fixture-suite-v2",
            "fixture_count": len(cases),
            "aggregate_pattern_counts": aggregate,
            "cases": cases,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(normalized, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    except (ContractError, OSError) as error:
        print(f"normalization error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

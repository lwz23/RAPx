#!/usr/bin/env python3
"""Normalize validated unit receipts into the deterministic fixture-suite artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from validate_receipt import ContractError, read_json, validate_oracle, validate_receipt


def normalized_case(case_id: str, receipt: dict) -> dict:
    return {
        "case_id": case_id,
        "package_name": receipt["package_name"],
        "package_version": receipt["package_version"],
        "crate_name": receipt["crate_name"],
        "crate_types": receipt["crate_types"],
        "target_kind": receipt["target_kind"],
        "target_triple": receipt["target_triple"],
        "cargo_supplied_rustc_commit": receipt["cargo_supplied_rustc_commit"],
        "rap_compiler_commit": receipt["rap_compiler_commit"],
        "rustc_commit": receipt["rustc_commit"],
        "pattern_counts": receipt["pattern_counts"],
        "finding_count": receipt["finding_count"],
        "findings": receipt["findings"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        schema = read_json(args.schema)
        manifest = read_json(args.manifest)
        rows = manifest.get("cases")
        if not isinstance(rows, list) or len(rows) != 27:
            raise ContractError("fixture manifest must contain exactly 27 cases")
        case_ids = sorted(row.get("case_id") for row in rows if isinstance(row, dict))
        if len(case_ids) != 27 or len(case_ids) != len(set(case_ids)) or not all(isinstance(case_id, str) and case_id for case_id in case_ids):
            raise ContractError("fixture manifest case ids are invalid")
        fixture_ids = sorted(path.name for path in args.fixture_root.iterdir() if path.is_dir())
        receipt_ids = sorted(path.name for path in args.receipt_root.iterdir() if path.is_dir())
        if fixture_ids != case_ids or receipt_ids != case_ids:
            raise ContractError("fixture or receipt directories differ from the manifest")
        cases = []
        aggregate = {f"pattern{number}": 0 for number in range(1, 7)}
        for case_id in case_ids:
            fixture_dir = args.fixture_root / case_id
            oracle = read_json(fixture_dir / "fixture.json")
            receipt_files = sorted((args.receipt_root / case_id).glob("*.json"))
            if len(receipt_files) != 1:
                raise ContractError(f"{case_id}: expected one receipt, found {len(receipt_files)}")
            receipt = validate_receipt(read_json(receipt_files[0]), schema)
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

#!/usr/bin/env python3
"""Run one bounded current-nightly core scan and build a review queue."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from noise_audit import family_id, family_key
from run_fixture_suite import (
    canonical_existing,
    sha256,
    toolchain_contract,
    verify_environment,
)
from run_stdlib_pilot import (
    PilotFailure,
    canonical_json,
    cargo_arguments,
    cargo_profile_contract,
    process_result_record,
    run_limited,
    scan_roots,
    select_unit_receipt,
    stdlib_source_root,
    verify_rap_binary_launch,
    write_driver,
)
from validate_receipt import ContractError, read_json


UNIT = "core"
STEPS = (("control", 0), ("scan", 1))
WALL_LIMIT_SECONDS = 900
PHYSICAL_LIMIT_BYTES = 8 * 1024**3
REVIEW_BUDGET = 20
HISTORICAL_POPULATION_RELATIVE = Path(
    "artifacts/unsoundaudit-v2/mac/stdlib-pilot/performance-v4/core.normalized.json"
)
CORE_CONTRACT = {
    "target_triple": "aarch64-apple-darwin",
    "units": {
        UNIT: {
            "build_std_components": [UNIT],
            "project_root": "library/core",
        }
    },
}


def fail(message: str) -> None:
    raise ContractError(message)


def output_names() -> tuple[str, ...]:
    return ("core.normalized.json", "core.summary.json", "core.execution.json")


def rustup_sysroot(channel: str, expected_commit: str) -> Path:
    rustup = shutil.which("rustup")
    if rustup is None:
        fail("rustup is unavailable for the historical source comparison")
    version = subprocess.run(
        [rustup, "run", channel, "rustc", "-Vv"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if version.returncode != 0 or f"commit-hash: {expected_commit}" not in version.stdout.splitlines():
        fail("historical rustc toolchain differs from the target contract")
    result = subprocess.run(
        [rustup, "run", channel, "rustc", "--print", "sysroot"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        fail("historical rustc sysroot query failed")
    return canonical_existing(Path(result.stdout.strip()), "historical rust sysroot")


def source_files(source_root: Path) -> dict[str, str]:
    core_root = canonical_existing(source_root / "library/core", "core source root")
    result = {}
    for path in sorted(core_root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            result[path.relative_to(source_root).as_posix()] = sha256(path)
    return result


def changed_source_files(historical_root: Path, current_root: Path) -> set[str]:
    historical = source_files(historical_root)
    current = source_files(current_root)
    return {
        path
        for path in set(historical) | set(current)
        if historical.get(path) != current.get(path)
    }


def finding_source_paths(finding: dict[str, Any]) -> set[str]:
    paths = set()
    for field in ("public_safe_root", "source", "first_contract_failure", "sink"):
        value = finding.get(field)
        span = value.get("span") if isinstance(value, dict) else None
        path = span.get("path") if isinstance(span, dict) else None
        if isinstance(path, str):
            paths.add(path)
    propagation = finding.get("propagation")
    witness = propagation.get("witness", []) if isinstance(propagation, dict) else []
    for step in witness:
        span = step.get("span") if isinstance(step, dict) else None
        path = span.get("path") if isinstance(span, dict) else None
        if isinstance(path, str):
            paths.add(path)
    return paths


def clustered_families(findings: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for finding in findings:
        key = family_key(UNIT, finding)
        identifier = family_id(key)
        previous = grouped.setdefault(identifier, key)
        if previous != key:
            fail("noise-family digest collision")
        members[identifier].append(finding)
    return {
        identifier: {
            "family_id": identifier,
            "family_key": grouped[identifier],
            "finding_count": len(rows),
            "finding_ids": sorted(row["finding_id"] for row in rows),
            "public_roots": sorted({row["public_safe_root"]["def_path"] for row in rows}),
            "source_paths": sorted(set().union(*(finding_source_paths(row) for row in rows))),
        }
        for identifier, rows in sorted(members.items())
    }


def differential_summary(
    current: dict[str, Any],
    historical: dict[str, Any],
    changed_files: set[str],
) -> dict[str, Any]:
    if historical.get("unit") != UNIT or not isinstance(historical.get("findings"), list):
        fail("historical core population is invalid")
    current_families = clustered_families(current["findings"])
    historical_families = clustered_families(historical["findings"])
    candidates = []
    unchanged = []
    for identifier, row in current_families.items():
        touched = sorted(set(row["source_paths"]) & changed_files)
        is_new = identifier not in historical_families
        if not is_new and not touched:
            unchanged.append(identifier)
            continue
        rule_id = row["family_key"]["rule_id"]
        raw_read = (
            rule_id == "P1.raw_read"
            or row["family_key"]["sink"]["kind"] == "raw_read"
        )
        rank = (
            0 if is_new else 1 if raw_read and touched else 2 if touched else 3,
            rule_id,
            identifier,
        )
        candidates.append(
            {
                **row,
                "new_family": is_new,
                "changed_source_files": touched,
                "review_rank": list(rank),
            }
        )
    candidates.sort(key=lambda row: tuple(row["review_rank"]))
    review_queue = [
        {**row, "status": "pending_feasibility_review"}
        for row in candidates[:REVIEW_BUDGET]
    ]
    remainder = [
        {**row, "status": "not_reviewed_feasibility_budget"}
        for row in candidates[REVIEW_BUDGET:]
    ]
    return {
        "schema_version": "unsoundaudit-v2-current-core-summary-v1",
        "unit": UNIT,
        "review_budget": REVIEW_BUDGET,
        "historical_finding_count": historical["finding_count"],
        "historical_family_count": len(historical_families),
        "current_finding_count": current["finding_count"],
        "current_family_count": len(current_families),
        "changed_source_file_count": len(changed_files),
        "candidate_family_count": len(candidates),
        "new_family_count": sum(row["new_family"] for row in candidates),
        "review_queue": review_queue,
        "not_reviewed_feasibility_budget": remainder,
        "not_reaudited_family_ids": sorted(unchanged),
        "classification_boundary": (
            "pending/not_reviewed/not_reaudited are prioritization states, not safety classifications"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--cargo", type=Path, required=True)
    parser.add_argument("--rap-bin-dir", type=Path, required=True)
    parser.add_argument("--rust-sysroot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        repo_root = canonical_existing(args.repo_root, "repository root")
        target_manifest = canonical_existing(args.target_manifest, "current target manifest")
        schema_path = canonical_existing(args.schema, "receipt schema")
        cargo = canonical_existing(args.cargo, "Cargo")
        rap_bin_dir = canonical_existing(args.rap_bin_dir, "RAP binary directory")
        rust_sysroot = canonical_existing(args.rust_sysroot, "current rust sysroot")
        output_root = args.output_root.expanduser().resolve()
        if output_root.exists():
            fail("current core output root must not already exist")
        if rap_bin_dir.name != "rapx-perf":
            fail("current core scan requires the rapx-perf binary directory")

        expected_toolchain = toolchain_contract(repo_root, target_manifest)
        cargo_rapx, rapx = verify_environment(
            repo_root, cargo, rap_bin_dir, expected_toolchain
        )
        cargo_profile_contract(repo_root / "rapx/Cargo.toml")
        verify_rap_binary_launch(cargo_rapx)
        target = read_json(target_manifest)
        if target["target_triple"] != CORE_CONTRACT["target_triple"]:
            fail("current core target triple drifted")
        current_source = stdlib_source_root(rust_sysroot)
        unit_root, current_source = scan_roots(current_source, CORE_CONTRACT, UNIT)
        historical_sysroot = rustup_sysroot(
            target["historical_baseline"]["channel"],
            target["historical_baseline"]["rustc_commit"],
        )
        historical_source = stdlib_source_root(historical_sysroot)
        changed_files = changed_source_files(historical_source, current_source)
        historical_population_path = canonical_existing(
            repo_root / HISTORICAL_POPULATION_RELATIVE,
            "historical v4 core population",
        )
        historical_population = read_json(historical_population_path)
        schema = read_json(schema_path)

        work_root = Path(tempfile.mkdtemp(prefix="rapx-current-core-feasibility.")).resolve()
        driver = work_root / "driver"
        logs = work_root / "logs"
        receipts = work_root / "receipts"
        driver.mkdir()
        logs.mkdir()
        receipts.mkdir()
        write_driver(driver, UNIT)
        arguments = cargo_arguments(CORE_CONTRACT, UNIT)
        executions = []
        receipt = None
        for phase, run_number in STEPS:
            target_dir = work_root / f"target-{phase}"
            target_dir.mkdir()
            environment = os.environ.copy()
            environment["CARGO_BUILD_JOBS"] = "1"
            environment["CARGO_TARGET_DIR"] = str(target_dir)
            if phase == "control":
                command = [str(cargo), "check", *arguments]
            else:
                environment["UNSOUND_SCANNER_RAP_JSON_DIR"] = str(receipts)
                environment["UNSOUND_SCANNER_PROJECT_ROOT"] = str(unit_root)
                environment["UNSOUND_SCANNER_SOURCE_ROOT"] = str(current_source)
                command = [str(cargo_rapx), "rapx", "-unsoundaudit", "--", *arguments]
            result = run_limited(
                command,
                driver,
                environment,
                logs / f"{phase}-{run_number}.log",
                WALL_LIMIT_SECONDS,
                PHYSICAL_LIMIT_BYTES,
                memory_limit_metric="physical_footprint",
            )
            executions.append(
                process_result_record(
                    phase, run_number, result, "physical_footprint"
                )
            )
            if result.classification != "success":
                raise PilotFailure(
                    result.classification,
                    f"current core {phase} {run_number} failed; work root: {work_root}",
                )
            if phase == "scan":
                receipt = select_unit_receipt(
                    receipts, schema, UNIT, expected_toolchain[1]
                )
        if receipt is None:
            fail("current core scan produced no selected receipt")

        normalized = {
            "schema_version": "unsoundaudit-v2-current-core-feasibility-v1",
            "unit": UNIT,
            "target_manifest_sha256": sha256(target_manifest),
            "toolchain": {
                "cargo_commit": expected_toolchain[0],
                "rustc_commit": expected_toolchain[1],
            },
            "pattern_counts": receipt["pattern_counts"],
            "finding_count": receipt["finding_count"],
            "findings": receipt["findings"],
        }
        summary = differential_summary(normalized, historical_population, changed_files)
        execution = {
            "schema_version": "unsoundaudit-v2-current-core-execution-v1",
            "unit": UNIT,
            "status": "success",
            "analyzer_profile": "rapx-perf",
            "cargo_rapx_sha256": sha256(cargo_rapx),
            "rapx_sha256": sha256(rapx),
            "memory_limit_metric": "physical_footprint",
            "wall_limit_seconds": WALL_LIMIT_SECONDS,
            "physical_limit_bytes": PHYSICAL_LIMIT_BYTES,
            "executions": executions,
            "raw_logs_committed": False,
        }
        output_root.mkdir(parents=True, exist_ok=False)
        (output_root / "core.normalized.json").write_bytes(canonical_json(normalized))
        (output_root / "core.summary.json").write_bytes(canonical_json(summary))
        (output_root / "core.execution.json").write_bytes(canonical_json(execution))
        return 0
    except (ContractError, OSError, ValueError) as error:
        print(f"current core feasibility error: {error}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

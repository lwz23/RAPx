#!/usr/bin/env python3
"""Run the frozen UnsoundAudit v2 fixtures serially and fail closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from normalize_receipts import normalized_case
from validate_receipt import ContractError, read_json, validate_oracle, validate_receipt


REQUIRED_ENVIRONMENT = (
    "CARGO_HOME",
    "LIBCLANG_PATH",
    "PKG_CONFIG_PATH",
    "DYLD_LIBRARY_PATH",
    "Z3_SYS_Z3_HEADER",
    "RUSTFLAGS",
    "RUST_SYSROOT",
    "UNSOUND_SCANNER_EXACT_CARGO_PATH",
    "UNSOUND_SCANNER_EXACT_CARGO_COMMIT",
    "RUSTC",
    "UNSOUND_SCANNER_EXPECTED_RUSTC_PATH",
    "UNSOUND_SCANNER_EXPECTED_RUSTC_COMMIT",
)
EXPECTED_FIXTURE_COUNT = 27
EXPECTED_COUNTS = {
    "pattern1": 3,
    "pattern2": 2,
    "pattern3": 2,
    "pattern4": 2,
    "pattern5": 1,
    "pattern6": 2,
}


def fail(message: str) -> None:
    raise ContractError(message)


def canonical_existing(path: Path, role: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        fail(f"{role} does not resolve: {error}")
    return resolved


def verify_environment(cargo: Path, rap_bin_dir: Path) -> tuple[Path, Path]:
    missing = [key for key in REQUIRED_ENVIRONMENT if not os.environ.get(key)]
    if missing:
        fail(f"exact environment is incomplete: {', '.join(missing)}")
    if os.environ.get("CARGO_ENCODED_RUSTFLAGS"):
        fail("CARGO_ENCODED_RUSTFLAGS must be unset for the exact environment")
    exact_cargo = canonical_existing(Path(os.environ["UNSOUND_SCANNER_EXACT_CARGO_PATH"]), "exact Cargo")
    if cargo != exact_cargo:
        fail("--cargo differs from UNSOUND_SCANNER_EXACT_CARGO_PATH")
    exact_rustc = canonical_existing(Path(os.environ["RUSTC"]), "exact rustc")
    expected_rustc = canonical_existing(Path(os.environ["UNSOUND_SCANNER_EXPECTED_RUSTC_PATH"]), "expected rustc")
    if exact_rustc != expected_rustc:
        fail("RUSTC differs from UNSOUND_SCANNER_EXPECTED_RUSTC_PATH")
    cargo_rapx = canonical_existing(rap_bin_dir / "cargo-rapx", "cargo-rapx")
    rapx = canonical_existing(rap_bin_dir / "rapx", "rapx")
    if cargo_rapx.parent != rapx.parent:
        fail("cargo-rapx and rapx must be sibling binaries")
    return cargo_rapx, rapx


def git_tracked(repo_root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(repo_root)
    except ValueError:
        return False
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", "--", relative.as_posix()],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_contract(repo_root: Path, fixture_root: Path, manifest_path: Path, manifest: dict) -> list[str]:
    case_rows = manifest.get("cases")
    if not isinstance(case_rows, list) or len(case_rows) != EXPECTED_FIXTURE_COUNT:
        fail(f"fixture manifest must contain exactly {EXPECTED_FIXTURE_COUNT} cases")
    if not all(isinstance(row, dict) and isinstance(row.get("case_id"), str) for row in case_rows):
        fail("fixture manifest cases must be objects with case_id")
    case_ids = [row["case_id"] for row in case_rows]
    if len(case_ids) != len(set(case_ids)) or not all(isinstance(case_id, str) and case_id for case_id in case_ids):
        fail("fixture manifest case ids must be unique non-empty strings")
    directories = sorted(path.name for path in fixture_root.iterdir() if path.is_dir())
    if sorted(case_ids) != directories:
        fail("fixture manifest and fixture directories differ")
    if manifest.get("aggregate_expected_pattern_counts") != EXPECTED_COUNTS:
        fail("fixture manifest aggregate oracle drifted")
    provenance = manifest.get("legacy_provenance_manifest")
    if not isinstance(provenance, dict) or set(provenance) != {"path", "sha256", "verified_file_count"}:
        fail("fixture manifest legacy provenance entry drifted")
    provenance_path = (manifest_path.parent / provenance["path"]).resolve(strict=True)
    if sha256(provenance_path) != provenance["sha256"] or provenance["verified_file_count"] != 31:
        fail("legacy provenance manifest hash or file count drifted")
    row_by_case = {row["case_id"]: row for row in case_rows}
    for case_id in sorted(case_ids):
        fixture = fixture_root / case_id
        required = (fixture / "Cargo.toml", fixture / "Cargo.lock", fixture / "src" / "lib.rs", fixture / "fixture.json")
        for path in required:
            if not path.is_file():
                fail(f"{case_id}: missing {path.relative_to(fixture)}")
            if not git_tracked(repo_root, path):
                fail(f"{case_id}: {path.relative_to(repo_root)} is not tracked by Git")
        if (fixture / "build.rs").exists():
            fail(f"{case_id}: build.rs is forbidden")
        cargo_text = (fixture / "Cargo.toml").read_text(encoding="utf-8")
        if "[dependencies]" in cargo_text or "[dev-dependencies]" in cargo_text or "[build-dependencies]" in cargo_text:
            fail(f"{case_id}: dependency sections are forbidden")
        oracle = read_json(fixture / "fixture.json")
        if oracle.get("case_id") != case_id:
            fail(f"{case_id}: oracle case_id differs from directory")
        row = row_by_case[case_id]
        expected_row_fields = {
            "case_id",
            "role",
            "package_name",
            "expected_primary_pattern",
            "expected_rule_id",
            "expected_pattern_counts",
            "files",
        }
        optional_row_fields = {"legacy_frozen_case", "paired_with"}
        if not expected_row_fields <= set(row) or not set(row) <= expected_row_fields | optional_row_fields:
            fail(f"{case_id}: fixture manifest row keys drifted")
        for key in ("role", "package_name", "expected_primary_pattern", "expected_rule_id", "expected_pattern_counts"):
            if row[key] != oracle.get(key):
                fail(f"{case_id}: fixture manifest {key} differs from oracle")
        file_rows = row["files"]
        if not isinstance(file_rows, dict) or set(file_rows) != {"manifest", "source", "lock", "oracle"}:
            fail(f"{case_id}: fixture file hash set drifted")
        expected_paths = {
            "manifest": fixture / "Cargo.toml",
            "source": fixture / "src" / "lib.rs",
            "lock": fixture / "Cargo.lock",
            "oracle": fixture / "fixture.json",
        }
        for role, path in expected_paths.items():
            file_row = file_rows[role]
            if not isinstance(file_row, dict) or set(file_row) != {"path", "sha256"}:
                fail(f"{case_id}: {role} hash entry drifted")
            expected_relative = path.relative_to(fixture_root.parent).as_posix()
            if file_row["path"] != expected_relative or sha256(path) != file_row["sha256"]:
                fail(f"{case_id}: {role} path or SHA-256 differs from fixture manifest")
    return sorted(case_ids)


def run_command(command: list[str], cwd: Path, environment: dict[str, str], log_path: Path) -> None:
    with log_path.open("wb") as log:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        fail(f"command failed with exit {result.returncode}; local log: {log_path}")


def fresh_target(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix, dir=os.environ.get("TMPDIR") or "/tmp")).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--cargo", type=Path, required=True)
    parser.add_argument("--rap-bin-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case", action="append", dest="selected_cases")
    args = parser.parse_args()
    try:
        repo_root = canonical_existing(args.repo_root, "repository root")
        fixture_root = canonical_existing(args.fixture_root, "fixture root")
        manifest_path = canonical_existing(args.manifest, "fixture manifest")
        schema_path = canonical_existing(args.schema, "receipt schema")
        cargo = canonical_existing(args.cargo, "Cargo")
        rap_bin_dir = canonical_existing(args.rap_bin_dir, "RAP binary directory")
        cargo_rapx, _ = verify_environment(cargo, rap_bin_dir)
        manifest = read_json(manifest_path)
        schema = read_json(schema_path)
        all_case_ids = verify_contract(repo_root, fixture_root, manifest_path, manifest)
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
        aggregate = {f"pattern{number}": 0 for number in range(1, 7)}
        for case_id in case_ids:
            fixture = (fixture_root / case_id).resolve(strict=True)
            control_target = fresh_target(f"rapx-v2-control-{case_id}.")
            scan_target = fresh_target(f"rapx-v2-scan-{case_id}.")
            receipt_dir = receipt_root / case_id
            receipt_dir.mkdir()
            if any(receipt_dir.iterdir()):
                fail(f"{case_id}: receipt directory was not empty before scan")

            control_environment = os.environ.copy()
            control_environment["CARGO_BUILD_JOBS"] = "1"
            control_environment["CARGO_TARGET_DIR"] = str(control_target)
            run_command(
                [str(cargo), "check", "--lib", "--locked", "--jobs", "1"],
                fixture,
                control_environment,
                log_root / f"{case_id}.control.log",
            )

            scan_environment = os.environ.copy()
            scan_environment["UNSOUND_SCANNER_RAP_JSON_DIR"] = str(receipt_dir.resolve())
            scan_environment["UNSOUND_SCANNER_PROJECT_ROOT"] = str(fixture)
            scan_environment["CARGO_BUILD_JOBS"] = "1"
            scan_environment["CARGO_TARGET_DIR"] = str(scan_target)
            run_command(
                [str(cargo_rapx), "rapx", "-unsoundaudit", "--", "--lib", "--locked", "--jobs", "1"],
                fixture,
                scan_environment,
                log_root / f"{case_id}.scan.log",
            )
            receipts = sorted(receipt_dir.glob("*.json"))
            if len(receipts) != 1:
                fail(f"{case_id}: expected one unit receipt, found {len(receipts)}")
            receipt = validate_receipt(read_json(receipts[0]), schema)
            oracle = read_json(fixture / "fixture.json")
            validate_oracle(receipt, oracle)
            case = normalized_case(case_id, receipt)
            cases.append(case)
            for pattern, count in case["pattern_counts"].items():
                aggregate[pattern] += count

        normalized = {
            "schema_version": "rap-fixture-suite-v2",
            "fixture_count": len(cases),
            "aggregate_pattern_counts": aggregate,
            "cases": cases,
        }
        if not args.selected_cases:
            if len(cases) != EXPECTED_FIXTURE_COUNT:
                fail(f"full suite produced {len(cases)} cases, expected {EXPECTED_FIXTURE_COUNT}")
            if aggregate != EXPECTED_COUNTS:
                fail(f"full suite aggregate differs from oracle: {aggregate}")
        (output_root / "fixture_results.normalized.json").write_text(
            json.dumps(normalized, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (ContractError, OSError) as error:
        print(f"fixture suite error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

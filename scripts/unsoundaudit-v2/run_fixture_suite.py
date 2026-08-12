#!/usr/bin/env python3
"""Run the frozen UnsoundAudit v2 fixtures serially and fail closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any

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
FORBIDDEN_DEPENDENCY_TABLES = frozenset({"dependencies", "dev-dependencies", "build-dependencies"})
BASELINE_MANIFEST_RELATIVE = Path("docs/unsoundaudit-v2/baseline_manifest_v1.json")
ENVIRONMENT_RECEIPT_RELATIVE = Path(
    "artifacts/unsoundaudit-v2/mac/exact-nightly-2024-10-12-arm64/environment_receipt.json"
)


def fail(message: str) -> None:
    raise ContractError(message)


def json_values_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(json_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(json_values_equal(a, b) for a, b in zip(left, right))
    return left == right


def valid_case_id(value: Any) -> bool:
    if not isinstance(value, str) or not value or "/" in value or "\\" in value or "\0" in value:
        return False
    pure = PurePosixPath(value)
    return (
        not pure.is_absolute()
        and re.match(r"^[A-Za-z]:", value) is None
        and pure.as_posix() == value
        and pure.parts == (value,)
        and value not in {".", ".."}
    )


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


def canonical_existing(path: Path, role: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        fail(f"{role} does not resolve: {error}")
    return resolved


def command_commit(executable: Path, role: str) -> str:
    result = subprocess.run(
        [str(executable), "-Vv"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        fail(f"{role} -Vv failed with exit {result.returncode}")
    fields = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition(": ")
        if separator:
            fields[key] = value
    commit = fields.get("commit-hash")
    if not isinstance(commit, str) or not commit:
        fail(f"{role} -Vv did not report commit-hash")
    return commit


def cargo_config_candidates(repo_root: Path, cargo_home: Path) -> tuple[Path, ...]:
    return (
        repo_root / ".cargo" / "config",
        repo_root / ".cargo" / "config.toml",
        cargo_home / "config",
        cargo_home / "config.toml",
        cargo_home / ".cargo" / "config",
        cargo_home / ".cargo" / "config.toml",
    )


def audit_cargo_configs(repo_root: Path, cargo_home: Path) -> None:
    for candidate in cargo_config_candidates(repo_root, cargo_home):
        if not candidate.exists():
            continue
        if not candidate.is_file():
            fail(f"Cargo configuration path is not a regular file: {candidate}")
        try:
            with candidate.open("rb") as source:
                config = tomllib.load(source)
        except (OSError, tomllib.TOMLDecodeError) as error:
            fail(f"Cargo configuration cannot be audited: {candidate}: {error}")

        def contains_rustflags(value: Any) -> bool:
            if not isinstance(value, dict):
                return False
            return "rustflags" in value or any(contains_rustflags(child) for child in value.values())

        if contains_rustflags(config.get("build")) or contains_rustflags(config.get("target")):
            fail(f"Cargo build/target configuration contains rustflags: {candidate}")


def audit_fixture_cargo_manifest(path: Path, case_id: str) -> None:
    try:
        with path.open("rb") as source:
            manifest = tomllib.load(source)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        fail(f"{case_id}: Cargo.toml cannot be audited: {error}")

    def contains_dependency_table(value: Any) -> bool:
        if isinstance(value, dict):
            return any(
                key in FORBIDDEN_DEPENDENCY_TABLES or contains_dependency_table(child)
                for key, child in value.items()
            )
        if isinstance(value, list):
            return any(contains_dependency_table(child) for child in value)
        return False

    if contains_dependency_table(manifest):
        fail(f"{case_id}: dependency sections are forbidden")


def approved_rustflags(repo_root: Path) -> str:
    receipt = read_json(repo_root / ENVIRONMENT_RECEIPT_RELATIVE)
    controls = receipt.get("environment_controls") if isinstance(receipt, dict) else None
    if (
        not isinstance(receipt, dict)
        or receipt.get("status") != "pass"
        or not isinstance(controls, dict)
        or controls.get("rustflags_role") != "z3_native_library_search_only"
    ):
        fail("Mac environment receipt does not approve the Z3-only RUSTFLAGS role")
    header = Path(os.environ["Z3_SYS_Z3_HEADER"]).expanduser()
    if not header.is_absolute() or ".." in header.parts or not header.is_file():
        fail("Z3_SYS_Z3_HEADER does not identify an existing file")
    if header.name != "z3.h" or header.parent.name != "include":
        fail("Z3_SYS_Z3_HEADER must identify the approved Z3 include/z3.h")
    expected_library = header.parent.parent / "lib"
    if not expected_library.is_dir():
        fail("approved Z3 library directory does not exist")
    return f"-Lnative={expected_library}"


def baseline_toolchain(repo_root: Path) -> tuple[str, str]:
    baseline = read_json(repo_root / BASELINE_MANIFEST_RELATIVE)
    toolchain = baseline.get("toolchain") if isinstance(baseline, dict) else None
    if not isinstance(toolchain, dict):
        fail("baseline manifest toolchain entry is missing")
    cargo_commit = toolchain.get("cargo_commit_hash")
    rustc_commit = toolchain.get("rustc_commit_hash")
    if not isinstance(cargo_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", cargo_commit):
        fail("baseline manifest Cargo commit is invalid")
    if not isinstance(rustc_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", rustc_commit):
        fail("baseline manifest rustc commit is invalid")
    return cargo_commit, rustc_commit


def verify_receipt_toolchain(receipt: dict, expected_rustc_commit: str) -> None:
    for field in ("cargo_supplied_rustc_commit", "rap_compiler_commit", "rustc_commit"):
        if receipt[field] != expected_rustc_commit:
            fail(f"receipt.{field} differs from the baseline rustc commit")


def verify_environment(repo_root: Path, cargo: Path, rap_bin_dir: Path) -> tuple[Path, Path]:
    missing = [key for key in REQUIRED_ENVIRONMENT if not os.environ.get(key)]
    if missing:
        fail(f"exact environment is incomplete: {', '.join(missing)}")
    if "CARGO_ENCODED_RUSTFLAGS" in os.environ:
        fail("CARGO_ENCODED_RUSTFLAGS must be unset for the exact environment")
    expected_cargo_commit, expected_rustc_commit = baseline_toolchain(repo_root)
    exact_cargo = canonical_existing(Path(os.environ["UNSOUND_SCANNER_EXACT_CARGO_PATH"]), "exact Cargo")
    if cargo != exact_cargo:
        fail("--cargo differs from UNSOUND_SCANNER_EXACT_CARGO_PATH")
    exact_rustc = canonical_existing(Path(os.environ["RUSTC"]), "exact rustc")
    expected_rustc = canonical_existing(Path(os.environ["UNSOUND_SCANNER_EXPECTED_RUSTC_PATH"]), "expected rustc")
    if exact_rustc != expected_rustc:
        fail("RUSTC differs from UNSOUND_SCANNER_EXPECTED_RUSTC_PATH")
    supplied_cargo_commit = os.environ["UNSOUND_SCANNER_EXACT_CARGO_COMMIT"]
    supplied_rustc_commit = os.environ["UNSOUND_SCANNER_EXPECTED_RUSTC_COMMIT"]
    if supplied_cargo_commit != expected_cargo_commit or command_commit(exact_cargo, "exact Cargo") != expected_cargo_commit:
        fail("exact Cargo commit differs from the baseline manifest")
    if supplied_rustc_commit != expected_rustc_commit or command_commit(exact_rustc, "exact rustc") != expected_rustc_commit:
        fail("exact rustc commit differs from the baseline manifest")
    if os.environ["RUSTFLAGS"] != approved_rustflags(repo_root):
        fail("RUSTFLAGS must contain only the approved Z3 native-library search path")
    cargo_home = canonical_existing(Path(os.environ["CARGO_HOME"]), "CARGO_HOME")
    audit_cargo_configs(repo_root, cargo_home)
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


def normalized_relative_path(value: Any, location: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        fail(f"{location} must be a non-empty normalized POSIX path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or re.match(r"^[A-Za-z]:/", value)
        or pure.as_posix() != value
        or "." in pure.parts
        or ".." in pure.parts
        or "//" in value
    ):
        fail(f"{location} must be a normalized relative POSIX path")
    return pure


def manifest_reference_path(value: Any, location: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value or "//" in value:
        fail(f"{location} must be a normalized POSIX path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or re.match(r"^[A-Za-z]:/", value) or pure.as_posix() != value or "/./" in f"/{value}/":
        fail(f"{location} must be a normalized relative POSIX path")
    return pure


def verify_legacy_provenance(repo_root: Path, manifest_path: Path, provenance: dict) -> None:
    repo_root = repo_root.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=False)
    provenance_relative = manifest_reference_path(provenance["path"], "legacy provenance path")
    try:
        provenance_path = (manifest_path.parent / provenance_relative).resolve(strict=True)
    except OSError as error:
        fail(f"legacy provenance manifest does not resolve: {error}")
    try:
        provenance_path.relative_to(repo_root)
    except ValueError:
        fail("legacy provenance manifest escapes the repository root")
    if sha256(provenance_path) != provenance["sha256"]:
        fail("legacy provenance manifest hash drifted")
    legacy = read_json(provenance_path)
    rows = legacy.get("files") if isinstance(legacy, dict) else None
    expected_count = provenance["verified_file_count"]
    if not isinstance(rows, list) or len(rows) != expected_count or expected_count != 31:
        fail("legacy provenance manifest must contain exactly 31 file rows")
    legacy_root = provenance_path.parent.resolve(strict=True)
    expected_paths: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"path", "size_bytes", "sha256"}:
            fail(f"legacy provenance files[{index}] keys drifted")
        if not isinstance(row["size_bytes"], int) or isinstance(row["size_bytes"], bool) or row["size_bytes"] < 0:
            fail(f"legacy provenance files[{index}].size_bytes is invalid")
        if not isinstance(row["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            fail(f"legacy provenance files[{index}].sha256 is invalid")
        relative = normalized_relative_path(row["path"], f"legacy provenance files[{index}].path")
        relative_text = relative.as_posix()
        if relative_text in expected_paths:
            fail(f"legacy provenance contains duplicate path: {relative_text}")
        expected_paths.add(relative_text)
        candidate = (legacy_root / relative).resolve(strict=False)
        try:
            candidate.relative_to(legacy_root)
        except ValueError:
            fail(f"legacy provenance path escapes its root: {relative_text}")
        if not candidate.is_file():
            fail(f"legacy provenance file is missing: {relative_text}")
        if candidate.stat().st_size != row["size_bytes"] or sha256(candidate) != row["sha256"]:
            fail(f"legacy provenance file size or SHA-256 differs: {relative_text}")
    actual_paths = {
        path.relative_to(legacy_root).as_posix()
        for path in legacy_root.rglob("*")
        if path.is_file() and path != provenance_path
    }
    if actual_paths != expected_paths:
        fail(
            "legacy provenance file set differs: "
            f"missing={sorted(expected_paths - actual_paths)}, extra={sorted(actual_paths - expected_paths)}"
        )


def verify_case_contract(
    repo_root: Path,
    case_root: Path,
    manifest_path: Path,
    manifest: dict,
    *,
    expected_schema: str,
    expected_count: int,
    expected_counts: dict[str, int],
    require_legacy_provenance: bool,
) -> list[str]:
    if not isinstance(expected_count, int) or isinstance(expected_count, bool) or expected_count <= 0:
        fail("expected case count must be a positive integer")
    if (
        not isinstance(expected_counts, dict)
        or set(expected_counts) != set(EXPECTED_COUNTS)
        or any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0
            for count in expected_counts.values()
        )
    ):
        fail("expected aggregate must contain exact non-negative integer pattern1-pattern6 counts")
    expected_manifest_keys = {
        "schema_version",
        "fixture_count",
        "role_counts",
        "aggregate_expected_pattern_counts",
        "cases",
    }
    if require_legacy_provenance:
        expected_manifest_keys.add("legacy_provenance_manifest")
    if not isinstance(manifest, dict) or set(manifest) != expected_manifest_keys:
        fail("case manifest top-level keys drifted")
    if manifest["schema_version"] != expected_schema:
        fail("case manifest schema drifted")
    fixture_count = manifest["fixture_count"]
    if not isinstance(fixture_count, int) or isinstance(fixture_count, bool) or fixture_count <= 0:
        fail("case manifest fixture_count must be a positive integer")
    if fixture_count != expected_count:
        fail(f"case manifest fixture_count must be exactly {expected_count}")
    manifest_counts = manifest["aggregate_expected_pattern_counts"]
    if (
        not isinstance(manifest_counts, dict)
        or set(manifest_counts) != set(expected_counts)
        or any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in manifest_counts.values()
        )
    ):
        fail("case manifest aggregate counts must be exact non-negative integers")
    if manifest_counts != expected_counts:
        fail("case manifest aggregate oracle drifted")

    manifest_role_counts = manifest["role_counts"]
    if (
        not isinstance(manifest_role_counts, dict)
        or not manifest_role_counts
        or any(not isinstance(role, str) or not role for role in manifest_role_counts)
        or any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0
            for count in manifest_role_counts.values()
        )
    ):
        fail("case manifest role counts must use non-empty keys and non-negative integer values")

    case_rows = manifest.get("cases")
    if not isinstance(case_rows, list) or len(case_rows) != expected_count:
        fail(f"case manifest must contain exactly {expected_count} cases")
    if not all(isinstance(row, dict) and valid_case_id(row.get("case_id")) for row in case_rows):
        fail("case manifest cases must be objects with case_id")
    case_ids = [row["case_id"] for row in case_rows]
    if len(case_ids) != len(set(case_ids)):
        fail("case manifest case ids must be unique non-empty directory names")
    directories = sorted(path.name for path in case_root.iterdir() if path.is_dir())
    if sorted(case_ids) != directories:
        fail("case manifest and case directories differ")

    role_counts: dict[str, int] = {}
    aggregate = {pattern: 0 for pattern in expected_counts}
    row_by_case = {row["case_id"]: row for row in case_rows}
    for case_id in sorted(case_ids):
        fixture = case_root / case_id
        if fixture.is_symlink():
            fail(f"{case_id}: fixture directory must not be a symbolic link")
        try:
            resolved_fixture = fixture.resolve(strict=True)
            resolved_fixture.relative_to(case_root.resolve(strict=True))
        except (OSError, ValueError) as error:
            fail(f"{case_id}: fixture directory escapes its root: {error}")
        required = (fixture / "Cargo.toml", fixture / "Cargo.lock", fixture / "src" / "lib.rs", fixture / "fixture.json")
        for path in required:
            if path.is_symlink() or not path.is_file():
                fail(f"{case_id}: missing {path.relative_to(fixture)}")
            try:
                path.resolve(strict=True).relative_to(resolved_fixture)
            except (OSError, ValueError) as error:
                fail(f"{case_id}: {path.relative_to(fixture)} escapes its fixture: {error}")
            if not git_tracked(repo_root, path):
                fail(f"{case_id}: {path.relative_to(repo_root)} is not tracked by Git")
        if (fixture / "build.rs").exists():
            fail(f"{case_id}: build.rs is forbidden")
        audit_fixture_cargo_manifest(fixture / "Cargo.toml", case_id)
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
            fail(f"{case_id}: case manifest row keys drifted")
        for key in ("role", "package_name", "expected_primary_pattern", "expected_rule_id", "expected_pattern_counts"):
            if key not in oracle or not json_values_equal(row[key], oracle[key]):
                fail(f"{case_id}: case manifest {key} differs from oracle")
        role = row["role"]
        if not isinstance(role, str) or not role:
            fail(f"{case_id}: role must be a non-empty string")
        role_counts[role] = role_counts.get(role, 0) + 1
        counts = row["expected_pattern_counts"]
        if not isinstance(counts, dict) or set(counts) != set(expected_counts):
            fail(f"{case_id}: expected pattern-count keys drifted")
        for pattern, count in counts.items():
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                fail(f"{case_id}: {pattern} count must be a non-negative integer")
            aggregate[pattern] += count
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
            expected_relative = path.relative_to(case_root.parent).as_posix()
            if (
                file_row["path"] != expected_relative
                or not isinstance(file_row["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", file_row["sha256"])
                or sha256(path) != file_row["sha256"]
            ):
                fail(f"{case_id}: {role} path or SHA-256 differs from case manifest")
    if set(manifest_role_counts) != set(role_counts) or manifest_role_counts != role_counts:
        fail("case manifest role totals drifted")
    if aggregate != expected_counts:
        fail("case rows do not sum to the expected aggregate")
    return sorted(case_ids)


def verify_contract(repo_root: Path, fixture_root: Path, manifest_path: Path, manifest: dict) -> list[str]:
    provenance = manifest.get("legacy_provenance_manifest") if isinstance(manifest, dict) else None
    if not isinstance(provenance, dict) or set(provenance) != {"path", "sha256", "verified_file_count"}:
        fail("fixture manifest legacy provenance entry drifted")
    verify_legacy_provenance(repo_root, manifest_path, provenance)
    return verify_case_contract(
        repo_root,
        fixture_root,
        manifest_path,
        manifest,
        expected_schema="unsoundaudit-v2-fixture-manifest-v1",
        expected_count=EXPECTED_FIXTURE_COUNT,
        expected_counts=EXPECTED_COUNTS,
        require_legacy_provenance=True,
    )


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
        cargo_rapx, _ = verify_environment(repo_root, cargo, rap_bin_dir)
        manifest = read_json(manifest_path)
        schema = read_json(schema_path)
        _, expected_rustc_commit = baseline_toolchain(repo_root)
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
            verify_receipt_toolchain(receipt, expected_rustc_commit)
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

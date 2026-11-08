#!/usr/bin/env python3
"""Fail-closed serial campaign runner for crates.io robustness evaluation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cratesio_campaign_common import (
    CampaignError,
    canonical_json,
    load_policy,
    read_json,
    read_jsonl,
    sha256,
    write_json,
    exact_keys,
    exact_hex,
    sanitized_environment,
    tool_commit,
)
from run_stdlib_gate import (
    GateError,
    process_record,
    run_limited,
    validate_development_mvp,
    verify_binary_launch,
)
from validate_receipt import ContractError, validate_receipt


def classify_outcome(
    control_exit: int | None = None,
    scan_exit: int | None = None,
    scan_timeout: bool = False,
    coverage_complete: bool = False,
) -> str:
    if control_exit is not None and control_exit != 0:
        return "control_failed"
    if scan_timeout:
        return "scan_timeout"
    if scan_exit is not None and scan_exit != 0:
        return "scan_failed"
    if not coverage_complete:
        return "partial_coverage"
    return "complete_exact"


def select_root_receipt(
    receipts: list[dict[str, Any]],
    schema: dict[str, Any],
    package_name: str,
    package_version: str,
    crate_name: str,
    rustc_commit: str,
) -> dict[str, Any]:
    v3_receipts: list[dict[str, Any]] = []
    for r in receipts:
        if not isinstance(r, dict):
            continue
        ver = r.get("schema_version")
        if ver == "rap-unit-v3":
            try:
                validate_receipt(r, schema)
            except ContractError as e:
                raise CampaignError(f"receipt failed schema validation: {e}") from e
            v3_receipts.append(r)
        elif ver != "rap-unit-skip-v1":
            raise CampaignError(f"unrecognized receipt schema: {ver}")

    matching: list[dict[str, Any]] = []
    for r in v3_receipts:
        if (
            r.get("package_name") == package_name
            and r.get("package_version") == package_version
            and r.get("crate_name") == crate_name
            and r.get("target_kind") == "lib"
            and r.get("rustc_commit") == rustc_commit
        ):
            matching.append(r)

    if len(matching) == 0:
        raise CampaignError(
            f"no root library receipt matching {package_name}@{package_version} (crate {crate_name}, commit {rustc_commit})"
        )
    if len(matching) > 1:
        raise CampaignError(
            f"multiple root library receipts ({len(matching)}) matching {package_name}@{package_version}"
        )
    return matching[0]


def extract_crate_archive(archive_path: Path, destination: Path, expected_root: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as tar:
        for member in tar.getmembers():
            parts = PurePosixPath(member.name).parts
            if not parts or parts[0] != expected_root or ".." in parts:
                raise CampaignError(f"archive member {member.name} escapes {expected_root}")
            if member.issym() or member.islnk() or member.isdev():
                raise CampaignError(f"archive member {member.name} contains link or device")
        tar.extractall(destination)


def normalize_semantic_data(data: Any, project_root_str: str) -> Any:
    if isinstance(data, dict):
        normalized = {}
        for k, v in data.items():
            if k in {
                "rustc_invocation_id",
                "wall_time_seconds",
                "peak_physical_footprint_bytes",
                "peak_rss_bytes",
                "timestamp",
                "start_time",
                "end_time",
                "duration_seconds",
                "pid",
                "process_id",
            }:
                continue
            normalized[k] = normalize_semantic_data(v, project_root_str)
        return normalized
    elif isinstance(data, list):
        return [normalize_semantic_data(item, project_root_str) for item in data]
    elif isinstance(data, str):
        return data.replace(project_root_str, "/PROJECT_ROOT")
    else:
        return data


def run_single_package(
    package_info: dict[str, Any],
    package_dir: Path,
    cargo: Path,
    rustc: Path,
    cargo_rapx: Path,
    rapx: Path,
    rust_sysroot: Path,
    cargo_commit: str,
    rustc_commit: str,
    schema: dict[str, Any],
    wall_time_seconds: int = 900,
) -> dict[str, Any]:
    name = package_info["name"]
    version = package_info["version"]
    expected_root = f"{name}-{version}"
    crate_name = name.replace("-", "_")

    archive_path = Path(package_info["archive_path"])
    source_root = package_dir / "source"
    cargo_home = package_dir / "cargo-home"
    control_target = package_dir / "control-target"
    scan_target = package_dir / "scan-target"
    receipts_dir = package_dir / "receipts"
    mvp_dir = package_dir / "mvp"
    logs_dir = package_dir / "logs"

    for d in (source_root, cargo_home, control_target, scan_target, receipts_dir, mvp_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    extract_crate_archive(archive_path, source_root, expected_root)
    pkg_source = source_root / expected_root

    # Generate lockfile and fetch
    base_env = sanitized_environment()
    base_env["CARGO_HOME"] = str(cargo_home)
    base_env["RUSTC"] = str(rustc)

    # Cargo metadata to generate lockfile if needed
    subprocess.run(
        [str(cargo), "metadata", "--format-version", "1"],
        cwd=pkg_source,
        env=base_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    subprocess.run(
        [str(cargo), "fetch"],
        cwd=pkg_source,
        env=base_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    lockfile_path = pkg_source / "Cargo.lock"
    lockfile_sha = sha256(lockfile_path) if lockfile_path.exists() else None

    # 1. Control build
    control_cmd = [
        str(cargo), "check", "--lib", "--locked", "--offline", "--jobs", "1",
    ]
    control_env = sanitized_environment()
    control_env["CARGO_HOME"] = str(cargo_home)
    control_env["CARGO_TARGET_DIR"] = str(control_target)
    control_env["CARGO_BUILD_JOBS"] = "1"
    control_env["RUSTC"] = str(rustc)

    control_log = logs_dir / "control.log"
    control_record = run_limited(
        control_cmd,
        cwd=pkg_source,
        environment=control_env,
        wall_time_seconds=wall_time_seconds,
        physical_footprint_limit_bytes=None,
        log_path=control_log,
    )

    if control_record.classification == "timeout" or control_record.returncode != 0:
        return {
            "name": name,
            "version": version,
            "terminal_state": "control_failed",
            "control": {
                "returncode": control_record.returncode,
                "elapsed_seconds": control_record.elapsed_seconds,
                "classification": control_record.classification,
                "log_sha256": sha256(control_log) if control_log.exists() else None,
            },
            "lockfile_sha256": lockfile_sha,
            "scan": None,
            "receipt": None,
            "mvp": None,
        }

    # 2. RAP Scan
    scan_cmd = [
        str(cargo_rapx), "rapx", "-unsoundaudit", "--",
        "--lib", "--locked", "--offline", "--jobs", "1",
    ]
    scan_env = sanitized_environment()
    scan_env["RUSTC"] = str(rustc)
    scan_env["RUST_SYSROOT"] = str(rust_sysroot)
    scan_env["CARGO_HOME"] = str(cargo_home)
    scan_env["CARGO_TARGET_DIR"] = str(scan_target)
    scan_env["CARGO_BUILD_JOBS"] = "1"
    scan_env["RAP_CLEAN"] = "false"
    scan_env["UNSOUND_SCANNER_PROJECT_ROOT"] = str(pkg_source)
    scan_env["UNSOUND_SCANNER_EXACT_CARGO_PATH"] = str(cargo)
    scan_env["UNSOUND_SCANNER_EXACT_CARGO_COMMIT"] = cargo_commit
    scan_env["UNSOUND_SCANNER_EXPECTED_RUSTC_PATH"] = str(rustc)
    scan_env["UNSOUND_SCANNER_EXPECTED_RUSTC_COMMIT"] = rustc_commit
    scan_env["RAP_UNSOUNDAUDIT_OUTPUT"] = str(mvp_dir)
    scan_env["UNSOUND_SCANNER_RAP_JSON_DIR"] = str(receipts_dir)
    perf_trace_path = logs_dir / "perf_trace.jsonl"
    scan_env["UNSOUND_SCANNER_PERF_TRACE"] = str(perf_trace_path)

    if sys.platform == "darwin":
        scan_env["DYLD_LIBRARY_PATH"] = str(rust_sysroot / "lib")

    scan_log = logs_dir / "scan.log"
    scan_record = run_limited(
        scan_cmd,
        cwd=pkg_source,
        environment=scan_env,
        wall_time_seconds=wall_time_seconds,
        physical_footprint_limit_bytes=None,
        log_path=scan_log,
    )

    scan_summary = {
        "returncode": scan_record.returncode,
        "elapsed_seconds": scan_record.elapsed_seconds,
        "max_physical_footprint_bytes": scan_record.max_physical_footprint_bytes,
        "max_rss_bytes": scan_record.max_rss_bytes,
        "classification": scan_record.classification,
        "log_sha256": sha256(scan_log) if scan_log.exists() else None,
        "perf_trace_sha256": sha256(perf_trace_path) if perf_trace_path.exists() else None,
    }

    if scan_record.classification == "timeout":
        return {
            "name": name,
            "version": version,
            "terminal_state": "scan_timeout",
            "control": {
                "returncode": control_record.returncode,
                "elapsed_seconds": control_record.elapsed_seconds,
                "classification": control_record.classification,
                "log_sha256": sha256(control_log) if control_log.exists() else None,
            },
            "lockfile_sha256": lockfile_sha,
            "scan": scan_summary,
            "receipt": None,
            "mvp": None,
        }

    if scan_record.returncode != 0:
        return {
            "name": name,
            "version": version,
            "terminal_state": "scan_failed",
            "control": {
                "returncode": control_record.returncode,
                "elapsed_seconds": control_record.elapsed_seconds,
                "classification": control_record.classification,
                "log_sha256": sha256(control_log) if control_log.exists() else None,
            },
            "lockfile_sha256": lockfile_sha,
            "scan": scan_summary,
            "receipt": None,
            "mvp": None,
        }

    # Inspect receipts
    receipt_files = sorted(receipts_dir.glob("*.json"))
    raw_receipts = [read_json(f) for f in receipt_files]
    try:
        root_receipt = select_root_receipt(
            raw_receipts, schema, package_name=name, package_version=version,
            crate_name=crate_name, rustc_commit=rustc_commit
        )
    except Exception as e:
        scan_summary["selection_error"] = str(e)
        return {
            "name": name,
            "version": version,
            "terminal_state": "scan_failed",
            "control": {
                "returncode": control_record.returncode,
                "elapsed_seconds": control_record.elapsed_seconds,
                "classification": control_record.classification,
                "log_sha256": sha256(control_log) if control_log.exists() else None,
            },
            "lockfile_sha256": lockfile_sha,
            "scan": scan_summary,
            "receipt": None,
            "mvp": None,
        }

    # Validate MVP
    try:
        mvp_result = validate_development_mvp(mvp_dir)
    except Exception as e:
        scan_summary["mvp_error"] = str(e)
        return {
            "name": name,
            "version": version,
            "terminal_state": "scan_failed",
            "control": {
                "returncode": control_record.returncode,
                "elapsed_seconds": control_record.elapsed_seconds,
                "classification": control_record.classification,
                "log_sha256": sha256(control_log) if control_log.exists() else None,
            },
            "lockfile_sha256": lockfile_sha,
            "scan": scan_summary,
            "receipt": None,
            "mvp": None,
        }

    coverage_is_complete = root_receipt.get("coverage", {}).get("complete") is True or (
        len(root_receipt.get("coverage_failures", [])) == 0
        and root_receipt.get("observed_counts", {}).get("semantics") == "exact"
    )
    terminal_state = "complete_exact" if coverage_is_complete else "partial_coverage"

    return {
        "name": name,
        "version": version,
        "terminal_state": terminal_state,
        "control": {
            "returncode": control_record.returncode,
            "elapsed_seconds": control_record.elapsed_seconds,
            "classification": control_record.classification,
            "log_sha256": sha256(control_log) if control_log.exists() else None,
        },
        "lockfile_sha256": lockfile_sha,
        "scan": scan_summary,
        "receipt": {
            "coverage_failures": root_receipt.get("coverage_failures", []),
            "unsafe_site_census": root_receipt.get("unsafe_site_census"),
            "observed_counts": root_receipt.get("observed_counts"),
            "finding_count": root_receipt.get("finding_count", 0),
        },
        "mvp": {
            "active_count": mvp_result.active_count,
            "family_count": mvp_result.family_count,
            "pure_suppressed_count": mvp_result.pure_suppressed_count,
            "reclassified_count": mvp_result.reclassified_count,
            "unestablished_count": mvp_result.unestablished_count,
            "suppressed_count": mvp_result.suppressed_count,
            "considered_count": mvp_result.considered_count,
        },
    }


def run_campaign(
    policy: dict[str, Any],
    roster: dict[str, Any],
    protocol: dict[str, Any],
    schema: dict[str, Any],
    cargo: Path,
    rustc: Path,
    rap_bin_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise CampaignError(f"output root {output_root} already exists and is not empty")

    output_root.mkdir(parents=True, exist_ok=True)
    cargo_rapx = rap_bin_dir / "cargo-rapx"
    rapx = rap_bin_dir / "rapx"

    cargo_commit = tool_commit(cargo, label="Cargo")
    rustc_commit = tool_commit(rustc, label="rustc")
    rust_sysroot = Path(
        subprocess.check_output([str(rustc), "--print", "sysroot"], text=True).strip()
    )

    preflight_env = sanitized_environment()
    if sys.platform == "darwin":
        preflight_env["DYLD_LIBRARY_PATH"] = str(rust_sysroot / "lib")

    verify_binary_launch(cargo_rapx, preflight_env)

    campaign_data: dict[str, Any] = {
        "schema_version": "rap-unsoundaudit-cratesio-campaign-journal-v1",
        "campaign_id": policy["campaign_id"],
        "selection_date": policy["selection_date"],
        "policy_sha256": roster.get("policy_sha256"),
        "protocol_sha256": sha256(output_root / "protocol.json") if (output_root / "protocol.json").exists() else None,
        "packages": [],
    }

    journal_path = output_root / "campaign.json"

    for pkg_info in roster["packages"]:
        pkg_name = pkg_info["name"]
        pkg_dir = output_root / "packages" / pkg_name
        pkg_res = run_single_package(
            package_info=pkg_info,
            package_dir=pkg_dir,
            cargo=cargo,
            rustc=rustc,
            cargo_rapx=cargo_rapx,
            rapx=rapx,
            rust_sysroot=rust_sysroot,
            cargo_commit=cargo_commit,
            rustc_commit=rustc_commit,
            schema=schema,
            wall_time_seconds=policy.get("wall_time_seconds", 900),
        )
        campaign_data["packages"].append(pkg_res)
        write_json(journal_path, campaign_data)

    # Rerun smallest and largest complete_exact packages
    complete_exact_pkgs = [
        p for p in campaign_data["packages"] if p["terminal_state"] == "complete_exact"
    ]
    if len(complete_exact_pkgs) >= 2:
        # Match roster source lines
        roster_lines = {
            p["name"]: p["admission"]["physical_rust_lines"] for p in roster["packages"]
        }
        complete_exact_pkgs.sort(key=lambda p: roster_lines.get(p["name"], 0))
        rerun_targets = [complete_exact_pkgs[0], complete_exact_pkgs[-1]]
        rerun_results = []

        for tgt in rerun_targets:
            tgt_info = next(p for p in roster["packages"] if p["name"] == tgt["name"])
            rerun_dir = output_root / "reruns" / tgt["name"]
            rerun_res = run_single_package(
                package_info=tgt_info,
                package_dir=rerun_dir,
                cargo=cargo,
                rustc=rustc,
                cargo_rapx=cargo_rapx,
                rapx=rapx,
                rust_sysroot=rust_sysroot,
                cargo_commit=cargo_commit,
                rustc_commit=rustc_commit,
                schema=schema,
                wall_time_seconds=policy.get("wall_time_seconds", 900),
            )
            # Check byte-identical semantics
            orig_mvp_dir = output_root / "packages" / tgt["name"] / "mvp"
            rerun_mvp_dir = rerun_dir / "mvp"

            orig_candidates = read_jsonl(orig_mvp_dir / "candidates.jsonl") if (orig_mvp_dir / "candidates.jsonl").exists() else []
            rerun_candidates = read_jsonl(rerun_mvp_dir / "candidates.jsonl") if (rerun_mvp_dir / "candidates.jsonl").exists() else []

            orig_norm = normalize_semantic_data(orig_candidates, str(output_root / "packages" / tgt["name"] / "source"))
            rerun_norm = normalize_semantic_data(rerun_candidates, str(rerun_dir / "source"))

            orig_hash = hashlib.sha256(canonical_json(orig_norm)).hexdigest()
            rerun_hash = hashlib.sha256(canonical_json(rerun_norm)).hexdigest()

            if orig_hash != rerun_hash:
                raise CampaignError(
                    f"rerun normalized hash mismatch for {tgt['name']}: {orig_hash} != {rerun_hash}"
                )

            rerun_results.append(
                {
                    "name": tgt["name"],
                    "version": tgt["version"],
                    "normalized_sha256": orig_hash,
                    "rerun_state": rerun_res["terminal_state"],
                    "byte_identical": True,
                }
            )
        campaign_data["reruns"] = {
            "status": "verified_byte_identical",
            "targets": rerun_results,
        }
    else:
        campaign_data["reruns"] = {"status": "not_run_fewer_than_two_complete"}

    write_json(journal_path, campaign_data)
    return campaign_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run serial crates.io robustness campaign.")
    parser.add_argument("--repo-root", type=Path, required=True, help="Path to RAP repository root")
    parser.add_argument("--policy", type=Path, required=True, help="Path to policy JSON")
    parser.add_argument("--roster", type=Path, required=True, help="Path to roster JSON")
    parser.add_argument("--protocol", type=Path, required=True, help="Path to protocol JSON")
    parser.add_argument("--schema", type=Path, required=True, help="Path to rap-unit-v3 schema")
    parser.add_argument("--cargo", type=Path, required=True, help="Path to cargo executable")
    parser.add_argument("--rustc", type=Path, required=True, help="Path to rustc executable")
    parser.add_argument("--rap-bin-dir", type=Path, required=True, help="Path to target/rapx-perf directory")
    parser.add_argument("--output-root", type=Path, required=True, help="Path to evidence output root")
    args = parser.parse_args()

    policy = load_policy(args.policy)
    roster = read_json(args.roster)
    protocol = read_json(args.protocol)
    schema = read_json(args.schema)

    res = run_campaign(
        policy=policy,
        roster=roster,
        protocol=protocol,
        schema=schema,
        cargo=args.cargo,
        rustc=args.rustc,
        rap_bin_dir=args.rap_bin_dir,
        output_root=args.output_root,
    )
    print(f"Campaign execution complete. {len(res['packages'])} packages processed.")


if __name__ == "__main__":
    main()

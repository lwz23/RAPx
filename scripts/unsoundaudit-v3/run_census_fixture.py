#!/usr/bin/env python3
"""Run one compiler-aligned census fixture twice and check its frozen oracle."""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def fail(message: str) -> "NoReturn":
    raise SystemExit(message)


def tool_commit(path: Path) -> str:
    result = subprocess.run(
        [str(path), "-Vv"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        fail(f"{path} -Vv failed with exit {result.returncode}")
    match = re.search(r"^commit-hash:\s*([0-9a-f]{40})$", result.stdout, re.MULTILINE)
    if match is None:
        fail(f"{path} -Vv did not contain a full commit hash")
    return match.group(1)


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"failed to read {path}: {error}")
    if not isinstance(value, dict):
        fail(f"{path} must contain a JSON object")
    return value


def site_projection(payload: dict) -> list[list[object]]:
    projected: list[list[object]] = []
    for owner in payload["owners"]:
        for site in owner["sites"]:
            projected.append(
                [
                    site["kind"],
                    site["owner"],
                    site["span"]["path"],
                    site["span"]["start"]["line"],
                ]
            )
    return sorted(projected)


def validate_payload(payload: dict, oracle: dict) -> None:
    if set(payload) != {
        "schema_version",
        "generation",
        "owners",
        "coverage_failures",
    }:
        fail("census sidecar top-level keys drifted")
    if payload["schema_version"] != "unsafe-site-census-v1" or payload["generation"] != 1:
        fail("census sidecar schema or invocation generation drifted")
    if payload["coverage_failures"] != []:
        fail(f"census fixture has coverage failures: {payload['coverage_failures']!r}")
    owners = payload["owners"]
    if len(owners) != oracle["owner_count"]:
        fail(f"owner count drifted: expected {oracle['owner_count']}, got {len(owners)}")
    sites = [site for owner in owners for site in owner["sites"]]
    if len(sites) != oracle["site_count"]:
        fail(f"site count drifted: expected {oracle['site_count']}, got {len(sites)}")
    actual_counts = dict(sorted(collections.Counter(site["kind"] for site in sites).items()))
    if actual_counts != dict(sorted(oracle["kind_counts"].items())):
        fail(f"unsafe-site kind counts drifted: {actual_counts!r}")
    if site_projection(payload) != sorted(oracle["sites"]):
        fail("unsafe-site kind/owner/span projection drifted")
    owner_rows = {owner["owner"]: owner for owner in owners}
    for owner in oracle["zero_site_owners"]:
        if owner not in owner_rows or owner_rows[owner]["sites"] != []:
            fail(f"safe/exempt owner is missing or contains unsafe sites: {owner}")
    ids = [site["site_id"] for site in sites]
    if len(ids) != len(set(ids)) or not all(re.fullmatch(r"[0-9a-f]{64}", site_id) for site_id in ids):
        fail("site IDs must be unique lowercase SHA-256 values")
    if any(site["adapter_id"] != oracle["adapter_id"] for site in sites):
        fail("site adapter ID drifted")


def run_compile_fail(rustc: Path, fixture: Path, rows: list[dict], output_root: Path) -> None:
    for index, row in enumerate(rows):
        source = fixture / row["path"]
        result = subprocess.run(
            [
                str(rustc),
                "--crate-name",
                f"census_compile_fail_{index}",
                "--crate-type",
                "lib",
                "--edition",
                "2024",
                "--error-format=json",
                "--out-dir",
                str(output_root / "compile-fail"),
                str(source),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 or f'"code":"{row["diagnostic_code"]}"' not in result.stderr:
            fail(f"compile-fail oracle did not produce {row['diagnostic_code']}: {source}")


def run_once(
    fixture: Path,
    output_root: Path,
    run_name: str,
    cargo_rapx: Path,
    cargo: Path,
    cargo_commit: str,
    rustc: Path,
    rustc_commit: str,
    rustdoc: Path,
    rust_sysroot: Path,
    census_only: bool,
) -> bytes:
    run_root = output_root / run_name
    receipts = run_root / "receipts"
    for directory in (run_root, receipts, run_root / "cargo-home", run_root / "target"):
        directory.mkdir(parents=True, exist_ok=False)
    census = run_root / "census.json"
    environment = os.environ.copy()
    environment.update(
        {
            "CARGO_HOME": str(run_root / "cargo-home"),
            "CARGO_TARGET_DIR": str(run_root / "target"),
            "CARGO_BUILD_JOBS": "1",
            "RUSTC": str(rustc),
            "RUSTDOC": str(rustdoc),
            "RUST_SYSROOT": str(rust_sysroot),
            "UNSOUND_SCANNER_EXACT_CARGO_PATH": str(cargo),
            "UNSOUND_SCANNER_EXACT_CARGO_COMMIT": cargo_commit,
            "UNSOUND_SCANNER_EXPECTED_RUSTC_PATH": str(rustc),
            "UNSOUND_SCANNER_EXPECTED_RUSTC_COMMIT": rustc_commit,
            "UNSOUND_SCANNER_RAP_BUILD_RUSTC_COMMIT": rustc_commit,
            "UNSOUND_SCANNER_PROJECT_ROOT": str(fixture),
            "UNSOUND_SCANNER_JSON_DIR": str(receipts),
            "UNSOUND_SCANNER_CENSUS_JSON_OUT": str(census),
        }
    )
    if census_only:
        environment["UNSOUND_SCANNER_CENSUS_ONLY"] = "1"
    else:
        environment.pop("UNSOUND_SCANNER_CENSUS_ONLY", None)
    result = subprocess.run(
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
        cwd=fixture,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (run_root / "scan.log").write_bytes(result.stdout)
    if result.returncode != 0 or not census.is_file():
        fail(f"{run_name} failed with exit {result.returncode}; see {run_root / 'scan.log'}")
    return census.read_bytes()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cargo-rapx", type=Path, required=True)
    parser.add_argument("--cargo", type=Path, required=True)
    parser.add_argument("--rustc", type=Path, required=True)
    parser.add_argument("--rustdoc", type=Path, required=True)
    parser.add_argument("--rust-sysroot", type=Path, required=True)
    args = parser.parse_args()

    fixture = args.fixture.resolve(strict=True)
    output_root = args.output_root.resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=False)
    oracle = load_json(fixture / "oracle.json")
    rustc_commit = tool_commit(args.rustc.resolve(strict=True))
    if rustc_commit != oracle["rustc_commit"]:
        fail(f"rustc commit differs from fixture oracle: {rustc_commit}")
    cargo = args.cargo.resolve(strict=True)
    cargo_commit = tool_commit(cargo)
    cargo_rapx = args.cargo_rapx.resolve(strict=True)
    rustdoc = args.rustdoc.resolve(strict=True)
    rust_sysroot = args.rust_sysroot.resolve(strict=True)

    first = run_once(
        fixture,
        output_root,
        "run-1",
        cargo_rapx,
        cargo,
        cargo_commit,
        args.rustc.resolve(strict=True),
        rustc_commit,
        rustdoc,
        rust_sysroot,
        False,
    )
    second = run_once(
        fixture,
        output_root,
        "run-2",
        cargo_rapx,
        cargo,
        cargo_commit,
        args.rustc.resolve(strict=True),
        rustc_commit,
        rustdoc,
        rust_sysroot,
        True,
    )
    if first != second:
        fail("full-analysis and census-only runs are not byte-identical")
    payload = json.loads(first)
    validate_payload(payload, oracle)
    compile_fail = oracle.get("compile_fail", [])
    (output_root / "compile-fail").mkdir(exist_ok=False)
    run_compile_fail(args.rustc, fixture, compile_fail, output_root)
    print(
        json.dumps(
            {
                "adapter_id": oracle["adapter_id"],
                "owners": oracle["owner_count"],
                "sites": oracle["site_count"],
                "two_run_byte_identical": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

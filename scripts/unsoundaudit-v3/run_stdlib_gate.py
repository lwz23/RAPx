#!/usr/bin/env python3
"""Run the compiler-aligned v3 analyzer on bounded standard-library units."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path
from typing import Any, Callable, NamedTuple

from validate_receipt import ContractError, read_json, validate_receipt


PROTOCOL_SCHEMA = "unsoundaudit-v3-stdlib-gate-protocol-v1"
BINARY_PROFILE_SCHEMA = "unsoundaudit-v3-rap-binary-profile-v1"
PROFILE_SIDECAR = "rapx-build-profile.json"
PROFILE_CONTRACT = {
    "profile": "rapx-perf",
    "opt_level": 3,
    "debug": 1,
    "debug_assertions": True,
    "overflow_checks": True,
    "incremental": False,
    "target_cargo_profile": "dev",
}
PROTOCOL_KEYS = {
    "schema_version",
    "analyzer_profile",
    "rap_source_commit",
    "rustc_commit",
    "cargo_commit",
    "target_triple",
    "unit_order",
    "runs_per_unit",
    "cargo_jobs",
    "fresh_targets",
    "trace_enabled",
    "units",
}
UNIT_KEYS = {
    "wall_time_seconds",
    "physical_footprint_limit_bytes",
    "build_std_components",
    "project_root",
}
PROFILE_KEYS = {
    "schema_version",
    *PROFILE_CONTRACT,
    "rap_source_commit",
    "rustc_commit",
    "cargo_rapx_sha256",
    "rapx_sha256",
}
KNOWN_UNITS = ("core", "alloc", "std")
RUSAGE_INFO_V4 = 4


class GateError(ValueError):
    """Raised when execution or evidence violates the closed resource gate."""


def fail(message: str) -> None:
    raise GateError(message)


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_hex(value: Any, width: int, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(f"[0-9a-f]{{{width}}}", value) is None:
        fail(f"{label} must be {width} lowercase hexadecimal characters")
    return value


def load_protocol(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict) or set(value) != PROTOCOL_KEYS:
        fail("stdlib gate protocol keys drifted")
    if value["schema_version"] != PROTOCOL_SCHEMA:
        fail("stdlib gate protocol schema drifted")
    if value["analyzer_profile"] != PROFILE_CONTRACT["profile"]:
        fail("stdlib gate analyzer profile drifted")
    exact_hex(value["rap_source_commit"], 40, "rap_source_commit")
    exact_hex(value["rustc_commit"], 40, "rustc_commit")
    exact_hex(value["cargo_commit"], 40, "cargo_commit")
    if value["target_triple"] != "aarch64-apple-darwin":
        fail("stdlib gate target triple drifted")
    if (
        value["runs_per_unit"] != 2
        or value["cargo_jobs"] != 1
        or value["fresh_targets"] is not True
        or value["trace_enabled"] is not False
    ):
        fail("stdlib gate execution contract drifted")
    order = value["unit_order"]
    units = value["units"]
    if (
        not isinstance(order, list)
        or not order
        or len(order) != len(set(order))
        or any(unit not in KNOWN_UNITS for unit in order)
        or not isinstance(units, dict)
        or set(units) != set(order)
    ):
        fail("stdlib gate unit set drifted")
    for unit in order:
        row = units[unit]
        if not isinstance(row, dict) or set(row) != UNIT_KEYS:
            fail("stdlib gate unit contract drifted")
        if (
            isinstance(row["wall_time_seconds"], bool)
            or not isinstance(row["wall_time_seconds"], int)
            or row["wall_time_seconds"] <= 0
            or isinstance(row["physical_footprint_limit_bytes"], bool)
            or not isinstance(row["physical_footprint_limit_bytes"], int)
            or row["physical_footprint_limit_bytes"] <= 0
            or not isinstance(row["build_std_components"], list)
            or not row["build_std_components"]
            or row["build_std_components"] != list(dict.fromkeys(row["build_std_components"]))
            or not isinstance(row["project_root"], str)
            or row["project_root"] != f"library/{unit}"
        ):
            fail("stdlib gate unit values drifted")
    return value


def cargo_profile_contract(manifest_path: Path) -> dict[str, Any]:
    try:
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise GateError(f"RAP analyzer profile manifest is unreadable: {error}") from error
    expected = {
        "inherits": "release",
        "opt-level": 3,
        "debug": 1,
        "debug-assertions": True,
        "overflow-checks": True,
        "incremental": False,
    }
    if manifest.get("profile", {}).get("rapx-perf") != expected:
        fail("RAP analyzer profile contract drifted")
    return PROFILE_CONTRACT.copy()


def load_binary_profile(
    binary_directory: Path,
    cargo_rapx: Path,
    rapx: Path,
    rap_source_commit: str,
    rustc_commit: str,
) -> dict[str, Any]:
    sidecar = binary_directory / PROFILE_SIDECAR
    if not sidecar.is_file():
        fail("RAP analyzer build-profile sidecar is missing")
    value = read_json(sidecar)
    if not isinstance(value, dict) or set(value) != PROFILE_KEYS:
        fail("RAP analyzer build-profile sidecar keys drifted")
    expected = {
        "schema_version": BINARY_PROFILE_SCHEMA,
        **PROFILE_CONTRACT,
        "rap_source_commit": rap_source_commit,
        "rustc_commit": rustc_commit,
        "cargo_rapx_sha256": sha256(cargo_rapx),
        "rapx_sha256": sha256(rapx),
    }
    if value != expected:
        fail("RAP analyzer build-profile contract or binary hash drifted")
    return value


def cargo_arguments(protocol: dict[str, Any], unit: str) -> list[str]:
    if unit not in protocol["units"]:
        fail(f"unit {unit} is not registered by this protocol")
    components = protocol["units"][unit]["build_std_components"]
    return [
        "-Z",
        f"build-std={','.join(components)}",
        "--target",
        protocol["target_triple"],
        "--lib",
        "--locked",
        "--jobs",
        "1",
    ]


def driver_files(unit: str) -> dict[str, str]:
    if unit not in KNOWN_UNITS:
        fail(f"unsupported standard-library unit: {unit}")
    package = f"unsoundaudit_v3_stdlib_driver_{unit}"
    source = {
        "core": "#![no_std]\n\npub fn gate_marker() {}\n",
        "alloc": "#![no_std]\nextern crate alloc;\n\npub fn gate_marker() {}\n",
        "std": "pub fn gate_marker() {}\n",
    }[unit]
    manifest = (
        "[package]\n"
        f"name = \"{package}\"\n"
        "version = \"0.0.0\"\n"
        "edition = \"2021\"\n"
        "publish = false\n\n"
        "[lib]\npath = \"src/lib.rs\"\n"
    )
    lock = (
        "# This file is automatically @generated by Cargo.\n"
        "version = 3\n\n"
        "[[package]]\n"
        f"name = \"{package}\"\n"
        "version = \"0.0.0\"\n"
    )
    return {"Cargo.toml": manifest, "Cargo.lock": lock, "src/lib.rs": source}


def write_driver(root: Path, unit: str) -> None:
    if any(root.iterdir()):
        fail("temporary driver directory must be empty")
    for relative, text in driver_files(unit).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def parse_verbose_commit(output: str, tool: str) -> str:
    matches = re.findall(r"^commit-hash:\s*([0-9a-f]{40})$", output, flags=re.MULTILINE)
    if len(matches) != 1:
        fail(f"cannot determine exact {tool} commit")
    return matches[0]


def tool_commit(path: Path, tool: str) -> str:
    result = subprocess.run(
        [str(path), "-Vv"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        fail(f"{tool} identity command failed")
    return parse_verbose_commit(result.stdout, tool)


def verify_source_commit(repo_root: Path, commit: str) -> None:
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=repo_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    unchanged = subprocess.run(
        ["git", "diff", "--quiet", commit, "HEAD", "--", "rapx"],
        cwd=repo_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if ancestor.returncode != 0 or unchanged.returncode != 0:
        fail("current RAP source differs from the protocol source commit")


def stdlib_source_root(rust_sysroot: Path) -> Path:
    root = (rust_sysroot / "lib/rustlib/src/rust").resolve(strict=True)
    required = [root / "library/Cargo.lock", *(root / f"library/{unit}/Cargo.toml" for unit in KNOWN_UNITS)]
    if any(not path.is_file() for path in required):
        fail("exact rust-src standard-library tree is incomplete")
    return root


def scan_roots(source_root: Path, protocol: dict[str, Any], unit: str) -> tuple[Path, Path]:
    canonical_source = source_root.resolve(strict=True)
    unit_root = (canonical_source / protocol["units"][unit]["project_root"]).resolve(strict=True)
    try:
        unit_root.relative_to(canonical_source)
    except ValueError as error:
        raise GateError("standard-library unit root escapes source root") from error
    return unit_root, canonical_source


class ProcessResult(NamedTuple):
    classification: str
    returncode: int | None
    elapsed_seconds: float
    max_physical_footprint_bytes: int
    max_rss_bytes: int


class ProcessGroupUsage(NamedTuple):
    physical_footprint_bytes: int
    rss_bytes: int
    sampled_processes: int


class ProcessDisappeared(ProcessLookupError):
    pass


class RusageInfoV4(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
        ("ri_cpu_time_qos_default", ctypes.c_uint64),
        ("ri_cpu_time_qos_maintenance", ctypes.c_uint64),
        ("ri_cpu_time_qos_background", ctypes.c_uint64),
        ("ri_cpu_time_qos_utility", ctypes.c_uint64),
        ("ri_cpu_time_qos_legacy", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_initiated", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_interactive", ctypes.c_uint64),
        ("ri_billed_system_time", ctypes.c_uint64),
        ("ri_serviced_system_time", ctypes.c_uint64),
        ("ri_logical_writes", ctypes.c_uint64),
        ("ri_lifetime_max_phys_footprint", ctypes.c_uint64),
        ("ri_instructions", ctypes.c_uint64),
        ("ri_cycles", ctypes.c_uint64),
        ("ri_billed_energy", ctypes.c_uint64),
        ("ri_serviced_energy", ctypes.c_uint64),
        ("ri_interval_max_phys_footprint", ctypes.c_uint64),
        ("ri_runnable_time", ctypes.c_uint64),
    ]


class DarwinPhysicalFootprint:
    def __init__(
        self,
        *,
        platform: str = sys.platform,
        library_loader: Callable[..., Any] = ctypes.CDLL,
    ) -> None:
        if platform != "darwin":
            fail("physical footprint sampling is unsupported")
        try:
            library = library_loader("/usr/lib/libproc.dylib", use_errno=True)
            function = library.proc_pid_rusage
            function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            function.restype = ctypes.c_int
        except (AttributeError, OSError) as error:
            raise GateError("physical footprint sampling API is unavailable") from error
        self._function = function

    def __call__(self, pid: int) -> int:
        usage = RusageInfoV4()
        ctypes.set_errno(0)
        if self._function(pid, RUSAGE_INFO_V4, ctypes.byref(usage)) != 0:
            number = ctypes.get_errno()
            if number == errno.ESRCH:
                raise ProcessDisappeared(pid)
            raise OSError(number, os.strerror(number), pid)
        return int(usage.ri_phys_footprint)


def parse_process_group_rows(output: str) -> list[tuple[int, int, int]]:
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        try:
            if len(fields) != 3:
                raise ValueError
            pid, group, rss = (int(field) for field in fields)
            if pid <= 0 or group <= 0 or rss < 0:
                raise ValueError
        except ValueError as error:
            raise GateError("process-group monitor output parse failed") from error
        rows.append((pid, group, rss))
    return rows


def aggregate_process_group_usage(
    output: str,
    process_group: int,
    footprint_reader: Callable[[int], int],
) -> ProcessGroupUsage:
    rows = [row for row in parse_process_group_rows(output) if row[1] == process_group]
    footprint = 0
    sampled = 0
    for pid, _, _ in rows:
        try:
            value = footprint_reader(pid)
            if value < 0:
                raise ValueError
        except ProcessDisappeared:
            continue
        except (OSError, ValueError) as error:
            raise GateError("physical footprint sampling failed") from error
        footprint += value
        sampled += 1
    return ProcessGroupUsage(footprint, sum(row[2] for row in rows) * 1024, sampled)


def process_group_usage(pid: int, reader: Callable[[int], int]) -> ProcessGroupUsage:
    result = subprocess.run(
        ["ps", "-axo", "pid=,pgid=,rss="],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        fail("process-group monitor failed")
    return aggregate_process_group_usage(result.stdout, pid, reader)


def terminate_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    time.sleep(0.2)
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_limited(
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    wall_time_seconds: float,
    physical_footprint_limit_bytes: int,
    *,
    usage_reader: Callable[[int], ProcessGroupUsage] | None = None,
) -> ProcessResult:
    reader = DarwinPhysicalFootprint() if usage_reader is None else None
    read_usage = usage_reader or (lambda pid: process_group_usage(pid, reader))
    start = time.monotonic()
    maximum_footprint = 0
    maximum_rss = 0
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        classification = "compiler_failure"
        returncode: int | None = None
        try:
            while True:
                returncode = process.poll()
                elapsed = time.monotonic() - start
                if returncode is not None:
                    classification = "success" if returncode == 0 else "compiler_failure"
                    break
                usage = read_usage(process.pid)
                if usage.sampled_processes <= 0:
                    if process.poll() is not None:
                        returncode = process.returncode
                        classification = "success" if returncode == 0 else "compiler_failure"
                        break
                    fail("running process group was absent from physical-footprint monitor")
                maximum_footprint = max(maximum_footprint, usage.physical_footprint_bytes)
                maximum_rss = max(maximum_rss, usage.rss_bytes)
                if elapsed > wall_time_seconds:
                    classification = "timeout"
                    terminate_group(process.pid)
                    returncode = process.wait()
                    break
                if maximum_footprint > physical_footprint_limit_bytes:
                    classification = "memory_limit"
                    terminate_group(process.pid)
                    returncode = process.wait()
                    break
                time.sleep(0.1)
        finally:
            if process.poll() is None:
                terminate_group(process.pid)
                process.wait()
    return ProcessResult(
        classification,
        returncode,
        time.monotonic() - start,
        maximum_footprint,
        maximum_rss,
    )


def normalized_unit(unit: str, receipt: dict[str, Any], protocol_sha256: str) -> dict[str, Any]:
    if receipt.get("analysis_status") != "complete":
        fail(f"{unit}: partial analysis is not admissible")
    if receipt.get("coverage_failures") != []:
        fail(f"{unit}: complete analysis contains coverage failures")
    observed = receipt.get("observed_counts")
    if not isinstance(observed, dict) or observed.get("semantics") != "exact":
        fail(f"{unit}: observed counts are not exact")
    return {
        "schema_version": "unsoundaudit-v3-stdlib-unit-v1",
        "unit": unit,
        "protocol_sha256": protocol_sha256,
        "analysis_status": "complete",
        "toolchain_adapter": receipt["toolchain_adapter"],
        "unsafe_site_census": receipt["unsafe_site_census"],
        "unsafe_sites": receipt["unsafe_sites"],
        "observed_counts": observed,
        "finding_count": receipt["finding_count"],
        "findings": receipt["findings"],
        "code_dictionary": receipt["code_dictionary"],
    }


def select_unit_receipt(
    receipt_dir: Path,
    schema: dict[str, Any],
    unit: str,
    rustc_commit: str,
) -> dict[str, Any]:
    selected = []
    for path in sorted(receipt_dir.glob("*.json")):
        value = read_json(path)
        version = value.get("schema_version") if isinstance(value, dict) else None
        if version == "rap-unit-v3":
            validate_receipt(value, schema)
            selected.append(value)
        elif version != "rap-unit-skip-v1":
            fail(f"unrecognized receipt schema: {path.name}")
    if len(selected) != 1:
        fail(f"expected exactly one complete {unit} receipt, found {len(selected)}")
    receipt = selected[0]
    if receipt["crate_name"] != unit or receipt["analysis_status"] != "complete":
        fail(f"selected receipt is not a complete {unit} analysis")
    for field in ("cargo_supplied_rustc_commit", "rap_compiler_commit", "rustc_commit"):
        if receipt[field] != rustc_commit:
            fail(f"selected receipt {field} differs from protocol")
    return receipt


def process_record(phase: str, run: int, result: ProcessResult) -> dict[str, Any]:
    return {
        "phase": phase,
        "run": run,
        "classification": result.classification,
        "returncode": result.returncode,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "memory_limit_metric": "physical_footprint",
        "max_physical_footprint_bytes": result.max_physical_footprint_bytes,
        "max_rss_bytes": result.max_rss_bytes,
    }


def verify_binary_launch(cargo_rapx: Path) -> None:
    result = subprocess.run(
        [str(cargo_rapx), "rapx", "--help"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        fail("RAP binary launch preflight failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--cargo", type=Path, required=True)
    parser.add_argument("--rap-bin-dir", type=Path, required=True)
    parser.add_argument("--rust-sysroot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--unit", choices=KNOWN_UNITS, required=True)
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_created = False
    executions: list[dict[str, Any]] = []
    try:
        if os.environ.get("UNSOUND_SCANNER_PERF_TRACE"):
            fail("formal stdlib gate requires performance trace to be disabled")
        repo_root = args.repo_root.expanduser().resolve(strict=True)
        protocol_path = args.protocol.expanduser().resolve(strict=True)
        schema_path = args.schema.expanduser().resolve(strict=True)
        cargo = args.cargo.expanduser().resolve(strict=True)
        binary_directory = args.rap_bin_dir.expanduser().resolve(strict=True)
        cargo_rapx = (binary_directory / "cargo-rapx").resolve(strict=True)
        rapx = (binary_directory / "rapx").resolve(strict=True)
        rust_sysroot = args.rust_sysroot.expanduser().resolve(strict=True)
        rustc = (rust_sysroot / "bin/rustc").resolve(strict=True)
        protocol = load_protocol(protocol_path)
        if args.unit not in protocol["units"]:
            fail(f"unit {args.unit} is outside this protocol")
        if tool_commit(cargo, "Cargo") != protocol["cargo_commit"]:
            fail("Cargo commit differs from protocol")
        if tool_commit(rustc, "rustc") != protocol["rustc_commit"]:
            fail("rustc commit differs from protocol")
        cargo_profile_contract(repo_root / "rapx/Cargo.toml")
        verify_source_commit(repo_root, protocol["rap_source_commit"])
        profile = load_binary_profile(
            binary_directory,
            cargo_rapx,
            rapx,
            protocol["rap_source_commit"],
            protocol["rustc_commit"],
        )
        verify_binary_launch(cargo_rapx)
        schema = read_json(schema_path)
        source_root = stdlib_source_root(rust_sysroot)
        unit_root, source_root = scan_roots(source_root, protocol, args.unit)
        if output_root.exists():
            fail("stdlib gate output root must not already exist")
        output_root.mkdir(parents=True, exist_ok=False)
        output_created = True
        (output_root / "logs").mkdir()
        (output_root / "receipts").mkdir()
        protocol_sha = sha256(protocol_path)
        unit_contract = protocol["units"][args.unit]
        normalized_runs: list[bytes] = []
        with tempfile.TemporaryDirectory(prefix=f"rapx-v3-stdlib-driver-{args.unit}.") as text:
            driver = Path(text).resolve()
            write_driver(driver, args.unit)
            arguments = cargo_arguments(protocol, args.unit)
            for phase, run in [("control", 0), ("scan", 1), ("scan", 2)]:
                target = Path(tempfile.mkdtemp(prefix=f"rapx-v3-stdlib-{args.unit}-{phase}-{run}."))
                try:
                    environment = os.environ.copy()
                    environment.update({
                        "CARGO_BUILD_JOBS": "1",
                        "CARGO_TARGET_DIR": str(target),
                        "RUSTC": str(rustc),
                        "RUST_SYSROOT": str(rust_sysroot),
                        "UNSOUND_SCANNER_EXACT_CARGO_PATH": str(cargo),
                        "UNSOUND_SCANNER_EXACT_CARGO_COMMIT": protocol["cargo_commit"],
                        "UNSOUND_SCANNER_EXPECTED_RUSTC_PATH": str(rustc),
                        "UNSOUND_SCANNER_EXPECTED_RUSTC_COMMIT": protocol["rustc_commit"],
                    })
                    if phase == "control":
                        command = [str(cargo), "check", *arguments]
                    else:
                        receipt_dir = output_root / "receipts" / f"run-{run}"
                        receipt_dir.mkdir()
                        environment.update({
                            "UNSOUND_SCANNER_PROJECT_ROOT": str(unit_root),
                            "UNSOUND_SCANNER_SOURCE_ROOT": str(source_root),
                            "UNSOUND_SCANNER_RAP_JSON_DIR": str(receipt_dir),
                        })
                        command = [str(cargo_rapx), "rapx", "-unsoundaudit", "--", *arguments]
                    result = run_limited(
                        command,
                        driver,
                        environment,
                        output_root / "logs" / f"{phase}-{run}.log",
                        unit_contract["wall_time_seconds"],
                        unit_contract["physical_footprint_limit_bytes"],
                    )
                    executions.append(process_record(phase, run, result))
                    if result.classification != "success":
                        fail(f"{phase} {run} failed: {result.classification}")
                    if phase == "scan":
                        receipt = select_unit_receipt(
                            receipt_dir, schema, args.unit, protocol["rustc_commit"]
                        )
                        normalized_runs.append(
                            canonical_json(normalized_unit(args.unit, receipt, protocol_sha))
                        )
                finally:
                    shutil.rmtree(target, ignore_errors=True)
        if len(normalized_runs) != 2 or normalized_runs[0] != normalized_runs[1]:
            fail("normalized v3 receipts are nondeterministic")
        (output_root / f"{args.unit}.normalized.json").write_bytes(normalized_runs[0])
        summary = {
            "schema_version": "unsoundaudit-v3-stdlib-gate-result-v1",
            "unit": args.unit,
            "status": "success",
            "protocol_sha256": protocol_sha,
            "analyzer_profile": profile,
            "executions": executions,
            "normalized_sha256": hashlib.sha256(normalized_runs[0]).hexdigest(),
        }
        (output_root / f"{args.unit}.resource.json").write_bytes(canonical_json(summary))
    except (ContractError, GateError, OSError, json.JSONDecodeError) as error:
        if output_created:
            failure = {
                "schema_version": "unsoundaudit-v3-stdlib-gate-failure-v1",
                "unit": args.unit,
                "status": "failed",
                "reason": str(error),
                "executions": executions,
            }
            (output_root / f"{args.unit}.failure.json").write_bytes(canonical_json(failure))
        print(f"v3 stdlib gate error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

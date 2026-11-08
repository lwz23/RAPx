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
from typing import Any, Callable, Iterator, NamedTuple

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
DEVELOPMENT_RESULT_SCHEMA = "unsoundaudit-v3-stdlib-development-single-run-v1"
DEVELOPMENT_FAILURE_SCHEMA = "unsoundaudit-v3-stdlib-development-failure-v1"
DEVELOPMENT_CORE_WALL_SECONDS = 900
DEVELOPMENT_CORE_PHYSICAL_LIMIT_BYTES = 8 * 1024**3
DEVELOPMENT_FAMILY_TARGET = 100
MVP_COVERAGE_SCHEMA = "rap-unsoundaudit-coverage-v3-mvp"
MVP_FAMILY_SCHEMA = "rap-unsoundaudit-family-v3-mvp"
MVP_SUPPRESSION_SCHEMA = "rap-unsoundaudit-suppression-v3-mvp"
MVP_UNESTABLISHED_SCHEMA = "rap-unsoundaudit-unestablished-v1"
MVP_ACCOUNTING_FIELDS = {
    "active_count",
    "pure_suppressed_count",
    "reclassified_count",
    "unestablished_count",
    "suppressed_count",
    "considered_count",
}
VALID_PATTERNS = {f"P{number}" for number in range(1, 7)}
VALID_SUPPRESSION_REASONS = {
    "contract_implied_by_type",
    "pointer_view_has_no_capability_gap",
    "root_has_no_relevant_capability_surface",
    "nonpersistent_public_field",
    "trait_not_downstream_implementable",
    "internal_behavior_controller",
    "unrealizable_call_context",
    "source_cannot_satisfy_requirement",
    "contract_discharged",
}
UNESTABLISHED_REASON_ORDER = (
    "missing_exact_context",
    "incomplete_normalization",
    "unresolved_controller",
    "unmodeled_unsafe_call",
    "unknown_capability_match",
    "opaque_transform",
    "havoced_provenance",
    "validation_state_unknown",
    "validity_obligation_unknown",
    "initialization_state_unknown",
    "derivation_obligation_unknown",
    "generic_witness_unknown",
)
UNESTABLISHED_REASON_RANK = {
    reason: index for index, reason in enumerate(UNESTABLISHED_REASON_ORDER)
}
DEVELOPMENT_ENVIRONMENT_EXACT_REMOVALS = {
    "CARGO_BUILD_JOBS",
    "CARGO_BUILD_TARGET",
    "CARGO_ENCODED_RUSTFLAGS",
    "CARGO_HOME",
    "CARGO_INCREMENTAL",
    "CARGO_TARGET_DIR",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "MACOSX_DEPLOYMENT_TARGET",
    "RAP_UNSOUNDAUDIT_OUTPUT",
    "RUSTC",
    "RUSTC_BOOTSTRAP",
    "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
    "RUSTDOCFLAGS",
    "RUSTFLAGS",
    "RUSTUP_TOOLCHAIN",
}


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


def load_development_binary_profile(
    binary_directory: Path,
    cargo_rapx: Path,
    rapx: Path,
    rustc_commit: str,
) -> dict[str, Any]:
    """Validate every binary-profile field except the frozen RAP source commit.

    This is intentionally separate from ``load_binary_profile`` so the formal
    default cannot accidentally inherit the development-only relaxation.
    """

    sidecar = binary_directory / PROFILE_SIDECAR
    if not sidecar.is_file():
        fail("RAP analyzer build-profile sidecar is missing")
    try:
        value = read_json(sidecar)
    except UnicodeError as error:
        raise GateError("RAP analyzer build-profile sidecar is not UTF-8") from error
    if not isinstance(value, dict) or set(value) != PROFILE_KEYS:
        fail("RAP analyzer build-profile sidecar keys drifted")
    source_commit = exact_hex(
        value.get("rap_source_commit"), 40, "binary profile rap_source_commit"
    )
    expected = {
        "schema_version": BINARY_PROFILE_SCHEMA,
        **PROFILE_CONTRACT,
        "rap_source_commit": source_commit,
        "rustc_commit": rustc_commit,
        "cargo_rapx_sha256": sha256(cargo_rapx),
        "rapx_sha256": sha256(rapx),
    }
    if value != expected:
        fail("development RAP profile, rustc commit, or binary hash drifted")
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


def development_environment(
    inherited: dict[str, str],
    *,
    cargo: Path,
    rustc: Path,
    rust_sysroot: Path,
    cargo_commit: str,
    rustc_commit: str,
    cargo_home: Path,
    target: Path,
    unit_root: Path,
    source_root: Path,
    mvp_output: Path,
    platform: str = sys.platform,
) -> dict[str, str]:
    environment = inherited.copy()
    for variable in tuple(environment):
        if (
            variable in DEVELOPMENT_ENVIRONMENT_EXACT_REMOVALS
            or variable.startswith("CARGO_BUILD_")
            or variable.startswith("CARGO_PROFILE_")
            or variable.startswith("CARGO_TARGET_")
            or variable.startswith("DYLD_")
            or variable.startswith("RAP_")
            or variable.startswith("UNSOUND_SCANNER_")
        ):
            environment.pop(variable, None)
    environment.update(
        {
            "CARGO_BUILD_JOBS": "1",
            "CARGO_HOME": str(cargo_home),
            "CARGO_TARGET_DIR": str(target),
            "RUSTC": str(rustc),
            "RUST_SYSROOT": str(rust_sysroot),
            "UNSOUND_SCANNER_EXACT_CARGO_PATH": str(cargo),
            "UNSOUND_SCANNER_EXACT_CARGO_COMMIT": cargo_commit,
            "UNSOUND_SCANNER_EXPECTED_RUSTC_PATH": str(rustc),
            "UNSOUND_SCANNER_EXPECTED_RUSTC_COMMIT": rustc_commit,
            "UNSOUND_SCANNER_PROJECT_ROOT": str(unit_root),
            "UNSOUND_SCANNER_SOURCE_ROOT": str(source_root),
            "RAP_UNSOUNDAUDIT_OUTPUT": str(mvp_output),
        }
    )
    if platform == "darwin":
        environment["DYLD_LIBRARY_PATH"] = str(rust_sysroot / "lib")
    return environment


def iter_jsonl_objects(path: Path, label: str) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        fail(f"development MVP output is missing {label}: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise GateError(
                        f"{path}:{line_number}: invalid {label} JSON: {error}"
                    ) from error
                if not isinstance(row, dict):
                    fail(f"{path}:{line_number}: {label} row is not an object")
                yield row
    except (OSError, UnicodeError) as error:
        raise GateError(f"cannot read development MVP {label}: {error}") from error


def canonical_sha256(value: Any, label: str) -> str:
    return exact_hex(value, 64, label)


def required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        fail(f"{label} must be a non-empty string")
    return value


def required_nonnegative_integer(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        fail(f"{label} must be a non-negative integer")
    return value


def span_token(value: Any, label: str) -> str:
    if not isinstance(value, dict) or set(value) != {"path", "start", "end"}:
        fail(f"{label} is not a canonical span object")
    path = required_string(value.get("path"), f"{label}.path")
    points = []
    for endpoint in ("start", "end"):
        point = value.get(endpoint)
        if not isinstance(point, dict) or set(point) != {"line", "column"}:
            fail(f"{label}.{endpoint} is not a canonical source position")
        line = required_nonnegative_integer(point.get("line"), f"{label}.{endpoint}.line")
        column = required_nonnegative_integer(
            point.get("column"), f"{label}.{endpoint}.column"
        )
        if line == 0 or column == 0:
            fail(f"{label}.{endpoint} is not one-based")
        points.append((line, column))
    if points[1] < points[0]:
        fail(f"{label} ends before it starts")
    return f"{path}:{points[0][0]}:{points[0][1]}-{points[1][0]}:{points[1][1]}"


def candidate_root_signature(candidate: dict[str, Any], label: str) -> tuple[str, str]:
    root = candidate.get("public_safe_root")
    if not isinstance(root, dict):
        fail(f"{label} lacks public_safe_root")
    return (
        required_string(root.get("owner"), f"{label} public root owner"),
        span_token(root.get("span"), f"{label} public root span"),
    )


def candidate_site_id(candidate: dict[str, Any], label: str) -> str:
    site = candidate.get("unsafe_site")
    if not isinstance(site, dict):
        fail(f"{label} lacks unsafe_site")
    if site.get("binding") != "bound_exact":
        fail(f"{label} unsafe site is not bound_exact")
    return canonical_sha256(site.get("site_id"), f"{label} unsafe site id")


def candidate_exposure_signature(
    candidate: dict[str, Any], label: str
) -> tuple[str, str, str] | None:
    if "exposure" not in candidate:
        fail(f"{label} lacks exposure")
    exposure = candidate["exposure"]
    if exposure is None:
        return None
    if not isinstance(exposure, dict):
        fail(f"{label} exposure is neither null nor an object")
    return (
        required_string(exposure.get("kind"), f"{label} exposure kind"),
        required_string(exposure.get("identity"), f"{label} exposure identity"),
        span_token(exposure.get("span"), f"{label} exposure span"),
    )


def candidate_witness_length(candidate: dict[str, Any], label: str) -> int:
    propagation = candidate.get("propagation")
    witness = propagation.get("witness") if isinstance(propagation, dict) else None
    if not isinstance(witness, list) or not witness:
        fail(f"{label} lacks a non-empty propagation witness")
    for index, step in enumerate(witness):
        if not isinstance(step, dict):
            fail(f"{label} witness step {index} is not an object")
        required_string(step.get("kind"), f"{label} witness step {index} kind")
        required_string(step.get("owner"), f"{label} witness step {index} owner")
        span_token(step.get("span"), f"{label} witness step {index} span")
    return len(witness)


def validate_candidate(candidate: dict[str, Any], label: str) -> str:
    candidate_id = canonical_sha256(candidate.get("finding_id"), f"{label} finding id")
    pattern = candidate.get("primary_pattern")
    if pattern not in VALID_PATTERNS:
        fail(f"{label} has invalid primary_pattern")
    candidate_root_signature(candidate, label)
    candidate_site_id(candidate, label)
    required_string(candidate.get("subject_role"), f"{label} subject role")
    exposure = candidate_exposure_signature(candidate, label)
    if pattern != "P5" and exposure is not None:
        fail(f"{label} has a non-P5 exposure")
    failure = candidate.get("first_contract_failure")
    if not isinstance(failure, dict):
        fail(f"{label} lacks first_contract_failure")
    required_string(failure.get("kind"), f"{label} first failure kind")
    span_token(failure.get("span"), f"{label} first failure span")
    source = candidate.get("source")
    if not isinstance(source, dict):
        fail(f"{label} lacks source")
    required_string(source.get("kind"), f"{label} source kind")
    required_string(source.get("identity"), f"{label} source identity")
    candidate_witness_length(candidate, label)
    return candidate_id


def candidate_family_key(candidate: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    pattern = required_string(candidate.get("primary_pattern"), "candidate primary pattern")
    failure = candidate["first_contract_failure"]
    contract = candidate.get("contract")
    if not isinstance(contract, dict):
        fail("candidate lacks contract")
    model_id = contract.get("model_id")
    if model_id is not None:
        model_id = required_string(model_id, "candidate contract model_id")
    obligation = contract.get("obligation")
    predicates_value = obligation.get("predicates") if isinstance(obligation, dict) else []
    if not isinstance(predicates_value, list):
        fail("candidate contract predicates are not an array")
    predicates = []
    for predicate in predicates_value:
        predicates.append(required_string(predicate, "candidate contract predicate"))
    predicates = sorted(set(predicates))
    source = candidate["source"]
    exposure_signature = candidate_exposure_signature(candidate, "candidate") if pattern == "P5" else None
    exposure_key = (
        None
        if exposure_signature is None
        else {
            "kind": exposure_signature[0],
            "identity": exposure_signature[1],
            "span": exposure_signature[2],
        }
    )
    site_id = candidate_site_id(candidate, "candidate")
    subject_role = required_string(candidate.get("subject_role"), "candidate subject role")
    failure_kind = required_string(failure.get("kind"), "candidate first failure kind")
    failure_span = span_token(failure.get("span"), "candidate first failure span")
    source_kind = required_string(source.get("kind"), "candidate source kind")
    source_identity = required_string(source.get("identity"), "candidate source identity")
    key = {
        "primary_pattern": pattern,
        "unsafe_site_id": site_id,
        "subject_role": subject_role,
        "first_contract_failure": {"kind": failure_kind, "span": failure_span},
        "contract": {"model_id": model_id, "predicates": predicates},
        "source": {"kind": source_kind, "identity": source_identity},
        "exposure": exposure_key,
    }
    exposure_fields = ("", "", "") if exposure_signature is None else exposure_signature
    fields = [
        pattern,
        site_id,
        subject_role,
        failure_kind,
        failure_span,
        model_id or "",
        *predicates,
        source_kind,
        source_identity,
        *exposure_fields,
    ]
    return key, fields


def family_id_for_candidate(candidate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    key, fields = candidate_family_key(candidate)
    payload = bytearray()
    for field in fields:
        encoded = field.encode("utf-8")
        payload.extend(str(len(encoded)).encode("ascii"))
        payload.extend(b":")
        payload.extend(encoded)
    return hashlib.sha256(payload).hexdigest(), key


class ReplacementSignature(NamedTuple):
    public_root: tuple[str, str]
    unsafe_site_id: str
    subject_role: str
    exposure: tuple[str, str, str] | None


def candidate_replacement_signature(
    candidate: dict[str, Any], label: str
) -> ReplacementSignature:
    return ReplacementSignature(
        public_root=candidate_root_signature(candidate, label),
        unsafe_site_id=candidate_site_id(candidate, label),
        subject_role=required_string(candidate.get("subject_role"), f"{label} subject role"),
        exposure=candidate_exposure_signature(candidate, label),
    )


class FamilyAccumulator:
    __slots__ = (
        "key",
        "raw_count",
        "roots",
        "shortest_rank",
        "shortest",
        "longest_length",
        "longest_id",
        "longest",
        "seen",
    )

    def __init__(
        self,
        key: dict[str, Any],
        candidate: dict[str, Any],
        candidate_id: str,
        root: tuple[str, str],
        witness_length: int,
    ) -> None:
        self.key = key
        self.raw_count = 1
        self.roots = {root}
        self.shortest_rank = (witness_length, candidate_id)
        self.shortest = candidate
        self.longest_length = witness_length
        self.longest_id = candidate_id
        self.longest = candidate
        self.seen = False

    def add(
        self,
        candidate: dict[str, Any],
        candidate_id: str,
        root: tuple[str, str],
        witness_length: int,
    ) -> None:
        self.raw_count += 1
        self.roots.add(root)
        rank = (witness_length, candidate_id)
        if rank < self.shortest_rank:
            self.shortest_rank = rank
            self.shortest = candidate
        if witness_length > self.longest_length or (
            witness_length == self.longest_length and candidate_id < self.longest_id
        ):
            self.longest_length = witness_length
            self.longest_id = candidate_id
            self.longest = candidate


def collect_active_candidates(
    candidates_path: Path,
) -> tuple[
    dict[str, ReplacementSignature],
    dict[str, FamilyAccumulator],
    dict[str, int],
    int,
]:
    active_by_id: dict[str, ReplacementSignature] = {}
    grouped: dict[str, FamilyAccumulator] = {}
    pattern_counts = {pattern: 0 for pattern in sorted(VALID_PATTERNS)}
    active_count = 0
    for index, candidate in enumerate(iter_jsonl_objects(candidates_path, "candidates")):
        candidate_id = validate_candidate(candidate, f"active candidate {index}")
        if candidate_id in active_by_id:
            fail(f"duplicate active candidate id {candidate_id}")
        signature = candidate_replacement_signature(candidate, f"active candidate {index}")
        active_by_id[candidate_id] = signature
        pattern_counts[candidate["primary_pattern"]] += 1
        family_id, key = family_id_for_candidate(candidate)
        witness_length = candidate_witness_length(candidate, "family member")
        accumulator = grouped.get(family_id)
        if accumulator is None:
            grouped[family_id] = FamilyAccumulator(
                key,
                candidate,
                candidate_id,
                signature.public_root,
                witness_length,
            )
        else:
            if accumulator.key != key:
                fail(f"family digest collision for {family_id}")
            accumulator.add(
                candidate,
                candidate_id,
                signature.public_root,
                witness_length,
            )
        active_count += 1
    return active_by_id, grouped, pattern_counts, active_count


def validate_families_stream(
    families_path: Path, grouped: dict[str, FamilyAccumulator]
) -> int:
    family_count = 0
    for index, family in enumerate(iter_jsonl_objects(families_path, "families")):
        if set(family) != {
            "schema_version",
            "family_id",
            "key",
            "raw_count",
            "public_root_fanout",
            "representatives",
        }:
            fail(f"family {index} fields drifted")
        if family["schema_version"] != MVP_FAMILY_SCHEMA:
            fail(f"family {index} schema drifted")
        family_id = canonical_sha256(family.get("family_id"), f"family {index} id")
        accumulator = grouped.get(family_id)
        if accumulator is None:
            fail(f"family {family_id} has no active members")
        if accumulator.seen:
            fail(f"duplicate family id {family_id}")
        accumulator.seen = True
        if family.get("key") != accumulator.key:
            fail(f"family {family_id} structural key drifted")
        if family.get("raw_count") != accumulator.raw_count:
            fail(f"family {family_id} raw_count drifted")
        if family.get("public_root_fanout") != len(accumulator.roots):
            fail(f"family {family_id} public_root_fanout drifted")
        representatives = family.get("representatives")
        expected_representatives = {
            "shortest": accumulator.shortest,
            "longest": accumulator.longest,
        }
        if representatives != expected_representatives:
            fail(f"family {family_id} representatives drifted")
        family_count += 1
    if any(not accumulator.seen for accumulator in grouped.values()):
        fail("active candidates and deterministic families do not form an exact partition")
    return family_count


def validate_replacement_compatibility(
    suppressed: dict[str, Any], replacement: dict[str, Any]
) -> None:
    comparisons = {
        "public root": (
            candidate_root_signature(suppressed, "suppressed candidate"),
            candidate_root_signature(replacement, "replacement candidate"),
        ),
        "unsafe site": (
            candidate_site_id(suppressed, "suppressed candidate"),
            candidate_site_id(replacement, "replacement candidate"),
        ),
        "subject role": (suppressed.get("subject_role"), replacement.get("subject_role")),
        "exposure": (
            candidate_exposure_signature(suppressed, "suppressed candidate"),
            candidate_exposure_signature(replacement, "replacement candidate"),
        ),
    }
    for field, (left, right) in comparisons.items():
        if left != right:
            fail(f"suppression replacement has a different {field}")


def validate_replacement_signature(
    suppressed: dict[str, Any], replacement: ReplacementSignature
) -> None:
    suppressed_signature = candidate_replacement_signature(
        suppressed, "suppressed candidate"
    )
    comparisons = {
        "public root": (suppressed_signature.public_root, replacement.public_root),
        "unsafe site": (suppressed_signature.unsafe_site_id, replacement.unsafe_site_id),
        "subject role": (suppressed_signature.subject_role, replacement.subject_role),
        "exposure": (suppressed_signature.exposure, replacement.exposure),
    }
    for field, (left, right) in comparisons.items():
        if left != right:
            fail(f"suppression replacement has a different {field}")


def validate_suppression_ledger_stream(
    ledger_path: Path, active_by_id: dict[str, ReplacementSignature]
) -> tuple[int, int, int, set[str]]:
    suppressed_ids: set[str] = set()
    pure_suppressed = 0
    reclassified = 0
    suppressed_count = 0
    expected_fields = {
        "schema_version",
        "candidate_id",
        "candidate",
        "reason",
        "evidence",
        "replacement_candidate_id",
    }
    for index, row in enumerate(
        iter_jsonl_objects(ledger_path, "suppression ledger")
    ):
        suppressed_count += 1
        if set(row) != expected_fields or row.get("schema_version") != MVP_SUPPRESSION_SCHEMA:
            fail(f"suppression row {index} schema or fields drifted")
        candidate_id = canonical_sha256(row.get("candidate_id"), f"suppression row {index} id")
        candidate = row.get("candidate")
        if not isinstance(candidate, dict):
            fail(f"suppression row {index} lacks its original candidate")
        if validate_candidate(candidate, f"suppressed candidate {index}") != candidate_id:
            fail(f"suppression row {index} candidate identity drifted")
        if candidate_id in active_by_id or candidate_id in suppressed_ids:
            fail(f"suppressed candidate identity {candidate_id} is not unique")
        suppressed_ids.add(candidate_id)
        if row.get("reason") not in VALID_SUPPRESSION_REASONS:
            fail(f"suppression row {index} has an invalid structural reason")
        evidence = row.get("evidence")
        if not isinstance(evidence, dict) or not evidence:
            fail(f"suppression row {index} lacks structural evidence")
        replacement_id = row["replacement_candidate_id"]
        if replacement_id is None:
            pure_suppressed += 1
            continue
        replacement_id = canonical_sha256(
            replacement_id, f"suppression row {index} replacement id"
        )
        replacement = active_by_id.get(replacement_id)
        if replacement is None:
            fail(f"suppression row {index} replacement is not active")
        validate_replacement_signature(candidate, replacement)
        reclassified += 1
    return pure_suppressed, reclassified, suppressed_count, suppressed_ids


def validate_unestablished_stream(
    path: Path,
    active_ids: set[str],
    suppressed_ids: set[str],
) -> tuple[int, set[str]]:
    expected_fields = {
        "schema_version",
        "decision_id",
        "attempted_pattern",
        "public_root",
        "unsafe_site_id",
        "operand_role",
        "source",
        "controller",
        "capabilities",
        "requirement",
        "primary_reason",
        "secondary_reasons",
        "exposure",
    }
    decision_ids: set[str] = set()
    previous_id: str | None = None
    for index, row in enumerate(iter_jsonl_objects(path, "unestablished decisions")):
        if set(row) != expected_fields or row.get("schema_version") != MVP_UNESTABLISHED_SCHEMA:
            fail(f"unestablished row {index} schema or fields drifted")
        decision_id = canonical_sha256(
            row.get("decision_id"), f"unestablished row {index} decision id"
        )
        if previous_id is not None and decision_id <= previous_id:
            fail("unestablished decision IDs are duplicated or not sorted")
        previous_id = decision_id
        if decision_id in active_ids or decision_id in suppressed_ids:
            fail(f"unestablished decision identity {decision_id} overlaps another population")
        decision_ids.add(decision_id)
        if row.get("attempted_pattern") not in {"P1", "P2", "P5", "P6", "P4", "P3"}:
            fail(f"unestablished row {index} attempts an invalid pattern")
        required_string(row.get("public_root"), f"unestablished row {index} public root")
        canonical_sha256(row.get("unsafe_site_id"), f"unestablished row {index} unsafe site")
        required_string(row.get("operand_role"), f"unestablished row {index} operand role")
        source = row.get("source")
        if not isinstance(source, dict) or set(source) != {"identity", "kind"}:
            fail(f"unestablished row {index} source fields drifted")
        required_string(source.get("identity"), f"unestablished row {index} source identity")
        required_string(source.get("kind"), f"unestablished row {index} source kind")
        required_string(row.get("controller"), f"unestablished row {index} controller")
        required_string(row.get("capabilities"), f"unestablished row {index} capabilities")
        requirement = row.get("requirement")
        if requirement is not None:
            required_string(requirement, f"unestablished row {index} requirement")
        primary = row.get("primary_reason")
        primary_rank = UNESTABLISHED_REASON_RANK.get(primary)
        if primary_rank is None:
            fail(f"unestablished row {index} has an invalid primary reason")
        secondary = row.get("secondary_reasons")
        if not isinstance(secondary, list):
            fail(f"unestablished row {index} secondary reasons are not an array")
        secondary_ranks = []
        for reason in secondary:
            rank = UNESTABLISHED_REASON_RANK.get(reason)
            if rank is None:
                fail(f"unestablished row {index} has an invalid secondary reason")
            secondary_ranks.append(rank)
        if secondary_ranks != sorted(set(secondary_ranks)) or any(
            rank <= primary_rank for rank in secondary_ranks
        ):
            fail(f"unestablished row {index} reason precedence drifted")
        if row.get("exposure") is not None:
            fail(f"unestablished row {index} has a non-null exposure")
    return len(decision_ids), decision_ids


def validate_unsafe_site_census(value: Any, coverage_exact: bool) -> dict[str, Any]:
    fields = {
        "discovered",
        "bound_exact",
        "optimized_away_exact",
        "unbound",
        "ambiguous",
        "elided",
        "by_kind",
    }
    if not isinstance(value, dict) or set(value) != fields:
        fail("development MVP unsafe-site census fields drifted")
    counts = {
        field: required_nonnegative_integer(value[field], f"unsafe_site_census.{field}")
        for field in (
            "discovered",
            "bound_exact",
            "optimized_away_exact",
            "unbound",
            "ambiguous",
            "elided",
        )
    }
    if counts["discovered"] != sum(
        counts[field]
        for field in (
            "bound_exact",
            "optimized_away_exact",
            "unbound",
            "ambiguous",
            "elided",
        )
    ):
        fail("development MVP unsafe-site census status partition drifted")
    if coverage_exact and (
        counts["bound_exact"] + counts["optimized_away_exact"] != counts["discovered"]
        or any(counts[field] for field in ("unbound", "ambiguous", "elided"))
    ):
        fail("exact development MVP coverage contains a nonexact unsafe-site binding")
    by_kind = value["by_kind"]
    if not isinstance(by_kind, list):
        fail("development MVP unsafe-site kind census is not an array")
    kind_counts: list[tuple[str, int]] = []
    for index, row in enumerate(by_kind):
        if not isinstance(row, dict) or set(row) != {"kind", "count"}:
            fail(f"unsafe-site kind census row {index} fields drifted")
        kind = required_string(row.get("kind"), f"unsafe-site kind census row {index} kind")
        count = required_nonnegative_integer(
            row.get("count"), f"unsafe-site kind census row {index} count"
        )
        if count == 0:
            fail(f"unsafe-site kind census row {index} is empty")
        kind_counts.append((kind, count))
    if kind_counts != sorted(kind_counts) or len({kind for kind, _ in kind_counts}) != len(
        kind_counts
    ):
        fail("development MVP unsafe-site kind census is not unique and sorted")
    if sum(count for _, count in kind_counts) != counts["discovered"]:
        fail("development MVP unsafe-site kind census total drifted")
    return value


class DevelopmentMvpSummary(NamedTuple):
    coverage_exact: bool
    coverage_semantics: str
    coverage_failure_count: int
    active_count: int
    family_count: int
    pure_suppressed_count: int
    reclassified_count: int
    unestablished_count: int
    suppressed_count: int
    considered_count: int
    pattern_counts: dict[str, int]
    unsafe_site_census: dict[str, Any]
    family_target_met: bool | None
    output_hashes: dict[str, str]


def _validate_development_mvp(output: Path) -> DevelopmentMvpSummary:
    candidates_path = output / "candidates.jsonl"
    families_path = output / "families.jsonl"
    ledger_path = output / "suppression-ledger.jsonl"
    unestablished_path = output / "unestablished.jsonl"
    coverage_path = output / "coverage.json"
    active_by_id, family_accumulators, pattern_counts, active_count = (
        collect_active_candidates(candidates_path)
    )
    family_count = validate_families_stream(families_path, family_accumulators)
    del family_accumulators
    pure_suppressed, reclassified, suppressed_count, suppressed_ids = (
        validate_suppression_ledger_stream(ledger_path, active_by_id)
    )
    unestablished_count, _ = validate_unestablished_stream(
        unestablished_path, set(active_by_id), suppressed_ids
    )
    del active_by_id
    del suppressed_ids

    try:
        coverage = read_json(coverage_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateError(f"cannot read development MVP coverage: {error}") from error
    if not isinstance(coverage, dict) or set(coverage) != {
        "schema_version",
        "complete",
        "semantics",
        "candidate_accounting",
        "unsafe_site_census",
        "coverage_failures",
    }:
        fail("development MVP coverage fields drifted")
    if coverage["schema_version"] != MVP_COVERAGE_SCHEMA:
        fail("development MVP coverage schema drifted")
    complete = coverage["complete"]
    semantics = coverage["semantics"]
    failures = coverage["coverage_failures"]
    if type(complete) is not bool or semantics not in {"exact", "lower_bound"}:
        fail("development MVP coverage status drifted")
    if not isinstance(failures, list) or any(not isinstance(row, dict) for row in failures):
        fail("development MVP coverage failures are not an array of objects")
    coverage_exact = complete and semantics == "exact" and not failures
    coverage_partial = not complete and semantics == "lower_bound" and bool(failures)
    if not coverage_exact and not coverage_partial:
        fail("development MVP coverage completeness and semantics are inconsistent")
    unsafe_site_census = validate_unsafe_site_census(
        coverage["unsafe_site_census"], coverage_exact
    )

    accounting = coverage["candidate_accounting"]
    if not isinstance(accounting, dict) or set(accounting) != MVP_ACCOUNTING_FIELDS:
        fail("development MVP candidate_accounting fields drifted")
    for field in MVP_ACCOUNTING_FIELDS:
        required_nonnegative_integer(accounting[field], f"candidate_accounting.{field}")
    observed = {
        "active_count": active_count,
        "pure_suppressed_count": pure_suppressed,
        "reclassified_count": reclassified,
        "unestablished_count": unestablished_count,
        "suppressed_count": suppressed_count,
        "considered_count": active_count + suppressed_count + unestablished_count,
    }
    if accounting["pure_suppressed_count"] + accounting["reclassified_count"] != accounting[
        "suppressed_count"
    ]:
        fail("writer suppression partition is inconsistent")
    if (
        accounting["active_count"]
        + accounting["suppressed_count"]
        + accounting["unestablished_count"]
        != accounting["considered_count"]
    ):
        fail("writer considered population is inconsistent")
    if accounting != observed:
        fail(f"development MVP candidate accounting drifted: writer={accounting}, observed={observed}")
    return DevelopmentMvpSummary(
        coverage_exact=coverage_exact,
        coverage_semantics=semantics,
        coverage_failure_count=len(failures),
        active_count=active_count,
        family_count=family_count,
        pure_suppressed_count=pure_suppressed,
        reclassified_count=reclassified,
        unestablished_count=unestablished_count,
        suppressed_count=suppressed_count,
        considered_count=active_count + suppressed_count + unestablished_count,
        pattern_counts=pattern_counts,
        unsafe_site_census=unsafe_site_census,
        family_target_met=(family_count <= DEVELOPMENT_FAMILY_TARGET)
        if coverage_exact
        else None,
        output_hashes={
            "candidates_sha256": sha256(candidates_path),
            "families_sha256": sha256(families_path),
            "suppression_ledger_sha256": sha256(ledger_path),
            "unestablished_sha256": sha256(unestablished_path),
            "coverage_sha256": sha256(coverage_path),
        },
    )


def validate_development_mvp(output: Path) -> DevelopmentMvpSummary:
    try:
        return _validate_development_mvp(output)
    except MemoryError as error:
        raise GateError("development MVP validation exhausted memory") from error


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
    physical_footprint_limit_bytes: int | None,
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
                if (
                    physical_footprint_limit_bytes is not None
                    and maximum_footprint > physical_footprint_limit_bytes
                ):
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
            if value["crate_name"] == unit:
                selected.append(value)
        elif version != "rap-unit-skip-v1":
            fail(f"unrecognized receipt schema: {path.name}")
    if len(selected) != 1:
        fail(f"expected exactly one complete {unit} receipt, found {len(selected)}")
    receipt = selected[0]
    if receipt["analysis_status"] != "complete":
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


def verify_binary_launch(
    cargo_rapx: Path, environment: dict[str, str] | None = None
) -> None:
    result = subprocess.run(
        [str(cargo_rapx), "rapx", "--help"],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        fail("RAP binary launch preflight failed")


def development_counts(summary: DevelopmentMvpSummary) -> dict[str, Any]:
    return {
        "active_count": summary.active_count,
        "family_count": summary.family_count,
        "pure_suppressed_count": summary.pure_suppressed_count,
        "reclassified_count": summary.reclassified_count,
        "unestablished_count": summary.unestablished_count,
        "suppressed_count": summary.suppressed_count,
        "considered_count": summary.considered_count,
        "coverage_failure_count": summary.coverage_failure_count,
        "pattern_counts": summary.pattern_counts,
    }


def existing_mvp_hashes(mvp_output: Path) -> dict[str, str]:
    result = {}
    for key, name in (
        ("candidates_sha256", "candidates.jsonl"),
        ("families_sha256", "families.jsonl"),
        ("suppression_ledger_sha256", "suppression-ledger.jsonl"),
        ("unestablished_sha256", "unestablished.jsonl"),
        ("coverage_sha256", "coverage.json"),
    ):
        path = mvp_output / name
        if path.is_file():
            result[key] = sha256(path)
    return result


def run_development_single_scan(
    *,
    repo_root: Path,
    protocol_path: Path,
    protocol: dict[str, Any],
    cargo: Path,
    binary_directory: Path,
    cargo_rapx: Path,
    rapx: Path,
    rust_sysroot: Path,
    rustc: Path,
    profile: dict[str, Any],
    source_root: Path,
    unit_root: Path,
    output_root: Path,
    unit: str,
    observe_memory_only: bool = False,
) -> None:
    if unit != "core":
        fail("development single-run mode is restricted to core")
    unit_contract = protocol["units"][unit]
    if (
        unit_contract["wall_time_seconds"] != DEVELOPMENT_CORE_WALL_SECONDS
        or unit_contract["physical_footprint_limit_bytes"]
        != DEVELOPMENT_CORE_PHYSICAL_LIMIT_BYTES
    ):
        fail("development core resource limits drifted from 900 seconds / 8 GiB")

    logs = output_root / "logs"
    logs.mkdir()
    scan_log = logs / "scan-1.log"
    performance_trace = logs / "perf.jsonl"
    mvp_output = output_root / "mvp"
    resource_path = output_root / f"{unit}.development.resource.json"
    arguments = cargo_arguments(protocol, unit)
    driver_context = tempfile.TemporaryDirectory(prefix="rapx-v3-stdlib-core-development-driver.")
    driver = Path(driver_context.name).resolve()
    target = Path(tempfile.mkdtemp(prefix="rapx-v3-stdlib-core-development-target.")).resolve()
    cargo_home = target / "cargo-home"
    cargo_home.mkdir()
    command = [str(cargo_rapx), "rapx", "-unsoundaudit", "--", *arguments]
    environment = development_environment(
        os.environ.copy(),
        cargo=cargo,
        rustc=rustc,
        rust_sysroot=rust_sysroot,
        cargo_commit=protocol["cargo_commit"],
        rustc_commit=protocol["rustc_commit"],
        cargo_home=cargo_home,
        target=target,
        unit_root=unit_root,
        source_root=source_root,
        mvp_output=mvp_output,
    )
    # The nonformal lane keeps a flushed phase trace even when the resource
    # monitor terminates the analyzer.  Install this only after sanitizing the
    # inherited environment so a caller cannot redirect or reuse the file.
    environment["UNSOUND_SCANNER_PERF_TRACE"] = str(performance_trace)
    resource: dict[str, Any] = {
        "schema_version": DEVELOPMENT_RESULT_SCHEMA,
        "unit": unit,
        "status": "running",
        "nonformal": True,
        "evidence_scope": "dirty_live_tree_development_measurement",
        "development_single_run": True,
        "runs_per_unit": 1,
        "fresh_target": True,
        "source_commit_gates": {
            "repository_source_commit_enforced": False,
            "binary_profile_source_commit_enforced": False,
        },
        "protocol_sha256": sha256(protocol_path),
        "protocol_rap_source_commit": protocol["rap_source_commit"],
        "binary_profile_rap_source_commit": profile.get("rap_source_commit"),
        "toolchain": {
            "cargo_commit": protocol["cargo_commit"],
            "rustc_commit": protocol["rustc_commit"],
            "embedded_rustc_commit": profile.get("rustc_commit"),
            "embedded_rustc_evidence": "trusted_build_profile_sidecar_bound_to_binary_hashes",
            "target_triple": protocol["target_triple"],
        },
        "analyzer_profile": profile,
        "limits": {
            "wall_time_seconds": DEVELOPMENT_CORE_WALL_SECONDS,
            "memory_limit_metric": "physical_footprint",
            "physical_footprint_limit_bytes": (
                None if observe_memory_only else DEVELOPMENT_CORE_PHYSICAL_LIMIT_BYTES
            ),
            "protocol_physical_footprint_limit_bytes": DEVELOPMENT_CORE_PHYSICAL_LIMIT_BYTES,
            "memory_limit_enforced": not observe_memory_only,
        },
        "command": command,
        "roots": {
            "repository": str(repo_root),
            "rust_sysroot": str(rust_sysroot),
            "rust_source": str(source_root),
            "project": str(unit_root),
            "driver": str(driver),
            "fresh_target": str(target),
            "isolated_cargo_home": str(cargo_home),
            "mvp_output": str(mvp_output),
            "performance_trace": str(performance_trace),
        },
        "hashes": {
            "cargo_sha256": sha256(cargo),
            "rustc_sha256": sha256(rustc),
            "cargo_rapx_sha256": sha256(cargo_rapx),
            "rapx_sha256": sha256(rapx),
            "binary_profile_sidecar_sha256": sha256(binary_directory / PROFILE_SIDECAR),
        },
        "execution": None,
        "coverage": None,
        "counts": None,
        "family_target": {
            "limit": DEVELOPMENT_FAMILY_TARGET,
            "met": None,
            "claim_admissible": False,
        },
    }
    try:
        write_driver(driver, unit)
        verify_binary_launch(cargo_rapx, environment)
        result = run_limited(
            command,
            driver,
            environment,
            scan_log,
            DEVELOPMENT_CORE_WALL_SECONDS,
            None if observe_memory_only else DEVELOPMENT_CORE_PHYSICAL_LIMIT_BYTES,
        )
        resource["execution"] = process_record("scan", 1, result)
        if performance_trace.is_file():
            resource["hashes"]["performance_trace_sha256"] = sha256(performance_trace)
        if not scan_log.is_file():
            fail("development scan did not preserve its diagnostic log")
        resource["hashes"]["scan_log_sha256"] = sha256(scan_log)
        if result.classification != "success":
            fail(f"development scan failed: {result.classification}")
        summary = validate_development_mvp(mvp_output)
        resource["hashes"].update(summary.output_hashes)
        resource["coverage"] = {
            "complete": summary.coverage_exact,
            "semantics": summary.coverage_semantics,
            "unsafe_site_census": summary.unsafe_site_census,
        }
        resource["counts"] = development_counts(summary)
        resource["family_target"] = {
            "limit": DEVELOPMENT_FAMILY_TARGET,
            "met": summary.family_target_met,
            "claim_admissible": summary.coverage_exact,
        }
        resource["status"] = "success" if summary.coverage_exact else "partial"
    except (
        ContractError,
        GateError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        MemoryError,
    ) as error:
        memory_failure = isinstance(error, MemoryError)
        reason = (
            "development MVP validation exhausted memory"
            if memory_failure
            else str(error)
        )
        resource["status"] = "failed"
        resource["reason"] = reason
        resource["hashes"].update(existing_mvp_hashes(mvp_output))
        if scan_log.is_file():
            resource["hashes"]["scan_log_sha256"] = sha256(scan_log)
        if performance_trace.is_file():
            resource["hashes"]["performance_trace_sha256"] = sha256(performance_trace)
        resource_path.write_bytes(canonical_json(resource))
        if memory_failure:
            raise GateError(reason) from error
        raise
    finally:
        driver_context.cleanup()
        shutil.rmtree(target, ignore_errors=True)

    resource_path.write_bytes(canonical_json(resource))
    if resource["status"] == "partial":
        fail("development MVP coverage is partial; family target is not admissible")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--cargo", type=Path, required=True)
    parser.add_argument("--rap-bin-dir", type=Path, required=True)
    parser.add_argument("--rust-sysroot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--unit", choices=KNOWN_UNITS, required=True)
    parser.add_argument(
        "--development-single-run",
        action="store_true",
        help="run one nonformal dirty-live-tree core MVP measurement",
    )
    parser.add_argument(
        "--development-observe-memory-only",
        action="store_true",
        help="record peak memory without enforcing the development memory limit",
    )
    args = parser.parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    output_created = False
    executions: list[dict[str, Any]] = []
    try:
        if not args.development_single_run and os.environ.get("UNSOUND_SCANNER_PERF_TRACE"):
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
        if args.development_single_run and args.unit != "core":
            fail("development single-run mode is restricted to core")
        if args.development_observe_memory_only and not args.development_single_run:
            fail("memory observation mode requires --development-single-run")
        if tool_commit(cargo, "Cargo") != protocol["cargo_commit"]:
            fail("Cargo commit differs from protocol")
        if tool_commit(rustc, "rustc") != protocol["rustc_commit"]:
            fail("rustc commit differs from protocol")
        cargo_profile_contract(repo_root / "rapx/Cargo.toml")
        schema = None
        if args.development_single_run:
            profile = load_development_binary_profile(
                binary_directory,
                cargo_rapx,
                rapx,
                protocol["rustc_commit"],
            )
        else:
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
        protocol_sha = sha256(protocol_path)
        if args.development_single_run:
            run_development_single_scan(
                repo_root=repo_root,
                protocol_path=protocol_path,
                protocol=protocol,
                cargo=cargo,
                binary_directory=binary_directory,
                cargo_rapx=cargo_rapx,
                rapx=rapx,
                rust_sysroot=rust_sysroot,
                rustc=rustc,
                profile=profile,
                source_root=source_root,
                unit_root=unit_root,
                output_root=output_root,
                unit=args.unit,
                observe_memory_only=args.development_observe_memory_only,
            )
            return 0

        (output_root / "logs").mkdir()
        (output_root / "receipts").mkdir()
        assert schema is not None
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
    except (ContractError, GateError, OSError, json.JSONDecodeError, MemoryError) as error:
        reason = (
            "development MVP validation exhausted memory"
            if isinstance(error, MemoryError) and args.development_single_run
            else str(error)
        )
        if output_created:
            if args.development_single_run:
                failure = {
                    "schema_version": DEVELOPMENT_FAILURE_SCHEMA,
                    "unit": args.unit,
                    "status": "failed",
                    "nonformal": True,
                    "reason": reason,
                }
                (output_root / f"{args.unit}.development.failure.json").write_bytes(
                    canonical_json(failure)
                )
            else:
                failure = {
                    "schema_version": "unsoundaudit-v3-stdlib-gate-failure-v1",
                    "unit": args.unit,
                    "status": "failed",
                    "reason": reason,
                    "executions": executions,
                }
                (output_root / f"{args.unit}.failure.json").write_bytes(canonical_json(failure))
        print(f"v3 stdlib gate error: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

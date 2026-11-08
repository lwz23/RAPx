#!/usr/bin/env python3
"""Run the lightweight 2024 P1-P6 mechanism pack once per crate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class PackError(RuntimeError):
    pass


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PackError(f"failed to read {path}: {error}") from error


def run(command: list[str], cwd: Path, environment: dict[str, str], log: Path) -> None:
    with log.open("wb") as stream:
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=900,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise PackError(f"command timed out; log={log}") from error
    if result.returncode != 0:
        raise PackError(f"command exited {result.returncode}; log={log}")


def tool_output(command: list[str], environment: dict[str, str], label: str) -> str:
    result = subprocess.run(
        command,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise PackError(f"failed to inspect {label}: {result.stderr.strip()}")
    return result.stdout


def commit_hash(executable: Path, environment: dict[str, str], label: str) -> str:
    for line in tool_output([str(executable), "-Vv"], environment, label).splitlines():
        if line.startswith("commit-hash: "):
            value = line.removeprefix("commit-hash: ").strip()
            if len(value) == 40 and all(character in "0123456789abcdef" for character in value):
                return value
    raise PackError(f"{label} did not report a full commit hash")


VALID_PATTERNS = {f"P{number}" for number in range(1, 7)}
MVP_FAMILY_SCHEMA = "rap-unsoundaudit-family-v3-mvp"
MVP_UNESTABLISHED_SCHEMA = "rap-unsoundaudit-unestablished-v1"
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
ENVIRONMENT_EXACT_REMOVALS = {
    "CARGO",
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
    "LIBRARY_PATH",
    "MACOSX_DEPLOYMENT_TARGET",
    "RUSTC",
    "RUSTC_BOOTSTRAP",
    "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
    "RUSTDOC",
    "RUSTDOCFLAGS",
    "RUSTFLAGS",
    "RUSTUP_TOOLCHAIN",
    "RUST_SYSROOT",
}
ENVIRONMENT_PREFIX_REMOVALS = (
    "CARGO_BUILD_",
    "CARGO_PROFILE_",
    "CARGO_TARGET_",
    "DYLD_",
    "LD_",
    "RAP_",
    "UNSOUND_SCANNER_",
)


def sanitized_base_environment(inherited: dict[str, str]) -> dict[str, str]:
    environment = inherited.copy()
    for variable in tuple(environment):
        if variable in ENVIRONMENT_EXACT_REMOVALS or variable.startswith(
            ENVIRONMENT_PREFIX_REMOVALS
        ):
            environment.pop(variable, None)
    return environment


def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PackError(f"missing {label} output: {path}")
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise PackError(f"{path}:{line_number}: invalid JSON: {error}") from error
                if not isinstance(row, dict):
                    raise PackError(f"{path}:{line_number}: {label} row is not an object")
                rows.append(row)
    except (OSError, UnicodeError) as error:
        raise PackError(f"failed to read {path}: {error}") from error
    return rows


def load_candidates(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path, "candidates")


def root_matches(owner: str, expected: str) -> bool:
    return owner == expected or owner.endswith(f"::{expected}")


def candidate_root(row: dict[str, Any], label: str) -> str:
    root = row.get("public_safe_root")
    owner = root.get("owner") if isinstance(root, dict) else None
    if not isinstance(owner, str) or not owner:
        raise PackError(f"{label} lacks public_safe_root.owner")
    return owner


def candidate_site_id(row: dict[str, Any], label: str) -> str:
    site = row.get("unsafe_site")
    if not isinstance(site, dict) or site.get("binding") != "bound_exact":
        raise PackError(f"{label} unsafe site is not bound_exact")
    site_id = site.get("site_id") if isinstance(site, dict) else None
    return canonical_sha256(site_id, f"{label} unsafe site id")


def candidate_exposure_identity(row: dict[str, Any], label: str) -> str | None:
    if "exposure" not in row:
        raise PackError(f"{label} lacks exposure")
    exposure = row.get("exposure")
    if exposure is None:
        return None
    identity = exposure.get("identity") if isinstance(exposure, dict) else None
    if not isinstance(identity, str) or not identity:
        raise PackError(f"{label} has an invalid exposure identity")
    return identity


def validate_replacement_compatibility(
    case: str, candidate: dict[str, Any], replacement: dict[str, Any]
) -> None:
    comparisons = {
        "public root owner": (
            candidate_root(candidate, "suppressed candidate"),
            candidate_root(replacement, "replacement candidate"),
        ),
        "unsafe site": (
            candidate_site_id(candidate, "suppressed candidate"),
            candidate_site_id(replacement, "replacement candidate"),
        ),
        "subject role": (candidate.get("subject_role"), replacement.get("subject_role")),
        "exposure identity": (
            candidate_exposure_identity(candidate, "suppressed candidate"),
            candidate_exposure_identity(replacement, "replacement candidate"),
        ),
    }
    for field, (suppressed_value, replacement_value) in comparisons.items():
        if field == "subject role" and (
            not isinstance(suppressed_value, str)
            or not suppressed_value
            or not isinstance(replacement_value, str)
            or not replacement_value
        ):
            raise PackError(f"{case}: replacement comparison has an invalid {field}")
        if suppressed_value != replacement_value:
            raise PackError(
                f"{case}: suppression replacement has a different {field}: "
                f"{suppressed_value!r} != {replacement_value!r}"
            )


def canonical_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PackError(f"{label} is not a canonical lowercase SHA-256")
    return value


def required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PackError(f"{label} must be a non-empty string")
    return value


def required_nonnegative_integer(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise PackError(f"{label} must be a non-negative integer")
    return value


def span_token(value: Any, label: str) -> str:
    if not isinstance(value, dict) or set(value) != {"path", "start", "end"}:
        raise PackError(f"{label} is not a canonical span object")
    path = required_string(value.get("path"), f"{label}.path")
    points = []
    for endpoint in ("start", "end"):
        point = value.get(endpoint)
        if not isinstance(point, dict) or set(point) != {"line", "column"}:
            raise PackError(f"{label}.{endpoint} is not a canonical source position")
        line = required_nonnegative_integer(
            point.get("line"), f"{label}.{endpoint}.line"
        )
        column = required_nonnegative_integer(
            point.get("column"), f"{label}.{endpoint}.column"
        )
        if line == 0 or column == 0:
            raise PackError(f"{label}.{endpoint} is not one-based")
        points.append((line, column))
    if points[1] < points[0]:
        raise PackError(f"{label} ends before it starts")
    return f"{path}:{points[0][0]}:{points[0][1]}-{points[1][0]}:{points[1][1]}"


def candidate_root_signature(candidate: dict[str, Any], label: str) -> tuple[str, str]:
    root = candidate.get("public_safe_root")
    if not isinstance(root, dict):
        raise PackError(f"{label} lacks public_safe_root")
    return (
        required_string(root.get("owner"), f"{label} public root owner"),
        span_token(root.get("span"), f"{label} public root span"),
    )


def candidate_exposure_signature(
    candidate: dict[str, Any], label: str
) -> tuple[str, str, str] | None:
    if "exposure" not in candidate:
        raise PackError(f"{label} lacks exposure")
    exposure = candidate["exposure"]
    if exposure is None:
        return None
    if not isinstance(exposure, dict):
        raise PackError(f"{label} exposure is neither null nor an object")
    return (
        required_string(exposure.get("kind"), f"{label} exposure kind"),
        required_string(exposure.get("identity"), f"{label} exposure identity"),
        span_token(exposure.get("span"), f"{label} exposure span"),
    )


def candidate_witness_length(candidate: dict[str, Any], label: str) -> int:
    propagation = candidate.get("propagation")
    witness = propagation.get("witness") if isinstance(propagation, dict) else None
    if not isinstance(witness, list) or not witness:
        raise PackError(f"{label} lacks a non-empty propagation witness")
    for index, step in enumerate(witness):
        if not isinstance(step, dict):
            raise PackError(f"{label} witness step {index} is not an object")
        required_string(step.get("kind"), f"{label} witness step {index} kind")
        required_string(step.get("owner"), f"{label} witness step {index} owner")
        span_token(step.get("span"), f"{label} witness step {index} span")
    return len(witness)


def candidate_family_key(candidate: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    pattern = required_string(candidate.get("primary_pattern"), "candidate primary pattern")
    if pattern not in VALID_PATTERNS:
        raise PackError("candidate primary pattern is invalid")
    candidate_root_signature(candidate, "candidate")
    site_id = candidate_site_id(candidate, "candidate")
    subject_role = required_string(candidate.get("subject_role"), "candidate subject role")
    exposure_signature = candidate_exposure_signature(candidate, "candidate")
    if pattern != "P3" and exposure_signature is not None:
        raise PackError("non-P3 candidate has an exposure")

    failure = candidate.get("first_contract_failure")
    if not isinstance(failure, dict):
        raise PackError("candidate lacks first_contract_failure")
    failure_kind = required_string(failure.get("kind"), "candidate first failure kind")
    failure_span = span_token(failure.get("span"), "candidate first failure span")

    contract = candidate.get("contract")
    if not isinstance(contract, dict):
        raise PackError("candidate lacks contract")
    model_id = contract.get("model_id")
    if model_id is not None:
        model_id = required_string(model_id, "candidate contract model_id")
    obligation = contract.get("obligation")
    predicates_value = obligation.get("predicates") if isinstance(obligation, dict) else []
    if not isinstance(predicates_value, list):
        raise PackError("candidate contract predicates are not an array")
    predicates = sorted(
        {
            required_string(predicate, "candidate contract predicate")
            for predicate in predicates_value
        }
    )

    source = candidate.get("source")
    if not isinstance(source, dict):
        raise PackError("candidate lacks source")
    source_kind = required_string(source.get("kind"), "candidate source kind")
    source_identity = required_string(source.get("identity"), "candidate source identity")
    candidate_witness_length(candidate, "candidate")

    exposure_key = (
        None
        if exposure_signature is None
        else {
            "kind": exposure_signature[0],
            "identity": exposure_signature[1],
            "span": exposure_signature[2],
        }
    )
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


def validate_family_accounting(
    case: str, candidates: list[dict[str, Any]], families: list[dict[str, Any]]
) -> None:
    active_by_id: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    expected_keys: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(candidates):
        candidate_id = canonical_sha256(
            candidate.get("finding_id"), f"active candidate {index} id"
        )
        if candidate_id in active_by_id:
            raise PackError(f"{case}: duplicate active candidate id {candidate_id}")
        active_by_id[candidate_id] = candidate
        family_id, key = family_id_for_candidate(candidate)
        if family_id in expected_keys and expected_keys[family_id] != key:
            raise PackError(f"{case}: family digest collision for {family_id}")
        expected_keys[family_id] = key
        grouped.setdefault(family_id, []).append(candidate)

    observed_ids: set[str] = set()
    expected_family_fields = {
        "schema_version",
        "family_id",
        "key",
        "raw_count",
        "public_root_fanout",
        "representatives",
    }
    for index, family in enumerate(families):
        if set(family) != expected_family_fields:
            raise PackError(f"{case}: family {index} fields drifted")
        if family.get("schema_version") != MVP_FAMILY_SCHEMA:
            raise PackError(f"{case}: family {index} has invalid schema_version")
        family_id = canonical_sha256(family.get("family_id"), f"family {index} id")
        if family_id in observed_ids:
            raise PackError(f"{case}: duplicate family id {family_id}")
        observed_ids.add(family_id)
        members = grouped.get(family_id)
        if not members:
            raise PackError(f"{case}: family {family_id} has no active members")
        if family.get("key") != expected_keys[family_id]:
            raise PackError(f"{case}: family {family_id} structural key drifted")
        raw_count = family.get("raw_count")
        if type(raw_count) is not int or raw_count != len(members):
            raise PackError(f"{case}: family {family_id} raw_count drifted")
        roots = {
            candidate_root_signature(member, "family member") for member in members
        }
        root_fanout = family.get("public_root_fanout")
        if type(root_fanout) is not int or root_fanout != len(roots):
            raise PackError(f"{case}: family {family_id} public_root_fanout drifted")
        sorted_members = sorted(
            members,
            key=lambda member: (
                candidate_witness_length(member, "family member"),
                member["finding_id"],
            ),
        )
        longest_length = candidate_witness_length(sorted_members[-1], "family member")
        longest = next(
            member
            for member in sorted_members
            if candidate_witness_length(member, "family member") == longest_length
        )
        expected_representatives = {
            "shortest": sorted_members[0],
            "longest": longest,
        }
        if family.get("representatives") != expected_representatives:
            raise PackError(f"{case}: family {family_id} representatives drifted")
    if observed_ids != set(grouped):
        raise PackError(
            f"{case}: active candidates and deterministic families do not form an exact partition"
        )


def validate_suppression_ledger(
    case: str,
    ledger: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    output: Path,
) -> dict[str, dict[str, Any]]:
    active_by_id: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        candidate_id = canonical_sha256(candidate.get("finding_id"), "active candidate id")
        if candidate_id in active_by_id:
            raise PackError(f"{case}: duplicate active candidate id {candidate_id}")
        active_by_id[candidate_id] = candidate
    suppressed_ids: set[str] = set()
    for row in ledger:
        if row.get("schema_version") != "rap-unsoundaudit-suppression-v3-mvp":
            raise PackError(f"{case}: suppression row has invalid schema_version")
        candidate_id = canonical_sha256(row.get("candidate_id"), "suppressed candidate id")
        candidate = row.get("candidate")
        if not isinstance(candidate, dict) or candidate.get("finding_id") != candidate_id:
            raise PackError(f"{case}: suppression row {candidate_id} does not preserve its candidate")
        if candidate_id in active_by_id:
            raise PackError(f"{case}: candidate {candidate_id} is both active and suppressed")
        if candidate_id in suppressed_ids:
            raise PackError(f"{case}: duplicate suppressed candidate id {candidate_id}")
        suppressed_ids.add(candidate_id)
        reason = row.get("reason")
        if reason not in VALID_SUPPRESSION_REASONS:
            raise PackError(f"{case}: suppression row {candidate_id} has invalid reason {reason!r}")
        evidence = row.get("evidence")
        if not isinstance(evidence, dict) or not evidence:
            raise PackError(f"{case}: suppression row {candidate_id} lacks structural evidence")
        if "replacement_candidate_id" not in row:
            raise PackError(
                f"{case}: suppression row {candidate_id} lacks replacement_candidate_id"
            )
        replacement_id = row.get("replacement_candidate_id")
        if replacement_id is not None:
            canonical_sha256(replacement_id, "replacement candidate id")
            if replacement_id not in active_by_id:
                raise PackError(
                    f"{case}: suppression row {candidate_id} references non-active "
                    f"replacement {replacement_id}"
                )
            validate_replacement_compatibility(
                case, candidate, active_by_id[replacement_id]
            )
            validate_candidate(active_by_id[replacement_id], output)
        validate_candidate(candidate, output)
    return active_by_id


def validate_candidate_accounting(
    case: str,
    candidates: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
    unestablished: list[dict[str, Any]],
    coverage: dict[str, Any],
) -> dict[str, int]:
    accounting = coverage.get("candidate_accounting")
    if not isinstance(accounting, dict):
        raise PackError(f"{case}: coverage lacks writer candidate_accounting")
    fields = (
        "active_count",
        "pure_suppressed_count",
        "reclassified_count",
        "unestablished_count",
        "suppressed_count",
        "considered_count",
    )
    if set(accounting) != set(fields):
        raise PackError(
            f"{case}: writer candidate_accounting fields drifted: "
            f"{sorted(accounting)}"
        )
    for field in fields:
        value = accounting[field]
        if type(value) is not int or value < 0:
            raise PackError(
                f"{case}: writer candidate_accounting.{field} is not a non-negative integer"
            )
    for index, row in enumerate(ledger):
        if "replacement_candidate_id" not in row:
            raise PackError(
                f"{case}: suppression row {index} lacks replacement_candidate_id"
            )
    active_count = len(candidates)
    pure_suppressed_count = sum(
        row.get("replacement_candidate_id") is None for row in ledger
    )
    reclassified_count = sum(
        row.get("replacement_candidate_id") is not None for row in ledger
    )
    suppressed_count = len(ledger)
    unestablished_count = len(unestablished)
    if pure_suppressed_count + reclassified_count != suppressed_count:
        raise PackError(
            f"{case}: suppression partition drift: pure={pure_suppressed_count} + "
            f"reclassified={reclassified_count} != suppressed={suppressed_count}"
        )
    considered_count = active_count + suppressed_count + unestablished_count
    observed = {
        "active_count": active_count,
        "pure_suppressed_count": pure_suppressed_count,
        "reclassified_count": reclassified_count,
        "unestablished_count": unestablished_count,
        "suppressed_count": suppressed_count,
        "considered_count": considered_count,
    }
    if accounting["pure_suppressed_count"] + accounting["reclassified_count"] != accounting[
        "suppressed_count"
    ]:
        raise PackError(
            f"{case}: writer suppression partition is inconsistent"
        )
    if (
        accounting["active_count"]
        + accounting["suppressed_count"]
        + accounting["unestablished_count"]
        != accounting["considered_count"]
    ):
        raise PackError(f"{case}: writer considered partition is inconsistent")
    if accounting != observed:
        raise PackError(
            f"{case}: writer candidate_accounting drift: "
            f"writer={accounting}, observed={observed}"
        )
    return observed


def validate_unestablished_rows(
    case: str,
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
) -> None:
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
    active_ids = {
        canonical_sha256(row.get("finding_id"), "active candidate id") for row in candidates
    }
    suppressed_ids = {
        canonical_sha256(row.get("candidate_id"), "suppressed candidate id") for row in ledger
    }
    previous_id: str | None = None
    for index, row in enumerate(rows):
        if set(row) != expected_fields or row.get("schema_version") != MVP_UNESTABLISHED_SCHEMA:
            raise PackError(f"{case}: unestablished row {index} schema or fields drifted")
        decision_id = canonical_sha256(
            row.get("decision_id"), f"{case} unestablished row {index} id"
        )
        if previous_id is not None and decision_id <= previous_id:
            raise PackError(f"{case}: unestablished IDs are duplicated or not sorted")
        previous_id = decision_id
        if decision_id in active_ids or decision_id in suppressed_ids:
            raise PackError(f"{case}: unestablished identity overlaps another population")
        if row.get("attempted_pattern") not in {"P1", "P2", "P3", "P4", "P5", "P6"}:
            raise PackError(f"{case}: unestablished row {index} attempts an invalid pattern")
        required_string(row.get("public_root"), f"{case} unestablished public root")
        canonical_sha256(row.get("unsafe_site_id"), f"{case} unestablished unsafe site")
        required_string(row.get("operand_role"), f"{case} unestablished operand role")
        source = row.get("source")
        if not isinstance(source, dict) or set(source) != {"identity", "kind"}:
            raise PackError(f"{case}: unestablished row {index} source fields drifted")
        required_string(source.get("identity"), f"{case} unestablished source identity")
        required_string(source.get("kind"), f"{case} unestablished source kind")
        required_string(row.get("controller"), f"{case} unestablished controller")
        required_string(row.get("capabilities"), f"{case} unestablished capabilities")
        requirement = row.get("requirement")
        if requirement is not None:
            required_string(requirement, f"{case} unestablished requirement")
        primary = row.get("primary_reason")
        primary_rank = UNESTABLISHED_REASON_RANK.get(primary)
        if primary_rank is None:
            raise PackError(f"{case}: unestablished row {index} primary reason is invalid")
        secondary = row.get("secondary_reasons")
        if not isinstance(secondary, list):
            raise PackError(f"{case}: unestablished row {index} secondary reasons are invalid")
        ranks = [UNESTABLISHED_REASON_RANK.get(reason) for reason in secondary]
        if any(rank is None for rank in ranks):
            raise PackError(f"{case}: unestablished row {index} secondary reason is invalid")
        numeric_ranks = [int(rank) for rank in ranks]
        if numeric_ranks != sorted(set(numeric_ranks)) or any(
            rank <= primary_rank for rank in numeric_ranks
        ):
            raise PackError(f"{case}: unestablished row {index} reason precedence drifted")
        if row.get("exposure") is not None:
            raise PackError(f"{case}: unestablished row {index} has non-null exposure")


def validate_complete_coverage(case: str, coverage: Any) -> None:
    if not isinstance(coverage, dict):
        raise PackError(f"{case}: coverage output is not an object")
    if coverage.get("schema_version") != "rap-unsoundaudit-coverage-v3-mvp":
        raise PackError(f"{case}: coverage has invalid schema_version")
    complete = coverage.get("complete")
    if complete is not True:
        raise PackError(
            f"{case}: mechanism coverage is not complete; partial results are a lower bound"
        )
    if coverage.get("semantics") != "exact":
        raise PackError(f"{case}: complete mechanism coverage must use exact semantics")
    failures = coverage.get("coverage_failures")
    if not isinstance(failures, list) or failures:
        raise PackError(f"{case}: complete mechanism coverage must have no coverage failures")
    census = coverage.get("unsafe_site_census")
    count_fields = {
        "discovered",
        "bound_exact",
        "optimized_away_exact",
        "unbound",
        "ambiguous",
        "elided",
    }
    if not isinstance(census, dict) or set(census) != {*count_fields, "by_kind"}:
        raise PackError(f"{case}: unsafe-site census fields drifted")
    counts = {
        field: required_nonnegative_integer(census.get(field), f"{case} census {field}")
        for field in count_fields
    }
    if counts["discovered"] != sum(counts[field] for field in count_fields - {"discovered"}):
        raise PackError(f"{case}: unsafe-site census partition drifted")
    if (
        counts["bound_exact"] + counts["optimized_away_exact"] != counts["discovered"]
        or counts["unbound"]
        or counts["ambiguous"]
        or counts["elided"]
    ):
        raise PackError(f"{case}: exact coverage contains a nonexact unsafe-site binding")


def validate_lifecycle_trace(
    case: str,
    events: list[dict[str, Any]],
    active_candidates: list[dict[str, Any]],
) -> dict[str, int]:
    if not events:
        raise PackError(f"{case}: candidate lifecycle trace is empty")
    emission_count = 0
    observed_emissions: dict[tuple[str, str, str], int] = {}
    for index, event in enumerate(events, start=1):
        stage = event.get("stage")
        owner = event.get("owner")
        if not isinstance(stage, str) or not stage:
            raise PackError(f"{case}: lifecycle event {index} lacks a stage")
        if not isinstance(owner, str) or not owner:
            raise PackError(f"{case}: lifecycle event {index} lacks an owner")
        if stage != "v3_emission":
            continue
        emission_count += 1
        if event.get("emitted") is not True:
            raise PackError(f"{case}: lifecycle emission {index} is not marked emitted")
        if event.get("primary_pattern") not in VALID_PATTERNS:
            raise PackError(f"{case}: lifecycle emission {index} has an invalid pattern")
        site_id = canonical_sha256(event.get("site_id"), "lifecycle emission site id")
        key = (owner, site_id, event["primary_pattern"])
        observed_emissions[key] = observed_emissions.get(key, 0) + 1
    expected_emissions: dict[tuple[str, str, str], int] = {}
    for candidate in active_candidates:
        pattern = candidate.get("primary_pattern")
        if pattern not in VALID_PATTERNS:
            raise PackError(f"{case}: active candidate has an invalid pattern")
        key = (
            candidate_root(candidate, "active candidate"),
            candidate_site_id(candidate, "active candidate"),
            pattern,
        )
        expected_emissions[key] = expected_emissions.get(key, 0) + 1
    if expected_emissions != observed_emissions:
        raise PackError(
            f"{case}: lifecycle/output terminal population drift: active={len(active_candidates)}, "
            f"emission_events={emission_count}; body_seed events are not candidate accounting"
        )
    return {"event_count": len(events), "emission_event_count": emission_count}


def validate_suppression_oracle(
    case: str,
    ledger: list[dict[str, Any]],
    active_by_id: dict[str, dict[str, Any]],
    oracle: dict[str, Any],
) -> None:
    declarations = oracle.get("suppression_roots", {})
    if not isinstance(declarations, dict):
        raise PackError(f"{case}: suppression_roots oracle is not an object")
    for root, declared in declarations.items():
        expectations = declared if isinstance(declared, list) else [declared]
        if not expectations or any(not isinstance(item, dict) for item in expectations):
            raise PackError(f"{case}: invalid suppression oracle for {root}")
        root_rows = [
            row
            for row in ledger
            if root_matches(candidate_root(row["candidate"], "suppressed candidate"), root)
        ]
        for expectation in expectations:
            reason = expectation.get("reason")
            if reason not in VALID_SUPPRESSION_REASONS:
                raise PackError(f"{case}: invalid suppression reason oracle for {root}")
            candidate_pattern = expectation.get("candidate_pattern")
            if candidate_pattern is not None and candidate_pattern not in VALID_PATTERNS:
                raise PackError(f"{case}: invalid suppressed candidate pattern oracle for {root}")
            expected_evidence = expectation.get("evidence_equals", {})
            if not isinstance(expected_evidence, dict):
                raise PackError(f"{case}: invalid suppression evidence oracle for {root}")
            replacement_absent = expectation.get("replacement_absent", False)
            if type(replacement_absent) is not bool:
                raise PackError(f"{case}: invalid replacement_absent oracle for {root}")
            replacement_pattern = expectation.get("replacement_pattern")
            if replacement_absent and replacement_pattern is not None:
                raise PackError(f"{case}: contradictory replacement oracle for {root}")
            matches = [
                row
                for row in root_rows
                if row.get("reason") == reason
                and (
                    candidate_pattern is None
                    or row["candidate"].get("primary_pattern") == candidate_pattern
                )
                and all(row.get("evidence", {}).get(key) == value for key, value in expected_evidence.items())
            ]
            minimum = expectation.get("min_count", 1)
            if type(minimum) is not int or minimum <= 0:
                raise PackError(f"{case}: invalid suppression min_count oracle for {root}")
            if len(matches) < minimum:
                raise PackError(
                    f"{case}: missing suppression for root={root}, reason={reason}, "
                    f"candidate_pattern={candidate_pattern}"
                )
            if replacement_absent:
                for row in matches:
                    if row.get("replacement_candidate_id") is not None:
                        raise PackError(
                            f"{case}: suppression for {root} unexpectedly has a replacement"
                        )
            if replacement_pattern is not None:
                if replacement_pattern not in VALID_PATTERNS:
                    raise PackError(f"{case}: invalid replacement pattern oracle for {root}")
                for row in matches:
                    replacement_id = row.get("replacement_candidate_id")
                    replacement = active_by_id.get(replacement_id)
                    if replacement is None:
                        raise PackError(
                            f"{case}: suppression for {root} lacks an active replacement"
                        )
                    if replacement.get("primary_pattern") != replacement_pattern:
                        raise PackError(
                            f"{case}: suppression for {root} replacement is "
                            f"{replacement.get('primary_pattern')}, expected {replacement_pattern}"
                        )
                    if not root_matches(
                        candidate_root(replacement, "replacement candidate"), root
                    ):
                        raise PackError(
                            f"{case}: suppression for {root} replacement belongs to another root"
                        )


def matching(rows: list[dict[str, Any]], root: str, pattern: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("primary_pattern") == pattern
        and isinstance(row.get("public_safe_root"), dict)
        and isinstance(row["public_safe_root"].get("owner"), str)
        and root_matches(row["public_safe_root"]["owner"], root)
    ]


def matching_root(rows: list[dict[str, Any]], root: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if isinstance(row.get("public_safe_root"), dict)
        and isinstance(row["public_safe_root"].get("owner"), str)
        and root_matches(row["public_safe_root"]["owner"], root)
    ]


def validate_candidate(row: dict[str, Any], output: Path) -> None:
    site = row.get("unsafe_site")
    if not isinstance(site, dict) or site.get("binding") != "bound_exact":
        raise PackError("target candidate is not bound_exact")
    failure = row.get("first_contract_failure")
    if not isinstance(failure, dict) or not isinstance(failure.get("span"), dict):
        raise PackError("target candidate lacks first contract failure")
    propagation = row.get("propagation")
    witness = propagation.get("witness") if isinstance(propagation, dict) else None
    if not isinstance(witness, list) or not witness:
        raise PackError("target candidate lacks witness")
    if witness[0].get("kind") != "entry" or witness[-1].get("kind") not in {
        "unsafe_site",
        "exposure",
    }:
        raise PackError("target candidate witness has invalid endpoints")
    refs = row.get("code_refs")
    if not isinstance(refs, list) or not refs:
        raise PackError("target candidate lacks code references")
    for code_id in refs:
        if not isinstance(code_id, str) or not (output / "code" / f"{code_id}.rs").is_file():
            raise PackError(f"target candidate references missing code block: {code_id}")


def warning_root_matches(owner: str, expected: str) -> bool:
    return owner == expected or owner.endswith(f"::{expected}")


def validate_warning_oracle(case: str, rows: list[dict], oracle: dict) -> None:
    present = oracle.get("warning_present_roots")
    absent = oracle.get("warning_absent_roots")
    if present is None and absent is None:
        return
    if present is None or absent is None:
        raise PackError(f"{case}: warning oracle must set both present and absent roots")
    if not isinstance(present, list) or not isinstance(absent, list):
        raise PackError(f"{case}: warning oracle roots must be arrays")
    owners = []
    for index, row in enumerate(rows):
        owner = row.get("public_root_owner")
        site_id = row.get("unsafe_site_id")
        predicate = row.get("predicate")
        sink_model = row.get("sink_model")
        if not isinstance(owner, str) or not owner:
            raise PackError(f"{case}: warning {index} lacks public_root_owner")
        if (
            not isinstance(site_id, str)
            or len(site_id) != 64
            or any(character not in "0123456789abcdef" for character in site_id)
        ):
            raise PackError(f"{case}: warning {index} has an invalid unsafe_site_id")
        if predicate not in {"nonnull", "in_bounds"}:
            raise PackError(f"{case}: warning {index} has an invalid predicate")
        if not isinstance(sink_model, str) or not sink_model:
            raise PackError(f"{case}: warning {index} lacks sink_model")
        owners.append(owner)
    for root in present:
        if not any(warning_root_matches(owner, root) for owner in owners):
            raise PackError(f"{case}: missing sanitizer warning for {root}")
    for root in absent:
        if any(warning_root_matches(owner, root) for owner in owners):
            raise PackError(f"{case}: unexpected sanitizer warning for {root}")


def validate_oracle(
    case: str, rows: list[dict[str, Any]], oracle: dict[str, Any], output: Path
) -> None:
    pattern = oracle.get("pattern")
    if pattern not in VALID_PATTERNS:
        raise PackError(f"{case}: invalid oracle pattern")
    for root in oracle.get("present_roots", []):
        matches = matching(rows, root, pattern)
        if not matches:
            raise PackError(f"{case}: missing {pattern} candidate for {root}")
        for row in matches:
            validate_candidate(row, output)
    for root in oracle.get("absent_roots", []):
        if matching(rows, root, pattern):
            raise PackError(f"{case}: unexpected {pattern} candidate for {root}")
    for root, expected_pattern in oracle.get("other_pattern_roots", {}).items():
        matches = matching(rows, root, expected_pattern)
        if not matches:
            raise PackError(f"{case}: missing {expected_pattern} candidate for {root}")
        for row in matches:
            validate_candidate(row, output)
    for root in oracle.get("zero_candidate_roots", []):
        matches = matching_root(rows, root)
        if matches:
            patterns = sorted({row.get("primary_pattern") for row in matches})
            raise PackError(f"{case}: unexpected candidate(s) for {root}: {patterns}")


def validate_unestablished_oracle(
    case: str, rows: list[dict[str, Any]], oracle: dict[str, Any]
) -> None:
    declarations = oracle.get("unestablished_roots", {})
    if not isinstance(declarations, dict):
        raise PackError(f"{case}: unestablished_roots oracle is not an object")
    for root, declared in declarations.items():
        expectations = declared if isinstance(declared, list) else [declared]
        if not expectations or any(not isinstance(item, dict) for item in expectations):
            raise PackError(f"{case}: invalid unestablished oracle for {root}")
        root_rows = [
            row
            for row in rows
            if isinstance(row.get("public_root"), str)
            and root_matches(row["public_root"], root)
        ]
        for expectation in expectations:
            pattern = expectation.get("attempted_pattern")
            reason = expectation.get("primary_reason")
            minimum = expectation.get("min_count", 1)
            if pattern not in VALID_PATTERNS:
                raise PackError(f"{case}: invalid unestablished pattern oracle for {root}")
            if reason not in UNESTABLISHED_REASON_RANK:
                raise PackError(f"{case}: invalid unestablished reason oracle for {root}")
            if type(minimum) is not int or minimum <= 0:
                raise PackError(f"{case}: invalid unestablished min_count oracle for {root}")
            matches = [
                row
                for row in root_rows
                if row.get("attempted_pattern") == pattern
                and row.get("primary_reason") == reason
            ]
            if len(matches) < minimum:
                raise PackError(
                    f"{case}: missing unestablished decision for root={root}, "
                    f"pattern={pattern}, reason={reason}"
                )
    absent = oracle.get("absent_unestablished_roots", [])
    if not isinstance(absent, list) or any(not isinstance(root, str) for root in absent):
        raise PackError(f"{case}: absent_unestablished_roots oracle is not an array")
    for root in absent:
        if any(
            isinstance(row.get("public_root"), str)
            and root_matches(row["public_root"], root)
            for row in rows
        ):
            raise PackError(f"{case}: unexpected unestablished decision for {root}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--cargo", type=Path, required=True)
    parser.add_argument("--rustc", type=Path, required=True)
    parser.add_argument("--cargo-rapx", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--case",
        choices=[f"p{number}" for number in range(1, 7)] + ["missing_sanitizer"],
        action="append",
    )
    args = parser.parse_args()

    repo = args.repo_root.resolve(strict=True)
    cargo = args.cargo.resolve(strict=True)
    rustc = args.rustc.resolve(strict=True)
    cargo_rapx = args.cargo_rapx.resolve(strict=True)
    base_environment = sanitized_base_environment(os.environ)
    cargo_commit = commit_hash(cargo, base_environment, "Cargo")
    rustc_commit = commit_hash(rustc, base_environment, "rustc")
    rust_sysroot = Path(
        tool_output([str(rustc), "--print", "sysroot"], base_environment, "rustc sysroot").strip()
    ).resolve(strict=True)
    base_environment.update(
        {
            "RUSTC": str(rustc),
            "RUST_SYSROOT": str(rust_sysroot),
            "UNSOUND_SCANNER_EXACT_CARGO_PATH": str(cargo),
            "UNSOUND_SCANNER_EXACT_CARGO_COMMIT": cargo_commit,
            "UNSOUND_SCANNER_EXPECTED_RUSTC_PATH": str(rustc),
            "UNSOUND_SCANNER_EXPECTED_RUSTC_COMMIT": rustc_commit,
        }
    )
    pack = repo / "tests/unsoundaudit-v3/mechanism-pack"
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logs = output_root / "logs"
    logs.mkdir(exist_ok=True)
    cases = args.case or [f"p{number}" for number in range(1, 7)] + ["missing_sanitizer"]
    results = []

    for case in cases:
        fixture = (pack / case).resolve(strict=True)
        oracle = read_json(fixture / "oracle.json")
        control_target = Path(tempfile.mkdtemp(prefix=f"rapx-v3-mvp-control-{case}."))
        scan_target = Path(tempfile.mkdtemp(prefix=f"rapx-v3-mvp-scan-{case}."))
        case_output = output_root / case
        if case_output.exists():
            shutil.rmtree(case_output)
        try:
            environment = base_environment.copy()
            environment.update({"CARGO_BUILD_JOBS": "1", "CARGO_TARGET_DIR": str(control_target)})
            run(
                [str(cargo), "check", "--lib", "--locked", "--jobs", "1"],
                fixture,
                environment,
                logs / f"{case}.control.log",
            )
            environment.update(
                {
                    "CARGO_TARGET_DIR": str(scan_target),
                    "UNSOUND_SCANNER_PROJECT_ROOT": str(fixture),
                    "RAP_UNSOUNDAUDIT_OUTPUT": str(case_output),
                    "UNSOUND_SCANNER_CENSUS_JSON_OUT": str(output_root / f"{case}.census.json"),
                    "UNSOUND_SCANNER_BINDING_JSON_OUT": str(output_root / f"{case}.binding.json"),
                    "UNSOUND_SCANNER_CANDIDATE_LIFECYCLE_TRACE": str(output_root / f"{case}.lifecycle.jsonl"),
                }
            )
            if os.uname().sysname == "Darwin":
                environment["DYLD_LIBRARY_PATH"] = str(rust_sysroot / "lib")
            else:
                environment["LD_LIBRARY_PATH"] = str(rust_sysroot / "lib")
            run(
                [str(cargo_rapx), "rapx", "-unsoundaudit", "--", "--lib", "--locked", "--jobs", "1"],
                fixture,
                environment,
                logs / f"{case}.scan.log",
            )
        finally:
            shutil.rmtree(control_target, ignore_errors=True)
            shutil.rmtree(scan_target, ignore_errors=True)

        rows = load_candidates(case_output / "candidates.jsonl")
        families = load_jsonl(case_output / "families.jsonl", "families")
        ledger = load_jsonl(case_output / "suppression-ledger.jsonl", "suppression ledger")
        unestablished = load_jsonl(
            case_output / "unestablished.jsonl", "unestablished decisions"
        )
        validate_oracle(case, rows, oracle, case_output)
        warning_rows = load_jsonl(
            case_output / "missing_sanitizer_warnings.jsonl",
            "missing sanitizer warnings",
        )
        validate_warning_oracle(case, warning_rows, oracle)
        validate_family_accounting(case, rows, families)
        active_by_id = validate_suppression_ledger(case, ledger, rows, case_output)
        validate_unestablished_rows(case, unestablished, rows, ledger)
        validate_suppression_oracle(case, ledger, active_by_id, oracle)
        validate_unestablished_oracle(case, unestablished, oracle)
        coverage = read_json(case_output / "coverage.json")
        validate_complete_coverage(case, coverage)
        accounting = validate_candidate_accounting(
            case,
            rows,
            ledger,
            unestablished,
            coverage,
        )
        lifecycle = load_jsonl(
            output_root / f"{case}.lifecycle.jsonl", "candidate lifecycle"
        )
        lifecycle_summary = validate_lifecycle_trace(case, lifecycle, rows)
        results.append(
            {
                "case": case,
                "pattern": oracle["pattern"],
            "candidate_count": len(rows),
            "unestablished_candidate_count": len(unestablished),
                "family_count": len(families),
                **accounting,
                "lifecycle_event_count": lifecycle_summary["event_count"],
                "lifecycle_emission_event_count": lifecycle_summary[
                    "emission_event_count"
                ],
                "coverage_complete": True,
            }
        )

    totals = {
        key: sum(result[key] for result in results)
        for key in (
            "active_count",
            "pure_suppressed_count",
            "reclassified_count",
            "unestablished_count",
            "suppressed_count",
            "considered_count",
            "family_count",
        )
    }
    (output_root / "capability_matrix.json").write_text(
        json.dumps(
            {
                "schema_version": "rap-unsoundaudit-capability-matrix-v3-mvp",
                "cases": results,
                "totals": totals,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"cases": results, "totals": totals}, sort_keys=True))
    return 0


def run_self_tests() -> None:
    def expect_rejected(label: str, action: Any) -> None:
        try:
            action()
        except PackError:
            return
        raise PackError(f"self-test: {label} was accepted")

    dirty_environment = {
        "PATH": "/usr/bin",
        "KEEP_ME": "yes",
        "CARGO": "/wrong/cargo",
        "CARGO_BUILD_TARGET": "wrong-target",
        "CARGO_PROFILE_DEV_OPT_LEVEL": "3",
        "CARGO_TARGET_DIR": "/wrong/target",
        "RUSTC": "/wrong/rustc",
        "RUSTFLAGS": "-C panic=abort",
        "RUSTUP_TOOLCHAIN": "stable",
        "RUST_SYSROOT": "/wrong/sysroot",
        "DYLD_LIBRARY_PATH": "/wrong/dyld",
        "DYLD_INSERT_LIBRARIES": "/wrong/injected.dylib",
        "LD_LIBRARY_PATH": "/wrong/ld",
        "LIBRARY_PATH": "/wrong/linker",
        "RAP_RECURSIVE": "1",
        "RAP_UNSOUNDAUDIT_OUTPUT": "/wrong/output",
        "UNSOUND_SCANNER_SOURCE_ROOT": "/wrong/source",
        "UNSOUND_SCANNER_PROJECT_ROOT": "/wrong/project",
    }
    sanitized = sanitized_base_environment(dirty_environment)
    if sanitized != {"PATH": "/usr/bin", "KEEP_ME": "yes"}:
        raise PackError(f"self-test: inherited environment was not closed: {sanitized}")

    active = [{"finding_id": "a" * 64}, {"finding_id": "b" * 64}]
    ledger = [
        {"replacement_candidate_id": None},
        {"replacement_candidate_id": "a" * 64},
    ]
    unestablished = [{"decision_id": "d" * 64}]
    expected = {
        "active_count": 2,
        "pure_suppressed_count": 1,
        "reclassified_count": 1,
        "unestablished_count": 1,
        "suppressed_count": 2,
        "considered_count": 5,
    }
    coverage = {"candidate_accounting": expected}
    if validate_candidate_accounting(
        "self-test", active, ledger, unestablished, coverage
    ) != expected:
        raise PackError("self-test: positive accounting result drifted")
    missing_accounting: dict[str, Any] = {}
    try:
        validate_candidate_accounting(
            "self-test", active, ledger, unestablished, missing_accounting
        )
    except PackError:
        pass
    else:
        raise PackError("self-test: missing writer accounting was accepted")
    drifted = {**expected, "considered_count": 6}
    try:
        validate_candidate_accounting(
            "self-test", active, ledger, unestablished, {"candidate_accounting": drifted}
        )
    except PackError:
        pass
    else:
        raise PackError("self-test: inconsistent writer accounting was accepted")
    try:
        validate_candidate_accounting(
            "self-test",
            active,
            [{}],
            unestablished,
            {"candidate_accounting": {**expected, "considered_count": 4}},
        )
    except PackError:
        pass
    else:
        raise PackError("self-test: missing replacement field was accepted")

    def self_test_span(line: int) -> dict[str, Any]:
        return {
            "path": "src/lib.rs",
            "start": {"line": line, "column": 1},
            "end": {"line": line, "column": 2},
        }

    def self_test_candidate(
        candidate_id: str, root: str, root_line: int, witness_length: int
    ) -> dict[str, Any]:
        witness = [
            {"kind": "entry", "owner": root, "span": self_test_span(root_line)}
        ]
        for line in range(2, witness_length):
            witness.append(
                {
                    "kind": "local_call",
                    "owner": "sample::middle",
                    "span": self_test_span(line),
                }
            )
        witness.append(
            {
                "kind": "unsafe_site",
                "owner": "sample::sink",
                "span": self_test_span(8),
            }
        )
        return {
            "finding_id": candidate_id * 64,
            "primary_pattern": "P1",
            "subject_role": "address",
            "public_safe_root": {"owner": root, "span": self_test_span(root_line)},
            "unsafe_site": {"site_id": "d" * 64, "binding": "bound_exact"},
            "first_contract_failure": {
                "kind": "precondition:raw_pointer_dereference",
                "span": self_test_span(8),
            },
            "contract": {
                "model_id": "raw_pointer_access_v1",
                "obligation": {"predicates": ["initialized", "aligned", "initialized"]},
            },
            "source": {
                "kind": "public_parameter",
                "identity": "sample::sink:formal:0",
            },
            "exposure": None,
            "propagation": {"witness": witness},
        }

    short = self_test_candidate("a", "sample::root_a", 1, 2)
    long = self_test_candidate("b", "sample::root_b", 2, 4)
    family_id, family_key = family_id_for_candidate(short)
    if family_id_for_candidate(long) != (family_id, family_key):
        raise PackError("self-test: non-family fields split a structural family")
    family = {
        "schema_version": MVP_FAMILY_SCHEMA,
        "family_id": family_id,
        "key": family_key,
        "raw_count": 2,
        "public_root_fanout": 2,
        "representatives": {"shortest": short, "longest": long},
    }
    validate_family_accounting("self-test", [short, long], [family])

    def mutated_family(path: str) -> dict[str, Any]:
        result = json.loads(json.dumps(family))
        if path == "family_id":
            result["family_id"] = "e" * 64
        elif path == "key":
            result["key"]["source"]["identity"] = "tampered"
        elif path == "raw_count":
            result["raw_count"] = 1
        elif path == "fanout":
            result["public_root_fanout"] = 1
        elif path == "shortest":
            result["representatives"]["shortest"] = long
        elif path == "longest":
            result["representatives"]["longest"] = short
        elif path == "fields":
            result["unexpected"] = True
        else:
            raise AssertionError(path)
        return result

    for mutation in (
        "family_id",
        "key",
        "raw_count",
        "fanout",
        "shortest",
        "longest",
        "fields",
    ):
        expect_rejected(
            f"family {mutation} drift",
            lambda mutation=mutation: validate_family_accounting(
                "self-test", [short, long], [mutated_family(mutation)]
            ),
        )
    expect_rejected(
        "missing family partition",
        lambda: validate_family_accounting("self-test", [short, long], []),
    )

    lifecycle = [
        {"stage": "body_seed", "owner": "sample::root"},
        {"stage": "body_seed", "owner": "sample::helper"},
        {
            "stage": "v3_emission",
            "owner": "sample::root",
            "emitted": True,
            "primary_pattern": "P1",
            "site_id": "c" * 64,
        },
    ]
    lifecycle_candidates = [
        {
            "public_safe_root": {"owner": "sample::root"},
            "unsafe_site": {"site_id": "c" * 64, "binding": "bound_exact"},
            "primary_pattern": "P1",
        }
    ]
    summary = validate_lifecycle_trace("self-test", lifecycle, lifecycle_candidates)
    if summary != {"event_count": 3, "emission_event_count": 1}:
        raise PackError("self-test: lifecycle summary drifted")
    try:
        validate_lifecycle_trace(
            "self-test", lifecycle, lifecycle_candidates + lifecycle_candidates
        )
    except PackError:
        pass
    else:
        raise PackError("self-test: lifecycle/output count drift was accepted")

    replacement_candidate = {
        "public_safe_root": {"owner": "sample::root"},
        "unsafe_site": {"site_id": "d" * 64, "binding": "bound_exact"},
        "subject_role": "address",
        "exposure": {"identity": "sample::root:return"},
    }
    validate_replacement_compatibility(
        "self-test", replacement_candidate, dict(replacement_candidate)
    )
    replacement_drifts = [
        {**replacement_candidate, "public_safe_root": {"owner": "sample::other"}},
        {
            **replacement_candidate,
            "unsafe_site": {"site_id": "e" * 64, "binding": "bound_exact"},
        },
        {**replacement_candidate, "subject_role": "argument:0"},
        {**replacement_candidate, "exposure": {"identity": "sample::other:return"}},
    ]
    for drifted_replacement in replacement_drifts:
        try:
            validate_replacement_compatibility(
                "self-test", replacement_candidate, drifted_replacement
            )
        except PackError:
            pass
        else:
            raise PackError("self-test: incompatible replacement was accepted")

    complete = {
        "schema_version": "rap-unsoundaudit-coverage-v3-mvp",
        "complete": True,
        "semantics": "exact",
        "coverage_failures": [],
        "candidate_accounting": expected,
        "unsafe_site_census": {
            "discovered": 2,
            "bound_exact": 1,
            "optimized_away_exact": 1,
            "unbound": 0,
            "ambiguous": 0,
            "elided": 0,
            "by_kind": [],
        },
    }
    validate_complete_coverage("self-test", complete)
    partial = {**complete, "complete": False, "semantics": "lower_bound"}
    try:
        validate_complete_coverage("self-test", partial)
    except PackError:
        pass
    else:
        raise PackError("self-test: partial coverage was accepted")


if __name__ == "__main__":
    try:
        if os.sys.argv[1:] == ["--self-test"]:
            run_self_tests()
            raise SystemExit(0)
        raise SystemExit(main())
    except PackError as error:
        print(f"mechanism-pack error: {error}", file=os.sys.stderr)
        raise SystemExit(1)

#!/usr/bin/env python3
"""Fail-closed validator for the frozen UnsoundAudit v2 fixture contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any


PATTERNS = tuple(f"pattern{number}" for number in range(1, 7))
PATTERN_SET = set(PATTERNS)
COMMITS = re.compile(r"^[0-9a-f]{40}$")
FINDING_IDS = re.compile(r"^[0-9a-f]{64}$")

TOP_KEYS = {
    "schema_version",
    "project_root",
    "source",
    "success",
    "unit_id",
    "rustc_invocation_id",
    "package_id",
    "package_name",
    "package_version",
    "crate_name",
    "crate_types",
    "target_kind",
    "target_triple",
    "manifest_path",
    "source_path",
    "extra_filename",
    "rustc_path",
    "cargo_supplied_rustc_path",
    "cargo_supplied_rustc_commit",
    "rap_compiler_commit",
    "rustc_commit",
    "pattern_counts",
    "finding_count",
    "findings",
}
FINDING_REQUIRED_KEYS = {
    "finding_id",
    "causal_key",
    "primary_pattern",
    "rule_id",
    "public_safe_root",
    "source",
    "first_contract_failure",
    "sink",
    "obligation",
    "validation_status",
    "propagation",
    "secondary_source_kinds",
    "heuristic_candidate_generator",
    "limitations",
}
FINDING_OPTIONAL_KEYS = {"family_support"}
SOURCE_KINDS = {
    "public_parameter",
    "literal_public_field",
    "internal_unsafe_origin",
    "internal_derived",
    "generic_nonempty_capability",
    "ffi_output",
    "open_behavior_output",
}
PREDICATES = {
    "valid_for_read",
    "in_bounds",
    "non_empty",
    "range_in_bounds",
    "nonnull",
    "referent_outlives_reference",
    "initialized",
    "valid_bool",
    "valid_utf8",
}
RULES = {
    "P1.raw_read",
    "P1.get_unchecked",
    "P2.raw_read",
    "P2.get_unchecked",
    "P3.lifetime_transmute",
    "P3.assume_init_bool",
    "P3.unchecked_utf8",
    "P4.bounds",
    "P4.offset",
    "P5.1.nonempty",
    "P6.ffi_out_param",
    "P6.open_trait_index",
}
OBLIGATION_ALIASES = {
    "initialized_and_valid_bool": ["initialized", "valid_bool"],
}


class ContractError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def exact_keys(value: Any, expected: set[str], location: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{location} must be an object")
    actual = set(value)
    require(
        actual == expected,
        f"{location} keys differ: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}",
    )
    return value


def nonempty_string(value: Any, location: str) -> str:
    require(isinstance(value, str) and bool(value), f"{location} must be a non-empty string")
    return value


def string_list(value: Any, location: str, *, nonempty: bool = False) -> list[str]:
    require(isinstance(value, list), f"{location} must be an array")
    require(not nonempty or bool(value), f"{location} must not be empty")
    require(all(isinstance(item, str) and bool(item) for item in value), f"{location} must contain non-empty strings")
    require(len(value) == len(set(value)), f"{location} must not contain duplicates")
    return value


def validate_position(value: Any, location: str) -> tuple[int, int]:
    position = exact_keys(value, {"line", "column"}, location)
    for key in ("line", "column"):
        require(isinstance(position[key], int) and not isinstance(position[key], bool), f"{location}.{key} must be an integer")
        require(position[key] >= 1, f"{location}.{key} must be at least 1")
    return position["line"], position["column"]


def validate_span(value: Any, location: str) -> tuple[str, tuple[int, int], tuple[int, int]]:
    span = exact_keys(value, {"path", "start", "end"}, location)
    path = nonempty_string(span["path"], f"{location}.path")
    require("\\" not in path, f"{location}.path must use '/' separators")
    pure = PurePosixPath(path)
    require(not pure.is_absolute(), f"{location}.path must be repository-relative")
    require(".." not in pure.parts and "." not in pure.parts, f"{location}.path must be normalized")
    start = validate_position(span["start"], f"{location}.start")
    end = validate_position(span["end"], f"{location}.end")
    require(end >= start, f"{location}.end precedes start")
    return path, start, end


def span_token(value: Any) -> str:
    path, start, end = validate_span(value, "causal span")
    return f"{path}:{start[0]}:{start[1]}-{end[0]}:{end[1]}"


def stable_identity(value: Any, location: str) -> str:
    identity = nonempty_string(value, location)
    require("\\" not in identity and "\0" not in identity, f"{location} must be machine-independent")
    require(not identity.startswith("/") and not re.match(r"^[A-Za-z]:[/\\]", identity), f"{location} must not be an absolute path")
    return identity


def validate_schema_contract(schema: Any) -> None:
    require(isinstance(schema, dict), "schema document must be an object")
    require(schema.get("additionalProperties") is False, "schema top-level must close additional properties")
    require(set(schema.get("required", [])) == TOP_KEYS, "schema top-level required keys drifted")
    require(set(schema.get("properties", {})) == TOP_KEYS, "schema top-level properties drifted")
    counts = schema["properties"]["pattern_counts"]
    require(counts.get("additionalProperties") is False, "pattern_counts must close additional properties")
    require(set(counts.get("required", [])) == PATTERN_SET, "schema pattern keys drifted")
    finding = schema.get("$defs", {}).get("finding", {})
    require(finding.get("additionalProperties") is False, "finding must close additional properties")
    require(set(finding.get("required", [])) == FINDING_REQUIRED_KEYS, "schema finding required keys drifted")
    require(set(finding.get("properties", {})) == FINDING_REQUIRED_KEYS | FINDING_OPTIONAL_KEYS, "schema finding properties drifted")


def validate_pattern_counts(value: Any, location: str = "pattern_counts") -> dict[str, int]:
    counts = exact_keys(value, PATTERN_SET, location)
    for pattern in PATTERNS:
        count = counts[pattern]
        require(isinstance(count, int) and not isinstance(count, bool) and count >= 0, f"{location}.{pattern} must be a non-negative integer")
    return counts


def validate_finding(value: Any, index: int, receipt_crate_name: str) -> tuple[str, str]:
    location = f"findings[{index}]"
    require(isinstance(value, dict), f"{location} must be an object")
    keys = set(value)
    require(FINDING_REQUIRED_KEYS <= keys, f"{location} is missing {sorted(FINDING_REQUIRED_KEYS - keys)}")
    require(keys <= FINDING_REQUIRED_KEYS | FINDING_OPTIONAL_KEYS, f"{location} has extra keys {sorted(keys - FINDING_REQUIRED_KEYS - FINDING_OPTIONAL_KEYS)}")

    finding_id = nonempty_string(value["finding_id"], f"{location}.finding_id")
    require(bool(FINDING_IDS.fullmatch(finding_id)), f"{location}.finding_id must be a lowercase SHA-256")
    primary = nonempty_string(value["primary_pattern"], f"{location}.primary_pattern")
    require(primary in PATTERN_SET, f"{location}.primary_pattern is unsupported")
    rule = nonempty_string(value["rule_id"], f"{location}.rule_id")
    require(rule in RULES, f"{location}.rule_id is outside the frozen registry")
    require(rule.startswith(f"P{primary[-1]}."), f"{location}.rule_id disagrees with primary pattern")

    causal = exact_keys(
        value["causal_key"],
        {"crate", "public_root", "source_origin", "first_contract_failure", "sink_or_exposure", "canonical_obligation"},
        f"{location}.causal_key",
    )
    for key, item in causal.items():
        stable_identity(item, f"{location}.causal_key.{key}")

    root = exact_keys(value["public_safe_root"], {"def_path", "span"}, f"{location}.public_safe_root")
    stable_identity(root["def_path"], f"{location}.public_safe_root.def_path")
    validate_span(root["span"], f"{location}.public_safe_root.span")

    source = exact_keys(value["source"], {"kind", "origin_key", "span"}, f"{location}.source")
    require(source["kind"] in SOURCE_KINDS, f"{location}.source.kind is outside the frozen registry")
    stable_identity(source["origin_key"], f"{location}.source.origin_key")
    validate_span(source["span"], f"{location}.source.span")

    for operation_name in ("first_contract_failure", "sink"):
        operation = exact_keys(value[operation_name], {"kind", "span"}, f"{location}.{operation_name}")
        nonempty_string(operation["kind"], f"{location}.{operation_name}.kind")
        validate_span(operation["span"], f"{location}.{operation_name}.span")

    obligation = exact_keys(value["obligation"], {"predicates", "subject"}, f"{location}.obligation")
    predicates = string_list(obligation["predicates"], f"{location}.obligation.predicates", nonempty=True)
    require(set(predicates) <= PREDICATES, f"{location}.obligation.predicates is outside the frozen registry")
    require(predicates == sorted(predicates), f"{location}.obligation.predicates must be sorted")
    stable_identity(obligation["subject"], f"{location}.obligation.subject")

    validation = exact_keys(value["validation_status"], {"status", "facts"}, f"{location}.validation_status")
    require(validation["status"] == "missing", f"{location}.validation_status.status must be missing for a finding")
    facts = string_list(validation["facts"], f"{location}.validation_status.facts")
    require(facts == sorted(facts), f"{location}.validation_status.facts must be sorted")

    propagation = exact_keys(value["propagation"], {"depth", "local_call_count", "boundaries", "witness"}, f"{location}.propagation")
    require(propagation["depth"] in {"intra_procedural", "inter_procedural", "recursive"}, f"{location}.propagation.depth is invalid")
    call_count = propagation["local_call_count"]
    require(call_count is None or (isinstance(call_count, int) and not isinstance(call_count, bool) and call_count >= 0), f"{location}.propagation.local_call_count is invalid")
    if propagation["depth"] == "recursive":
        require(call_count is None, f"{location}.recursive propagation must use null local_call_count")
    else:
        require(call_count is not None, f"{location}.non-recursive propagation needs local_call_count")
        require((call_count == 0) == (propagation["depth"] == "intra_procedural"), f"{location}.propagation depth/count disagree")
    string_list(propagation["boundaries"], f"{location}.propagation.boundaries")
    witness = propagation["witness"]
    require(isinstance(witness, list) and len(witness) >= 2, f"{location}.propagation.witness must contain at least two steps")
    for witness_index, step_value in enumerate(witness):
        step = exact_keys(step_value, {"kind", "function", "span"}, f"{location}.propagation.witness[{witness_index}]")
        require(step["kind"] in {"entry", "source", "local_call", "scc_cycle", "sink", "exposure"}, f"{location}.propagation.witness[{witness_index}].kind is invalid")
        nonempty_string(step["function"], f"{location}.propagation.witness[{witness_index}].function")
        validate_span(step["span"], f"{location}.propagation.witness[{witness_index}].span")

    secondary = string_list(value["secondary_source_kinds"], f"{location}.secondary_source_kinds")
    require(secondary == sorted(secondary), f"{location}.secondary_source_kinds must be sorted")
    require(value["heuristic_candidate_generator"] is True, f"{location} must identify itself as a heuristic candidate generator")
    string_list(value["limitations"], f"{location}.limitations", nonempty=True)

    if primary == "pattern5":
        require(value.get("family_support") == "partial", f"{location} P5 finding must declare partial family support")
    else:
        require("family_support" not in value, f"{location} non-P5 finding must not declare family_support")
    if rule == "P6.ffi_out_param":
        require(source["kind"] == "ffi_output", f"{location} FFI rule requires ffi_output source")
    if rule == "P6.open_trait_index":
        require(source["kind"] == "open_behavior_output", f"{location} open-trait rule requires open_behavior_output source")

    expected_causal = {
        "crate": receipt_crate_name,
        "public_root": root["def_path"],
        "source_origin": f"{source['kind']}:{source['origin_key']}@{span_token(source['span'])}",
        "first_contract_failure": f"{value['first_contract_failure']['kind']}@{span_token(value['first_contract_failure']['span'])}",
        "sink_or_exposure": f"{value['sink']['kind']}@{span_token(value['sink']['span'])}",
        "canonical_obligation": f"{'+'.join(predicates)}:{obligation['subject']}",
    }
    require(causal == expected_causal, f"{location}.causal_key does not match the structured finding")

    canonical = json.dumps(causal, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    expected_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    require(finding_id == expected_id, f"{location}.finding_id does not hash the canonical causal key")
    return finding_id, canonical


def validate_receipt(receipt: Any, schema: Any) -> dict[str, Any]:
    validate_schema_contract(schema)
    unit = exact_keys(receipt, TOP_KEYS, "receipt")
    require(unit["schema_version"] == "rap-unit-v2", "receipt.schema_version must be rap-unit-v2")
    require(unit["source"] == "rap", "receipt.source must be rap")
    require(unit["success"] is True, "receipt.success must be true")
    for key in TOP_KEYS - {"success", "crate_types", "pattern_counts", "finding_count", "findings"}:
        require(isinstance(unit[key], str), f"receipt.{key} must be a string")
    for key in ("cargo_supplied_rustc_commit", "rap_compiler_commit", "rustc_commit"):
        require(bool(COMMITS.fullmatch(unit[key])), f"receipt.{key} must be a lowercase 40-digit commit")
    crate_types = string_list(unit["crate_types"], "receipt.crate_types", nonempty=True)
    require(crate_types == sorted(crate_types), "receipt.crate_types must be sorted")
    counts = validate_pattern_counts(unit["pattern_counts"])
    finding_count = unit["finding_count"]
    require(isinstance(finding_count, int) and not isinstance(finding_count, bool) and finding_count >= 0, "receipt.finding_count must be a non-negative integer")
    require(isinstance(unit["findings"], list), "receipt.findings must be an array")
    require(finding_count == len(unit["findings"]), "receipt.finding_count disagrees with findings length")
    observed_counts = {pattern: 0 for pattern in PATTERNS}
    finding_ids: list[str] = []
    causal_keys: list[str] = []
    for index, finding in enumerate(unit["findings"]):
        finding_id, causal = validate_finding(finding, index, unit["crate_name"])
        finding_ids.append(finding_id)
        causal_keys.append(causal)
        observed_counts[finding["primary_pattern"]] += 1
    require(counts == observed_counts, "receipt.pattern_counts disagrees with primary findings")
    require(len(finding_ids) == len(set(finding_ids)), "receipt contains duplicate finding_id values")
    require(len(causal_keys) == len(set(causal_keys)), "receipt contains duplicate causal keys")
    require(causal_keys == sorted(causal_keys), "receipt.findings must be sorted by canonical causal key")
    return unit


def validate_oracle(receipt: dict[str, Any], oracle: Any) -> None:
    require(isinstance(oracle, dict), "oracle must be an object")
    case_id = nonempty_string(oracle.get("case_id"), "oracle.case_id")
    require(receipt["package_name"] == oracle.get("package_name"), f"{case_id}: package_name differs from oracle")
    expected_counts = validate_pattern_counts(oracle.get("expected_pattern_counts"), "oracle.expected_pattern_counts")
    require(receipt["pattern_counts"] == expected_counts, f"{case_id}: pattern counts differ: expected={expected_counts}, actual={receipt['pattern_counts']}")
    require(receipt["finding_count"] == oracle.get("expected_finding_count"), f"{case_id}: finding count differs")
    for forbidden in oracle.get("forbidden_extra_patterns", []):
        require(forbidden in PATTERN_SET and receipt["pattern_counts"][forbidden] == 0, f"{case_id}: forbidden pattern {forbidden} was reported")
    expected_primary = oracle.get("expected_primary_pattern")
    if expected_primary is None:
        require(not receipt["findings"], f"{case_id}: negative/noise oracle must have no findings")
        return
    require(len(receipt["findings"]) == 1, f"{case_id}: positive oracle requires exactly one finding")
    finding = receipt["findings"][0]
    require(finding["primary_pattern"] == expected_primary, f"{case_id}: primary pattern differs")
    require(finding["rule_id"] == oracle.get("expected_rule_id"), f"{case_id}: rule id differs")
    require(finding["source"]["kind"] == oracle.get("source"), f"{case_id}: source kind differs")
    require(finding["sink"]["kind"] == oracle.get("sink"), f"{case_id}: sink kind differs")
    expected_predicates = OBLIGATION_ALIASES.get(oracle.get("obligation"), [oracle.get("obligation")])
    require(finding["obligation"]["predicates"] == sorted(expected_predicates), f"{case_id}: obligation differs")
    expected_depth = oracle.get("propagation_depth")
    if expected_depth is not None:
        require(finding["propagation"]["depth"] == expected_depth, f"{case_id}: propagation depth differs")
    expected_support = oracle.get("family_support")
    require(finding.get("family_support") == expected_support, f"{case_id}: family support differs")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"failed to read JSON {path}: {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--oracle", type=Path)
    args = parser.parse_args()
    try:
        receipt = validate_receipt(read_json(args.receipt), read_json(args.schema))
        if args.oracle is not None:
            validate_oracle(receipt, read_json(args.oracle))
    except ContractError as error:
        print(f"receipt contract error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

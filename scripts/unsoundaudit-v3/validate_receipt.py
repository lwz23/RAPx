#!/usr/bin/env python3
"""Fail-closed validator for compiler-aligned rap-unit-v3 receipts."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """Raised when a receipt violates the closed v3 contract."""


HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
SITE_KINDS = {
    "unsafe_call",
    "inline_assembly",
    "restricted_validity_initialization",
    "unsafe_field_initialization",
    "unsafe_field_use",
    "mutable_static_access",
    "extern_static_access",
    "raw_pointer_dereference",
    "union_field_access",
    "layout_constrained_mutation",
    "layout_constrained_borrow",
    "target_feature_call",
    "unsafe_binder_cast",
    "explicit_drop_call",
}
PATTERN_SOURCE = {
    "P1": "public_parameter",
    "P2": "public_field",
    "P5": "internal_unsafe_origin",
    "P6": "internal_derivation",
    "P4": "generic_capability_gap",
    "P3": "external_behavior_output",
}
BINDINGS = {"bound_exact", "optimized_away_exact", "unbound", "ambiguous", "elided"}
COVERAGE_REASON_BINDING = {
    "unbound_unsafe_site": "unbound",
    "ambiguous_unsafe_site": "ambiguous",
    "elided_unsafe_site": "elided",
}
TOP_LEVEL_KEYS = {
    "schema_version",
    "source",
    "success",
    "analysis_status",
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
    "cargo_supplied_rustc_commit",
    "rap_compiler_commit",
    "rustc_commit",
    "toolchain_adapter",
    "unsafe_site_census",
    "unsafe_sites",
    "coverage_failures",
    "observed_counts",
    "finding_count",
    "findings",
    "missing_sanitizer_warnings",
    "code_dictionary",
}


def read_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fail(message: str) -> None:
    raise ContractError(message)


def exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        fail(f"{label} keys differ: missing={sorted(expected - actual)} extra={sorted(actual - expected)}")
    return value


def nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        fail(f"{label} must be a non-empty string")
    return value


def integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        fail(f"{label} must be a non-negative integer")
    return value


def enum(value: Any, choices: set[str], label: str) -> str:
    if value not in choices:
        fail(f"{label} is not one of {sorted(choices)}")
    return value


def digest(value: Any, width: int, label: str) -> str:
    pattern = HEX40 if width == 40 else HEX64
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        fail(f"{label} must be {width} lowercase hexadecimal characters")
    return value


def relative_path(value: Any, label: str) -> str:
    path = nonempty(value, label)
    if (
        path.startswith("/")
        or re.match(r"[A-Za-z]:", path)
        or "\\" in path
        or "//" in path
        or any(part in {".", ".."} for part in path.split("/"))
    ):
        fail(f"{label} must be a normalized relative path")
    return path


def sorted_unique(values: Any, label: str, *, nonempty_items: bool = True) -> list[str]:
    if not isinstance(values, list):
        fail(f"{label} must be an array")
    if nonempty_items and not values:
        fail(f"{label} must not be empty")
    for index, value in enumerate(values):
        nonempty(value, f"{label}[{index}]")
    if values != sorted(set(values)):
        fail(f"{label} must be sorted and unique")
    return values


def validate_position(value: Any, label: str) -> tuple[int, int]:
    item = exact_keys(value, {"line", "column"}, label)
    line = integer(item["line"], f"{label}.line")
    column = integer(item["column"], f"{label}.column")
    if line == 0 or column == 0:
        fail(f"{label} is one-based")
    return line, column


def validate_span(value: Any, label: str) -> tuple[str, tuple[int, int], tuple[int, int]]:
    item = exact_keys(value, {"path", "start", "end"}, label)
    path = relative_path(item["path"], f"{label}.path")
    start = validate_position(item["start"], f"{label}.start")
    end = validate_position(item["end"], f"{label}.end")
    if end < start:
        fail(f"{label} end precedes start")
    return path, start, end


def validate_schema_contract(schema: Any) -> None:
    if not isinstance(schema, dict):
        fail("schema must be an object")
    if schema.get("$id") != "https://rapx.dev/schemas/rap-unit-v3.schema.json":
        fail("unexpected schema id")
    if schema.get("additionalProperties") is not False:
        fail("top-level schema must be closed")
    if set(schema.get("required", [])) != TOP_LEVEL_KEYS:
        fail("schema top-level required keys drifted")
    properties = schema.get("properties")
    if not isinstance(properties, dict) or set(properties) != TOP_LEVEL_KEYS:
        fail("schema top-level properties drifted")
    if properties.get("schema_version", {}).get("const") != "rap-unit-v3":
        fail("schema version drifted")


def validate_adapter(value: Any, rustc_commit: str) -> None:
    item = exact_keys(value, {"adapter_id", "rustc_commit", "checker_sha256", "emitted_kinds"}, "toolchain_adapter")
    nonempty(item["adapter_id"], "toolchain_adapter.adapter_id")
    if digest(item["rustc_commit"], 40, "toolchain_adapter.rustc_commit") != rustc_commit:
        fail("toolchain adapter rustc commit does not match receipt")
    digest(item["checker_sha256"], 64, "toolchain_adapter.checker_sha256")
    kinds = sorted_unique(item["emitted_kinds"], "toolchain_adapter.emitted_kinds")
    for kind in kinds:
        enum(kind, SITE_KINDS, "toolchain_adapter.emitted_kinds item")


def validate_census(value: Any) -> dict[str, Any]:
    item = exact_keys(value, {"discovered", "bound_exact", "optimized_away_exact", "unbound", "ambiguous", "elided", "by_kind"}, "unsafe_site_census")
    counts = {name: integer(item[name], f"unsafe_site_census.{name}") for name in ("discovered", "bound_exact", "optimized_away_exact", "unbound", "ambiguous", "elided")}
    if counts["discovered"] != sum(counts[name] for name in ("bound_exact", "optimized_away_exact", "unbound", "ambiguous", "elided")):
        fail("unsafe site census does not reconcile")
    by_kind = item["by_kind"]
    if not isinstance(by_kind, list):
        fail("unsafe_site_census.by_kind must be an array")
    kinds: list[str] = []
    total = 0
    for index, raw in enumerate(by_kind):
        entry = exact_keys(raw, {"kind", "count"}, f"unsafe_site_census.by_kind[{index}]")
        kinds.append(enum(entry["kind"], SITE_KINDS, f"unsafe_site_census.by_kind[{index}].kind"))
        count = integer(entry["count"], f"unsafe_site_census.by_kind[{index}].count")
        if count == 0:
            fail("unsafe_site_census.by_kind must omit zero-count kinds")
        total += count
    if kinds != sorted(set(kinds)):
        fail("unsafe_site_census.by_kind must be sorted and unique")
    if total != counts["discovered"]:
        fail("unsafe_site_census.by_kind does not sum to discovered")
    return counts


def validate_sites(value: Any, census: dict[str, Any], emitted_kinds: set[str]) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list) or len(value) != census["discovered"]:
        fail("unsafe_sites must contain exactly every discovered site")
    sites: dict[str, dict[str, Any]] = {}
    binding_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    for index, raw in enumerate(value):
        site = exact_keys(raw, {"site_id", "kind", "owner", "span", "binding", "candidate_count"}, f"unsafe_sites[{index}]")
        site_id = digest(site["site_id"], 64, f"unsafe_sites[{index}].site_id")
        if site_id in sites:
            fail("unsafe site ids must be unique")
        kind = enum(site["kind"], SITE_KINDS, f"unsafe_sites[{index}].kind")
        if kind not in emitted_kinds:
            fail("unsafe site kind is not emitted by the selected adapter")
        nonempty(site["owner"], f"unsafe_sites[{index}].owner")
        validate_span(site["span"], f"unsafe_sites[{index}].span")
        binding = enum(site["binding"], BINDINGS, f"unsafe_sites[{index}].binding")
        integer(site["candidate_count"], f"unsafe_sites[{index}].candidate_count")
        sites[site_id] = site
        binding_counts[binding] += 1
        kind_counts[kind] += 1
    for binding in BINDINGS:
        if binding_counts[binding] != census[binding]:
            fail(f"unsafe_sites binding count differs for {binding}")
    return sites


def validate_coverage(value: Any, sites: dict[str, dict[str, Any]], status: str) -> None:
    if not isinstance(value, list):
        fail("coverage_failures must be an array")
    seen: set[tuple[Any, ...]] = set()
    site_failures: set[str] = set()
    for index, raw in enumerate(value):
        failure = exact_keys(raw, {"reason", "owner", "site_id", "candidate_count"}, f"coverage_failures[{index}]")
        reason = enum(
            failure["reason"],
            {"collector_missing_after_steal", "collector_failed", *COVERAGE_REASON_BINDING},
            f"coverage_failures[{index}].reason",
        )
        owner = nonempty(failure["owner"], f"coverage_failures[{index}].owner")
        count = integer(failure["candidate_count"], f"coverage_failures[{index}].candidate_count")
        site_id = failure["site_id"]
        if reason in COVERAGE_REASON_BINDING:
            digest(site_id, 64, f"coverage_failures[{index}].site_id")
            site = sites.get(site_id)
            if site is None:
                fail("coverage failure references an unknown site")
            if site["binding"] != COVERAGE_REASON_BINDING[reason] or site["owner"] != owner:
                fail("coverage failure does not match its unsafe site terminal state")
            if site["candidate_count"] != count:
                fail("coverage failure candidate count differs from unsafe site")
            site_failures.add(site_id)
        elif site_id is not None:
            fail("collector coverage failure must not invent a site id")
        key = (reason, owner, site_id, count)
        if key in seen:
            fail("coverage failures must be unique")
        seen.add(key)
    expected_site_failures = {
        site_id
        for site_id, site in sites.items()
        if site["binding"] not in {"bound_exact", "optimized_away_exact"}
    }
    if site_failures != expected_site_failures:
        fail("every non-bound site must have exactly one coverage failure")
    if status == "complete" and value:
        fail("complete receipt cannot contain coverage failures")
    if status == "partial" and not value:
        fail("partial receipt requires a coverage failure")


def validate_counts(value: Any, status: str) -> dict[str, int]:
    keys = {"semantics", "pattern1", "pattern2", "pattern5", "pattern6", "pattern4", "pattern3"}
    item = exact_keys(value, keys, "observed_counts")
    expected_semantics = "exact" if status == "complete" else "lower_bound"
    if item["semantics"] != expected_semantics:
        fail(f"{status} receipt must use {expected_semantics} counts")
    return {f"P{number}": integer(item[f"pattern{number}"], f"observed_counts.pattern{number}") for number in range(1, 7)}


def validate_code_dictionary(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        fail("code_dictionary must be an array")
    blocks: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for index, raw in enumerate(value):
        block = exact_keys(raw, {"code_id", "kind", "owner", "path", "span", "sha256", "text"}, f"code_dictionary[{index}]")
        code_id = digest(block["code_id"], 64, f"code_dictionary[{index}].code_id")
        if code_id in blocks:
            fail("code dictionary ids must be unique")
        enum(block["kind"], {"function_body", "signature", "safety_doc"}, f"code_dictionary[{index}].kind")
        nonempty(block["owner"], f"code_dictionary[{index}].owner")
        relative_path(block["path"], f"code_dictionary[{index}].path")
        validate_span(block["span"], f"code_dictionary[{index}].span")
        expected_sha = hashlib.sha256(block["text"].encode("utf-8")).hexdigest() if isinstance(block["text"], str) else ""
        if digest(block["sha256"], 64, f"code_dictionary[{index}].sha256") != expected_sha:
            fail("code dictionary text hash mismatch")
        blocks[code_id] = block
        order.append(code_id)
    if order != sorted(order):
        fail("code_dictionary must be sorted by code_id")
    return blocks


def validate_findings(value: Any, finding_count: int, pattern_counts: dict[str, int], sites: dict[str, dict[str, Any]], blocks: dict[str, dict[str, Any]]) -> None:
    if not isinstance(value, list) or len(value) != finding_count:
        fail("findings length differs from finding_count")
    ids: set[str] = set()
    actual_patterns: Counter[str] = Counter()
    site_references: Counter[str] = Counter()
    referenced_code: set[str] = set()
    for index, raw in enumerate(value):
        finding = exact_keys(raw, {"finding_id", "primary_pattern", "mechanism_id", "classification_confidence", "public_safe_root", "source", "first_contract_failure", "unsafe_site", "exposure", "contract", "propagation", "code_refs"}, f"findings[{index}]")
        finding_id = digest(finding["finding_id"], 64, f"findings[{index}].finding_id")
        if finding_id in ids:
            fail("finding ids must be unique")
        ids.add(finding_id)
        pattern = enum(finding["primary_pattern"], set(PATTERN_SOURCE), f"findings[{index}].primary_pattern")
        actual_patterns[pattern] += 1
        nonempty(finding["mechanism_id"], f"findings[{index}].mechanism_id")
        enum(finding["classification_confidence"], {"exact", "structural"}, f"findings[{index}].classification_confidence")

        root = exact_keys(finding["public_safe_root"], {"owner", "span", "code_id"}, f"findings[{index}].public_safe_root")
        nonempty(root["owner"], f"findings[{index}].public_safe_root.owner")
        validate_span(root["span"], f"findings[{index}].public_safe_root.span")
        root_code = digest(root["code_id"], 64, f"findings[{index}].public_safe_root.code_id")
        referenced_code.add(root_code)

        source_keys = {"kind", "identity", "span"}
        if "generic_capability" in finding["source"]:
            source_keys.add("generic_capability")
        source = exact_keys(finding["source"], source_keys, f"findings[{index}].source")
        if source["kind"] != PATTERN_SOURCE[pattern]:
            fail("primary pattern and causal source kind disagree")
        nonempty(source["identity"], f"findings[{index}].source.identity")
        validate_span(source["span"], f"findings[{index}].source.span")
        if source.get("generic_capability") is not None and not isinstance(source["generic_capability"], dict):
            fail("generic_capability must be an object or null")
        failure = exact_keys(finding["first_contract_failure"], {"kind", "span"}, f"findings[{index}].first_contract_failure")
        nonempty(failure["kind"], f"findings[{index}].first_contract_failure.kind")
        validate_span(failure["span"], f"findings[{index}].first_contract_failure.span")

        site_ref = exact_keys(finding["unsafe_site"], {"site_id", "kind", "owner", "span", "callee", "operands", "destination", "binding"}, f"findings[{index}].unsafe_site")
        site_id = digest(site_ref["site_id"], 64, f"findings[{index}].unsafe_site.site_id")
        site = sites.get(site_id)
        if site is None:
            fail("finding references unknown unsafe site")
        enum(site_ref["binding"], {"bound_exact", "low_confidence"}, f"findings[{index}].unsafe_site.binding")
        if site_ref["kind"] != site["kind"] or site_ref["owner"] != site["owner"] or site_ref["span"] != site["span"]:
            fail("finding unsafe site identity differs from census")
        if site["binding"] == "bound_exact" and site_ref["binding"] != "bound_exact":
            fail("exactly bound site must remain exact in finding")
        if site["binding"] != "bound_exact" and site_ref["binding"] != "low_confidence":
            fail("coverage-gap finding must be marked low confidence")
        if site_ref["callee"] is not None:
            nonempty(site_ref["callee"], f"findings[{index}].unsafe_site.callee")
        sorted_unique(site_ref["operands"], f"findings[{index}].unsafe_site.operands", nonempty_items=False)
        if site_ref["destination"] is not None:
            nonempty(site_ref["destination"], f"findings[{index}].unsafe_site.destination")
        validate_span(site_ref["span"], f"findings[{index}].unsafe_site.span")
        site_references[site_id] += 1

        if finding["exposure"] is not None:
            exposure = exact_keys(finding["exposure"], {"kind", "span"}, f"findings[{index}].exposure")
            nonempty(exposure["kind"], f"findings[{index}].exposure.kind")
            validate_span(exposure["span"], f"findings[{index}].exposure.span")
        if pattern == "P5" and finding["exposure"] is None:
            fail("P5 finding requires an exposure")

        contract = exact_keys(finding["contract"], {"precision", "model_id", "relevant_roles", "obligation", "safety_doc"}, f"findings[{index}].contract")
        precision = enum(contract["precision"], {"precise", "generic", "opaque"}, f"findings[{index}].contract.precision")
        if precision == "opaque":
            if contract["model_id"] is not None or contract["obligation"] is not None:
                fail("opaque contract cannot invent a model or machine obligation")
        else:
            nonempty(contract["model_id"], f"findings[{index}].contract.model_id")
            if not isinstance(contract["obligation"], dict):
                fail("precise/generic contract requires an obligation object")
        sorted_unique(contract["relevant_roles"], f"findings[{index}].contract.relevant_roles")
        safety_doc = exact_keys(contract["safety_doc"], {"status", "code_id", "reason"}, f"findings[{index}].contract.safety_doc")
        if safety_doc["status"] == "available":
            doc_id = digest(safety_doc["code_id"], 64, f"findings[{index}].contract.safety_doc.code_id")
            if safety_doc["reason"] is not None:
                fail("available safety doc cannot have an unavailable reason")
            referenced_code.add(doc_id)
        elif safety_doc["status"] == "unavailable":
            if safety_doc["code_id"] is not None:
                fail("unavailable safety doc cannot reference code")
            nonempty(safety_doc["reason"], f"findings[{index}].contract.safety_doc.reason")
        else:
            fail("unknown safety doc status")

        propagation = exact_keys(finding["propagation"], {"depth", "boundaries", "witness"}, f"findings[{index}].propagation")
        enum(propagation["depth"], {"intra_procedural", "inter_procedural", "recursive"}, f"findings[{index}].propagation.depth")
        sorted_unique(propagation["boundaries"], f"findings[{index}].propagation.boundaries", nonempty_items=False)
        witness = propagation["witness"]
        if not isinstance(witness, list) or len(witness) < 2:
            fail("propagation witness requires at least entry and endpoint")
        for step_index, raw_step in enumerate(witness):
            step = exact_keys(raw_step, {"kind", "owner", "span"}, f"findings[{index}].propagation.witness[{step_index}]")
            enum(step["kind"], {"entry", "source", "local_call", "callback", "scc_cycle", "unsafe_site", "exposure"}, f"findings[{index}].propagation.witness[{step_index}].kind")
            nonempty(step["owner"], f"findings[{index}].propagation.witness[{step_index}].owner")
            validate_span(step["span"], f"findings[{index}].propagation.witness[{step_index}].span")
        if witness[0]["kind"] != "entry" or not any(step["kind"] == "unsafe_site" for step in witness):
            fail("witness must begin at entry and reach the unsafe site")
        if pattern == "P5" and witness[-1]["kind"] != "exposure":
            fail("P5 witness must terminate at exposure")

        code_refs = sorted_unique(finding["code_refs"], f"findings[{index}].code_refs")
        for code_id in code_refs:
            digest(code_id, 64, f"findings[{index}].code_refs item")
            referenced_code.add(code_id)
        if root_code not in code_refs:
            fail("public root code must be present in code_refs")

    if dict(actual_patterns) != {pattern: count for pattern, count in pattern_counts.items() if count}:
        fail("observed pattern counts differ from findings")
    for site_id, site in sites.items():
        if site_references[site_id] != site["candidate_count"]:
            fail("unsafe site candidate_count differs from emitted findings")
    missing = referenced_code - set(blocks)
    if missing:
        fail(f"finding references missing code blocks: {sorted(missing)}")
    unused = set(blocks) - referenced_code
    if unused:
        fail(f"code dictionary contains unreferenced blocks: {sorted(unused)}")


def validate_missing_sanitizer_warnings(value: Any, findings: Any) -> None:
    if not isinstance(value, list):
        fail("missing_sanitizer_warnings must be an array")
    claimed = set()
    if not isinstance(findings, list):
        fail("findings must be an array")
    for finding in findings:
        if not isinstance(finding, dict):
            fail("finding must be an object")
        root = finding.get("public_safe_root")
        site = finding.get("unsafe_site")
        owner = root.get("owner") if isinstance(root, dict) else None
        site_id = site.get("site_id") if isinstance(site, dict) else None
        if isinstance(owner, str) and isinstance(site_id, str):
            claimed.add((owner, site_id))
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(value):
        item = exact_keys(
            row,
            {"public_root_owner", "unsafe_site_id", "predicate", "sink_model"},
            f"missing_sanitizer_warnings[{index}]",
        )
        owner = nonempty(item["public_root_owner"], f"missing_sanitizer_warnings[{index}].public_root_owner")
        site_id = digest(item["unsafe_site_id"], 64, f"missing_sanitizer_warnings[{index}].unsafe_site_id")
        enum(item["predicate"], {"nonnull", "in_bounds"}, f"missing_sanitizer_warnings[{index}].predicate")
        nonempty(item["sink_model"], f"missing_sanitizer_warnings[{index}].sink_model")
        key = (owner, site_id)
        if key in seen:
            fail(f"duplicate missing sanitizer warning for {owner} {site_id}")
        seen.add(key)
        if key in claimed:
            fail("missing sanitizer warning duplicates an active finding")


def validate_receipt(receipt: Any, schema: Any) -> None:
    validate_schema_contract(schema)
    item = exact_keys(receipt, TOP_LEVEL_KEYS, "receipt")
    if item["schema_version"] != "rap-unit-v3" or item["source"] != "rap" or item["success"] is not True:
        fail("receipt discriminator fields are invalid")
    status = enum(item["analysis_status"], {"complete", "partial"}, "analysis_status")
    for field in ("unit_id", "rustc_invocation_id", "package_id", "package_name", "package_version", "crate_name", "target_kind", "target_triple"):
        nonempty(item[field], field)
    sorted_unique(item["crate_types"], "crate_types")
    relative_path(item["manifest_path"], "manifest_path")
    relative_path(item["source_path"], "source_path")
    digest(item["cargo_supplied_rustc_commit"], 40, "cargo_supplied_rustc_commit")
    digest(item["rap_compiler_commit"], 40, "rap_compiler_commit")
    rustc_commit = digest(item["rustc_commit"], 40, "rustc_commit")
    validate_adapter(item["toolchain_adapter"], rustc_commit)
    census = validate_census(item["unsafe_site_census"])
    emitted_kinds = set(item["toolchain_adapter"]["emitted_kinds"])
    sites = validate_sites(item["unsafe_sites"], census, emitted_kinds)
    expected_by_kind = Counter(site["kind"] for site in sites.values())
    actual_by_kind = {entry["kind"]: entry["count"] for entry in item["unsafe_site_census"]["by_kind"]}
    if dict(expected_by_kind) != actual_by_kind:
        fail("unsafe_site_census.by_kind differs from unsafe_sites")
    validate_coverage(item["coverage_failures"], sites, status)
    pattern_counts = validate_counts(item["observed_counts"], status)
    finding_count = integer(item["finding_count"], "finding_count")
    if sum(pattern_counts.values()) != finding_count:
        fail("finding_count differs from observed pattern counts")
    blocks = validate_code_dictionary(item["code_dictionary"])
    validate_findings(item["findings"], finding_count, pattern_counts, sites, blocks)
    validate_missing_sanitizer_warnings(item["missing_sanitizer_warnings"], item["findings"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--schema", type=Path, required=True)
    args = parser.parse_args()
    validate_receipt(read_json(args.receipt), read_json(args.schema))

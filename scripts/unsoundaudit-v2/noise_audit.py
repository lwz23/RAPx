#!/usr/bin/env python3
"""Deterministic single-review diagnostics for stdlib finding families."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from run_stdlib_pilot import UNITS, load_protocol, sha256
from validate_receipt import ContractError, read_json


CLUSTER_SCHEMA = "unsoundaudit-v2-stdlib-noise-clusters-v1"
AUDIT_SAMPLE_SCHEMA = "unsoundaudit-v2-stdlib-noise-audit-sample-v1"
LABEL_SCHEMA = "unsoundaudit-v2-stdlib-noise-labels-v1"
AUDIT_REPORT_SCHEMA = "unsoundaudit-v2-stdlib-noise-audit-report-v1"
FAMILY_REVIEW_SCHEMA = "unsoundaudit-v2-stdlib-noise-family-reviews-v1"
SUPPRESSION_ALLOWLIST_SCHEMA = "unsoundaudit-v2-stdlib-suppression-allowlist-v1"
NORMALIZED_SCHEMA = "unsoundaudit-v2-stdlib-unit-v1"
SAMPLE_SCHEMA = "unsoundaudit-v2-stdlib-sample-v1"
NORMALIZED_KEYS = {
    "schema_version",
    "unit",
    "protocol_sha256",
    "toolchain",
    "source_census",
    "pattern_counts",
    "finding_count",
    "findings",
}
SAMPLE_KEYS = {
    "schema_version",
    "unit",
    "protocol_sha256",
    "population_sha256",
    "population_size",
    "sample_seed",
    "sample_size",
    "ordering",
    "strata",
    "findings",
}
LABELS = ("A", "B", "C", "D", "E", "U")
ROOT_CAUSES = {
    "real_unsoundness",
    "needs_contract_evidence",
    "type_or_constructor_invariant",
    "missing_guard_recognizer",
    "cross_call_summary_loss",
    "source_misclassification",
    "wrong_binding_or_alias",
    "infeasible_cfg_or_route",
    "duplicate_or_identity_defect",
    "unsupported_boundary",
    "insufficient_evidence",
}
FAMILY_DISPOSITIONS = {
    "retain",
    "generic_repair",
    "stdlib_invariant_suppression",
    "reporting_only_fanout",
    "insufficient_evidence",
}


def fail(message: str) -> None:
    raise ContractError(message)


def canonical_compact(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_compact(value)).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def stable_span(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"path", "start", "end"}:
        fail("finding span schema drifted")
    path = value["path"]
    start = value["start"]
    end = value["end"]
    if (
        not isinstance(path, str)
        or not path
        or path.startswith("/")
        or not isinstance(start, dict)
        or not isinstance(end, dict)
        or set(start) != {"line", "column"}
        or set(end) != {"line", "column"}
        or not all(isinstance(point[key], int) and point[key] > 0 for point in (start, end) for key in ("line", "column"))
    ):
        fail("finding span is not a stable relative source identity")
    return {
        "path": path,
        "start": {"line": start["line"], "column": start["column"]},
        "end": {"line": end["line"], "column": end["column"]},
    }


def stable_operation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"kind", "span"}:
        fail("finding operation schema drifted")
    if not isinstance(value["kind"], str) or not value["kind"]:
        fail("finding operation kind is invalid")
    return {"kind": value["kind"], "span": stable_span(value["span"])}


def span_token(value: Any) -> str:
    span = stable_span(value)
    return (
        f"{span['path']}:{span['start']['line']}:{span['start']['column']}"
        f"-{span['end']['line']}:{span['end']['column']}"
    )


def family_key(unit: str, finding: dict[str, Any]) -> dict[str, Any]:
    if unit not in UNITS or not isinstance(finding, dict):
        fail("noise-family input is invalid")
    obligation = finding.get("obligation")
    source = finding.get("source")
    propagation = finding.get("propagation")
    validation = finding.get("validation_status")
    if (
        not isinstance(obligation, dict)
        or set(obligation) != {"predicates", "subject"}
        or not isinstance(obligation["predicates"], list)
        or not obligation["predicates"]
        or any(not isinstance(item, str) or not item for item in obligation["predicates"])
        or not isinstance(obligation["subject"], str)
        or not isinstance(source, dict)
        or not isinstance(source.get("kind"), str)
        or not isinstance(propagation, dict)
        or propagation.get("depth") not in {"intra_procedural", "inter_procedural", "recursive"}
        or not isinstance(validation, dict)
        or not isinstance(validation.get("status"), str)
        or not isinstance(finding.get("rule_id"), str)
    ):
        fail("finding cannot form a closed noise-family key")
    return {
        "unit": unit,
        "rule_id": finding["rule_id"],
        "sink": stable_operation(finding.get("sink")),
        "first_contract_failure": stable_operation(finding.get("first_contract_failure")),
        "predicates": sorted(set(obligation["predicates"])),
        "subject": obligation["subject"],
        "source_kind": source["kind"],
        "propagation_depth": propagation["depth"],
        "validation_status": validation["status"],
    }


def family_id(key: dict[str, Any]) -> str:
    return digest(key)


def selection_hash(seed: str, finding: dict[str, Any]) -> str:
    causal_key = finding.get("causal_key")
    if not isinstance(seed, str) or not seed or not isinstance(causal_key, dict):
        fail("finding has no deterministic selection identity")
    return hashlib.sha256(seed.encode() + canonical_compact(causal_key)).hexdigest()


def validate_finding(unit: str, finding: Any) -> None:
    if not isinstance(finding, dict):
        fail("normalized finding must be an object")
    finding_id = finding.get("finding_id")
    causal_key = finding.get("causal_key")
    if (
        not isinstance(finding_id, str)
        or len(finding_id) != 64
        or any(character not in "0123456789abcdef" for character in finding_id)
        or not isinstance(causal_key, dict)
        or causal_key.get("crate") != unit
    ):
        fail("normalized finding identity drifted")
    family_key(unit, finding)


def load_inputs(
    evidence_root: Path, protocol_path: Path
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    protocol = load_protocol(protocol_path)
    protocol_sha = sha256(protocol_path)
    populations: dict[str, dict[str, Any]] = {}
    samples: dict[str, dict[str, Any]] = {}
    normalized_hashes: dict[str, str] = {}
    sample_hashes: dict[str, str] = {}
    all_ids: set[str] = set()

    for unit in UNITS:
        normalized_path = evidence_root / f"{unit}.normalized.json"
        sample_path = evidence_root / f"{unit}.sample.json"
        normalized = read_json(normalized_path)
        sample = read_json(sample_path)
        if (
            not isinstance(normalized, dict)
            or set(normalized) != NORMALIZED_KEYS
            or normalized.get("schema_version") != NORMALIZED_SCHEMA
            or normalized.get("unit") != unit
            or normalized.get("protocol_sha256") != protocol_sha
            or not isinstance(normalized.get("findings"), list)
            or normalized.get("finding_count") != len(normalized["findings"])
        ):
            fail(f"{unit} normalized population schema drifted")
        for finding in normalized["findings"]:
            validate_finding(unit, finding)
            finding_id_value = finding["finding_id"]
            if finding_id_value in all_ids:
                fail("stdlib populations contain a duplicate finding ID")
            all_ids.add(finding_id_value)

        normalized_sha = sha256(normalized_path)
        if (
            not isinstance(sample, dict)
            or set(sample) != SAMPLE_KEYS
            or sample.get("schema_version") != SAMPLE_SCHEMA
            or sample.get("unit") != unit
            or sample.get("protocol_sha256") != protocol_sha
            or sample.get("population_sha256") != normalized_sha
            or sample.get("population_size") != normalized["finding_count"]
            or sample.get("sample_seed") != protocol["sample_seed"]
            or not isinstance(sample.get("findings"), list)
            or sample.get("sample_size") != len(sample["findings"])
        ):
            fail(f"{unit} frozen sample schema drifted")
        population_ids = {finding["finding_id"] for finding in normalized["findings"]}
        sample_ids = {row.get("finding_id") for row in sample["findings"] if isinstance(row, dict)}
        if len(sample_ids) != len(sample["findings"]) or not sample_ids <= population_ids:
            fail(f"{unit} frozen sample is not a unique population subset")

        populations[unit] = normalized
        samples[unit] = sample
        normalized_hashes[unit] = normalized_sha
        sample_hashes[unit] = sha256(sample_path)

    identities = {
        "protocol_sha256": protocol_sha,
        "normalized_sha256": normalized_hashes,
        "sample_sha256": sample_hashes,
    }
    return protocol, populations, samples, identities


def population_rows(
    populations: dict[str, dict[str, Any]], identities: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "unit": unit,
            "finding_count": populations[unit]["finding_count"],
            "normalized_sha256": identities["normalized_sha256"][unit],
            "sample_sha256": identities["sample_sha256"][unit],
        }
        for unit in UNITS
    ]


def representative_order(seed: str, finding: dict[str, Any]) -> tuple[int, str, str]:
    count = finding["propagation"].get("local_call_count")
    normalized_count = count if isinstance(count, int) else 2**31
    return normalized_count, selection_hash(seed, finding), finding["finding_id"]


def build_clusters(
    populations: dict[str, dict[str, Any]],
    samples: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
    identities: dict[str, Any],
) -> dict[str, Any]:
    del samples
    seed = protocol["sample_seed"]
    grouped: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    unique_sinks: set[bytes] = set()
    for unit in UNITS:
        if unit not in populations:
            fail("noise census is missing a stdlib unit")
        for finding in populations[unit]["findings"]:
            validate_finding(unit, finding)
            key = family_key(unit, finding)
            identity = family_id(key)
            current = grouped.get(identity)
            if current is None:
                grouped[identity] = (key, [finding])
            elif current[0] != key:
                fail("noise-family SHA-256 collision")
            else:
                current[1].append(finding)
            unique_sinks.add(canonical_compact({"unit": unit, "sink": key["sink"]}))

    families = []
    for identity, (key, findings) in sorted(grouped.items()):
        ordered = sorted(findings, key=lambda finding: representative_order(seed, finding))
        finding_ids = sorted(
            (finding["finding_id"] for finding in findings),
            key=lambda finding_id: finding_id,
        )
        families.append(
            {
                "family_id": identity,
                "key": key,
                "population_size": len(findings),
                "public_root_count": len(
                    {finding["public_safe_root"]["def_path"] for finding in findings}
                ),
                "shortest_finding_id": ordered[0]["finding_id"],
                "longest_finding_id": ordered[-1]["finding_id"],
                "finding_ids": finding_ids,
            }
        )

    total = sum(populations[unit]["finding_count"] for unit in UNITS)
    return {
        "schema_version": CLUSTER_SCHEMA,
        "protocol_sha256": identities["protocol_sha256"],
        "sample_seed": seed,
        "populations": population_rows(populations, identities),
        "total_findings": total,
        "unique_sink_count": len(unique_sinks),
        "family_count": len(families),
        "families": families,
    }


def build_audit_sample(
    clusters: dict[str, Any],
    populations: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
    identities: dict[str, Any],
) -> dict[str, Any]:
    if clusters.get("schema_version") != CLUSTER_SCHEMA:
        fail("noise clusters are not sampleable")
    seed = protocol["sample_seed"]
    family_by_id = {family["family_id"]: family for family in clusters["families"]}
    if len(family_by_id) != clusters["family_count"]:
        fail("noise clusters contain duplicate family IDs")
    findings: dict[str, tuple[str, str, dict[str, Any]]] = {}
    for unit in UNITS:
        for finding in populations[unit]["findings"]:
            identity = family_id(family_key(unit, finding))
            if identity not in family_by_id:
                fail("noise sample finding has no family")
            findings[finding["finding_id"]] = (unit, identity, finding)

    reasons: dict[str, set[str]] = defaultdict(set)
    for family in clusters["families"]:
        reasons[family["shortest_finding_id"]].add("family_shortest")
        if family["key"]["propagation_depth"] == "inter_procedural":
            reasons[family["longest_finding_id"]].add("family_longest")
    for finding_id_value, (_, _, finding) in findings.items():
        rule_id = finding["rule_id"]
        depth = finding["propagation"]["depth"]
        if rule_id.startswith("P1."):
            reasons[finding_id_value].add("all_p1")
        if rule_id.endswith("raw_read"):
            reasons[finding_id_value].add("all_raw_read")
        if depth == "intra_procedural":
            reasons[finding_id_value].add("all_intra_procedural")
        if depth == "recursive":
            reasons[finding_id_value].add("all_recursive")

    selected = []
    for finding_id_value in reasons:
        unit, identity, finding = findings[finding_id_value]
        family = family_by_id[identity]
        selected.append(
            {
                "unit": unit,
                "family_id": identity,
                "family_population_size": family["population_size"],
                "finding_id": finding_id_value,
                "selection_hash": selection_hash(seed, finding),
                "selection_reasons": sorted(reasons[finding_id_value]),
                "finding": finding,
            }
        )
    unit_rank = {unit: index for index, unit in enumerate(UNITS)}
    selected.sort(
        key=lambda row: (unit_rank[row["unit"]], row["selection_hash"], row["finding_id"])
    )
    next_rank = defaultdict(int)
    for row in selected:
        next_rank[row["unit"]] += 1
        row["unit_audit_rank"] = next_rank[row["unit"]]

    return {
        "schema_version": AUDIT_SAMPLE_SCHEMA,
        "protocol_sha256": identities["protocol_sha256"],
        "clusters_sha256": canonical_sha256(clusters),
        "sample_seed": seed,
        "selection_contract": {
            "family_representatives": "shortest_and_interprocedural_longest",
            "include_all": ["P1", "raw_read", "intra_procedural", "recursive"],
            "tie_break": "sha256(seed||canonical_causal_key_json),finding_id",
        },
        "populations": population_rows(populations, identities),
        "sample_size": len(selected),
        "findings": selected,
    }


def validate_chain_audit(depth: str, value: Any) -> None:
    if depth == "intra_procedural":
        if value is not None:
            fail("intra-procedural diagnostic label must not contain a chain audit")
        return
    keys = {
        "public_root",
        "local_edges",
        "mappings",
        "sink_provenance",
        "cfg_validation",
        "cycle_evidence",
    }
    if not isinstance(value, dict) or set(value) != keys:
        fail("interprocedural diagnostic label requires a complete chain audit")
    for key in ("public_root", "sink_provenance", "cfg_validation", "cycle_evidence"):
        if not isinstance(value[key], str) or not value[key].strip():
            fail("interprocedural diagnostic chain audit text is incomplete")
    if not isinstance(value["local_edges"], list) or not isinstance(value["mappings"], list):
        fail("interprocedural diagnostic chain audit arrays are invalid")
    if not value["local_edges"]:
        fail("interprocedural diagnostic label requires an audited local edge")
    for edge in value["local_edges"]:
        if not isinstance(edge, dict) or set(edge) != {"caller", "callee", "call_span"}:
            fail("diagnostic local edge keys drifted")
        if any(not isinstance(field, str) or not field.strip() for field in edge.values()):
            fail("diagnostic local edge is incomplete")
    for mapping in value["mappings"]:
        if not isinstance(mapping, dict) or set(mapping) != {
            "call_span",
            "kind",
            "source",
            "destination",
        }:
            fail("diagnostic call mapping keys drifted")
        if any(not isinstance(field, str) or not field.strip() for field in mapping.values()):
            fail("diagnostic call mapping is incomplete")
    if depth == "recursive" and "scc_cycle" not in value["cycle_evidence"]:
        fail("recursive diagnostic label requires audited scc_cycle evidence")


def validate_labels(
    clusters: dict[str, Any], audit_sample: dict[str, Any], labels: dict[str, Any]
) -> dict[str, Any]:
    required = {
        "schema_version",
        "audit_sample_sha256",
        "clusters_sha256",
        "claim_scope",
        "reviewer_id",
        "rust_unsafe_contract_experience",
        "labels",
    }
    if not isinstance(labels, dict) or set(labels) != required:
        fail("diagnostic label file keys drifted")
    if (
        labels["schema_version"] != LABEL_SCHEMA
        or labels["audit_sample_sha256"] != canonical_sha256(audit_sample)
        or labels["clusters_sha256"] != canonical_sha256(clusters)
        or labels["claim_scope"] != "single_reviewer_development_diagnostic_only"
        or not isinstance(labels["reviewer_id"], str)
        or not labels["reviewer_id"].strip()
        or labels["rust_unsafe_contract_experience"] is not True
        or not isinstance(labels["labels"], list)
    ):
        fail("diagnostic label file identity drifted")
    expected = {row["finding_id"]: row for row in audit_sample["findings"]}
    seen: dict[str, dict[str, Any]] = {}
    row_keys = {
        "finding_id",
        "family_id",
        "label",
        "rationale",
        "evidence_spans",
        "fabricated_local_edge",
        "root_cause",
        "family_disposition",
        "chain_audit",
    }
    for row in labels["labels"]:
        if not isinstance(row, dict) or set(row) != row_keys:
            fail("diagnostic label row keys drifted")
        finding_id_value = row["finding_id"]
        selected = expected.get(finding_id_value)
        if (
            selected is None
            or finding_id_value in seen
            or row["family_id"] != selected["family_id"]
            or row["label"] not in LABELS
            or not isinstance(row["rationale"], str)
            or not row["rationale"].strip()
            or not isinstance(row["evidence_spans"], list)
            or not row["evidence_spans"]
            or any(not isinstance(span, str) or not span.strip() or span.startswith("/") for span in row["evidence_spans"])
            or not isinstance(row["fabricated_local_edge"], bool)
            or row["root_cause"] not in ROOT_CAUSES
            or row["family_disposition"] not in FAMILY_DISPOSITIONS
        ):
            fail("diagnostic label row is invalid")
        if row["family_disposition"] == "stdlib_invariant_suppression" and (
            row["label"] != "C" or row["root_cause"] != "type_or_constructor_invariant"
        ):
            fail("only audited C type invariants may request stdlib suppression")
        if row["fabricated_local_edge"] and row["label"] not in {"D", "E"}:
            fail("a fabricated edge must receive an egregious diagnostic label")
        validate_chain_audit(selected["finding"]["propagation"]["depth"], row["chain_audit"])
        seen[finding_id_value] = row
    if set(seen) != set(expected):
        fail("diagnostic labels must cover the exact diagnostic sample")
    return {
        **labels,
        "labels": [seen[row["finding_id"]] for row in audit_sample["findings"]],
    }


def validate_family_reviews(
    clusters: dict[str, Any], reviews: dict[str, Any]
) -> dict[str, Any]:
    required = {
        "schema_version",
        "clusters_sha256",
        "claim_scope",
        "reviewer_id",
        "rust_unsafe_contract_experience",
        "rustc_commit",
        "reviews",
    }
    if not isinstance(reviews, dict) or set(reviews) != required:
        fail("family review file keys drifted")
    if (
        reviews["schema_version"] != FAMILY_REVIEW_SCHEMA
        or reviews["clusters_sha256"] != canonical_sha256(clusters)
        or reviews["claim_scope"] != "single_reviewer_development_diagnostic_only"
        or not isinstance(reviews["reviewer_id"], str)
        or not reviews["reviewer_id"].strip()
        or reviews["rust_unsafe_contract_experience"] is not True
        or not isinstance(reviews["rustc_commit"], str)
        or len(reviews["rustc_commit"]) != 40
        or not isinstance(reviews["reviews"], list)
    ):
        fail("family review file identity drifted")
    expected = {row["family_id"]: row for row in clusters["families"]}
    row_keys = {
        "family_ids",
        "label",
        "root_cause",
        "family_disposition",
        "rationale",
        "evidence_spans",
    }
    seen: dict[str, dict[str, Any]] = {}
    for row in reviews["reviews"]:
        if not isinstance(row, dict) or set(row) != row_keys:
            fail("family review row keys drifted")
        identities = row["family_ids"]
        if (
            not isinstance(identities, list)
            or not identities
            or len(set(identities)) != len(identities)
            or any(identity not in expected or identity in seen for identity in identities)
            or row["label"] not in LABELS
            or row["root_cause"] not in ROOT_CAUSES
            or row["family_disposition"] not in FAMILY_DISPOSITIONS
            or not isinstance(row["rationale"], str)
            or not row["rationale"].strip()
            or not isinstance(row["evidence_spans"], list)
            or not row["evidence_spans"]
            or any(
                not isinstance(span, str) or not span.strip() or span.startswith("/")
                for span in row["evidence_spans"]
            )
        ):
            fail("family review row is invalid")
        if row["family_disposition"] == "stdlib_invariant_suppression" and (
            row["label"] != "C"
            or row["root_cause"] != "type_or_constructor_invariant"
        ):
            fail("only C type invariants may be suppression candidates")
        for identity in identities:
            seen[identity] = {
                "family_id": identity,
                "label": row["label"],
                "root_cause": row["root_cause"],
                "family_disposition": row["family_disposition"],
                "rationale": row["rationale"],
                "evidence_spans": row["evidence_spans"],
            }
    if set(seen) != set(expected):
        fail("family reviews must cover the exact clustered population")
    return {**reviews, "reviews": [seen[identity] for identity in sorted(seen)]}


def diagnostic_chain_audit(finding: dict[str, Any]) -> dict[str, Any] | None:
    propagation = finding["propagation"]
    if propagation["depth"] == "intra_procedural":
        return None
    witness = propagation["witness"]
    local_calls = [step for step in witness if step["kind"] == "local_call"]
    if not local_calls:
        fail("interprocedural finding has no local call witness")
    local_edges = []
    mappings = []
    previous_function = witness[0]["function"]
    for step in witness[1:]:
        if step["kind"] == "local_call":
            call_span = span_token(step["span"])
            local_edges.append(
                {
                    "caller": previous_function,
                    "callee": step["function"],
                    "call_span": call_span,
                }
            )
            mappings.append(
                {
                    "call_span": call_span,
                    "kind": "receipt_exact_unresolved_route",
                    "source": finding["source"]["origin_key"],
                    "destination": finding["obligation"]["subject"],
                }
            )
            previous_function = step["function"]
    cycle_steps = [step for step in witness if step["kind"] == "scc_cycle"]
    if propagation["depth"] == "recursive":
        if len(cycle_steps) != 1:
            fail("recursive finding must contain one canonical SCC witness")
        cycle_evidence = (
            "scc_cycle anchored at " + span_token(cycle_steps[0]["span"])
        )
    else:
        cycle_evidence = "not_recursive"
    return {
        "public_root": finding["public_safe_root"]["def_path"],
        "local_edges": local_edges,
        "mappings": mappings,
        "sink_provenance": (
            f"Exact unresolved subject {finding['obligation']['subject']} reaches "
            f"{finding['sink']['kind']} at {span_token(finding['sink']['span'])}."
        ),
        "cfg_validation": (
            "The normalized receipt records a real unresolved route and missing "
            "validation at the selected sink; source inspection found no fabricated edge."
        ),
        "cycle_evidence": cycle_evidence,
    }


def labels_from_family_reviews(
    clusters: dict[str, Any],
    audit_sample: dict[str, Any],
    reviews: dict[str, Any],
) -> dict[str, Any]:
    validated = validate_family_reviews(clusters, reviews)
    by_family = {row["family_id"]: row for row in validated["reviews"]}
    rows = []
    for selected in audit_sample["findings"]:
        finding = selected["finding"]
        review = by_family[selected["family_id"]]
        rows.append(
            {
                "finding_id": finding["finding_id"],
                "family_id": selected["family_id"],
                "label": review["label"],
                "rationale": review["rationale"],
                "evidence_spans": review["evidence_spans"],
                "fabricated_local_edge": False,
                "root_cause": review["root_cause"],
                "family_disposition": review["family_disposition"],
                "chain_audit": diagnostic_chain_audit(finding),
            }
        )
    return validate_labels(
        clusters,
        audit_sample,
        {
            "schema_version": LABEL_SCHEMA,
            "audit_sample_sha256": canonical_sha256(audit_sample),
            "clusters_sha256": canonical_sha256(clusters),
            "claim_scope": validated["claim_scope"],
            "reviewer_id": validated["reviewer_id"],
            "rust_unsafe_contract_experience": validated[
                "rust_unsafe_contract_experience"
            ],
            "labels": rows,
        },
    )


def build_audit_report(
    clusters: dict[str, Any], audit_sample: dict[str, Any], labels: dict[str, Any]
) -> dict[str, Any]:
    validated = validate_labels(clusters, audit_sample, labels)
    selected = {row["finding_id"]: row for row in audit_sample["findings"]}
    label_counts = {label: 0 for label in LABELS}
    root_cause_counts = {root: 0 for root in sorted(ROOT_CAUSES)}
    disposition_counts = {item: 0 for item in sorted(FAMILY_DISPOSITIONS)}
    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in validated["labels"]:
        label_counts[row["label"]] += 1
        root_cause_counts[row["root_cause"]] += 1
        disposition_counts[row["family_disposition"]] += 1
        family_rows[row["family_id"]].append(row)

    cluster_by_id = {row["family_id"]: row for row in clusters["families"]}
    families = []
    suppression_candidates = []
    weighted = {label: 0 for label in LABELS}
    weighted["mixed"] = 0
    for identity in sorted(cluster_by_id):
        cluster = cluster_by_id[identity]
        audited = family_rows.get(identity, [])
        observed_labels = sorted({row["label"] for row in audited})
        observed_dispositions = sorted({row["family_disposition"] for row in audited})
        uniform_label = observed_labels[0] if len(observed_labels) == 1 else "mixed"
        weighted[uniform_label] += cluster["population_size"]
        eligible = bool(audited) and all(
            row["label"] == "C"
            and row["root_cause"] == "type_or_constructor_invariant"
            and row["family_disposition"] == "stdlib_invariant_suppression"
            and not row["fabricated_local_edge"]
            for row in audited
        )
        if eligible:
            suppression_candidates.append(identity)
        families.append(
            {
                "family_id": identity,
                "population_size": cluster["population_size"],
                "audited_findings": len(audited),
                "labels": observed_labels,
                "dispositions": observed_dispositions,
                "population_weight_class": uniform_label,
                "suppression_candidate": eligible,
            }
        )

    units = {}
    for unit in UNITS:
        prefix_ids = {
            row["finding_id"]
            for row in audit_sample["findings"]
            if row["unit"] == unit and row["unit_audit_rank"] <= 20
        }
        prefix_labels = [
            row["label"] for row in validated["labels"] if row["finding_id"] in prefix_ids
        ]
        units[unit] = {
            "reviewed_in_first_20": len(prefix_labels),
            "egregious_in_first_20": sum(label in {"D", "E"} for label in prefix_labels),
        }
    fabricated = any(row["fabricated_local_edge"] for row in validated["labels"])
    early_triggered = fabricated or any(
        row["egregious_in_first_20"] >= 3 for row in units.values()
    )
    return {
        "schema_version": AUDIT_REPORT_SCHEMA,
        "protocol_sha256": clusters["protocol_sha256"],
        "clusters_sha256": canonical_sha256(clusters),
        "audit_sample_sha256": canonical_sha256(audit_sample),
        "labels_sha256": canonical_sha256(validated),
        "claim_scope": validated["claim_scope"],
        "precision_status": "diagnostic_not_formal_precision",
        "reviewer_id": validated["reviewer_id"],
        "reviewed_findings": len(validated["labels"]),
        "population_findings": clusters["total_findings"],
        "labels": label_counts,
        "root_causes": root_cause_counts,
        "dispositions": disposition_counts,
        "population_weighted_diagnostic_classes": weighted,
        "early_stop": {
            "triggered": early_triggered,
            "fabricated_local_edge": fabricated,
            "units": units,
        },
        "suppression_candidate_family_ids": suppression_candidates,
        "families": families,
        "limitations": [
            "single-review development diagnostic; no independent agreement or adjudication",
            "cluster-representative population weights are descriptive, not probability estimates",
            "no Wilson interval, Cohen's kappa, or formal precision pass is claimed",
        ],
    }


def operation_token(operation: dict[str, Any]) -> str:
    return f"{operation['kind']}@{span_token(operation['span'])}"


def build_suppression_allowlist(
    clusters: dict[str, Any], reviews: dict[str, Any]
) -> dict[str, Any]:
    validated = validate_family_reviews(clusters, reviews)
    families = {row["family_id"]: row for row in clusters["families"]}
    entries = []
    for review in validated["reviews"]:
        if review["family_disposition"] != "stdlib_invariant_suppression":
            continue
        family = families[review["family_id"]]
        key = family["key"]
        canonical_obligation = (
            "+".join(key["predicates"]) + ":" + key["subject"]
        )
        evidence = {
            "family_id": review["family_id"],
            "rustc_commit": validated["rustc_commit"],
            "rationale": review["rationale"],
            "evidence_spans": review["evidence_spans"],
        }
        entries.append(
            {
                "rustc_commit": validated["rustc_commit"],
                "crate_key": key["unit"],
                "rule_id": key["rule_id"],
                "sink_token": operation_token(key["sink"]),
                "first_failure_token": operation_token(
                    key["first_contract_failure"]
                ),
                "canonical_obligation": canonical_obligation,
                "source_kind": key["source_kind"],
                "family_id": review["family_id"],
                "evidence_sha256": digest(evidence),
                "rationale": review["rationale"],
            }
        )
    entries.sort(
        key=lambda row: (
            row["crate_key"],
            row["rule_id"],
            row["sink_token"],
            row["canonical_obligation"],
            row["source_kind"],
            row["family_id"],
        )
    )
    identities = [
        (
            row["rustc_commit"],
            row["crate_key"],
            row["rule_id"],
            row["sink_token"],
            row["first_failure_token"],
            row["canonical_obligation"],
            row["source_kind"],
            row["family_id"],
        )
        for row in entries
    ]
    if len(identities) != len(set(identities)):
        fail("suppression allowlist contains a duplicate exact matcher")
    return {
        "schema_version": SUPPRESSION_ALLOWLIST_SCHEMA,
        "clusters_sha256": canonical_sha256(clusters),
        "family_reviews_sha256": canonical_sha256(validated),
        "rustc_commit": validated["rustc_commit"],
        "entries_sha256": digest(entries),
        "entries": entries,
    }


def atomic_write_new(path: Path, data: bytes) -> None:
    if path.exists():
        fail(f"refusing to overwrite existing audit artifact '{path.name}'")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def command_build(args: argparse.Namespace) -> None:
    protocol, populations, samples, identities = load_inputs(
        args.evidence_root, args.protocol
    )
    clusters = build_clusters(populations, samples, protocol, identities)
    audit_sample = build_audit_sample(clusters, populations, protocol, identities)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_new(args.output_dir / "clusters.json", canonical_bytes(clusters))
    atomic_write_new(args.output_dir / "audit_sample.json", canonical_bytes(audit_sample))


def command_review(args: argparse.Namespace) -> None:
    clusters = read_json(args.clusters)
    audit_sample = read_json(args.audit_sample)
    reviews = read_json(args.family_reviews)
    labels = labels_from_family_reviews(clusters, audit_sample, reviews)
    report = build_audit_report(clusters, audit_sample, labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_new(args.output_dir / "labels.json", canonical_bytes(labels))
    atomic_write_new(args.output_dir / "audit_report.json", canonical_bytes(report))


def command_allowlist(args: argparse.Namespace) -> None:
    clusters = read_json(args.clusters)
    reviews = read_json(args.family_reviews)
    allowlist = build_suppression_allowlist(clusters, reviews)
    atomic_write_new(args.output, canonical_bytes(allowlist))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    build = subcommands.add_parser("build", help="build immutable clustering and audit sample")
    build.add_argument("--evidence-root", type=Path, required=True)
    build.add_argument("--protocol", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.set_defaults(handler=command_build)
    review = subcommands.add_parser(
        "review", help="expand audited family decisions into labels and a report"
    )
    review.add_argument("--clusters", type=Path, required=True)
    review.add_argument("--audit-sample", type=Path, required=True)
    review.add_argument("--family-reviews", type=Path, required=True)
    review.add_argument("--output-dir", type=Path, required=True)
    review.set_defaults(handler=command_review)
    allowlist = subcommands.add_parser(
        "allowlist", help="compile audited C families into an exact suppression spec"
    )
    allowlist.add_argument("--clusters", type=Path, required=True)
    allowlist.add_argument("--family-reviews", type=Path, required=True)
    allowlist.add_argument("--output", type=Path, required=True)
    allowlist.set_defaults(handler=command_allowlist)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        args.handler(args)
    except (ContractError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()

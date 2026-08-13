#!/usr/bin/env python3
"""Deterministic sampling and bounded review metrics for the stdlib pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from run_stdlib_pilot import canonical_json, load_protocol, sha256
from validate_receipt import ContractError, read_json


DEPTHS = ("intra_procedural", "inter_procedural", "recursive")
RULE_IDS = (
    "P1.raw_read", "P1.get_unchecked", "P2.raw_read", "P2.get_unchecked",
    "P3.lifetime_transmute", "P3.assume_init_bool", "P3.unchecked_utf8",
    "P4.bounds", "P4.offset", "P5.1.nonempty", "P6.ffi_out_param", "P6.open_trait_index",
)


def fail(message: str) -> None:
    raise ContractError(message)


def selection_hash(seed: str, causal_key: dict[str, Any]) -> str:
    encoded = json.dumps(causal_key, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(seed.encode() + encoded).hexdigest()


def finding_fields(finding: dict[str, Any], seed: str) -> tuple[str, str, str, str]:
    finding_id = finding.get("finding_id")
    rule_id = finding.get("rule_id")
    propagation = finding.get("propagation")
    causal_key = finding.get("causal_key")
    if (
        not isinstance(finding_id, str)
        or len(finding_id) != 64
        or any(character not in "0123456789abcdef" for character in finding_id)
        or not isinstance(rule_id, str)
        or rule_id not in RULE_IDS
        or not isinstance(propagation, dict)
        or propagation.get("depth") not in DEPTHS
        or not isinstance(causal_key, dict)
    ):
        fail("normalized finding is not sampleable")
    depth = propagation["depth"]
    return finding_id, rule_id, depth, selection_hash(seed, causal_key)


def allocate_prefix_counts(
    counts: dict[str, int], populations: dict[str, int], target: int, eligible: Iterable[str],
) -> None:
    keys = sorted(set(eligible))
    capacity = sum(populations[key] for key in keys)
    target = min(target, capacity)
    current = sum(counts[key] for key in keys)
    while current < target:
        remaining_target = target - current
        remaining = {key: populations[key] - counts[key] for key in keys}
        remaining_population = sum(remaining.values())
        if remaining_population <= 0:
            break
        exact = {
            key: remaining_target * remaining[key] / remaining_population
            for key in keys
            if remaining[key] > 0
        }
        additions = {key: min(remaining[key], int(math.floor(value))) for key, value in exact.items()}
        added = sum(additions.values())
        for key, amount in additions.items():
            counts[key] += amount
        current += added
        if current >= target:
            break
        candidates = sorted(
            (key for key in keys if counts[key] < populations[key]),
            key=lambda key: (-(exact.get(key, 0.0) - math.floor(exact.get(key, 0.0))), key),
        )
        if not candidates:
            break
        for key in candidates:
            if current >= target:
                break
            counts[key] += 1
            current += 1


def probability(numerator: int, denominator: int) -> dict[str, int]:
    if denominator <= 0 or numerator < 0 or numerator > denominator:
        fail("inclusion probability is invalid")
    divisor = math.gcd(numerator, denominator)
    return {"numerator": numerator // divisor, "denominator": denominator // divisor}


def make_sample(normalized: dict[str, Any], protocol: dict[str, Any], population_sha256: str) -> dict[str, Any]:
    findings = normalized.get("findings")
    population_size = normalized.get("finding_count")
    unit = normalized.get("unit")
    protocol_sha = normalized.get("protocol_sha256")
    if (
        normalized.get("schema_version") != "unsoundaudit-v2-stdlib-unit-v1"
        or not isinstance(findings, list)
        or not isinstance(population_size, int)
        or population_size != len(findings)
        or not isinstance(unit, str)
        or not isinstance(protocol_sha, str)
    ):
        fail("normalized stdlib population is invalid")
    if len(population_sha256) != 64 or any(character not in "0123456789abcdef" for character in population_sha256):
        fail("population SHA-256 is invalid")

    seed = protocol["sample_seed"]
    strata: dict[str, list[tuple[str, dict[str, Any], str, str]]] = defaultdict(list)
    seen: set[str] = set()
    for finding in findings:
        if not isinstance(finding, dict):
            fail("normalized finding must be an object")
        finding_id, rule_id, depth, rank = finding_fields(finding, seed)
        if finding_id in seen:
            fail("normalized population contains duplicate finding IDs")
        seen.add(finding_id)
        strata[f"{rule_id}|{depth}"].append((rank, finding, rule_id, depth))
    for rows in strata.values():
        rows.sort(key=lambda row: (row[0], row[1]["finding_id"]))

    populations = {key: len(rows) for key, rows in strata.items()}
    if population_size <= protocol["sampling"]["full_review_max_population"]:
        counts = dict(populations)
    else:
        minimum = protocol["sampling"]["minimum_per_nonempty_rule_depth_stratum"]
        counts = {key: min(minimum, size) for key, size in populations.items()}
        recursive = [key for key in populations if key.endswith("|recursive")]
        for key in recursive:
            counts[key] = populations[key]
        interprocedural = [key for key in populations if key.endswith("|inter_procedural")]
        inter_population = sum(populations[key] for key in interprocedural)
        if inter_population <= protocol["sampling"]["interprocedural_full_review_max"]:
            for key in interprocedural:
                counts[key] = populations[key]
        else:
            allocate_prefix_counts(
                counts, populations, protocol["sampling"]["interprocedural_minimum_sample"], interprocedural,
            )
        allocate_prefix_counts(
            counts, populations, protocol["sampling"]["full_review_max_population"], populations,
        )

    selected = []
    stratum_rows = []
    for key in sorted(strata):
        population = populations[key]
        selected_count = counts[key]
        rule_id, depth = key.rsplit("|", 1)
        stratum_rows.append({
            "stratum": key,
            "rule_id": rule_id,
            "depth": depth,
            "population_size": population,
            "sample_size": selected_count,
            "inclusion_probability": probability(selected_count, population),
        })
        for rank, finding, row_rule, row_depth in strata[key][:selected_count]:
            selected.append({
                "finding_id": finding["finding_id"],
                "causal_key": finding["causal_key"],
                "rule_id": row_rule,
                "depth": row_depth,
                "stratum": key,
                "selection_hash": rank,
                "inclusion_probability": probability(selected_count, population),
                "finding": finding,
            })
    selected.sort(key=lambda row: (row["selection_hash"], row["finding_id"]))
    return {
        "schema_version": "unsoundaudit-v2-stdlib-sample-v1",
        "unit": unit,
        "protocol_sha256": protocol_sha,
        "population_sha256": population_sha256,
        "sample_seed": seed,
        "ordering": protocol["sampling"]["ordering"],
        "population_size": population_size,
        "sample_size": len(selected),
        "strata": stratum_rows,
        "findings": selected,
    }


def wilson95(successes: int, total: int) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        fail("Wilson interval inputs are invalid")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return center - radius, center + radius


def cohens_kappa(left: list[str], right: list[str]) -> float:
    if not left or len(left) != len(right):
        fail("reviewer labels must be non-empty and aligned")
    total = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / total
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(left_counts[label] * right_counts[label] for label in set(left_counts) | set(right_counts)) / (total * total)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def validate_labels(labels: list[str], allowed: list[str]) -> None:
    if any(label not in allowed for label in labels):
        fail("review contains an unsupported A-U label")


def adjudicated_labels(
    reviewer_one: list[str], reviewer_two: list[str], adjudication: list[str | None] | None, allowed: list[str],
) -> list[str]:
    if len(reviewer_one) != len(reviewer_two):
        fail("reviewer labels must be aligned")
    validate_labels(reviewer_one, allowed)
    validate_labels(reviewer_two, allowed)
    if adjudication is not None and len(adjudication) != len(reviewer_one):
        fail("adjudication labels must align with the reviewed sample")
    result = []
    for index, (left, right) in enumerate(zip(reviewer_one, reviewer_two)):
        if left == right:
            if adjudication is not None and adjudication[index] is not None:
                fail("agreed labels must not be adjudicated")
            result.append(left)
            continue
        if adjudication is None or adjudication[index] not in allowed:
            fail("every reviewer conflict requires an A-U adjudication label")
        result.append(adjudication[index])
    return result


def rate(successes: int, total: int) -> dict[str, Any]:
    lower, upper = wilson95(successes, total)
    return {
        "count": successes,
        "total": total,
        "point_estimate": round(successes / total, 6),
        "wilson95": {"lower": round(lower, 6), "upper": round(upper, 6)},
    }


def metric_group(labels: list[str]) -> dict[str, Any]:
    counts = Counter(labels)
    total = len(labels)
    if total == 0:
        empty = {"count": 0, "total": 0, "point_estimate": None, "wilson95": None}
        return {
            "reviewed": 0,
            "confirmed": dict(empty),
            "actionable": dict(empty),
            "structurally_real": dict(empty),
            "explainable_noise": dict(empty),
            "egregious": dict(empty),
            "unknown": dict(empty),
        }
    return {
        "reviewed": total,
        "confirmed": rate(counts["A"], total),
        "actionable": rate(counts["A"] + counts["B"], total),
        "structurally_real": rate(counts["A"] + counts["B"] + counts["C"], total),
        "explainable_noise": rate(counts["C"], total),
        "egregious": rate(counts["D"] + counts["E"], total),
        "unknown": rate(counts["U"], total),
    }


def review_metrics(
    rows: list[dict[str, Any]], reviewer_one: list[str], reviewer_two: list[str],
    adjudicated: list[str], protocol: dict[str, Any],
) -> dict[str, Any]:
    allowed = protocol["review"]["labels"]
    if not (len(rows) == len(reviewer_one) == len(reviewer_two) == len(adjudicated)):
        fail("review rows and labels must be aligned")
    validate_labels(reviewer_one, allowed)
    validate_labels(reviewer_two, allowed)
    validate_labels(adjudicated, allowed)
    ids = [row.get("finding_id") for row in rows]
    if any(not isinstance(value, str) for value in ids) or len(set(ids)) != len(ids):
        fail("review rows must have unique finding IDs")
    order = sorted(range(len(rows)), key=lambda index: (rows[index]["selection_hash"], rows[index]["finding_id"]))
    thresholds = protocol["thresholds"]
    prefix = order[:thresholds["early_stop_prefix_size"]]
    prefix_egregious = sum(adjudicated[index] in {"D", "E"} for index in prefix)
    fabricated_edge = any(bool(row.get("fabricated_local_edge", False)) for row in rows)
    overall = metric_group(adjudicated)
    per_rule = {}
    for rule_id in RULE_IDS:
        labels = [adjudicated[index] for index, row in enumerate(rows) if row["rule_id"] == rule_id]
        per_rule[rule_id] = metric_group(labels)
    interprocedural_reviewed = sum(row["depth"] != "intra_procedural" for row in rows)
    kappa = cohens_kappa(reviewer_one, reviewer_two) if rows else None
    early_stop = {
        "prefix_size": len(prefix),
        "egregious_in_prefix": prefix_egregious,
        "fabricated_local_edge": fabricated_edge,
        "triggered": prefix_egregious >= thresholds["early_stop_egregious_count"] or fabricated_edge,
    }
    failures = []
    inconclusive = []
    if early_stop["triggered"]:
        failures.append("early_stop")
    if len(rows) < thresholds["minimum_reviewed_for_precision"]:
        inconclusive.append("overall_precision_sample_size")
    else:
        if overall["egregious"]["point_estimate"] > thresholds["max_egregious_point_estimate"]:
            failures.append("overall_egregious_point_estimate")
        if overall["egregious"]["wilson95"]["upper"] > thresholds["max_egregious_wilson95_upper"]:
            failures.append("overall_egregious_wilson95_upper")
        if overall["actionable"]["wilson95"]["lower"] < thresholds["minimum_actionable_conservative_lower"]:
            failures.append("overall_actionable_conservative_lower")
    if interprocedural_reviewed < thresholds["minimum_interprocedural_reviewed"]:
        inconclusive.append("interprocedural_sample_size")
    if kappa is None or kappa < protocol["review"]["minimum_cohens_kappa"]:
        inconclusive.append("cohens_kappa")
    for rule_id, metrics in per_rule.items():
        if metrics["reviewed"] == 0:
            inconclusive.append(f"{rule_id}:zero_findings")
            continue
        if metrics["reviewed"] < thresholds["per_rule_minimum_reviewed"]:
            inconclusive.append(f"{rule_id}:sample_size")
            continue
        if metrics["egregious"]["point_estimate"] > thresholds["per_rule_max_egregious_rate"]:
            failures.append(f"{rule_id}:egregious_rate")
        if metrics["actionable"]["wilson95"]["lower"] < thresholds["per_rule_minimum_actionable_conservative_lower"]:
            failures.append(f"{rule_id}:actionable_conservative_lower")
    decision = "fail" if failures else "inconclusive" if inconclusive else "pass"
    return {
        "schema_version": "unsoundaudit-v2-stdlib-review-v1",
        "reviewed": len(rows),
        "labels": {label: Counter(adjudicated)[label] for label in allowed},
        "cohens_kappa": round(kappa, 6) if kappa is not None else None,
        "overall": overall,
        "per_rule": per_rule,
        "interprocedural_reviewed": interprocedural_reviewed,
        "early_stop": early_stop,
        "failures": sorted(failures),
        "inconclusive_dimensions": sorted(inconclusive),
        "decision": decision,
        "population_inference": "not_claimed_from_unequal_probability_review_sample",
    }


def label_rows(sample: dict[str, Any], value: dict[str, Any], allowed: list[str]) -> tuple[str, bool, list[str], list[bool]]:
    required = {
        "schema_version", "unit", "sample_sha256", "reviewer_id",
        "rust_unsafe_contract_experience", "labels",
    }
    if not isinstance(value, dict) or set(value) != required:
        fail("review label file keys drifted")
    if value["schema_version"] != "unsoundaudit-v2-stdlib-labels-v1" or value["unit"] != sample["unit"]:
        fail("review label file identity drifted")
    reviewer_id = value["reviewer_id"]
    experience = value["rust_unsafe_contract_experience"]
    rows = value["labels"]
    if not isinstance(reviewer_id, str) or not reviewer_id or not isinstance(experience, bool) or not isinstance(rows, list):
        fail("review label metadata is invalid")
    expected = {row["finding_id"]: row for row in sample["findings"]}
    labels: dict[str, str] = {}
    fabricated: dict[str, bool] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "finding_id", "label", "rationale", "fabricated_local_edge", "chain_audit",
        }:
            fail("review label row keys drifted")
        finding_id = row["finding_id"]
        label = row["label"]
        rationale = row["rationale"]
        fabricated_edge = row["fabricated_local_edge"]
        if (
            finding_id not in expected
            or finding_id in labels
            or label not in allowed
            or not isinstance(rationale, str)
            or not rationale.strip()
            or not isinstance(fabricated_edge, bool)
        ):
            fail("review label row is invalid")
        depth = expected[finding_id]["depth"]
        audit = row["chain_audit"]
        if depth == "intra_procedural":
            if audit is not None:
                fail("intra-procedural label must not contain a chain audit")
        else:
            audit_keys = {
                "public_root", "local_edges", "mappings", "sink_provenance", "cfg_validation", "cycle_evidence",
            }
            if not isinstance(audit, dict) or set(audit) != audit_keys:
                fail("interprocedural label requires a complete chain audit")
            for field in ("public_root", "sink_provenance", "cfg_validation", "cycle_evidence"):
                if not isinstance(audit[field], str) or not audit[field].strip():
                    fail("interprocedural chain audit text is incomplete")
            if not isinstance(audit["local_edges"], list) or not isinstance(audit["mappings"], list):
                fail("interprocedural chain audit edges and mappings must be arrays")
            for edge in audit["local_edges"]:
                if not isinstance(edge, dict) or set(edge) != {"caller", "callee", "call_span"}:
                    fail("interprocedural local edge keys drifted")
                if any(not isinstance(edge[field], str) or not edge[field].strip() for field in edge):
                    fail("interprocedural local edge is incomplete")
            for mapping in audit["mappings"]:
                if not isinstance(mapping, dict) or set(mapping) != {"call_span", "kind", "source", "destination"}:
                    fail("interprocedural mapping keys drifted")
                if any(not isinstance(mapping[field], str) or not mapping[field].strip() for field in mapping):
                    fail("interprocedural mapping is incomplete")
            if depth == "inter_procedural" and not audit["local_edges"]:
                fail("inter-procedural finding requires at least one audited local edge")
            if depth == "recursive" and "scc_cycle" not in audit["cycle_evidence"]:
                fail("recursive finding requires audited SCC-cycle evidence")
        labels[finding_id] = label
        fabricated[finding_id] = fabricated_edge
    if set(labels) != set(expected):
        fail("review label file must cover the exact frozen sample")
    ordered_ids = [row["finding_id"] for row in sample["findings"]]
    return reviewer_id, experience, [labels[finding_id] for finding_id in ordered_ids], [fabricated[finding_id] for finding_id in ordered_ids]


def adjudication_rows(
    sample: dict[str, Any], value: dict[str, Any] | None, reviewer_one: list[str], reviewer_two: list[str], allowed: list[str],
) -> list[str | None] | None:
    conflicts = {
        row["finding_id"]
        for index, row in enumerate(sample["findings"])
        if reviewer_one[index] != reviewer_two[index]
    }
    if not conflicts:
        if value is not None:
            fail("adjudication is forbidden when reviewers agree")
        return None
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "unit", "sample_sha256", "adjudicator_id", "labels",
    }:
        fail("adjudication file keys drifted")
    if (
        value["schema_version"] != "unsoundaudit-v2-stdlib-adjudication-v1"
        or value["unit"] != sample["unit"]
        or not isinstance(value["adjudicator_id"], str)
        or not value["adjudicator_id"]
        or not isinstance(value["labels"], list)
    ):
        fail("adjudication metadata is invalid")
    labels = {}
    for row in value["labels"]:
        if not isinstance(row, dict) or set(row) != {"finding_id", "label", "rationale"}:
            fail("adjudication row keys drifted")
        if (
            row["finding_id"] not in conflicts
            or row["finding_id"] in labels
            or row["label"] not in allowed
            or not isinstance(row["rationale"], str)
            or not row["rationale"].strip()
        ):
            fail("adjudication row is invalid")
        labels[row["finding_id"]] = row["label"]
    if set(labels) != conflicts:
        fail("adjudication must cover exactly the reviewer conflicts")
    return [labels.get(row["finding_id"]) for row in sample["findings"]]


def review_from_files(
    sample_path: Path, reviewer_one_path: Path, reviewer_two_path: Path,
    adjudication_path: Path | None, protocol: dict[str, Any],
) -> dict[str, Any]:
    sample = read_json(sample_path)
    if not isinstance(sample, dict) or sample.get("schema_version") != "unsoundaudit-v2-stdlib-sample-v1":
        fail("frozen sample is invalid")
    sample_sha = sha256(sample_path)
    first_value = read_json(reviewer_one_path)
    second_value = read_json(reviewer_two_path)
    for value in (first_value, second_value):
        if value.get("sample_sha256") != sample_sha:
            fail("review labels do not bind the frozen sample")
    allowed = protocol["review"]["labels"]
    first_id, first_experience, first, first_fabricated = label_rows(sample, first_value, allowed)
    second_id, second_experience, second, second_fabricated = label_rows(sample, second_value, allowed)
    if first_id == second_id or not (first_experience or second_experience):
        fail("reviews must be independent and include Rust unsafe-contract experience")
    adjudication_value = read_json(adjudication_path) if adjudication_path is not None else None
    if adjudication_value is not None and adjudication_value.get("sample_sha256") != sample_sha:
        fail("adjudication does not bind the frozen sample")
    if adjudication_value is not None and adjudication_value.get("adjudicator_id") in {first_id, second_id}:
        fail("adjudicator must be independent from both reviewers")
    adjudication = adjudication_rows(sample, adjudication_value, first, second, allowed)
    final = adjudicated_labels(first, second, adjudication, allowed)
    metric_rows = []
    for index, row in enumerate(sample["findings"]):
        metric_rows.append({
            "finding_id": row["finding_id"],
            "rule_id": row["rule_id"],
            "depth": row["depth"],
            "selection_hash": row["selection_hash"],
            "fabricated_local_edge": first_fabricated[index] or second_fabricated[index],
        })
    result = review_metrics(metric_rows, first, second, final, protocol)
    result.update({
        "unit": sample["unit"],
        "sample_sha256": sample_sha,
        "reviewer_ids": sorted([first_id, second_id]),
        "adjudication_sha256": sha256(adjudication_path) if adjudication_path is not None else None,
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    sample = subparsers.add_parser("sample")
    sample.add_argument("--protocol", type=Path, required=True)
    sample.add_argument("--normalized", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True)
    review_parser = subparsers.add_parser("review")
    review_parser.add_argument("--protocol", type=Path, required=True)
    review_parser.add_argument("--sample", type=Path, required=True)
    review_parser.add_argument("--reviewer-one", type=Path, required=True)
    review_parser.add_argument("--reviewer-two", type=Path, required=True)
    review_parser.add_argument("--adjudication", type=Path)
    review_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "sample":
            if args.output.exists():
                fail("sample output must not already exist")
            protocol = load_protocol(args.protocol.resolve(strict=True))
            normalized_path = args.normalized.resolve(strict=True)
            normalized = read_json(normalized_path)
            if not isinstance(normalized, dict) or normalized.get("protocol_sha256") != sha256(args.protocol):
                fail("normalized population does not bind the selected protocol")
            value = make_sample(normalized, protocol, sha256(normalized_path))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(canonical_json(value))
            return 0
        if args.command == "review":
            if args.output.exists():
                fail("review output must not already exist")
            protocol = load_protocol(args.protocol.resolve(strict=True))
            value = review_from_files(
                args.sample.resolve(strict=True),
                args.reviewer_one.resolve(strict=True),
                args.reviewer_two.resolve(strict=True),
                args.adjudication.resolve(strict=True) if args.adjudication is not None else None,
                protocol,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(canonical_json(value))
            return 0
        fail("unsupported review command")
    except (ContractError, OSError, ValueError) as error:
        print(f"standard-library review error: {error}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""No-build tests for deterministic standard-library noise-family auditing."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import noise_audit as audit


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = (
    REPO_ROOT / "artifacts/unsoundaudit-v2/mac/stdlib-pilot/performance-v4"
)
PROTOCOL = REPO_ROOT / "tests/unsoundaudit-v2/stdlib_pilot_protocol_v4.json"
FAMILY_REVIEWS = EVIDENCE_ROOT / "noise-audit/family_reviews.json"


class NoiseFamilyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        (
            cls.protocol,
            cls.populations,
            cls.samples,
            cls.identities,
        ) = audit.load_inputs(EVIDENCE_ROOT, PROTOCOL)

    def test_family_identity_omits_public_root_but_keeps_semantics(self) -> None:
        first = copy.deepcopy(self.populations["core"]["findings"][0])
        second = copy.deepcopy(first)
        second["finding_id"] = "f" * 64
        second["causal_key"]["public_root"] = "core::another_public_root"
        second["public_safe_root"]["def_path"] = "core::another_public_root"

        first_key = audit.family_key("core", first)
        second_key = audit.family_key("core", second)
        self.assertEqual(first_key, second_key)
        self.assertEqual(audit.family_id(first_key), audit.family_id(second_key))

        second["obligation"]["subject"] = "arg:999"
        second["causal_key"]["canonical_obligation"] = "in_bounds:arg:999"
        self.assertNotEqual(first_key, audit.family_key("core", second))

    def test_real_v4_census_is_stable_and_complete(self) -> None:
        clusters = audit.build_clusters(
            self.populations, self.samples, self.protocol, self.identities
        )
        self.assertEqual(clusters["total_findings"], 876)
        self.assertEqual(clusters["unique_sink_count"], 53)
        self.assertEqual(clusters["family_count"], 77)
        self.assertEqual(
            {row["unit"]: row["finding_count"] for row in clusters["populations"]},
            {"core": 269, "alloc": 501, "std": 106},
        )
        clustered = {
            finding_id
            for family in clusters["families"]
            for finding_id in family["finding_ids"]
        }
        expected = {
            finding["finding_id"]
            for population in self.populations.values()
            for finding in population["findings"]
        }
        self.assertEqual(clustered, expected)

    def test_real_v4_audit_panel_has_exact_selection_contract(self) -> None:
        clusters = audit.build_clusters(
            self.populations, self.samples, self.protocol, self.identities
        )
        sample = audit.build_audit_sample(
            clusters, self.populations, self.protocol, self.identities
        )
        self.assertEqual(sample["sample_size"], 124)
        rows = sample["findings"]
        self.assertEqual(len({row["finding_id"] for row in rows}), 124)

        selected = {row["finding_id"] for row in rows}
        all_findings = [
            finding
            for population in self.populations.values()
            for finding in population["findings"]
        ]
        mandatory = {
            finding["finding_id"]
            for finding in all_findings
            if finding["rule_id"].startswith("P1.")
            or finding["rule_id"].endswith("raw_read")
            or finding["propagation"]["depth"] in {"intra_procedural", "recursive"}
        }
        self.assertTrue(mandatory <= selected)

        for family in clusters["families"]:
            reasons = {
                reason
                for row in rows
                if row["family_id"] == family["family_id"]
                for reason in row["selection_reasons"]
            }
            self.assertIn("family_shortest", reasons)
            if family["key"]["propagation_depth"] == "inter_procedural":
                self.assertIn("family_longest", reasons)

    def test_reversed_populations_and_findings_are_byte_identical(self) -> None:
        forward_clusters = audit.build_clusters(
            self.populations, self.samples, self.protocol, self.identities
        )
        forward_sample = audit.build_audit_sample(
            forward_clusters, self.populations, self.protocol, self.identities
        )

        reversed_populations = {
            unit: {
                **copy.deepcopy(self.populations[unit]),
                "findings": list(reversed(self.populations[unit]["findings"])),
            }
            for unit in reversed(tuple(self.populations))
        }
        reversed_clusters = audit.build_clusters(
            reversed_populations, self.samples, self.protocol, self.identities
        )
        reversed_sample = audit.build_audit_sample(
            reversed_clusters, reversed_populations, self.protocol, self.identities
        )

        self.assertEqual(
            audit.canonical_bytes(forward_clusters),
            audit.canonical_bytes(reversed_clusters),
        )
        self.assertEqual(
            audit.canonical_bytes(forward_sample),
            audit.canonical_bytes(reversed_sample),
        )

    def test_outputs_bind_exact_v4_hashes_without_absolute_paths(self) -> None:
        clusters = audit.build_clusters(
            self.populations, self.samples, self.protocol, self.identities
        )
        encoded = audit.canonical_bytes(clusters)
        self.assertNotIn(str(REPO_ROOT).encode(), encoded)
        self.assertEqual(
            {row["unit"]: row["normalized_sha256"] for row in clusters["populations"]},
            {
                "core": "27dca139986fa55673ce1869da439fb1241c01214bae2a8975b950202d2dd38d",
                "alloc": "82b8e405d40baf62a443bee953d7d55d4fca7bc5be4814d2bae2831b2bdcaae9",
                "std": "84da88c52c5cf4a139bfc6f37a8602125a5c2f86b9906a8a78243ac7f5b80e3f",
            },
        )
        self.assertEqual(json.loads(encoded)["protocol_sha256"], self.identities["protocol_sha256"])


class DiagnosticLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol, cls.populations, cls.samples, cls.identities = audit.load_inputs(
            EVIDENCE_ROOT, PROTOCOL
        )
        cls.clusters = audit.build_clusters(
            cls.populations, cls.samples, cls.protocol, cls.identities
        )
        cls.sample = audit.build_audit_sample(
            cls.clusters, cls.populations, cls.protocol, cls.identities
        )

    def labels(self, default: str = "C") -> dict[str, object]:
        rows = []
        for selected in self.sample["findings"]:
            finding = selected["finding"]
            depth = finding["propagation"]["depth"]
            chain_audit = None
            if depth != "intra_procedural":
                witness = finding["propagation"]["witness"]
                calls = [step for step in witness if step["kind"] == "local_call"]
                chain_audit = {
                    "public_root": finding["public_safe_root"]["def_path"],
                    "local_edges": [
                        {
                            "caller": witness[index]["function"],
                            "callee": step["function"],
                            "call_span": audit.span_token(step["span"]),
                        }
                        for index, step in enumerate(witness)
                        if step["kind"] == "local_call"
                    ],
                    "mappings": [
                        {
                            "call_span": audit.span_token(step["span"]),
                            "kind": "audited_exact_mapping",
                            "source": finding["source"]["origin_key"],
                            "destination": finding["obligation"]["subject"],
                        }
                        for step in calls
                    ],
                    "sink_provenance": "The selected obligation reaches the exact sink operand.",
                    "cfg_validation": "The selected route reaches the sink with validation missing.",
                    "cycle_evidence": (
                        "scc_cycle is anchored to the selected real recursive edge."
                        if depth == "recursive"
                        else "not_recursive"
                    ),
                }
            root_cause = {
                "A": "real_unsoundness",
                "B": "needs_contract_evidence",
                "C": "type_or_constructor_invariant",
                "D": "infeasible_cfg_or_route",
                "E": "duplicate_or_identity_defect",
                "U": "insufficient_evidence",
            }[default]
            disposition = {
                "A": "retain",
                "B": "retain",
                "C": "stdlib_invariant_suppression",
                "D": "generic_repair",
                "E": "generic_repair",
                "U": "insufficient_evidence",
            }[default]
            rows.append(
                {
                    "finding_id": finding["finding_id"],
                    "family_id": selected["family_id"],
                    "label": default,
                    "rationale": "Source inspection establishes the diagnostic classification.",
                    "evidence_spans": [audit.span_token(finding["sink"]["span"])],
                    "fabricated_local_edge": False,
                    "root_cause": root_cause,
                    "family_disposition": disposition,
                    "chain_audit": chain_audit,
                }
            )
        return {
            "schema_version": audit.LABEL_SCHEMA,
            "audit_sample_sha256": audit.canonical_sha256(self.sample),
            "clusters_sha256": audit.canonical_sha256(self.clusters),
            "claim_scope": "single_reviewer_development_diagnostic_only",
            "reviewer_id": "single-rust-reviewer",
            "rust_unsafe_contract_experience": True,
            "labels": rows,
        }

    def test_single_reviewer_labels_cover_exact_sample_and_emit_no_formal_metrics(self) -> None:
        labels = self.labels()
        validated = audit.validate_labels(self.clusters, self.sample, labels)
        report = audit.build_audit_report(self.clusters, self.sample, validated)
        self.assertEqual(report["reviewed_findings"], 124)
        self.assertEqual(report["labels"], {"A": 0, "B": 0, "C": 124, "D": 0, "E": 0, "U": 0})
        self.assertEqual(report["claim_scope"], "single_reviewer_development_diagnostic_only")
        self.assertEqual(report["precision_status"], "diagnostic_not_formal_precision")
        def keys(value: object) -> set[str]:
            if isinstance(value, dict):
                return set(value) | {key for child in value.values() for key in keys(child)}
            if isinstance(value, list):
                return {key for child in value for key in keys(child)}
            return set()

        self.assertTrue({"cohens_kappa", "wilson95", "adjudication"}.isdisjoint(keys(report)))

    def test_labels_reject_missing_rows_and_incomplete_chain_audits(self) -> None:
        labels = self.labels()
        labels["labels"].pop()
        with self.assertRaisesRegex(Exception, "exact diagnostic sample"):
            audit.validate_labels(self.clusters, self.sample, labels)

        labels = self.labels()
        interprocedural = next(
            row for row in labels["labels"] if row["chain_audit"] is not None
        )
        interprocedural["chain_audit"] = None
        with self.assertRaisesRegex(Exception, "complete chain audit"):
            audit.validate_labels(self.clusters, self.sample, labels)

    def test_diagnostic_early_stop_is_per_unit_and_fabricated_edges_are_zero_tolerance(self) -> None:
        labels = self.labels()
        core_prefix = [
            row
            for selected in self.sample["findings"]
            if selected["unit"] == "core" and selected["unit_audit_rank"] <= 20
            for row in labels["labels"]
            if row["finding_id"] == selected["finding_id"]
        ]
        for row in core_prefix[:3]:
            row["label"] = "D"
            row["root_cause"] = "infeasible_cfg_or_route"
            row["family_disposition"] = "generic_repair"
        report = audit.build_audit_report(
            self.clusters, self.sample, audit.validate_labels(self.clusters, self.sample, labels)
        )
        self.assertTrue(report["early_stop"]["triggered"])
        self.assertEqual(report["early_stop"]["units"]["core"]["egregious_in_first_20"], 3)

        labels = self.labels()
        labels["labels"][0]["label"] = "E"
        labels["labels"][0]["root_cause"] = "duplicate_or_identity_defect"
        labels["labels"][0]["family_disposition"] = "generic_repair"
        labels["labels"][0]["fabricated_local_edge"] = True
        report = audit.build_audit_report(
            self.clusters, self.sample, audit.validate_labels(self.clusters, self.sample, labels)
        )
        self.assertTrue(report["early_stop"]["fabricated_local_edge"])

    def test_suppression_candidates_require_consistent_c_only_family_evidence(self) -> None:
        labels = self.labels()
        first_family = labels["labels"][0]["family_id"]
        affected = [row for row in labels["labels"] if row["family_id"] == first_family]
        affected[0]["label"] = "B"
        affected[0]["root_cause"] = "needs_contract_evidence"
        affected[0]["family_disposition"] = "retain"
        report = audit.build_audit_report(
            self.clusters, self.sample, audit.validate_labels(self.clusters, self.sample, labels)
        )
        self.assertNotIn(first_family, report["suppression_candidate_family_ids"])

        labels = self.labels()
        report = audit.build_audit_report(
            self.clusters, self.sample, audit.validate_labels(self.clusters, self.sample, labels)
        )
        self.assertEqual(len(report["suppression_candidate_family_ids"]), 77)

    def test_real_single_reviewer_decisions_cover_all_families_without_early_stop(self) -> None:
        reviews = audit.read_json(FAMILY_REVIEWS)
        validated_reviews = audit.validate_family_reviews(self.clusters, reviews)
        labels = audit.labels_from_family_reviews(self.clusters, self.sample, reviews)
        report = audit.build_audit_report(self.clusters, self.sample, labels)

        self.assertEqual(len(validated_reviews["reviews"]), 77)
        self.assertEqual(len(labels["labels"]), 124)
        self.assertEqual(
            report["labels"], {"A": 0, "B": 33, "C": 91, "D": 0, "E": 0, "U": 0}
        )
        self.assertFalse(report["early_stop"]["triggered"])
        self.assertEqual(len(report["suppression_candidate_family_ids"]), 54)
        population_by_family = {
            row["family_id"]: row["population_size"]
            for row in self.clusters["families"]
        }
        self.assertEqual(
            sum(
                population_by_family[identity]
                for identity in report["suppression_candidate_family_ids"]
            ),
            839,
        )

    def test_review_expansion_is_independent_of_group_and_sample_order(self) -> None:
        reviews = audit.read_json(FAMILY_REVIEWS)
        forward = audit.labels_from_family_reviews(self.clusters, self.sample, reviews)
        reversed_reviews = {
            **copy.deepcopy(reviews),
            "reviews": list(reversed(reviews["reviews"])),
        }
        reversed_sample = {
            **copy.deepcopy(self.sample),
            "findings": list(reversed(self.sample["findings"])),
        }
        reversed_labels = audit.labels_from_family_reviews(
            self.clusters, reversed_sample, reversed_reviews
        )
        self.assertEqual(
            {row["finding_id"]: row for row in forward["labels"]},
            {row["finding_id"]: row for row in reversed_labels["labels"]},
        )

    def test_family_review_table_rejects_missing_and_non_c_suppressions(self) -> None:
        reviews = audit.read_json(FAMILY_REVIEWS)
        reviews["reviews"][0]["family_ids"].pop()
        with self.assertRaisesRegex(Exception, "exact clustered population"):
            audit.validate_family_reviews(self.clusters, reviews)

        reviews = audit.read_json(FAMILY_REVIEWS)
        suppressing = next(
            row
            for row in reviews["reviews"]
            if row["family_disposition"] == "stdlib_invariant_suppression"
        )
        suppressing["label"] = "B"
        with self.assertRaisesRegex(Exception, "only C type invariants"):
            audit.validate_family_reviews(self.clusters, reviews)

    def test_exact_suppression_spec_contains_only_audited_c_families(self) -> None:
        reviews = audit.read_json(FAMILY_REVIEWS)
        allowlist = audit.build_suppression_allowlist(self.clusters, reviews)
        report = audit.build_audit_report(
            self.clusters,
            self.sample,
            audit.labels_from_family_reviews(self.clusters, self.sample, reviews),
        )
        entries = allowlist["entries"]
        self.assertEqual(len(entries), 54)
        self.assertEqual(
            {row["family_id"] for row in entries},
            set(report["suppression_candidate_family_ids"]),
        )
        self.assertEqual(allowlist["rustc_commit"], "1bc403daadbebb553ccc211a0a8eebb73989665f")
        self.assertEqual(allowlist["entries_sha256"], audit.digest(entries))
        self.assertTrue(
            all(
                set(row)
                == {
                    "rustc_commit",
                    "crate_key",
                    "rule_id",
                    "sink_token",
                    "first_failure_token",
                    "canonical_obligation",
                    "source_kind",
                    "family_id",
                    "evidence_sha256",
                    "rationale",
                }
                for row in entries
            )
        )

    def test_suppression_spec_is_independent_of_review_group_order(self) -> None:
        reviews = audit.read_json(FAMILY_REVIEWS)
        reversed_reviews = {
            **copy.deepcopy(reviews),
            "reviews": list(reversed(reviews["reviews"])),
        }
        self.assertEqual(
            audit.canonical_bytes(
                audit.build_suppression_allowlist(self.clusters, reviews)
            ),
            audit.canonical_bytes(
                audit.build_suppression_allowlist(self.clusters, reversed_reviews)
            ),
        )


if __name__ == "__main__":
    unittest.main()

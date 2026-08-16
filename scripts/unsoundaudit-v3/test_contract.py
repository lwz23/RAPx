#!/usr/bin/env python3
"""Fail-closed contract tests for the compiler-aligned rap-unit-v3 receipt."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "tests/unsoundaudit-v3/rap-unit-v3.schema.json"
VALIDATOR_PATH = Path(__file__).with_name("validate_receipt.py")
ANALYZER_ENTRY_PATH = REPO_ROOT / "rapx/src/analysis/unsoundaudit.rs"
MIR_PATH = REPO_ROOT / "rapx/src/analysis/unsoundaudit/mir.rs"


def load_validator():
    spec = importlib.util.spec_from_file_location("unsoundaudit_v3_validator", VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load v3 validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def span(line: int) -> dict:
    return {
        "path": "src/lib.rs",
        "start": {"line": line, "column": 1},
        "end": {"line": line, "column": 2},
    }


def minimal_receipt() -> dict:
    commit = "a" * 40
    checker = "b" * 64
    return {
        "schema_version": "rap-unit-v3",
        "source": "rap",
        "success": True,
        "analysis_status": "complete",
        "unit_id": "sample-unit",
        "rustc_invocation_id": "sample-invocation",
        "package_id": "sample@0.1.0#Cargo.toml",
        "package_name": "sample",
        "package_version": "0.1.0",
        "crate_name": "sample",
        "crate_types": ["lib"],
        "target_kind": "lib",
        "target_triple": "aarch64-apple-darwin",
        "manifest_path": "Cargo.toml",
        "source_path": "src/lib.rs",
        "cargo_supplied_rustc_commit": commit,
        "rap_compiler_commit": commit,
        "rustc_commit": commit,
        "toolchain_adapter": {
            "adapter_id": "rustc-2024-1bc403",
            "rustc_commit": commit,
            "checker_sha256": checker,
            "emitted_kinds": ["raw_pointer_dereference"],
        },
        "unsafe_site_census": {
            "discovered": 1,
            "bound_exact": 1,
            "unbound": 0,
            "ambiguous": 0,
            "elided": 0,
            "by_kind": [{"kind": "raw_pointer_dereference", "count": 1}],
        },
        "unsafe_sites": [{
            "site_id": "c" * 64,
            "kind": "raw_pointer_dereference",
            "owner": "sample::root",
            "span": span(1),
            "binding": "bound_exact",
            "candidate_count": 0,
        }],
        "coverage_failures": [],
        "observed_counts": {
            "semantics": "exact",
            "pattern1": 0,
            "pattern2": 0,
            "pattern3": 0,
            "pattern4": 0,
            "pattern5": 0,
            "pattern6": 0,
        },
        "finding_count": 0,
        "findings": [],
        "code_dictionary": [],
    }


def opaque_finding() -> dict:
    code = "pub fn root(p: *const u8) { unsafe { custom(p) } }\n"
    code_sha = hashlib.sha256(code.encode()).hexdigest()
    code_id = hashlib.sha256(b"function:sample::root").hexdigest()
    causal = {
        "pattern": "P1",
        "site": "c" * 64,
        "source": "formal:0",
        "failure": "opaque_unsafe_call",
        "root": "sample::root",
        "exposure": None,
    }
    return {
        "finding_id": hashlib.sha256(
            json.dumps(causal, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "primary_pattern": "P1",
        "mechanism_id": "parameter_to_unsafe_operand",
        "classification_confidence": "structural",
        "public_safe_root": {
            "owner": "sample::root",
            "span": span(1),
            "code_id": code_id,
        },
        "source": {
            "kind": "public_parameter",
            "identity": "formal:0",
            "span": span(1),
        },
        "first_contract_failure": {
            "kind": "opaque_unsafe_call",
            "span": span(1),
        },
        "unsafe_site": {
            "site_id": "c" * 64,
            "kind": "unsafe_call",
            "owner": "sample::root",
            "span": span(1),
            "callee": "sample::custom",
            "operands": ["argument:0"],
            "destination": None,
            "binding": "bound_exact",
        },
        "exposure": None,
        "contract": {
            "precision": "opaque",
            "model_id": None,
            "relevant_roles": ["argument:0"],
            "obligation": None,
            "safety_doc": {
                "status": "unavailable",
                "code_id": None,
                "reason": "external_metadata_has_no_docs",
            },
        },
        "propagation": {
            "depth": "intra_procedural",
            "boundaries": [],
            "witness": [
                {"kind": "entry", "owner": "sample::root", "span": span(1)},
                {"kind": "unsafe_site", "owner": "sample::root", "span": span(1)},
            ],
        },
        "code_refs": [code_id],
    }, {
        "code_id": code_id,
        "kind": "function_body",
        "owner": "sample::root",
        "path": "src/lib.rs",
        "span": span(1),
        "sha256": code_sha,
        "text": code,
    }


class RapUnitV3ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.validator = load_validator()
        cls.schema = cls.validator.read_json(SCHEMA_PATH)

    def assert_rejected(self, receipt: dict) -> None:
        with self.assertRaises(self.validator.ContractError):
            self.validator.validate_receipt(receipt, self.schema)

    def test_minimal_complete_receipt_passes(self) -> None:
        self.validator.validate_receipt(minimal_receipt(), self.schema)

    def test_unknown_top_level_key_is_rejected(self) -> None:
        receipt = minimal_receipt()
        receipt["unexpected"] = True
        self.assert_rejected(receipt)

    def test_production_writer_and_discovery_authority_are_v3_only(self) -> None:
        entry = ANALYZER_ENTRY_PATH.read_text()
        mir = MIR_PATH.read_text()
        compatibility_marker = (
            "// These v2 serializers remain only as immutable compatibility test fixtures."
        )
        production, compatibility = entry.split(compatibility_marker, 1)

        self.assertIn("v3_output::build_receipt(metadata, analysis)", production)
        self.assertNotIn("rap-unit-v2", production)
        self.assertIn('#[cfg(test)]\nmod stdlib_suppressions;', production)
        self.assertIn('#[cfg(test)]\nfn unit_json', compatibility)
        self.assertNotIn("legacy_findings", entry)
        self.assertNotIn("is_modeled_registry_call", mir)
        self.assertNotIn("adjusted_rule", mir)

    def test_census_must_reconcile_and_site_kinds_are_sorted_unique(self) -> None:
        receipt = minimal_receipt()
        receipt["unsafe_site_census"]["discovered"] = 2
        self.assert_rejected(receipt)

        receipt = minimal_receipt()
        receipt["unsafe_site_census"]["by_kind"] *= 2
        self.assert_rejected(receipt)

    def test_partial_receipt_requires_lower_bound_counts_and_coverage_failure(self) -> None:
        receipt = minimal_receipt()
        receipt["analysis_status"] = "partial"
        self.assert_rejected(receipt)

        receipt["observed_counts"]["semantics"] = "lower_bound"
        receipt["unsafe_site_census"] = {
            "discovered": 1,
            "bound_exact": 0,
            "unbound": 1,
            "ambiguous": 0,
            "elided": 0,
            "by_kind": [{"kind": "raw_pointer_dereference", "count": 1}],
        }
        receipt["unsafe_sites"] = [{
            "site_id": "d" * 64,
            "kind": "raw_pointer_dereference",
            "owner": "sample::root",
            "span": span(1),
            "binding": "unbound",
            "candidate_count": 0,
        }]
        receipt["coverage_failures"] = [{
            "reason": "unbound_unsafe_site",
            "owner": "sample::root",
            "site_id": "d" * 64,
            "candidate_count": 0,
        }]
        self.validator.validate_receipt(receipt, self.schema)

    def test_complete_receipt_rejects_coverage_failures(self) -> None:
        receipt = minimal_receipt()
        receipt["coverage_failures"] = [{
            "reason": "collector_missing_after_steal",
            "owner": "sample::root",
            "site_id": None,
            "candidate_count": 0,
        }]
        self.assert_rejected(receipt)

    def test_duplicate_site_or_code_ids_are_rejected(self) -> None:
        receipt = minimal_receipt()
        receipt["unsafe_sites"].append(copy.deepcopy(receipt["unsafe_sites"][0]))
        self.assert_rejected(receipt)

        receipt = minimal_receipt()
        finding, code = opaque_finding()
        receipt["unsafe_sites"][0]["kind"] = "unsafe_call"
        receipt["unsafe_sites"][0]["candidate_count"] = 2
        receipt["unsafe_site_census"]["by_kind"] = [{"kind": "unsafe_call", "count": 1}]
        receipt["findings"] = [finding, copy.deepcopy(finding)]
        receipt["finding_count"] = 2
        receipt["observed_counts"]["pattern1"] = 2
        receipt["code_dictionary"] = [code]
        self.assert_rejected(receipt)

        receipt = minimal_receipt()
        receipt["code_dictionary"] = [code, copy.deepcopy(code)]
        self.assert_rejected(receipt)

    def test_opaque_contract_is_a_p1_p6_finding_not_a_seventh_pattern(self) -> None:
        receipt = minimal_receipt()
        finding, code = opaque_finding()
        receipt["unsafe_sites"][0]["kind"] = "unsafe_call"
        receipt["unsafe_sites"][0]["candidate_count"] = 1
        receipt["toolchain_adapter"]["emitted_kinds"] = ["unsafe_call"]
        receipt["unsafe_site_census"]["by_kind"] = [{"kind": "unsafe_call", "count": 1}]
        receipt["findings"] = [finding]
        receipt["finding_count"] = 1
        receipt["observed_counts"]["pattern1"] = 1
        receipt["code_dictionary"] = [code]
        self.validator.validate_receipt(receipt, self.schema)

    def test_paths_hashes_counts_and_code_references_fail_closed(self) -> None:
        finding, code = opaque_finding()
        for field, value in (
            ("path", "/Users/example/src/lib.rs"),
            ("path", "src//lib.rs"),
            ("sha256", "not-a-hash"),
        ):
            with self.subTest(field=field, value=value):
                receipt = minimal_receipt()
                broken = copy.deepcopy(code)
                broken[field] = value
                receipt["code_dictionary"] = [broken]
                self.assert_rejected(receipt)

        receipt = minimal_receipt()
        receipt["unsafe_sites"][0]["kind"] = "unsafe_call"
        receipt["unsafe_sites"][0]["candidate_count"] = 1
        receipt["toolchain_adapter"]["emitted_kinds"] = ["unsafe_call"]
        receipt["unsafe_site_census"]["by_kind"] = [{"kind": "unsafe_call", "count": 1}]
        receipt["findings"] = [finding]
        receipt["finding_count"] = 1
        receipt["observed_counts"]["pattern1"] = 1
        self.assert_rejected(receipt)

        schema_pattern = self.schema["$defs"]["relative_path"]["pattern"]
        self.assertIsNone(re.fullmatch(schema_pattern, "C:src/lib.rs"))


if __name__ == "__main__":
    unittest.main()

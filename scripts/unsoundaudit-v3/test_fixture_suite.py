#!/usr/bin/env python3
"""No-build contracts for the rap-unit-v3 fixture runner."""

from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path

from test_contract import minimal_receipt, opaque_finding


RUNNER_PATH = Path(__file__).with_name("run_fixture_suite.py")


def load_runner():
    spec = importlib.util.spec_from_file_location("unsoundaudit_v3_fixture_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load v3 fixture runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class V3FixtureRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_runner()

    def receipt_with_finding(self) -> dict:
        receipt = minimal_receipt()
        finding, code = opaque_finding()
        receipt["unsafe_sites"][0]["kind"] = "unsafe_call"
        receipt["unsafe_sites"][0]["candidate_count"] = 1
        receipt["toolchain_adapter"]["emitted_kinds"] = ["unsafe_call"]
        receipt["unsafe_site_census"]["by_kind"] = [
            {"kind": "unsafe_call", "count": 1}
        ]
        receipt["findings"] = [finding]
        receipt["finding_count"] = 1
        receipt["observed_counts"]["pattern1"] = 1
        receipt["code_dictionary"] = [code]
        return receipt

    def test_complete_receipt_normalizes_to_p1_p6_counts(self) -> None:
        normalized = self.runner.normalized_case("sample", self.receipt_with_finding())
        self.assertEqual(normalized["pattern_counts"], {
            "pattern1": 1,
            "pattern2": 0,
            "pattern3": 0,
            "pattern4": 0,
            "pattern5": 0,
            "pattern6": 0,
        })
        self.assertEqual(normalized["finding_count"], 1)
        self.assertEqual(normalized["semantic_findings"][0]["primary_pattern"], "P1")

    def test_partial_or_lower_bound_receipt_is_rejected(self) -> None:
        receipt = minimal_receipt()
        receipt["analysis_status"] = "partial"
        receipt["observed_counts"]["semantics"] = "lower_bound"
        with self.assertRaises(self.runner.RunnerError):
            self.runner.normalized_case("partial", receipt)

    def test_suite_order_and_projection_are_deterministic(self) -> None:
        first = self.runner.normalized_case("z", self.receipt_with_finding())
        changed = self.receipt_with_finding()
        changed["findings"][0]["propagation"]["witness"].reverse()
        changed["code_dictionary"][0]["text"] += "// packaging-only\n"
        second = self.runner.normalized_case("a", changed)

        forward = self.runner.normalized_suite("challenge", [first, second])
        reverse = self.runner.normalized_suite("challenge", [second, first])
        self.assertEqual(forward, reverse)
        self.assertEqual([case["case_id"] for case in forward["cases"]], ["a", "z"])
        self.assertEqual(forward["aggregate_pattern_counts"]["pattern1"], 2)
        self.assertEqual(first["semantic_findings"], second["semantic_findings"])

    def test_oracle_checks_pattern_counts_not_legacy_rule_ids(self) -> None:
        case = self.runner.normalized_case("sample", self.receipt_with_finding())
        oracle = {
            "expected_pattern_counts": case["pattern_counts"],
            "expected_primary_pattern": "pattern1",
            "expected_rule_id": "legacy.rule.must.not.control.v3",
        }
        self.runner.validate_semantic_oracle("sample", case, oracle)

        wrong = copy.deepcopy(oracle)
        wrong["expected_pattern_counts"]["pattern1"] = 0
        with self.assertRaises(self.runner.RunnerError):
            self.runner.validate_semantic_oracle("sample", case, wrong)


if __name__ == "__main__":
    unittest.main()

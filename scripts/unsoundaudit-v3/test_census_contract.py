#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "unsoundaudit-v3" / "census-fixtures"
RUNNER = Path(__file__).with_name("run_census_fixture.py")


class CensusFixtureContractTests(unittest.TestCase):
    def test_fixture_contract_is_closed_and_covers_adapter_kinds(self) -> None:
        expected = {
            "legacy-2024": {
                "unsafe_call",
                "inline_assembly",
                "restricted_validity_initialization",
                "mutable_static_access",
                "extern_static_access",
                "raw_pointer_dereference",
                "union_field_access",
                "layout_constrained_mutation",
                "layout_constrained_borrow",
                "target_feature_call",
            },
            "current-2026": {
                "unsafe_call",
                "inline_assembly",
                "unsafe_field_initialization",
                "unsafe_field_use",
                "mutable_static_access",
                "extern_static_access",
                "raw_pointer_dereference",
                "union_field_access",
                "target_feature_call",
                "unsafe_binder_cast",
                "explicit_drop_call",
            },
        }
        for fixture_name, expected_kinds in expected.items():
            fixture = FIXTURES / fixture_name
            manifest = tomllib.loads((fixture / "Cargo.toml").read_text(encoding="utf-8"))
            self.assertEqual(manifest.get("dependencies"), {})
            self.assertNotIn("dev-dependencies", manifest)
            oracle = json.loads((fixture / "oracle.json").read_text(encoding="utf-8"))
            self.assertEqual(
                set(oracle),
                {
                    "adapter_id",
                    "rustc_commit",
                    "owner_count",
                    "site_count",
                    "kind_counts",
                    "sites",
                    "zero_site_owners",
                }
                | ({"compile_fail"} if fixture_name == "current-2026" else set()),
            )
            covered = set(oracle["kind_counts"])
            covered.update(row["expected_kind"] for row in oracle.get("compile_fail", []))
            self.assertEqual(covered, expected_kinds)
            self.assertEqual(sum(oracle["kind_counts"].values()), oracle["site_count"])
            self.assertEqual(len(oracle["sites"]), oracle["site_count"])
            self.assertEqual(len(oracle["sites"]), len({tuple(row) for row in oracle["sites"]}))
            self.assertRegex(oracle["rustc_commit"], r"^[0-9a-f]{40}$")

    def test_fixtures_exercise_ownership_and_exemptions_without_ast_fallback(self) -> None:
        legacy = (FIXTURES / "legacy-2024" / "src" / "lib.rs").read_text(encoding="utf-8")
        current = (FIXTURES / "current-2026" / "src" / "lib.rs").read_text(encoding="utf-8")
        for source in (legacy, current):
            for token in ("include!(", "let closure =", "const {", "macro_rules!", "nested_explicit"):
                self.assertIn(token, source)
            self.assertIn("let _ = unsafe {};", source)
        self.assertIn("raw_borrow_deref", current)
        self.assertIn("raw_borrow_union", current)
        for adapter in (
            ROOT / "rapx" / "src" / "analysis" / "unsoundaudit" / "unsafe_site" / "legacy_2024.rs",
            ROOT / "rapx" / "src" / "analysis" / "unsoundaudit" / "unsafe_site" / "current_2026.rs",
        ):
            source = adapter.read_text(encoding="utf-8")
            self.assertNotIn("rustc_ast::visit", source)
            self.assertNotIn("intravisit", source)

    def test_current_compile_fail_oracle_is_exact(self) -> None:
        fixture = FIXTURES / "current-2026"
        oracle = json.loads((fixture / "oracle.json").read_text(encoding="utf-8"))
        self.assertEqual(
            oracle["compile_fail"],
            [
                {
                    "path": "compile-fail/explicit_drop.rs",
                    "expected_kind": "explicit_drop_call",
                    "diagnostic_code": "E0040",
                }
            ],
        )
        source = (fixture / oracle["compile_fail"][0]["path"]).read_text(encoding="utf-8")
        self.assertIn("Drop::drop(&mut value);", source)

    def test_runner_exposes_strict_oracle_validation(self) -> None:
        spec = importlib.util.spec_from_file_location("run_census_fixture", RUNNER)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        self.assertTrue(callable(module.validate_payload))
        self.assertTrue(callable(module.validate_binding))
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("full-analysis and census-only runs are not byte-identical", source)
        self.assertIn("full-analysis binding sidecars are not byte-identical", source)
        self.assertIn('site["status"] != "bound_exact"', source)
        self.assertIn('environment["UNSOUND_SCANNER_CENSUS_ONLY"] = "1"', source)
        self.assertIn("unsafe-site kind/owner/span projection drifted", source)
        self.assertNotRegex(source, re.compile(r"nearest|best.?effort", re.IGNORECASE))


if __name__ == "__main__":
    unittest.main()

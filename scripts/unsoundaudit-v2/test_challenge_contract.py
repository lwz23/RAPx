#!/usr/bin/env python3
"""Pure standard-library contract tests for the adversarial challenge suite."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_receipt import ContractError, read_json


RUNNER_PATH = SCRIPT_DIR / "run_challenge_suite.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_challenge_suite", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ChallengeRunnerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_runner()
        cls.manifest_path = REPO_ROOT / "tests/unsoundaudit-v2/challenge_manifest.json"
        cls.case_root = REPO_ROOT / "tests/unsoundaudit-v2/challenges"

    def verify_production_contract(self) -> list[str]:
        return self.runner.verify_challenge_contract(
            REPO_ROOT,
            self.case_root,
            self.manifest_path,
            read_json(self.manifest_path),
        )

    def copied_contract(self, directory: str) -> tuple[Path, Path, Path]:
        repo = Path(directory) / "repo"
        source_parent = REPO_ROOT / "tests/unsoundaudit-v2"
        destination_parent = repo / "tests/unsoundaudit-v2"
        destination_parent.mkdir(parents=True)
        shutil.copytree(source_parent / "challenges", destination_parent / "challenges")
        shutil.copy2(source_parent / "challenge_manifest.json", destination_parent / "challenge_manifest.json")
        return repo, destination_parent / "challenges", destination_parent / "challenge_manifest.json"

    def test_production_contract_is_closed_and_sorted(self) -> None:
        case_ids = self.verify_production_contract()
        self.assertEqual(len(case_ids), self.runner.CHALLENGE_COUNT)
        self.assertEqual(case_ids, sorted(case_ids))
        self.assertEqual(self.runner.CHALLENGE_COUNT, 27)
        self.assertEqual(
            self.runner.CHALLENGE_COUNTS,
            {"pattern1": 13, "pattern2": 0, "pattern3": 0, "pattern4": 0, "pattern5": 0, "pattern6": 0},
        )

    def test_schema_and_legacy_provenance_drift_are_rejected(self) -> None:
        manifest = read_json(self.manifest_path)
        manifest["schema_version"] = "unsoundaudit-v2-fixture-manifest-v1"
        with self.assertRaises(ContractError):
            self.runner.verify_challenge_contract(REPO_ROOT, self.case_root, self.manifest_path, manifest)
        manifest = read_json(self.manifest_path)
        manifest["legacy_provenance_manifest"] = {}
        with self.assertRaises(ContractError):
            self.runner.verify_challenge_contract(REPO_ROOT, self.case_root, self.manifest_path, manifest)

    def test_manifest_requires_exact_case_directory_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, case_root, manifest_path = self.copied_contract(directory)
            (case_root / "unexpected_case").mkdir()
            with mock.patch.object(self.runner, "git_tracked", return_value=True):
                with self.assertRaises(ContractError):
                    self.runner.verify_challenge_contract(repo, case_root, manifest_path, read_json(manifest_path))

    def test_original_fixture_tree_cannot_be_used_as_challenge_root(self) -> None:
        with self.assertRaises(ContractError):
            self.runner.verify_challenge_contract(
                REPO_ROOT,
                REPO_ROOT / "tests/unsoundaudit-v2/fixtures",
                self.manifest_path,
                read_json(self.manifest_path),
            )

    def test_extra_file_hash_tamper_and_undeclared_file_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, case_root, manifest_path = self.copied_contract(directory)
            macro_file = case_root / "span_macro_stable_positive/src/generated_macro.rs"
            macro_file.write_text(macro_file.read_text(encoding="utf-8") + "// tampered\n", encoding="utf-8")
            with mock.patch.object(self.runner, "git_tracked", return_value=True):
                with self.assertRaises(ContractError):
                    self.runner.verify_challenge_contract(repo, case_root, manifest_path, read_json(manifest_path))
            shutil.rmtree(case_root)
            shutil.copytree(REPO_ROOT / "tests/unsoundaudit-v2/challenges", case_root)
            (case_root / "cfg_diamond_all_paths_guard_negative/src/undeclared.rs").write_text(
                "// not in manifest\n", encoding="utf-8"
            )
            with mock.patch.object(self.runner, "git_tracked", return_value=True):
                with self.assertRaises(ContractError):
                    self.runner.verify_challenge_contract(repo, case_root, manifest_path, read_json(manifest_path))

    def test_symlink_and_path_escape_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, case_root, manifest_path = self.copied_contract(directory)
            target = case_root / "span_include_stable_positive/src/included_sink.rs"
            target.unlink()
            target.symlink_to(Path("../../../../outside.rs"))
            (repo / "tests/outside.rs").write_text("pub fn outside() {}\n", encoding="utf-8")
            with mock.patch.object(self.runner, "git_tracked", return_value=True):
                with self.assertRaises(ContractError):
                    self.runner.verify_challenge_contract(repo, case_root, manifest_path, read_json(manifest_path))

    def test_dependencies_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, case_root, manifest_path = self.copied_contract(directory)
            cargo_toml = case_root / "opaque_external_negative/Cargo.toml"
            cargo_toml.write_text(cargo_toml.read_text(encoding="utf-8") + "\n[dependencies]\n", encoding="utf-8")
            with mock.patch.object(self.runner, "git_tracked", return_value=True):
                with self.assertRaises(ContractError):
                    self.runner.verify_challenge_contract(repo, case_root, manifest_path, read_json(manifest_path))

    def test_runner_requires_exact_environment_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cargo = root / "cargo"
            rap_bin_dir = root / "rap-bin"
            output = root / "output"
            cargo.write_text("not invoked\n", encoding="utf-8")
            rap_bin_dir.mkdir()
            arguments = [
                "run_challenge_suite.py",
                "--repo-root", str(REPO_ROOT),
                "--challenge-root", str(self.case_root),
                "--manifest", str(self.manifest_path),
                "--schema", str(REPO_ROOT / "tests/unsoundaudit-v2/rap-unit-v2.schema.json"),
                "--cargo", str(cargo),
                "--rap-bin-dir", str(rap_bin_dir),
                "--output-root", str(output),
            ]
            with mock.patch.object(sys, "argv", arguments), mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(self.runner.main(), 1)
            self.assertFalse(output.exists())

    def test_normalized_output_sorts_case_ids_and_closes_aggregate(self) -> None:
        cases = [
            {"case_id": "z", "pattern_counts": {f"pattern{n}": (1 if n == 1 else 0) for n in range(1, 7)}},
            {"case_id": "a", "pattern_counts": {f"pattern{n}": 0 for n in range(1, 7)}},
        ]
        normalized = self.runner.normalized_suite(cases)
        self.assertEqual([case["case_id"] for case in normalized["cases"]], ["a", "z"])
        self.assertEqual(normalized["schema_version"], "rap-challenge-suite-v1")
        self.assertEqual(normalized["challenge_count"], 2)
        self.assertEqual(normalized["aggregate_pattern_counts"]["pattern1"], 1)

    def test_macro_and_include_oracles_freeze_distinct_span_paths(self) -> None:
        macro = read_json(self.case_root / "span_macro_stable_positive/fixture.json")
        included = read_json(self.case_root / "span_include_stable_positive/fixture.json")
        self.assertEqual(
            macro["expected_finding_spans"],
            [
                {"finding_field": "first_contract_failure", "path": "src/lib.rs"},
                {"finding_field": "sink", "path": "src/lib.rs"},
            ],
        )
        self.assertEqual(
            included["expected_finding_spans"],
            [
                {"finding_field": "first_contract_failure", "path": "src/included_sink.rs"},
                {"finding_field": "sink", "path": "src/included_sink.rs"},
            ],
        )

    def test_local_out_and_opaque_oracles_keep_the_frozen_boundary(self) -> None:
        local_out = read_json(self.case_root / "call_local_out_positive/fixture.json")
        self.assertEqual(local_out["source"], "public_parameter")
        self.assertEqual(local_out["expected_rule_id"], "P1.get_unchecked")
        self.assertEqual(local_out["propagation_depth"], "intra_procedural")
        for case_id in (
            "opaque_fn_pointer_negative",
            "opaque_closure_negative",
            "opaque_dyn_negative",
            "opaque_external_negative",
        ):
            oracle = read_json(self.case_root / case_id / "fixture.json")
            self.assertEqual(oracle["expected_finding_count"], 0)
            self.assertEqual(oracle["fix_fact"], "opaque_unsupported_no_edge")
            self.assertIn("unsupported opaque boundary", oracle["limitations"])

    def test_positive_depths_follow_the_actual_sink_function(self) -> None:
        expected = {
            "call_local_out_positive": "intra_procedural",
            "call_wrappers_0_hop_positive": "intra_procedural",
            "call_wrappers_1_hop_positive": "inter_procedural",
            "call_wrappers_3_hop_positive": "inter_procedural",
            "call_wrappers_5_hop_positive": "inter_procedural",
            "call_wrong_value_guard_positive": "inter_procedural",
            "cfg_diamond_one_branch_guard_positive": "intra_procedural",
            "cfg_guard_then_one_path_write_positive": "intra_procedural",
            "cfg_loop_reassignment_positive": "intra_procedural",
            "recursion_direct_positive": "recursive",
            "recursion_mutual_positive": "recursive",
            "span_include_stable_positive": "inter_procedural",
            "span_macro_stable_positive": "intra_procedural",
        }
        observed = {}
        for oracle_path in self.case_root.glob("*/fixture.json"):
            oracle = read_json(oracle_path)
            if oracle["role"] == "positive":
                observed[oracle["case_id"]] = oracle["propagation_depth"]
        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()

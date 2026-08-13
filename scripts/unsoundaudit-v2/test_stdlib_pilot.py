#!/usr/bin/env python3
"""No-build contract tests for the bounded standard-library pilot."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path, PurePosixPath

import run_stdlib_pilot as pilot
from validate_receipt import ContractError


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = REPO_ROOT / "tests/unsoundaudit-v2/stdlib_pilot_protocol.json"


class PilotProtocolTests(unittest.TestCase):
    def test_protocol_is_closed_and_pre_registered(self) -> None:
        value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(
            set(value),
            {
                "schema_version",
                "rap_source_commit",
                "rustc_commit",
                "cargo_commit",
                "target_triple",
                "unit_order",
                "sample_seed",
                "cargo_jobs",
                "runs_per_unit",
                "fresh_targets",
                "units",
                "sampling",
                "review",
                "thresholds",
            },
        )
        self.assertEqual(value["schema_version"], "unsoundaudit-v2-stdlib-pilot-protocol-v1")
        self.assertEqual(value["rap_source_commit"], "fd0a64cc794fb94f4b54022f8b8917b90cfb1e9c")
        self.assertEqual(value["rustc_commit"], "1bc403daadbebb553ccc211a0a8eebb73989665f")
        self.assertEqual(value["cargo_commit"], "15fbd2f607d4defc87053b8b76bf5038f2483cf4")
        self.assertEqual(value["unit_order"], ["core", "alloc", "std"])
        self.assertEqual(value["cargo_jobs"], 1)
        self.assertEqual(value["runs_per_unit"], 2)
        self.assertEqual(value["fresh_targets"], {"control": True, "scan": True})
        self.assertRegex(value["sample_seed"], r"^[A-Za-z0-9._-]+$")


class PilotRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = pilot.load_protocol(PROTOCOL)

    def test_driver_is_minimal_untracked_and_has_no_build_script(self) -> None:
        for unit in ("core", "alloc", "std"):
            with self.subTest(unit=unit), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pilot.write_driver(root, unit)
                self.assertEqual(
                    sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()),
                    ["Cargo.lock", "Cargo.toml", "src/lib.rs"],
                )
                self.assertFalse((root / "build.rs").exists())
                self.assertIn("--locked", pilot.cargo_arguments(self.protocol, unit))
                self.assertEqual(pilot.cargo_arguments(self.protocol, unit)[-2:], ["--jobs", "1"])
                self.assertIn(f"build-std={','.join(self.protocol['units'][unit]['build_std_components'])}", pilot.cargo_arguments(self.protocol, unit))

    def test_rss_parser_sums_only_the_selected_process_group(self) -> None:
        output = " 10 99 1024\n 11 99 2048\n 12 100 4096\ninvalid\n"
        self.assertEqual(pilot.parse_process_group_rss(output, 99), 3072 * 1024)

    def test_resource_runner_classifies_success_timeout_and_memory(self) -> None:
        environment = os.environ.copy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            success = pilot.run_limited(
                ["/bin/sh", "-c", "exit 0"], root, environment, root / "success.log", 2, 1024**3,
            )
            timeout = pilot.run_limited(
                ["/bin/sh", "-c", "sleep 5"], root, environment, root / "timeout.log", 0.05, 1024**3,
            )
            memory = pilot.run_limited(
                ["/bin/sh", "-c", "sleep 5"], root, environment, root / "memory.log", 2, 1,
            )
        self.assertEqual(success.classification, "success")
        self.assertEqual(timeout.classification, "timeout")
        self.assertEqual(memory.classification, "memory_limit")

    def test_normalized_output_drops_machine_paths_and_resource_noise(self) -> None:
        receipt = {
            "cargo_supplied_rustc_commit": "a" * 40,
            "rap_compiler_commit": "a" * 40,
            "rustc_commit": "a" * 40,
            "finding_count": 0,
            "pattern_counts": {f"pattern{number}": 0 for number in range(1, 7)},
            "findings": [],
            "project_root": "/private/machine",
        }
        normalized = pilot.normalized_unit("core", receipt, "b" * 64, {"rust_file_count": 1, "rust_line_count": 2, "rust_kloc": 0.002})
        encoded = pilot.canonical_json(normalized)
        self.assertNotIn(b"/private", encoded)
        self.assertNotIn(b"elapsed", encoded)

    def test_nonempty_driver_and_unsupported_unit_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "occupied").write_text("x", encoding="utf-8")
            with self.assertRaises(ContractError):
                pilot.write_driver(root, "core")
        with self.assertRaises(ContractError):
            pilot.driver_files("other")

    def test_rss_monitor_failure_is_not_treated_as_zero_usage(self) -> None:
        original = pilot.subprocess.run

        def failed_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="monitor failed")

        pilot.subprocess.run = failed_run
        try:
            with self.assertRaisesRegex(pilot.PilotFailure, "RSS monitor failed"):
                pilot.process_group_rss(123)
        finally:
            pilot.subprocess.run = original

    def test_running_process_missing_from_rss_monitor_fails_closed(self) -> None:
        original = pilot.process_group_rss
        pilot.process_group_rss = lambda _pid: 0
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaisesRegex(pilot.PilotFailure, "absent from RSS monitor"):
                    pilot.run_limited(
                        ["/bin/sh", "-c", "sleep 5"], root, os.environ.copy(), root / "rss.log", 2, 1024**3,
                    )
        finally:
            pilot.process_group_rss = original

    def test_units_have_exact_routes_and_frozen_limits(self) -> None:
        value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        expected = {
            "core": (3600, 12 * 1024**3, ["core"], "library/core"),
            "alloc": (5400, 16 * 1024**3, ["core", "alloc"], "library/alloc"),
            "std": (10800, 24 * 1024**3, ["core", "alloc", "std", "panic_abort"], "library/std"),
        }
        self.assertEqual(set(value["units"]), set(expected))
        for unit, (wall, rss, components, root) in expected.items():
            row = value["units"][unit]
            self.assertEqual(
                set(row),
                {"wall_time_seconds", "rss_limit_bytes", "build_std_components", "project_root"},
            )
            self.assertEqual(
                (row["wall_time_seconds"], row["rss_limit_bytes"], row["build_std_components"], row["project_root"]),
                (wall, rss, components, root),
            )
            path = PurePosixPath(root)
            self.assertFalse(path.is_absolute())
            self.assertNotIn("..", path.parts)

    def test_sampling_review_and_stop_lines_are_frozen(self) -> None:
        value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(
            value["sampling"],
            {
                "full_review_max_population": 250,
                "minimum_per_nonempty_rule_depth_stratum": 20,
                "interprocedural_full_review_max": 150,
                "interprocedural_minimum_sample": 150,
                "include_all_recursive": True,
                "ordering": "sha256(seed||canonical_causal_key_json)",
            },
        )
        self.assertEqual(value["review"]["labels"], ["A", "B", "C", "D", "E", "U"])
        self.assertEqual(value["review"]["independent_reviewers"], 2)
        self.assertEqual(value["review"]["minimum_cohens_kappa"], 0.7)
        self.assertEqual(
            value["thresholds"],
            {
                "max_unique_findings_total": 500,
                "max_findings_per_kloc": 0.5,
                "max_egregious_point_estimate": 0.03,
                "max_egregious_wilson95_upper": 0.05,
                "per_rule_minimum_reviewed": 20,
                "per_rule_max_egregious_rate": 0.1,
                "minimum_actionable_conservative_lower": 0.6,
                "per_rule_minimum_actionable_conservative_lower": 0.4,
                "early_stop_prefix_size": 20,
                "early_stop_egregious_count": 3,
                "minimum_reviewed_for_precision": 50,
                "minimum_interprocedural_reviewed": 20,
                "zero_tolerance": [
                    "nondeterminism",
                    "schema_violation",
                    "duplicate_finding_id",
                    "fabricated_local_edge",
                    "hidden_truncation",
                ],
            },
        )


if __name__ == "__main__":
    unittest.main()

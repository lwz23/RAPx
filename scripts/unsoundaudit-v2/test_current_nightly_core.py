from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "tests/unsoundaudit-v2/latest_stdlib_target.json"
CORE_SUMMARY = (
    ROOT
    / "artifacts/unsoundaudit-v2/mac/stdlib-pilot/current-core-feasibility/core.summary.json"
)
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


class CurrentNightlyTargetTests(unittest.TestCase):
    def test_current_target_is_exact(self) -> None:
        value = json.loads(TARGET.read_text(encoding="utf-8"))
        self.assertEqual(
            set(value),
            {
                "schema_version",
                "nightly",
                "stable",
                "historical_baseline",
                "target_triple",
                "rust_src_tree_sha256",
            },
        )
        self.assertEqual(
            value["schema_version"], "unsoundaudit-v2-latest-stdlib-target-v1"
        )
        self.assertEqual(
            value["nightly"],
            {
                "cargo_commit": "eb98b54bc9f3c74519f43d066cb3fd02ebc88df0",
                "channel": "nightly-2026-08-14",
                "miri_version": "miri 0.1.0 (ba28ff76f3 2026-08-13)",
                "rustc_commit": "ba28ff76f353a722f31c4f3dd2ac4e437d36411b",
            },
        )
        self.assertEqual(
            value["stable"],
            {
                "cargo_commit": "c980f4866141969fab6254a680546a277789d6f0",
                "channel": "1.97.1",
                "rustc_commit": "8bab26f4f68e0e26f0bb7960be334d5b520ea452",
            },
        )
        self.assertEqual(
            value["historical_baseline"],
            {
                "channel": "nightly-2024-10-12",
                "rustc_commit": "1bc403daadbebb553ccc211a0a8eebb73989665f",
            },
        )
        self.assertEqual(value["target_triple"], "aarch64-apple-darwin")
        self.assertEqual(
            value["rust_src_tree_sha256"],
            {
                "alloc": "3d40f86c6a4fbfa21b58b89a0d8e8b9029e94afb315597109031dbe58b0cdbcf",
                "core": "c0a1e218b985dbcff02409dcef163588799dd79ccec21b2d8361f0c8f9370707",
                "std": "d758f147c839f3a2f5d672490864f83603d8d1b720ee284b4d493968a9d9945f",
            },
        )


class CurrentCoreRunnerTests(unittest.TestCase):
    def test_runner_is_one_control_plus_one_core_scan(self) -> None:
        import run_current_core_feasibility as runner

        self.assertEqual(runner.UNIT, "core")
        self.assertEqual(runner.STEPS, (("control", 0), ("scan", 1)))
        self.assertEqual(runner.WALL_LIMIT_SECONDS, 900)
        self.assertEqual(runner.PHYSICAL_LIMIT_BYTES, 8 * 1024**3)

    def test_runner_never_names_alloc_or_std_outputs(self) -> None:
        import run_current_core_feasibility as runner

        names = runner.output_names()
        self.assertEqual(
            names,
            ("core.normalized.json", "core.summary.json", "core.execution.json"),
        )
        self.assertTrue(all("alloc" not in name and "std" not in name for name in names))


class CurrentCoreReviewTests(unittest.TestCase):
    def test_bounded_review_closes_exactly_the_selected_queue(self) -> None:
        summary = json.loads(CORE_SUMMARY.read_text(encoding="utf-8"))
        queue = summary["review_queue"]
        reviews = summary["review_results"]

        self.assertEqual(summary["review_budget"], 20)
        self.assertEqual(len(queue), 20)
        self.assertEqual(len(reviews), 20)
        self.assertEqual(
            {row["family_id"] for row in reviews},
            {row["family_id"] for row in queue},
        )
        self.assertEqual(len({row["family_id"] for row in reviews}), 20)
        self.assertEqual(sum(row["finding_count"] for row in queue), 124)
        self.assertTrue(
            all(row["status"] == "reviewed_confirmed_safe_current" for row in queue)
        )
        self.assertTrue(
            all(row["disposition"] == "confirmed_safe_current" for row in reviews)
        )
        self.assertTrue(all(row["remaining_hypothesis"] == "none" for row in reviews))
        self.assertEqual(summary["reviewed_family_count"], 20)
        self.assertEqual(summary["reviewed_finding_count"], 124)
        self.assertEqual(summary["high_value_signal_count"], 0)
        self.assertEqual(
            summary["value_stop_gate"],
            "current_core_no_high_value_signal_in_bounded_review",
        )
        self.assertEqual(summary["next_units"], {"alloc": "not_started", "std": "not_started"})
        self.assertEqual(summary["poc_status"], "not_started_no_remaining_hypothesis")

    def test_review_proofs_are_pinned_and_closed(self) -> None:
        summary = json.loads(CORE_SUMMARY.read_text(encoding="utf-8"))
        target = json.loads(TARGET.read_text(encoding="utf-8"))
        proof_groups = summary["proof_groups"]

        self.assertEqual(summary["reviewed_rustc_commit"], target["nightly"]["rustc_commit"])
        self.assertEqual(
            set(proof_groups),
            {
                "array_into_iter_alive_range",
                "ascii_fixed_chunks",
                "binary_search_indices",
                "initialized_value_moves",
                "integer_format_buffer",
                "stable_sort_pivot",
                "str_range_normalization",
                "two_way_string_search",
            },
        )
        self.assertTrue(all(group["safe_public_boundary"] for group in proof_groups.values()))
        self.assertTrue(all(group["same_object_proof"] for group in proof_groups.values()))
        self.assertTrue(all(group["unwind_proof"] for group in proof_groups.values()))
        self.assertTrue(
            all(row["proof_group"] in proof_groups for row in summary["review_results"])
        )


if __name__ == "__main__":
    unittest.main()

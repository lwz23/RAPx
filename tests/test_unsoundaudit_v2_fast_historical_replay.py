import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "unsoundaudit_v2_fast_historical_replay.py"
SPEC = importlib.util.spec_from_file_location("fast_historical_replay", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load replay module from {SCRIPT}")
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


class FastHistoricalReplayTests(unittest.TestCase):
    def test_case_contract_is_exact_and_closed(self) -> None:
        self.assertEqual(10, len(REPLAY.CASES))
        self.assertEqual(list(range(1, 11)), [case["case"] for case in REPLAY.CASES])
        self.assertEqual(
            ["P1", "P1", "P2", "P3", "P3", "P4", "P4", "P5.1", "P6", "P6"],
            [case["expected_pattern"] for case in REPLAY.CASES],
        )
        self.assertEqual(
            "0f3aacb99ccfe0cbbf0bb83ecf736f67339ce8ae^",
            REPLAY.CASES[0]["checkout"],
        )
        self.assertEqual(
            "49aa163f1a9c793da6d591e10481fbfb6c044b51^",
            REPLAY.CASES[-1]["checkout"],
        )

    def test_revision_expression_rejects_any_nonfrozen_shape(self) -> None:
        self.assertEqual("a" * 40, REPLAY.validate_revision_expression("a" * 40))
        self.assertEqual("b" * 40 + "^", REPLAY.validate_revision_expression("b" * 40 + "^"))
        for value in ("main", "a" * 39, "a" * 40 + "~2", "A" * 40):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    REPLAY.validate_revision_expression(value)

    def test_markdown_table_preserves_zero_as_no_finding(self) -> None:
        row = {
            "case": 1,
            "expected_pattern": "P1",
            "repo_issue": "owner/repo #1",
            "detached_revision": "a" * 40,
            "control": "ok",
            "rap": "ok",
            "pattern_counts": {f"pattern{i}": 0 for i in range(1, 7)},
            "result_dir": "case-01",
        }
        table = REPLAY.markdown_table([row])
        self.assertIn("| 1 | P1 | owner/repo #1 |", table)
        self.assertIn("| 0 | 0 | 0 | 0 | 0 | 0 | case-01 | no_finding |", table)

    def test_frozen_evidence_records_inconclusive_gate_without_raw_logs(self) -> None:
        runtime = Path(tempfile.mkdtemp(prefix="rapx-history-runtime-"))
        output = runtime.parent / f"{runtime.name}-evidence"
        rows = []
        for case in REPLAY.CASES:
            row = REPLAY._base_row(case)
            row["failure"] = "lockfile_missing"
            rows.append(row)
        (runtime / "results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (runtime / "results.md").write_text(REPLAY.markdown_table(rows), encoding="utf-8")
        (runtime / "environment.json").write_text("{}\n", encoding="utf-8")
        logs = runtime / "cases" / "case-01" / "logs"
        logs.mkdir(parents=True)
        (logs / "control.log").write_text("failed\n", encoding="utf-8")

        manifest = REPLAY.freeze_evidence(runtime, output)
        self.assertEqual("inconclusive_input_or_execution_gate", manifest["interpretation"])
        self.assertEqual(10, manifest["failure_counts"]["lockfile_missing"])
        self.assertFalse(manifest["raw_logs_or_receipts_committed"])
        self.assertFalse((output / "cases").exists())


if __name__ == "__main__":
    unittest.main()

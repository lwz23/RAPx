import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "unsoundaudit_v2_fast_historical_replay_v2.py"
MANIFEST = ROOT / "tests" / "unsoundaudit-v2" / "historical_replay_v2_cases.json"
EVIDENCE = ROOT / "docs" / "unsoundaudit-v2" / "fast_historical_replay_v2_evidence"
SPEC = importlib.util.spec_from_file_location("fast_historical_replay_v2", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load replay module from {SCRIPT}")
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


class FastHistoricalReplayV2ContractTests(unittest.TestCase):
    def test_manifest_is_closed_and_binds_exact_vulnerable_targets(self) -> None:
        payload = REPLAY.load_case_manifest(MANIFEST)
        self.assertEqual("rapx-fast-historical-replay-cases-v2", payload["schema"])
        self.assertEqual(list(range(1, 11)), [case["case_id"] for case in payload["cases"]])
        required = {
            "case_id",
            "repository",
            "repo_issue",
            "fix_commit",
            "pre_fix_commit",
            "manifest_path",
            "package",
            "target_kind",
            "target_name",
            "target_triple",
            "default_features",
            "features",
            "vulnerable_path",
            "vulnerable_snippet_sha256",
            "fixed_snippet_sha256",
            "fix_diff_sha256",
            "expected_rule_id",
            "capability_tier",
            "validation_status",
        }
        self.assertTrue(all(set(case) == required for case in payload["cases"]))
        gitoxide = payload["cases"][1]
        self.assertEqual("gix-attributes/Cargo.toml", gitoxide["manifest_path"])
        self.assertEqual("gix-attributes", gitoxide["package"])
        self.assertEqual("P3.unchecked_utf8", gitoxide["expected_rule_id"])
        rand_779 = payload["cases"][6]
        self.assertEqual("9828cdf0201f816e4e4bff07834d849a10be80e9", rand_779["pre_fix_commit"])
        self.assertEqual("0c0f9311bbce0c41a6705db2b0957ff772111ce7", rand_779["fix_commit"])
        self.assertEqual("P1.raw_read", rand_779["expected_rule_id"])
        self.assertEqual("out_of_scope", rand_779["capability_tier"])
        rand_1158 = payload["cases"][7]
        self.assertEqual("P4.bounds", rand_1158["expected_rule_id"])
        self.assertFalse(payload["cases"][8]["default_features"])
        self.assertEqual("out_of_scope", payload["cases"][8]["capability_tier"])
        self.assertFalse(payload["cases"][9]["default_features"])
        self.assertEqual("out_of_scope", payload["cases"][9]["capability_tier"])

    def test_general_uninitialized_rule_is_additive_and_schema_bound(self) -> None:
        schema = json.loads(
            (ROOT / "tests" / "unsoundaudit-v2" / "rap-unit-v2.schema.json").read_text(
                encoding="utf-8"
            )
        )
        rule_ids = schema["$defs"]["finding"]["properties"]["rule_id"]["enum"]
        self.assertEqual(1, rule_ids.count("P3.assume_init_uninitialized"))
        validator_path = ROOT / "scripts" / "unsoundaudit-v2" / "validate_receipt.py"
        validator_spec = importlib.util.spec_from_file_location(
            "historical_replay_v2_validator", validator_path
        )
        if validator_spec is None or validator_spec.loader is None:
            self.fail("cannot load receipt validator")
        validator = importlib.util.module_from_spec(validator_spec)
        validator_spec.loader.exec_module(validator)
        self.assertEqual(
            (
                "pattern3",
                "internal_unsafe_origin",
                "assume_init",
                "invalid_value_exposure",
                ("initialized",),
            ),
            validator.RULE_MATRIX["P3.assume_init_uninitialized"],
        )
        self.assertEqual(
            (
                "pattern6",
                "ffi_output",
                "invalid_value_exposure",
                "invalid_value_exposure",
                ("valid_for_type",),
            ),
            validator.RULE_MATRIX["P6.ffi_out_invalid_value"],
        )

    def test_non_success_rows_have_nullable_counts_and_markdown_dashes(self) -> None:
        row = REPLAY.new_unit_result(
            case_id=2,
            package="gix-attributes",
            target_kind="lib",
            target_name="gix_attributes",
            status="partial_timeout",
        )
        self.assertIsNone(row["finding_count"])
        self.assertIsNone(row["pattern_counts"])
        self.assertEqual("pre_fix", row["revision"])
        table = REPLAY.markdown_table([row])
        self.assertIn("| partial_timeout | — | — | — | — | — | — | — |", table)
        with self.assertRaisesRegex(ValueError, "successful result"):
            REPLAY.new_unit_result(
                case_id=2,
                package="gix-attributes",
                target_kind="lib",
                target_name="gix_attributes",
                status="success",
            )
        with self.assertRaisesRegex(ValueError, "revision"):
            REPLAY.new_unit_result(
                case_id=2,
                package="gix-attributes",
                target_kind="lib",
                target_name="gix_attributes",
                status="invalid_input",
                revision="unknown",
            )

    def test_manifest_rejects_wrong_hashes_and_escaping_paths(self) -> None:
        valid = {
            "schema": "rapx-fast-historical-replay-cases-v2",
            "cases": [
                {
                    "case_id": 1,
                    "repository": "owner/repo",
                    "repo_issue": "owner/repo #1",
                    "fix_commit": "a" * 40,
                    "pre_fix_commit": "b" * 40,
                    "manifest_path": "Cargo.toml",
                    "package": "demo",
                    "target_kind": "lib",
                    "target_name": "demo",
                    "target_triple": None,
                    "default_features": True,
                    "vulnerable_path": "src/lib.rs",
                    "vulnerable_snippet_sha256": "c" * 64,
                    "fixed_snippet_sha256": "d" * 64,
                    "fix_diff_sha256": "e" * 64,
                    "expected_rule_id": "P1.raw_read",
                    "features": [],
                    "capability_tier": "existing_rule",
                    "validation_status": "validated",
                }
            ],
        }
        root = Path(tempfile.mkdtemp(prefix="rapx-history-v2-manifest-"))
        path = root / "cases.json"
        path.write_text(json.dumps(valid), encoding="utf-8")
        self.assertEqual(1, len(REPLAY.load_case_manifest(path)["cases"]))
        for field, value in (
            ("manifest_path", "../Cargo.toml"),
            ("vulnerable_path", "/src/lib.rs"),
            ("fix_diff_sha256", "not-a-hash"),
        ):
            with self.subTest(field=field):
                changed = json.loads(json.dumps(valid))
                changed["cases"][0][field] = value
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValueError):
                    REPLAY.load_case_manifest(path)

    def test_validated_out_of_scope_is_not_counted_as_a_successful_zero(self) -> None:
        case = {
            "expected_rule_id": "P1.raw_read",
            "capability_tier": "out_of_scope",
            "default_features": True,
            "features": [],
            "target_triple": None,
            "vulnerable_path": "src/lib.rs",
        }
        row = REPLAY.new_unit_result(
            case_id=1,
            package="demo",
            target_kind="lib",
            target_name="demo",
            status="success",
            finding_count=0,
            pattern_counts={f"pattern{number}": 0 for number in range(1, 7)},
        )
        row["findings"] = []
        REPLAY.attach_recall_outcome(row, case, "pre_fix")
        self.assertEqual("validated_out_of_scope", row["recall_outcome"])

    def test_v1_evidence_remains_byte_for_byte_frozen(self) -> None:
        expected = {
            "environment.json": "762c491a189a50cc8c7fe353d843709d09058b30a5f1a0ea6e31ef3deceed315",
            "evidence_manifest.json": "1dde5ab0acacb756858dc1b50f2ffe05d100a4d72450b89d78776ef9e17ad190",
            "results.jsonl": "2913fa8bf97b654b218a5a1d3a1075988f10371a615238e66dec8ace8f637d27",
            "results.md": "269121a7d74a094e29576d0e1d036c20cbf67cfe71329e40a7ae87b2d4858391",
        }
        evidence = ROOT / "docs" / "unsoundaudit-v2" / "fast_historical_replay_v1_evidence"
        actual = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in evidence.iterdir()
            if path.is_file()
        }
        self.assertEqual(expected, actual)

    def test_v2_evidence_separates_execution_recall_and_capability(self) -> None:
        summary = json.loads((EVIDENCE / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual("rapx-fast-historical-replay-evidence-v2", summary["schema"])
        self.assertEqual(
            {
                "input_invalid": 3,
                "not_run": 0,
                "partial": 0,
                "successful_zero": 0,
                "detected": 3,
                "out_of_scope": 4,
            },
            summary["classification_counts"],
        )
        self.assertEqual(
            {"detected": 3, "denominator": 3, "rate": 1.0},
            summary["in_scope_recall"],
        )
        cases = summary["cases"]
        self.assertEqual(list(range(1, 11)), [case["case_id"] for case in cases])
        self.assertEqual(
            {2: 2, 4: 6, 8: 2},
            {
                case["case_id"]: case["pre_fix"]["expected_rule_finding_count"]
                for case in cases
                if case["pre_fix"].get("outcome") == "detected"
            },
        )
        self.assertEqual(
            {3, 7, 9, 10},
            {
                case["case_id"]
                for case in cases
                if case["pre_fix"].get("outcome") == "validated_out_of_scope"
            },
        )
        self.assertTrue(
            all(
                case["fixed"].get("expected_rule_finding_count") == 0
                for case in cases
                if case["fixed"]["status"] == "success"
            )
        )
        self.assertTrue(
            all(
                "finding_count" not in revision
                for case in cases
                for revision in (case["pre_fix"], case["fixed"])
                if revision["status"] != "success"
            )
        )
        self.assertEqual(
            hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            summary["inputs"]["case_manifest_sha256"],
        )
        self.assertEqual(
            hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
            summary["inputs"]["runner_sha256"],
        )
        schema = ROOT / "tests" / "unsoundaudit-v2" / "rap-unit-v2.schema.json"
        self.assertEqual(
            hashlib.sha256(schema.read_bytes()).hexdigest(),
            summary["inputs"]["new_schema_sha256"],
        )
        report = (EVIDENCE / "report.md").read_text(encoding="utf-8")
        self.assertIn("one measured pre-fix/fixed pass", report)
        self.assertNotIn("ecosystem recall is 100%", report)

    def test_exact_historical_target_beats_lexical_workspace_order(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="rapx-history-v2-target-"))
        (root / "aaa").mkdir()
        (root / "gix-attributes").mkdir()
        for manifest in (root / "aaa/Cargo.toml", root / "gix-attributes/Cargo.toml"):
            manifest.write_text("[package]\nname='fixture'\nversion='0.0.0'\n", encoding="utf-8")
        metadata = {
            "workspace_root": str(root),
            "workspace_members": ["aaa 0.0.0", "gix-attributes 0.0.0"],
            "workspace_default_members": ["aaa 0.0.0"],
            "resolve": {"root": None},
            "packages": [
                {
                    "id": "aaa 0.0.0",
                    "name": "aaa",
                    "manifest_path": str(root / "aaa/Cargo.toml"),
                    "targets": [{"name": "aaa", "kind": ["lib"]}],
                },
                {
                    "id": "gix-attributes 0.0.0",
                    "name": "gix-attributes",
                    "manifest_path": str(root / "gix-attributes/Cargo.toml"),
                    "targets": [{"name": "gix_attributes", "kind": ["lib"]}],
                },
            ],
        }
        case = REPLAY.load_case_manifest(MANIFEST)["cases"][1]
        selected = REPLAY.select_historical_target(metadata, root, case)
        self.assertEqual("gix-attributes", selected["package"])
        self.assertEqual("gix-attributes 0.0.0", selected["package_id"])
        self.assertEqual("gix-attributes/Cargo.toml", selected["manifest_path"])
        self.assertEqual("gix_attributes", selected["name"])
        self.assertEqual(
            "gix-attributes 0.0.0",
            REPLAY.scan_argv(Path("/bin/cargo-rapx"), selected)[-2],
        )
        self.assertEqual(
            ["--features", "sqlite"],
            REPLAY.scan_argv(Path("/bin/cargo-rapx"), selected, ["sqlite"])[-2:],
        )
        self.assertEqual(
            ["--no-default-features", "--features", "socket"],
            REPLAY.scan_argv(
                Path("/bin/cargo-rapx"),
                selected,
                ["socket"],
                default_features=False,
            )[-3:],
        )
        self.assertEqual(
            ["--target", "x86_64-unknown-linux-gnu"],
            REPLAY.scan_argv(
                Path("/bin/cargo-rapx"),
                selected,
                [],
                "x86_64-unknown-linux-gnu",
            )[-2:],
        )
        environment = REPLAY._command_environment(
            cargo=Path("/pinned/cargo"),
            rustc=Path("/pinned/rustc"),
            cargo_home=Path("/tmp/cargo-home"),
            target_dir=Path("/tmp/target"),
        )
        self.assertTrue(environment["RUSTFLAGS"].endswith("--cap-lints allow"))

        defaults = REPLAY.select_default_targets(metadata, root)
        self.assertEqual(["aaa"], [target["package"] for target in defaults])
        all_targets = REPLAY.select_default_targets(metadata, root, workspace_all=True)
        self.assertEqual(["aaa", "gix-attributes"], [target["package"] for target in all_targets])

    def test_lockless_library_generates_lock_only_in_disposable_copy(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="rapx-history-v2-lockless-"))
        source = root / "source"
        source.mkdir()
        (source / "Cargo.toml").write_text(
            "[package]\nname='lockless'\nversion='0.0.0'\n", encoding="utf-8"
        )
        (source / "src").mkdir()
        (source / "src/lib.rs").write_text("pub fn value() -> usize { 1 }\n", encoding="utf-8")
        before = REPLAY.tree_digest(source)

        def fake_run(argv: list[str], cwd: Path, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
            self.assertEqual(
                ["/pinned/cargo", "-Z", "msrv-policy", "generate-lockfile"], argv[:4]
            )
            self.assertEqual(
                "fallback", environment["CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS"]
            )
            self.assertNotEqual(source, cwd)
            (cwd / "Cargo.lock").write_text(
                "# This file is automatically @generated by Cargo.\nversion = 3\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(argv, 0, "", "")

        prepared = REPLAY.prepare_locked_workspace(
            repository=source,
            workspace_root=source,
            manifest_path="Cargo.toml",
            copy_root=root / "copy",
            cargo=Path("/pinned/cargo"),
            environment={},
            run_command=fake_run,
        )
        self.assertTrue(prepared["generated"])
        self.assertTrue((prepared["repository"] / "Cargo.lock").is_file())
        self.assertFalse((source / "Cargo.lock").exists())
        self.assertEqual(before, REPLAY.tree_digest(source))
        self.assertRegex(prepared["lock_sha256"], r"^[0-9a-f]{64}$")

    def test_receipt_aggregation_preserves_partial_timeout_and_rejects_duplicates(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="rapx-history-v2-receipts-"))
        receipt_dir = root / "receipts"
        receipt_dir.mkdir()
        target = {
            "package": "gix-attributes",
            "kind": "lib",
            "name": "gix_attributes",
            "manifest_path": "gix-attributes/Cargo.toml",
        }
        receipt = {
            "schema_version": "rap-unit-v2",
            "success": True,
            "package_name": "gix-attributes",
            "target_kind": "lib",
            "target_name": None,
            "finding_count": 2,
            "pattern_counts": {
                "pattern1": 0,
                "pattern2": 0,
                "pattern3": 2,
                "pattern4": 0,
                "pattern5": 0,
                "pattern6": 0,
            },
            "findings": [
                {
                    "finding_id": "a" * 64,
                    "rule_id": "P3.unchecked_utf8",
                    "first_contract_failure": {
                        "span": {"path": "gix-attributes/src/state.rs"}
                    },
                },
                {
                    "finding_id": "b" * 64,
                    "rule_id": "P3.unchecked_utf8",
                    "first_contract_failure": {
                        "span": {"path": "gix-attributes/src/unrelated.rs"}
                    },
                },
            ],
        }
        (receipt_dir / "gix.json").write_text(json.dumps(receipt), encoding="utf-8")
        aggregated = REPLAY.collect_repository_receipts(
            receipt_dir, repository_packages={"gix-attributes"}, selected_target=target
        )
        self.assertEqual(1, len(aggregated["repository_receipts"]))
        self.assertEqual("gix-attributes", aggregated["selected_receipt"]["package_name"])
        result = REPLAY.result_from_execution(
            case_id=2,
            target=target,
            execution_status="timeout",
            receipts=aggregated,
        )
        self.assertEqual("partial_timeout", result["status"])
        self.assertIsNone(result["finding_count"])
        self.assertEqual(1, result["partial_receipt_count"])

        fixed = REPLAY.result_from_execution(
            case_id=2,
            target=target,
            execution_status="success",
            receipts=aggregated,
            revision="fixed",
        )
        fixed["findings"] = aggregated["selected_receipt"]["findings"]
        case = REPLAY.load_case_manifest(MANIFEST)["cases"][1]
        REPLAY.attach_recall_outcome(fixed, case, "fixed")
        self.assertEqual("fixed_rule_remaining", fixed["recall_outcome"])
        self.assertEqual(1, fixed["expected_rule_finding_count"])

        (receipt_dir / "duplicate.json").write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly one selected receipt"):
            REPLAY.collect_repository_receipts(
                receipt_dir, repository_packages={"gix-attributes"}, selected_target=target
            )

        missing = root / "missing-selected"
        missing.mkdir()
        dependency = dict(receipt)
        dependency["package_name"] = "gix-utils"
        (missing / "dependency.json").write_text(json.dumps(dependency), encoding="utf-8")
        partial = REPLAY.collect_repository_receipts(
            missing,
            repository_packages={"gix-attributes", "gix-utils"},
            selected_target=target,
            require_selected=False,
        )
        self.assertIsNone(partial["selected_receipt"])
        self.assertEqual(1, len(partial["repository_receipts"]))


if __name__ == "__main__":
    unittest.main()

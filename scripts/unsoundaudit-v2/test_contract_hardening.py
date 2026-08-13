#!/usr/bin/env python3
"""Pure standard-library tamper tests for the frozen v2 contract."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import normalize_receipts
import run_fixture_suite as runner
from validate_receipt import ContractError, read_json, validate_receipt


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "tests/unsoundaudit-v2/rap-unit-v2.schema.json"
ZERO_COUNTS = {f"pattern{number}": 0 for number in range(1, 7)}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def span(line: int) -> dict:
    return {
        "path": "src/lib.rs",
        "start": {"line": line, "column": 1},
        "end": {"line": line, "column": 2},
    }


def valid_receipt() -> dict:
    source_span = span(1)
    sink_span = span(2)
    causal = {
        "crate": "sample",
        "public_root": "sample::root",
        "source_origin": "public_parameter:arg:1@src/lib.rs:1:1-1:2",
        "first_contract_failure": "raw_read@src/lib.rs:2:1-2:2",
        "sink_or_exposure": "raw_read@src/lib.rs:2:1-2:2",
        "canonical_obligation": "valid_for_read:arg:1",
    }
    canonical = json.dumps(causal, sort_keys=True, separators=(",", ":"))
    finding = {
        "finding_id": hashlib.sha256(canonical.encode()).hexdigest(),
        "causal_key": causal,
        "primary_pattern": "pattern1",
        "rule_id": "P1.raw_read",
        "public_safe_root": {"def_path": "sample::root", "span": source_span},
        "source": {"kind": "public_parameter", "origin_key": "arg:1", "span": source_span},
        "first_contract_failure": {"kind": "raw_read", "span": sink_span},
        "sink": {"kind": "raw_read", "span": sink_span},
        "obligation": {"predicates": ["valid_for_read"], "subject": "arg:1"},
        "validation_status": {"status": "missing", "facts": []},
        "propagation": {
            "depth": "intra_procedural",
            "local_call_count": 0,
            "boundaries": [],
            "witness": [
                {"kind": "entry", "function": "sample::root", "span": source_span},
                {"kind": "sink", "function": "sample::root", "span": sink_span},
            ],
        },
        "secondary_source_kinds": [],
        "heuristic_candidate_generator": True,
        "limitations": ["fixture-closed heuristic candidate generator"],
    }
    counts = {f"pattern{number}": 0 for number in range(1, 7)}
    counts["pattern1"] = 1
    return {
        "schema_version": "rap-unit-v2",
        "project_root": "/machine/path",
        "source": "rap",
        "success": True,
        "unit_id": "unit",
        "rustc_invocation_id": "invocation",
        "package_id": "sample 0.1.0",
        "package_name": "sample",
        "package_version": "0.1.0",
        "crate_name": "sample",
        "crate_types": ["lib"],
        "target_kind": "lib",
        "target_triple": "aarch64-apple-darwin",
        "manifest_path": "/machine/path/Cargo.toml",
        "source_path": "/machine/path/src/lib.rs",
        "extra_filename": "",
        "rustc_path": "/toolchain/rustc",
        "cargo_supplied_rustc_path": "/toolchain/rustc",
        "cargo_supplied_rustc_commit": "a" * 40,
        "rap_compiler_commit": "a" * 40,
        "rustc_commit": "a" * 40,
        "pattern_counts": counts,
        "finding_count": 1,
        "findings": [finding],
    }


class ReceiptTamperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = read_json(SCHEMA_PATH)

    def assert_rejected(self, receipt: dict) -> None:
        with self.assertRaises(ContractError):
            validate_receipt(receipt, self.schema)

    def test_valid_minimal_receipt_passes(self) -> None:
        validate_receipt(valid_receipt(), self.schema)

    def test_rule_matrix_tamper_is_rejected(self) -> None:
        receipt = valid_receipt()
        receipt["findings"][0]["first_contract_failure"]["kind"] = "get_unchecked"
        self.assert_rejected(receipt)

    def test_noncanonical_span_is_rejected(self) -> None:
        schema_pattern = self.schema["$defs"]["span"]["properties"]["path"]["pattern"]
        for invalid in ("src//lib.rs", "./src/lib.rs", "C:/src/lib.rs", "C:src/lib.rs"):
            with self.subTest(path=invalid):
                self.assertIsNone(re.fullmatch(schema_pattern, invalid))
                receipt = valid_receipt()
                receipt["findings"][0]["public_safe_root"]["span"]["path"] = invalid
                self.assert_rejected(receipt)

    def test_absolute_stable_identity_is_rejected(self) -> None:
        receipt = valid_receipt()
        receipt["findings"][0]["source"]["origin_key"] = "arg:/Users/example/private"
        self.assert_rejected(receipt)

    def test_witness_call_count_tamper_is_rejected(self) -> None:
        receipt = valid_receipt()
        propagation = receipt["findings"][0]["propagation"]
        propagation["depth"] = "inter_procedural"
        propagation["local_call_count"] = 2
        propagation["boundaries"] = ["sample::helper"]
        propagation["witness"].insert(
            1,
            {"kind": "local_call", "function": "sample::helper", "span": span(1)},
        )
        propagation["witness"][-1]["function"] = "sample::helper"
        self.assert_rejected(receipt)

    def test_recursive_duplicate_cycle_tamper_is_rejected(self) -> None:
        receipt = valid_receipt()
        propagation = receipt["findings"][0]["propagation"]
        propagation["depth"] = "recursive"
        propagation["local_call_count"] = None
        cycle = {"kind": "scc_cycle", "function": "scc:[sample::root]", "span": span(1)}
        propagation["witness"].insert(1, copy.deepcopy(cycle))
        propagation["witness"].insert(1, copy.deepcopy(cycle))
        self.assert_rejected(receipt)

    def test_unsorted_boundaries_are_rejected(self) -> None:
        receipt = valid_receipt()
        propagation = receipt["findings"][0]["propagation"]
        propagation["depth"] = "inter_procedural"
        propagation["local_call_count"] = 1
        propagation["boundaries"] = ["sample::z", "sample::helper"]
        propagation["witness"].insert(
            1,
            {"kind": "local_call", "function": "sample::helper", "span": span(1)},
        )
        propagation["witness"][-1]["function"] = "sample::helper"
        self.assert_rejected(receipt)


class FrozenInputTamperTests(unittest.TestCase):
    def create_legacy_tree(self, root: Path) -> tuple[Path, dict, list[Path]]:
        repo = root / "repo"
        legacy_root = repo / "docs/frozen"
        legacy_root.mkdir(parents=True)
        files = []
        rows = []
        for index in range(31):
            path = legacy_root / f"file-{index:02}.txt"
            path.write_text(f"frozen-{index}\n", encoding="utf-8")
            files.append(path)
            rows.append(
                {
                    "path": path.relative_to(legacy_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": runner.sha256(path),
                }
            )
        provenance_path = legacy_root / "legacy.json"
        write_json(provenance_path, {"files": rows})
        fixture_manifest = repo / "tests/unsoundaudit-v2/fixture_manifest.json"
        fixture_manifest.parent.mkdir(parents=True)
        descriptor = {
            "path": "../../docs/frozen/legacy.json",
            "sha256": runner.sha256(provenance_path),
            "verified_file_count": 31,
        }
        return fixture_manifest, descriptor, files

    def test_legacy_file_content_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_manifest, descriptor, files = self.create_legacy_tree(root)
            runner.verify_legacy_provenance(root / "repo", fixture_manifest, descriptor)
            files[0].write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(ContractError):
                runner.verify_legacy_provenance(root / "repo", fixture_manifest, descriptor)

    def test_legacy_extra_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_manifest, descriptor, _ = self.create_legacy_tree(root)
            (root / "repo/docs/frozen/extra.txt").write_text("extra\n", encoding="utf-8")
            with self.assertRaises(ContractError):
                runner.verify_legacy_provenance(root / "repo", fixture_manifest, descriptor)

    def test_legacy_path_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_manifest, descriptor, _ = self.create_legacy_tree(root)
            provenance_path = root / "repo/docs/frozen/legacy.json"
            provenance = read_json(provenance_path)
            provenance["files"][0]["path"] = "../outside.txt"
            (root / "repo/docs/outside.txt").write_text("outside\n", encoding="utf-8")
            write_json(provenance_path, provenance)
            descriptor["sha256"] = runner.sha256(provenance_path)
            with self.assertRaises(ContractError):
                runner.verify_legacy_provenance(root / "repo", fixture_manifest, descriptor)

    def test_cargo_target_rustflags_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            cargo_home = root / "cargo-home"
            repo.mkdir()
            cargo_home.mkdir()
            (cargo_home / "config.toml").write_text(
                '[target.aarch64-apple-darwin]\nrustflags = ["-C", "target-cpu=native"]\n',
                encoding="utf-8",
            )
            with self.assertRaises(ContractError):
                runner.audit_cargo_configs(repo, cargo_home)

    def test_environment_commit_tamper_is_rejected_without_invoking_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            cargo_home = Path(directory) / "cargo-home"
            bin_dir = Path(directory) / "rap-bin"
            toolchain = Path(directory) / "toolchain"
            z3 = Path(directory) / "z3"
            cargo_home.mkdir(parents=True)
            bin_dir.mkdir()
            toolchain.mkdir()
            (z3 / "include").mkdir(parents=True)
            (z3 / "lib").mkdir()
            (z3 / "include/z3.h").write_text("/* z3 */\n", encoding="utf-8")
            for path in (bin_dir / "cargo-rapx", bin_dir / "rapx", toolchain / "cargo", toolchain / "rustc"):
                path.write_text("tool\n", encoding="utf-8")
            expected_cargo = "1" * 40
            expected_rustc = "2" * 40
            write_json(
                repo / runner.BASELINE_MANIFEST_RELATIVE,
                {"toolchain": {"cargo_commit_hash": expected_cargo, "rustc_commit_hash": expected_rustc}},
            )
            write_json(
                repo / runner.ENVIRONMENT_RECEIPT_RELATIVE,
                {"status": "pass", "environment_controls": {"rustflags_role": "z3_native_library_search_only"}},
            )
            environment = {
                "CARGO_HOME": str(cargo_home),
                "LIBCLANG_PATH": "/approved/libclang",
                "PKG_CONFIG_PATH": "/approved/pkgconfig",
                "DYLD_LIBRARY_PATH": "/approved/dylib",
                "Z3_SYS_Z3_HEADER": str(z3 / "include/z3.h"),
                "RUSTFLAGS": f"-Lnative={z3 / 'lib'}",
                "RUST_SYSROOT": "/approved/sysroot",
                "UNSOUND_SCANNER_EXACT_CARGO_PATH": str(toolchain / "cargo"),
                "UNSOUND_SCANNER_EXACT_CARGO_COMMIT": expected_cargo,
                "RUSTC": str(toolchain / "rustc"),
                "UNSOUND_SCANNER_EXPECTED_RUSTC_PATH": str(toolchain / "rustc"),
                "UNSOUND_SCANNER_EXPECTED_RUSTC_COMMIT": expected_rustc,
            }
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
                runner, "command_commit", return_value="f" * 40
            ) as commit_probe:
                with self.assertRaises(ContractError):
                    runner.verify_environment(repo, (toolchain / "cargo").resolve(), bin_dir.resolve())
                commit_probe.assert_called_once_with((toolchain / "cargo").resolve(), "exact Cargo")
            environment["UNSOUND_SCANNER_EXACT_CARGO_COMMIT"] = "f" * 40
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
                runner, "command_commit"
            ) as commit_probe:
                with self.assertRaises(ContractError):
                    runner.verify_environment(repo, (toolchain / "cargo").resolve(), bin_dir.resolve())
                commit_probe.assert_not_called()
            environment["UNSOUND_SCANNER_EXACT_CARGO_COMMIT"] = expected_cargo
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
                runner, "command_commit", side_effect=[expected_cargo, expected_rustc]
            ):
                runner.verify_environment(repo, (toolchain / "cargo").resolve(), bin_dir.resolve())
            environment["CARGO_ENCODED_RUSTFLAGS"] = ""
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
                runner, "command_commit"
            ) as commit_probe:
                with self.assertRaises(ContractError):
                    runner.verify_environment(repo, (toolchain / "cargo").resolve(), bin_dir.resolve())
                commit_probe.assert_not_called()
            del environment["CARGO_ENCODED_RUSTFLAGS"]
            environment["RUSTFLAGS"] += " -Ctarget-cpu=native"
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
                runner, "command_commit", side_effect=[expected_cargo, expected_rustc]
            ):
                with self.assertRaises(ContractError):
                    runner.verify_environment(repo, (toolchain / "cargo").resolve(), bin_dir.resolve())

    def test_receipt_toolchain_tamper_is_rejected(self) -> None:
        receipt = valid_receipt()
        runner.verify_receipt_toolchain(receipt, "a" * 40)
        receipt["rustc_commit"] = "b" * 40
        with self.assertRaises(ContractError):
            runner.verify_receipt_toolchain(receipt, "a" * 40)

    def test_standalone_normalizer_uses_full_contract_verifier(self) -> None:
        self.assertIs(normalize_receipts.verify_contract, runner.verify_contract)
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            (repo / ".git").mkdir(parents=True)
            shutil.copytree(
                REPO_ROOT / "tests/unsoundaudit-v2",
                repo / "tests/unsoundaudit-v2",
            )
            shutil.copytree(
                REPO_ROOT / "docs/unsoundaudit-v2/frozen-v1-fixtures",
                repo / "docs/unsoundaudit-v2/frozen-v1-fixtures",
            )
            manifest_path = repo / "tests/unsoundaudit-v2/fixture_manifest.json"
            fixture_root = repo / "tests/unsoundaudit-v2/fixtures"
            with mock.patch.object(runner, "git_tracked", return_value=True):
                normalize_receipts.verify_contract(
                    repo,
                    fixture_root,
                    manifest_path,
                    read_json(manifest_path),
                )
                source = fixture_root / "legacy_p1_param_direct_positive/src/lib.rs"
                source.write_text(source.read_text(encoding="utf-8") + "// tampered\n", encoding="utf-8")
                with self.assertRaises(ContractError):
                    normalize_receipts.verify_contract(
                        repo,
                        fixture_root,
                        manifest_path,
                        read_json(manifest_path),
                    )


class GenericCaseContractTests(unittest.TestCase):
    def create_case_contract(self, root: Path) -> tuple[Path, Path, dict]:
        repo = root / "repo"
        case_root = repo / "tests/unsoundaudit-v2/challenges"
        manifest_path = case_root.parent / "challenge_manifest.json"
        aggregate = dict(ZERO_COUNTS)
        aggregate["pattern1"] = 1
        rows = []
        for case_id, role, pattern, rule_id in (
            ("a", "positive", "pattern1", "P1.raw_read"),
            ("b", "negative", None, None),
        ):
            fixture = case_root / case_id
            (fixture / "src").mkdir(parents=True)
            (fixture / "Cargo.toml").write_text(
                f'[package]\nname = "challenge_{case_id}"\nversion = "0.1.0"\nedition = "2021"\n\n'
                '[lib]\npath = "src/lib.rs"\n',
                encoding="utf-8",
            )
            (fixture / "Cargo.lock").write_text(
                f'version = 3\n\n[[package]]\nname = "challenge_{case_id}"\nversion = "0.1.0"\n',
                encoding="utf-8",
            )
            (fixture / "src/lib.rs").write_text(
                f"pub fn {case_id}() {{}}\n",
                encoding="utf-8",
            )
            counts = dict(ZERO_COUNTS)
            if pattern is not None:
                counts[pattern] = 1
            oracle = {
                "case_id": case_id,
                "role": role,
                "package_name": f"challenge_{case_id}",
                "expected_primary_pattern": pattern,
                "expected_rule_id": rule_id,
                "expected_pattern_counts": counts,
            }
            write_json(fixture / "fixture.json", oracle)
            paths = {
                "manifest": fixture / "Cargo.toml",
                "source": fixture / "src/lib.rs",
                "lock": fixture / "Cargo.lock",
                "oracle": fixture / "fixture.json",
            }
            rows.append(
                {
                    "case_id": case_id,
                    "role": role,
                    "package_name": f"challenge_{case_id}",
                    "expected_primary_pattern": pattern,
                    "expected_rule_id": rule_id,
                    "expected_pattern_counts": counts,
                    "files": {
                        file_role: {
                            "path": path.relative_to(case_root.parent).as_posix(),
                            "sha256": runner.sha256(path),
                        }
                        for file_role, path in paths.items()
                    },
                }
            )
        manifest = {
            "schema_version": "unsoundaudit-v2-challenge-manifest-v1",
            "fixture_count": 2,
            "role_counts": {"positive": 1, "negative": 1},
            "aggregate_expected_pattern_counts": aggregate,
            "cases": rows,
        }
        write_json(manifest_path, manifest)
        return repo, case_root, manifest

    def verify_generic(self, repo: Path, case_root: Path, manifest: dict, **overrides: object) -> list[str]:
        options = {
            "expected_schema": "unsoundaudit-v2-challenge-manifest-v1",
            "expected_count": 2,
            "expected_counts": {"pattern1": 1, **{f"pattern{number}": 0 for number in range(2, 7)}},
            "require_legacy_provenance": False,
        }
        options.update(overrides)
        return runner.verify_case_contract(  # type: ignore[arg-type]
            repo_root=repo,
            case_root=case_root,
            manifest_path=case_root.parent / "challenge_manifest.json",
            manifest=manifest,
            **options,
        )

    def add_extra_file(
        self,
        case_root: Path,
        manifest: dict,
        relative: str = "challenges/a/src/included.rs",
    ) -> Path:
        path = case_root.parent / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pub fn included() {}\n", encoding="utf-8")
        manifest["cases"][0]["extra_files"] = [
            {"path": relative, "sha256": runner.sha256(path)}
        ]
        return path

    def test_generic_contract_is_closed_and_tamper_resistant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, case_root, manifest = self.create_case_contract(Path(directory))
            with mock.patch.object(runner, "git_tracked", return_value=True) as tracked:
                self.assertEqual(self.verify_generic(repo, case_root, manifest), ["a", "b"])
                self.assertEqual(tracked.call_count, 8)
                self.assertEqual(
                    {path.relative_to(case_root).as_posix() for _, path in (call.args for call in tracked.call_args_list)},
                    {
                        "a/Cargo.toml",
                        "a/Cargo.lock",
                        "a/src/lib.rs",
                        "a/fixture.json",
                        "b/Cargo.toml",
                        "b/Cargo.lock",
                        "b/src/lib.rs",
                        "b/fixture.json",
                    },
                )

            (case_root / "extra").mkdir()
            with mock.patch.object(runner, "git_tracked", return_value=True), self.assertRaises(ContractError):
                self.verify_generic(repo, case_root, manifest)
            (case_root / "extra").rmdir()

            source = case_root / "a/src/lib.rs"
            source.write_text("pub fn tampered() {}\n", encoding="utf-8")
            with mock.patch.object(runner, "git_tracked", return_value=True), self.assertRaises(ContractError):
                self.verify_generic(repo, case_root, manifest)
            source.write_text("pub fn a() {}\n", encoding="utf-8")

            untracked_lock = (case_root / "b/Cargo.lock").resolve()
            with mock.patch.object(
                runner,
                "git_tracked",
                side_effect=lambda _repo, path: path.resolve() != untracked_lock,
            ), self.assertRaises(ContractError):
                self.verify_generic(repo, case_root, manifest)

            role_drift = copy.deepcopy(manifest)
            role_drift["role_counts"] = {"positive": 2}
            with mock.patch.object(runner, "git_tracked", return_value=True), self.assertRaises(ContractError):
                self.verify_generic(repo, case_root, role_drift)

            aggregate_drift = copy.deepcopy(manifest)
            aggregate_drift["aggregate_expected_pattern_counts"]["pattern2"] = 1
            with mock.patch.object(runner, "git_tracked", return_value=True), self.assertRaises(ContractError):
                self.verify_generic(repo, case_root, aggregate_drift)

            row_drift = copy.deepcopy(manifest)
            row_drift["cases"][0]["package_name"] = "different"
            with mock.patch.object(runner, "git_tracked", return_value=True), self.assertRaises(ContractError):
                self.verify_generic(repo, case_root, row_drift)

            cargo_toml = case_root / "a/Cargo.toml"
            original_cargo = cargo_toml.read_text(encoding="utf-8")
            cargo_toml.write_text(original_cargo + '\n[dependencies]\nserde = "1"\n', encoding="utf-8")
            dependency_manifest = copy.deepcopy(manifest)
            dependency_manifest["cases"][0]["files"]["manifest"]["sha256"] = runner.sha256(cargo_toml)
            with mock.patch.object(runner, "git_tracked", return_value=True), self.assertRaises(ContractError):
                self.verify_generic(repo, case_root, dependency_manifest)

    def test_dependency_subtables_are_rejected(self) -> None:
        for dependency_section in (
            '[dependencies.serde]\nversion = "1"\n',
            '[target.\'cfg(unix)\'.dependencies]\nserde = "1"\n',
        ):
            with self.subTest(section=dependency_section), tempfile.TemporaryDirectory() as directory:
                repo, case_root, manifest = self.create_case_contract(Path(directory))
                cargo_toml = case_root / "a/Cargo.toml"
                cargo_toml.write_text(
                    cargo_toml.read_text(encoding="utf-8") + "\n" + dependency_section,
                    encoding="utf-8",
                )
                manifest["cases"][0]["files"]["manifest"]["sha256"] = runner.sha256(cargo_toml)
                with mock.patch.object(runner, "git_tracked", return_value=True), self.assertRaises(ContractError):
                    self.verify_generic(repo, case_root, manifest)

    def test_malformed_cargo_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, case_root, manifest = self.create_case_contract(Path(directory))
            cargo_toml = case_root / "a/Cargo.toml"
            cargo_toml.write_text(
                cargo_toml.read_text(encoding="utf-8") + "\ninvalid = [\n",
                encoding="utf-8",
            )
            manifest["cases"][0]["files"]["manifest"]["sha256"] = runner.sha256(cargo_toml)
            with mock.patch.object(runner, "git_tracked", return_value=True), self.assertRaises(ContractError):
                self.verify_generic(repo, case_root, manifest)

    def test_numeric_contract_fields_require_non_bool_integers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, case_root, manifest = self.create_case_contract(Path(directory))
            tampered_fields = (
                ("fixture_count_float", "fixture_count", 2.0),
                ("role_count_bool", "role_counts", {"positive": True, "negative": 1}),
                (
                    "aggregate_bool",
                    "aggregate_expected_pattern_counts",
                    {"pattern1": True, **{f"pattern{number}": 0 for number in range(2, 7)}},
                ),
            )
            for label, field, value in tampered_fields:
                with self.subTest(field=label):
                    tampered = copy.deepcopy(manifest)
                    tampered[field] = value
                    with mock.patch.object(runner, "git_tracked", return_value=True), self.assertRaises(
                        ContractError
                    ):
                        self.verify_generic(repo, case_root, tampered)

            invalid_expected_counts = {
                "pattern1": True,
                **{f"pattern{number}": 0 for number in range(2, 7)},
            }
            with self.subTest(field="expected_counts_bool"), mock.patch.object(
                runner, "git_tracked", return_value=True
            ), self.assertRaises(ContractError):
                self.verify_generic(repo, case_root, manifest, expected_counts=invalid_expected_counts)

        with tempfile.TemporaryDirectory() as directory:
            repo, case_root, manifest = self.create_case_contract(Path(directory))
            shutil.rmtree(case_root / "b")
            manifest["fixture_count"] = 1
            manifest["role_counts"] = {"positive": 1}
            manifest["cases"] = manifest["cases"][:1]
            with self.subTest(field="expected_count_bool"), mock.patch.object(
                runner, "git_tracked", return_value=True
            ), self.assertRaises(ContractError):
                self.verify_generic(repo, case_root, manifest, expected_count=True)

    def test_oracle_boolean_count_cannot_equal_manifest_integer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, case_root, manifest = self.create_case_contract(Path(directory))
            oracle_path = case_root / "a/fixture.json"
            oracle = read_json(oracle_path)
            oracle["expected_pattern_counts"]["pattern1"] = True
            write_json(oracle_path, oracle)
            manifest["cases"][0]["files"]["oracle"]["sha256"] = runner.sha256(oracle_path)
            with mock.patch.object(runner, "git_tracked", return_value=True), self.assertRaises(ContractError):
                self.verify_generic(repo, case_root, manifest)

    def test_json_values_equal_is_type_sensitive_and_recursive(self) -> None:
        self.assertFalse(runner.json_values_equal(1, True))
        self.assertFalse(runner.json_values_equal(1, 1.0))
        self.assertTrue(runner.json_values_equal({"a": [1, {"b": False}]}, {"a": [1, {"b": False}]}))
        self.assertFalse(runner.json_values_equal({"a": [1, {"b": 0}]}, {"a": [1, {"b": False}]}))
        self.assertFalse(runner.json_values_equal({"a": [1]}, {"a": [1, 2]}))
        self.assertFalse(runner.json_values_equal({"a": 1}, {"b": 1}))

    def test_case_ids_are_single_normalized_relative_components(self) -> None:
        self.assertTrue(runner.valid_case_id("case-a_1"))
        for invalid in ("", "/", "//", "a/b", "a\\b", "a\0b", ".", "..", "/absolute", "C:/case", "C:case"):
            with self.subTest(case_id=repr(invalid)):
                self.assertFalse(runner.valid_case_id(invalid))
        self.assertFalse(runner.valid_case_id(1))

    def test_case_files_cannot_escape_through_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, case_root, manifest = self.create_case_contract(Path(directory))
            outside = Path(directory) / "outside.rs"
            outside.write_text("pub fn outside() {}\n", encoding="utf-8")
            source = case_root / "a/src/lib.rs"
            source.unlink()
            source.symlink_to(outside)
            manifest["cases"][0]["files"]["source"]["sha256"] = runner.sha256(source)
            with mock.patch.object(runner, "git_tracked", return_value=True), self.assertRaises(ContractError):
                self.verify_generic(repo, case_root, manifest)

    def test_extra_files_close_the_regular_file_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, case_root, manifest = self.create_case_contract(Path(directory))
            extra = self.add_extra_file(case_root, manifest)
            with mock.patch.object(runner, "git_tracked", return_value=True) as tracked:
                self.assertEqual(self.verify_generic(repo, case_root, manifest), ["a", "b"])
                self.assertIn(extra, [call.args[1] for call in tracked.call_args_list])

            manifest["cases"][0].pop("extra_files")
            with mock.patch.object(runner, "git_tracked", return_value=True), self.assertRaises(ContractError):
                self.verify_generic(repo, case_root, manifest)

    def test_extra_file_rows_reject_invalid_paths_and_metadata(self) -> None:
        invalid_paths = (
            "challenges/a/Cargo.toml",
            "challenges/a/Cargo.lock",
            "challenges/a/src/lib.rs",
            "challenges/a/fixture.json",
            "challenges/a/build.rs",
            "challenges/a/../outside.rs",
            "/absolute.rs",
            "C:/outside.rs",
            "challenges\\a\\outside.rs",
            "challenges/a/bad\0name.rs",
        )
        for invalid in invalid_paths:
            with self.subTest(path=repr(invalid)), tempfile.TemporaryDirectory() as directory:
                repo, case_root, manifest = self.create_case_contract(Path(directory))
                manifest["cases"][0]["extra_files"] = [{"path": invalid, "sha256": "0" * 64}]
                with mock.patch.object(runner, "git_tracked", return_value=True), self.assertRaises(ContractError):
                    self.verify_generic(repo, case_root, manifest)

        with tempfile.TemporaryDirectory() as directory:
            repo, case_root, manifest = self.create_case_contract(Path(directory))
            self.add_extra_file(case_root, manifest)
            manifest["cases"][0]["extra_files"] *= 2
            with mock.patch.object(runner, "git_tracked", return_value=True), self.assertRaises(ContractError):
                self.verify_generic(repo, case_root, manifest)

        for mutation in ("wrong_hash", "untracked", "symlink", "hidden_file"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                repo, case_root, manifest = self.create_case_contract(root)
                extra = self.add_extra_file(case_root, manifest)
                tracked = mock.patch.object(runner, "git_tracked", return_value=True)
                if mutation == "wrong_hash":
                    manifest["cases"][0]["extra_files"][0]["sha256"] = "0" * 64
                elif mutation == "untracked":
                    tracked = mock.patch.object(
                        runner,
                        "git_tracked",
                        side_effect=lambda _repo, path: path != extra,
                    )
                elif mutation == "symlink":
                    target = root / "outside.rs"
                    target.write_text("pub fn outside() {}\n", encoding="utf-8")
                    extra.unlink()
                    extra.symlink_to(target)
                    manifest["cases"][0]["extra_files"][0]["sha256"] = runner.sha256(extra)
                else:
                    (case_root / "a/.unfrozen.rs").write_text("pub fn hidden() {}\n", encoding="utf-8")
                with tracked, self.assertRaises(ContractError):
                    self.verify_generic(repo, case_root, manifest)

    def test_frozen_contract_still_rejects_a_twenty_eighth_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            shutil.copytree(
                REPO_ROOT / "tests/unsoundaudit-v2",
                repo / "tests/unsoundaudit-v2",
            )
            shutil.copytree(
                REPO_ROOT / "docs/unsoundaudit-v2/frozen-v1-fixtures",
                repo / "docs/unsoundaudit-v2/frozen-v1-fixtures",
            )
            manifest_path = repo / "tests/unsoundaudit-v2/fixture_manifest.json"
            fixture_root = repo / "tests/unsoundaudit-v2/fixtures"
            manifest = read_json(manifest_path)
            extra = copy.deepcopy(manifest["cases"][0])
            extra["case_id"] = "twenty_eighth_case"
            manifest["cases"].append(extra)
            manifest["fixture_count"] = 28
            with mock.patch.object(runner, "git_tracked", return_value=True), self.assertRaises(ContractError):
                runner.verify_contract(repo, fixture_root, manifest_path, manifest)


if __name__ == "__main__":
    unittest.main()

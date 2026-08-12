#!/usr/bin/env python3
"""Pure standard-library tamper tests for the frozen v2 contract."""

from __future__ import annotations

import copy
import hashlib
import json
import os
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
        for invalid in ("src//lib.rs", "./src/lib.rs", "C:/src/lib.rs"):
            with self.subTest(path=invalid):
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


if __name__ == "__main__":
    unittest.main()

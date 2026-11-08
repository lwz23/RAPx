#!/usr/bin/env python3
"""No-build contracts for the compiler-aligned v3 stdlib resource gate."""

from __future__ import annotations

import copy
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_contract import minimal_receipt


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = Path(__file__).with_name("run_stdlib_gate.py")
PROTOCOL_2024 = REPO_ROOT / "tests/unsoundaudit-v3/stdlib_gate_2024.json"
PROTOCOL_CURRENT = REPO_ROOT / "tests/unsoundaudit-v3/stdlib_gate_current.json"


def load_runner():
    spec = importlib.util.spec_from_file_location("unsoundaudit_v3_stdlib_gate", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load v3 stdlib gate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def span(line: int = 1) -> dict[str, object]:
    return {
        "path": "library/core/src/lib.rs",
        "start": {"line": line, "column": 1},
        "end": {"line": line, "column": 2},
    }


def family_digest(fields: list[str]) -> str:
    payload = bytearray()
    for field in fields:
        encoded = field.encode()
        payload.extend(str(len(encoded)).encode())
        payload.extend(b":")
        payload.extend(encoded)
    return hashlib.sha256(payload).hexdigest()


def mvp_candidate(
    finding_id: str = "a" * 64,
    *,
    pattern: str = "P1",
    root: str = "core::root",
) -> dict[str, object]:
    location = span()
    return {
        "finding_id": finding_id,
        "primary_pattern": pattern,
        "subject_role": "argument:0",
        "public_safe_root": {"owner": root, "span": location},
        "unsafe_site": {"site_id": "b" * 64, "binding": "bound_exact", "span": location},
        "first_contract_failure": {"kind": "precondition:unsafe_call", "span": location},
        "contract": {
            "model_id": "ptr_read_v1",
            "obligation": {"predicates": ["non_null"]},
        },
        "source": {"kind": "public_parameter", "identity": "core::root::argument:0"},
        "exposure": None,
        "propagation": {
            "witness": [{"kind": "entry", "owner": root, "span": location}],
        },
    }


def mvp_family(candidate: dict[str, object]) -> dict[str, object]:
    location = "library/core/src/lib.rs:1:1-1:2"
    key = {
        "primary_pattern": candidate["primary_pattern"],
        "unsafe_site_id": "b" * 64,
        "subject_role": "argument:0",
        "first_contract_failure": {
            "kind": "precondition:unsafe_call",
            "span": location,
        },
        "contract": {"model_id": "ptr_read_v1", "predicates": ["non_null"]},
        "source": {"kind": "public_parameter", "identity": "core::root::argument:0"},
        "exposure": None,
    }
    identifier = family_digest(
        [
            str(candidate["primary_pattern"]),
            "b" * 64,
            "argument:0",
            "precondition:unsafe_call",
            location,
            "ptr_read_v1",
            "non_null",
            "public_parameter",
            "core::root::argument:0",
            "",
            "",
            "",
        ]
    )
    return {
        "schema_version": "rap-unsoundaudit-family-v3-mvp",
        "family_id": identifier,
        "key": key,
        "raw_count": 1,
        "public_root_fanout": 1,
        "representatives": {"shortest": candidate, "longest": candidate},
    }


def write_mvp_package(root: Path, *, complete: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    candidate = mvp_candidate()
    family = mvp_family(candidate)
    suppressed = mvp_candidate("c" * 64, pattern="P4")
    ledger = {
        "schema_version": "rap-unsoundaudit-suppression-v3-mvp",
        "candidate_id": "c" * 64,
        "candidate": suppressed,
        "reason": "pointer_view_has_no_capability_gap",
        "evidence": {"status": "impossible_by_compiler_facts"},
        "replacement_candidate_id": "a" * 64,
    }
    rows = {
        "candidates.jsonl": [candidate],
        "families.jsonl": [family],
        "suppression-ledger.jsonl": [ledger],
        "unestablished.jsonl": [],
    }
    for name, values in rows.items():
        (root / name).write_text("".join(json.dumps(value) + "\n" for value in values))
    coverage = {
        "schema_version": "rap-unsoundaudit-coverage-v3-mvp",
        "complete": complete,
        "semantics": "exact" if complete else "lower_bound",
        "candidate_accounting": {
            "active_count": 1,
            "pure_suppressed_count": 0,
            "reclassified_count": 1,
            "unestablished_count": 0,
            "suppressed_count": 1,
            "considered_count": 2,
        },
        "unsafe_site_census": {
            "discovered": 1 if complete else 2,
            "bound_exact": 1,
            "optimized_away_exact": 0,
            "unbound": 0,
            "ambiguous": 0,
            "elided": 0 if complete else 1,
            "by_kind": [{"kind": "unsafe_call", "count": 1 if complete else 2}],
        },
        "coverage_failures": [] if complete else [{"reason": "elided_unsafe_site"}],
    }
    (root / "coverage.json").write_text(json.dumps(coverage))


def write_large_mvp_package(
    root: Path, *, active_count: int, suppressed_count: int, unestablished_count: int = 0
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    root_fanout = min(active_count, 128)

    def active_candidate(index: int) -> dict[str, object]:
        return mvp_candidate(
            f"{index:064x}", root=f"core::root_{index % root_fanout}"
        )

    representative = active_candidate(1)
    with (root / "candidates.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(1, active_count + 1):
            handle.write(json.dumps(active_candidate(index), separators=(",", ":")) + "\n")

    family = mvp_family(representative)
    family["raw_count"] = active_count
    family["public_root_fanout"] = root_fanout
    family["representatives"] = {
        "shortest": representative,
        "longest": representative,
    }
    with (root / "families.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(family, separators=(",", ":")) + "\n")

    replacement_id = str(representative["finding_id"])
    pure_suppressed = 0
    reclassified = 0
    with (root / "suppression-ledger.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for index in range(suppressed_count):
            candidate_id = f"{(1 << 255) + index:064x}"
            candidate = mvp_candidate(
                candidate_id,
                pattern="P4",
                root=str(representative["public_safe_root"]["owner"]),
            )
            replacement = replacement_id if index % 2 else None
            pure_suppressed += replacement is None
            reclassified += replacement is not None
            row = {
                "schema_version": "rap-unsoundaudit-suppression-v3-mvp",
                "candidate_id": candidate_id,
                "candidate": candidate,
                "reason": "pointer_view_has_no_capability_gap",
                "evidence": {"status": "impossible_by_compiler_facts"},
                "replacement_candidate_id": replacement,
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    with (root / "unestablished.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(unestablished_count):
            decision_id = f"{(1 << 254) + index:064x}"
            row = {
                "schema_version": "rap-unsoundaudit-unestablished-v1",
                "decision_id": decision_id,
                "attempted_pattern": "P3",
                "public_root": "core::root",
                "unsafe_site_id": "b" * 64,
                "operand_role": "argument:0",
                "source": {
                    "identity": f"core::root:external:{index}",
                    "kind": "external_behavior_output",
                },
                "controller": "unknown",
                "capabilities": "unknown",
                "requirement": None,
                "primary_reason": "missing_exact_context",
                "secondary_reasons": ["opaque_transform"],
                "exposure": None,
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    coverage = {
        "schema_version": "rap-unsoundaudit-coverage-v3-mvp",
        "complete": True,
        "semantics": "exact",
        "candidate_accounting": {
            "active_count": active_count,
            "pure_suppressed_count": pure_suppressed,
            "reclassified_count": reclassified,
            "unestablished_count": unestablished_count,
            "suppressed_count": suppressed_count,
            "considered_count": active_count + suppressed_count + unestablished_count,
        },
        "unsafe_site_census": {
            "discovered": 1,
            "bound_exact": 1,
            "optimized_away_exact": 0,
            "unbound": 0,
            "ambiguous": 0,
            "elided": 0,
            "by_kind": [{"kind": "unsafe_call", "count": 1}],
        },
        "coverage_failures": [],
    }
    (root / "coverage.json").write_text(json.dumps(coverage))


class V3StdlibGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gate = load_runner()

    def test_protocols_pin_toolchains_units_and_hard_limits(self) -> None:
        legacy = self.gate.load_protocol(PROTOCOL_2024)
        current = self.gate.load_protocol(PROTOCOL_CURRENT)
        self.assertEqual(legacy["rustc_commit"], "1bc403daadbebb553ccc211a0a8eebb73989665f")
        self.assertEqual(legacy["cargo_commit"], "15fbd2f607d4defc87053b8b76bf5038f2483cf4")
        self.assertEqual(legacy["unit_order"], ["core", "alloc", "std"])
        self.assertEqual(
            {
                unit: (row["wall_time_seconds"], row["physical_footprint_limit_bytes"])
                for unit, row in legacy["units"].items()
            },
            {
                "core": (900, 8 * 1024**3),
                "alloc": (1800, 10 * 1024**3),
                "std": (3600, 12 * 1024**3),
            },
        )
        self.assertEqual(current["rustc_commit"], "ba28ff76f353a722f31c4f3dd2ac4e437d36411b")
        self.assertEqual(current["cargo_commit"], "eb98b54bc9f3c74519f43d066cb3fd02ebc88df0")
        self.assertEqual(current["unit_order"], ["core"])
        self.assertEqual(set(current["units"]), {"core"})
        self.assertEqual(current["units"]["core"]["wall_time_seconds"], 900)
        self.assertEqual(current["units"]["core"]["physical_footprint_limit_bytes"], 8 * 1024**3)
        for protocol in (legacy, current):
            self.assertEqual(protocol["analyzer_profile"], "rapx-perf")
            self.assertEqual(protocol["runs_per_unit"], 2)
            self.assertEqual(protocol["cargo_jobs"], 1)
            self.assertFalse(protocol["trace_enabled"])

    def test_target_cargo_is_dev_check_and_profile_contract_is_closed(self) -> None:
        protocol = self.gate.load_protocol(PROTOCOL_2024)
        for unit in protocol["unit_order"]:
            arguments = self.gate.cargo_arguments(protocol, unit)
            self.assertIn("--locked", arguments)
            self.assertEqual(arguments[-2:], ["--jobs", "1"])
            self.assertNotIn("--release", arguments)
            self.assertNotIn("--profile", arguments)
            self.assertNotIn("-r", arguments)
        self.assertEqual(
            self.gate.cargo_profile_contract(REPO_ROOT / "rapx/Cargo.toml"),
            {
                "profile": "rapx-perf",
                "opt_level": 3,
                "debug": 1,
                "debug_assertions": True,
                "overflow_checks": True,
                "incremental": False,
                "target_cargo_profile": "dev",
            },
        )

    def test_binary_sidecar_binds_profile_source_toolchain_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cargo_rapx = root / "cargo-rapx"
            rapx = root / "rapx"
            cargo_rapx.write_bytes(b"cargo wrapper")
            rapx.write_bytes(b"analyzer")
            value = {
                "schema_version": self.gate.BINARY_PROFILE_SCHEMA,
                **self.gate.PROFILE_CONTRACT,
                "rap_source_commit": "a" * 40,
                "rustc_commit": "b" * 40,
                "cargo_rapx_sha256": self.gate.sha256(cargo_rapx),
                "rapx_sha256": self.gate.sha256(rapx),
            }
            (root / self.gate.PROFILE_SIDECAR).write_bytes(self.gate.canonical_json(value))
            self.assertEqual(
                self.gate.load_binary_profile(root, cargo_rapx, rapx, "a" * 40, "b" * 40),
                value,
            )
            for mutation in (
                {"profile": "release"},
                {"rustc_commit": "c" * 40},
                {"rapx_sha256": "0" * 64},
                {"extra": True},
            ):
                changed = value | mutation
                (root / self.gate.PROFILE_SIDECAR).write_bytes(self.gate.canonical_json(changed))
                with self.assertRaises(self.gate.GateError):
                    self.gate.load_binary_profile(root, cargo_rapx, rapx, "a" * 40, "b" * 40)

    def test_development_profile_skips_only_the_source_commit_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cargo_rapx = root / "cargo-rapx"
            rapx = root / "rapx"
            cargo_rapx.write_bytes(b"cargo wrapper")
            rapx.write_bytes(b"analyzer")
            value = {
                "schema_version": self.gate.BINARY_PROFILE_SCHEMA,
                **self.gate.PROFILE_CONTRACT,
                "rap_source_commit": "c" * 40,
                "rustc_commit": "b" * 40,
                "cargo_rapx_sha256": self.gate.sha256(cargo_rapx),
                "rapx_sha256": self.gate.sha256(rapx),
            }
            (root / self.gate.PROFILE_SIDECAR).write_bytes(self.gate.canonical_json(value))
            with self.assertRaises(self.gate.GateError):
                self.gate.load_binary_profile(root, cargo_rapx, rapx, "a" * 40, "b" * 40)
            self.assertEqual(
                self.gate.load_development_binary_profile(
                    root, cargo_rapx, rapx, "b" * 40
                ),
                value,
            )
            for mutation in (
                {"profile": "release"},
                {"rustc_commit": "d" * 40},
                {"cargo_rapx_sha256": "0" * 64},
            ):
                (root / self.gate.PROFILE_SIDECAR).write_bytes(
                    self.gate.canonical_json(value | mutation)
                )
                with self.assertRaises(self.gate.GateError):
                    self.gate.load_development_binary_profile(
                        root, cargo_rapx, rapx, "b" * 40
                    )

    def test_development_environment_clears_inherited_build_and_output_controls(self) -> None:
        environment = self.gate.development_environment(
            {
                "PATH": "/usr/bin",
                "RUSTFLAGS": "-C target-cpu=native",
                "CARGO_ENCODED_RUSTFLAGS": "bad",
                "CARGO_HOME": "/untrusted-cargo-home",
                "RUSTC_WRAPPER": "wrapper",
                "DYLD_INSERT_LIBRARIES": "inject.dylib",
                "DYLD_FALLBACK_LIBRARY_PATH": "/also/untrusted",
                "DYLD_LIBRARY_PATH": "/untrusted",
                "CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS": "bad target flags",
                "RAP_RECURSIVE": "deep",
                "RAP_CLEAN": "false",
                "RAP_LOG": "trace",
                "UNSOUND_SCANNER_RAP_JSON_DIR": "/wrong/receipt",
                "UNSOUND_SCANNER_PROJECT_ROOT": "/wrong/project",
                "RAP_UNSOUNDAUDIT_OUTPUT": "/wrong/output",
            },
            cargo=Path("/toolchain/bin/cargo"),
            rustc=Path("/toolchain/bin/rustc"),
            rust_sysroot=Path("/toolchain"),
            cargo_commit="a" * 40,
            rustc_commit="b" * 40,
            cargo_home=Path("/isolated-cargo-home"),
            target=Path("/fresh-target"),
            unit_root=Path("/rust-src/library/core"),
            source_root=Path("/rust-src"),
            mvp_output=Path("/evidence/mvp"),
            platform="darwin",
        )
        self.assertEqual(environment["PATH"], "/usr/bin")
        self.assertEqual(environment["CARGO_HOME"], "/isolated-cargo-home")
        self.assertEqual(environment["CARGO_TARGET_DIR"], "/fresh-target")
        self.assertEqual(environment["UNSOUND_SCANNER_PROJECT_ROOT"], "/rust-src/library/core")
        self.assertEqual(environment["UNSOUND_SCANNER_SOURCE_ROOT"], "/rust-src")
        self.assertEqual(environment["RAP_UNSOUNDAUDIT_OUTPUT"], "/evidence/mvp")
        self.assertEqual(environment["DYLD_LIBRARY_PATH"], "/toolchain/lib")
        for removed in (
            "RUSTFLAGS",
            "CARGO_ENCODED_RUSTFLAGS",
            "RUSTC_WRAPPER",
            "DYLD_INSERT_LIBRARIES",
            "DYLD_FALLBACK_LIBRARY_PATH",
            "CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS",
            "RAP_RECURSIVE",
            "RAP_CLEAN",
            "RAP_LOG",
            "UNSOUND_SCANNER_RAP_JSON_DIR",
        ):
            self.assertNotIn(removed, environment)

    def test_footprint_monitor_aggregates_process_group_and_fails_closed(self) -> None:
        output = " 10 99 1024\n 11 99 2048\n 12 100 4096\n"
        usage = self.gate.aggregate_process_group_usage(
            output, 99, {10: 4 * 1024**2, 11: 7 * 1024**2}.__getitem__
        )
        self.assertEqual(usage.physical_footprint_bytes, 11 * 1024**2)
        self.assertEqual(usage.rss_bytes, 3072 * 1024)
        self.assertEqual(usage.sampled_processes, 2)
        with self.assertRaisesRegex(self.gate.GateError, "parse"):
            self.gate.aggregate_process_group_usage("bad row\n", 99, lambda _pid: 1)
        with self.assertRaisesRegex(self.gate.GateError, "sampling"):
            self.gate.aggregate_process_group_usage(
                " 10 99 1\n", 99, lambda _pid: (_ for _ in ()).throw(OSError("bad"))
            )

    def test_limited_runner_classifies_timeout_and_physical_limit(self) -> None:
        environment = os.environ.copy()
        usage = self.gate.ProcessGroupUsage(4096, 8192, 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            success = self.gate.run_limited(
                ["/bin/sh", "-c", "exit 0"], root, environment, root / "ok.log", 2, 1 << 30,
                usage_reader=lambda _pid: usage,
            )
            timeout = self.gate.run_limited(
                ["/bin/sh", "-c", "sleep 5"], root, environment, root / "timeout.log", 0.03, 1 << 30,
                usage_reader=lambda _pid: usage,
            )
            memory = self.gate.run_limited(
                ["/bin/sh", "-c", "sleep 5"], root, environment, root / "memory.log", 2, 1,
                usage_reader=lambda _pid: usage,
            )
            observed_only = self.gate.run_limited(
                ["/bin/sh", "-c", "sleep 0.15"],
                root,
                environment,
                root / "observed-only.log",
                2,
                None,
                usage_reader=lambda _pid: usage,
            )
        self.assertEqual(success.classification, "success")
        self.assertEqual(timeout.classification, "timeout")
        self.assertEqual(memory.classification, "memory_limit")
        self.assertEqual(memory.max_physical_footprint_bytes, 4096)
        self.assertEqual(memory.max_rss_bytes, 8192)
        self.assertEqual(observed_only.classification, "success")
        self.assertEqual(observed_only.max_physical_footprint_bytes, 4096)

    def test_complete_receipt_normalizes_but_partial_is_rejected(self) -> None:
        receipt = minimal_receipt()
        normalized = self.gate.normalized_unit("core", receipt, "d" * 64)
        self.assertEqual(normalized["analysis_status"], "complete")
        self.assertEqual(normalized["observed_counts"]["semantics"], "exact")
        partial = copy.deepcopy(receipt)
        partial["analysis_status"] = "partial"
        partial["observed_counts"]["semantics"] = "lower_bound"
        with self.assertRaisesRegex(self.gate.GateError, "partial"):
            self.gate.normalized_unit("core", partial, "d" * 64)

    def test_receipt_selection_requires_exactly_one_complete_target_unit(self) -> None:
        schema = self.gate.read_json(REPO_ROOT / "tests/unsoundaudit-v3/rap-unit-v3.schema.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = minimal_receipt()
            receipt["crate_name"] = "core"
            receipt["package_name"] = "core"
            receipt["rustc_commit"] = "a" * 40
            receipt["rap_compiler_commit"] = "a" * 40
            receipt["cargo_supplied_rustc_commit"] = "a" * 40
            receipt["toolchain_adapter"]["rustc_commit"] = "a" * 40
            (root / "core.json").write_text(json.dumps(receipt))
            dependency = copy.deepcopy(receipt)
            dependency["crate_name"] = "env_filter"
            dependency["package_name"] = "env_filter"
            (root / "dependency.json").write_text(json.dumps(dependency))
            selected = self.gate.select_unit_receipt(root, schema, "core", "a" * 40)
            self.assertEqual(selected["crate_name"], "core")
            (root / "duplicate.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(self.gate.GateError, "exactly one"):
                self.gate.select_unit_receipt(root, schema, "core", "a" * 40)

    def test_development_mvp_accounting_and_replacement_correlation_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_mvp_package(root)
            summary = self.gate.validate_development_mvp(root)
            self.assertTrue(summary.coverage_exact)
            self.assertEqual(summary.active_count, 1)
            self.assertEqual(summary.family_count, 1)
            self.assertEqual(summary.reclassified_count, 1)
            self.assertTrue(summary.family_target_met)

            ledger_path = root / "suppression-ledger.jsonl"
            ledger = json.loads(ledger_path.read_text())
            ledger["candidate"]["public_safe_root"]["owner"] = "core::other_root"
            ledger_path.write_text(json.dumps(ledger) + "\n")
            with self.assertRaisesRegex(self.gate.GateError, "public root"):
                self.gate.validate_development_mvp(root)

    def test_development_mvp_unestablished_schema_precedence_and_overlap_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_mvp_package(root)
            row = {
                "schema_version": "rap-unsoundaudit-unestablished-v1",
                "decision_id": "d" * 64,
                "attempted_pattern": "P3",
                "public_root": "core::root",
                "unsafe_site_id": "b" * 64,
                "operand_role": "argument:0",
                "source": {
                    "identity": "core::root:external:0",
                    "kind": "external_behavior_output",
                },
                "controller": "unknown",
                "capabilities": "unknown",
                "requirement": None,
                "primary_reason": "missing_exact_context",
                "secondary_reasons": ["opaque_transform"],
                "exposure": None,
            }
            unestablished = root / "unestablished.jsonl"
            unestablished.write_text(json.dumps(row) + "\n")
            coverage_path = root / "coverage.json"
            coverage = json.loads(coverage_path.read_text())
            coverage["candidate_accounting"]["unestablished_count"] = 1
            coverage["candidate_accounting"]["considered_count"] = 3
            coverage_path.write_text(json.dumps(coverage))
            summary = self.gate.validate_development_mvp(root)
            self.assertEqual(summary.unestablished_count, 1)

            for pattern in sorted(self.gate.VALID_PATTERNS):
                row["attempted_pattern"] = pattern
                unestablished.write_text(json.dumps(row) + "\n")
                summary = self.gate.validate_development_mvp(root)
                self.assertEqual(summary.unestablished_count, 1)

            row["attempted_pattern"] = "P3"
            row["exposure"] = {"identity": "forbidden"}
            unestablished.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(self.gate.GateError, "non-null exposure"):
                self.gate.validate_development_mvp(root)

            row["exposure"] = None
            row["primary_reason"] = "opaque_transform"
            row["secondary_reasons"] = ["missing_exact_context"]
            unestablished.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(self.gate.GateError, "precedence drifted"):
                self.gate.validate_development_mvp(root)

            row["primary_reason"] = "missing_exact_context"
            row["secondary_reasons"] = []
            row["decision_id"] = "a" * 64
            unestablished.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(self.gate.GateError, "overlaps another population"):
                self.gate.validate_development_mvp(root)

    def test_development_mvp_rejects_family_and_writer_accounting_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_mvp_package(root)
            family_path = root / "families.jsonl"
            family = json.loads(family_path.read_text())
            family["family_id"] = "f" * 64
            family_path.write_text(json.dumps(family) + "\n")
            with self.assertRaisesRegex(self.gate.GateError, "no active members"):
                self.gate.validate_development_mvp(root)

            write_mvp_package(root)
            coverage_path = root / "coverage.json"
            coverage = json.loads(coverage_path.read_text())
            coverage["candidate_accounting"]["active_count"] = 2
            coverage["candidate_accounting"]["considered_count"] = 3
            coverage_path.write_text(json.dumps(coverage))
            with self.assertRaisesRegex(self.gate.GateError, "accounting drifted"):
                self.gate.validate_development_mvp(root)

            write_mvp_package(root)
            coverage = json.loads(coverage_path.read_text())
            coverage["unsafe_site_census"].update(bound_exact=0, unbound=1)
            coverage_path.write_text(json.dumps(coverage))
            with self.assertRaisesRegex(self.gate.GateError, "nonexact unsafe-site binding"):
                self.gate.validate_development_mvp(root)

            write_mvp_package(root)
            ledger_path = root / "suppression-ledger.jsonl"
            ledger = json.loads(ledger_path.read_text())
            ledger["reason"] = "unregistered_reason"
            ledger_path.write_text(json.dumps(ledger) + "\n")
            with self.assertRaisesRegex(self.gate.GateError, "invalid structural reason"):
                self.gate.validate_development_mvp(root)

            write_mvp_package(root)
            candidates_path = root / "candidates.jsonl"
            candidate = json.loads(candidates_path.read_text())
            candidate["unsafe_site"]["binding"] = "unbound"
            candidates_path.write_text(json.dumps(candidate) + "\n")
            with self.assertRaisesRegex(self.gate.GateError, "not bound_exact"):
                self.gate.validate_development_mvp(root)

    def test_streamed_family_validation_preserves_exact_accumulators(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_mvp_package(root)
            shortest = mvp_candidate("a" * 64, root="core::root_a")
            longest_high_id = mvp_candidate("c" * 64, root="core::root_c")
            longest_low_id = mvp_candidate("b" * 64, root="core::root_b")
            for candidate in (longest_high_id, longest_low_id):
                witness = candidate["propagation"]["witness"]
                witness.extend(copy.deepcopy(witness[0]) for _ in range(2))
            candidates = [longest_high_id, shortest, longest_low_id]
            (root / "candidates.jsonl").write_text(
                "".join(json.dumps(candidate) + "\n" for candidate in candidates)
            )
            family = mvp_family(shortest)
            family.update(
                raw_count=3,
                public_root_fanout=3,
                representatives={"shortest": shortest, "longest": longest_low_id},
            )
            family_path = root / "families.jsonl"
            family_path.write_text(json.dumps(family) + "\n")
            (root / "suppression-ledger.jsonl").write_text("")
            coverage_path = root / "coverage.json"
            coverage = json.loads(coverage_path.read_text())
            coverage["candidate_accounting"] = {
                "active_count": 3,
                "pure_suppressed_count": 0,
                "reclassified_count": 0,
                "unestablished_count": 0,
                "suppressed_count": 0,
                "considered_count": 3,
            }
            coverage_path.write_text(json.dumps(coverage))
            self.assertEqual(self.gate.validate_development_mvp(root).family_count, 1)

            mutations = {
                "root_fanout": lambda row: row.update(public_root_fanout=2),
                "representatives": lambda row: row["representatives"].update(
                    longest=longest_high_id
                ),
                "structural key": lambda row: row["key"]["source"].update(
                    identity="core::different"
                ),
            }
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    changed = copy.deepcopy(family)
                    mutate(changed)
                    family_path.write_text(json.dumps(changed) + "\n")
                    with self.assertRaisesRegex(self.gate.GateError, label):
                        self.gate.validate_development_mvp(root)
            family_path.write_text("")
            with self.assertRaisesRegex(self.gate.GateError, "exact partition"):
                self.gate.validate_development_mvp(root)

    def test_development_mvp_jsonl_rows_fail_closed(self) -> None:
        for name in (
            "candidates.jsonl",
            "families.jsonl",
            "suppression-ledger.jsonl",
            "unestablished.jsonl",
        ):
            with self.subTest(name=name, failure="utf8"), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_mvp_package(root)
                (root / name).write_bytes(b"\xff\n")
                with self.assertRaisesRegex(self.gate.GateError, "cannot read"):
                    self.gate.validate_development_mvp(root)

            with self.subTest(name=name, failure="non-object"), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_mvp_package(root)
                (root / name).write_text("[]\n")
                with self.assertRaisesRegex(self.gate.GateError, "row is not an object"):
                    self.gate.validate_development_mvp(root)

            with self.subTest(name=name, failure="json"), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_mvp_package(root)
                (root / name).write_text("{\n")
                with self.assertRaisesRegex(self.gate.GateError, "invalid .* JSON"):
                    self.gate.validate_development_mvp(root)

    def test_development_mvp_streams_large_jsonl_without_path_read_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_large_mvp_package(
                root, active_count=2048, suppressed_count=8192, unestablished_count=4096
            )
            with mock.patch.object(
                Path,
                "read_text",
                side_effect=AssertionError("JSONL validation must stream from an open file"),
            ):
                summary = self.gate.validate_development_mvp(root)
            self.assertEqual(summary.active_count, 2048)
            self.assertEqual(summary.family_count, 1)
            self.assertEqual(summary.pure_suppressed_count, 4096)
            self.assertEqual(summary.reclassified_count, 4096)
            self.assertEqual(summary.suppressed_count, 8192)
            self.assertEqual(summary.unestablished_count, 4096)
            self.assertEqual(summary.considered_count, 14336)
            self.assertEqual(summary.pattern_counts["P1"], 2048)

    def test_development_mvp_invalid_utf8_coverage_is_a_gate_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_mvp_package(root)
            (root / "coverage.json").write_bytes(b"\xff\n")
            with self.assertRaisesRegex(
                self.gate.GateError, "cannot read development MVP coverage"
            ):
                self.gate.validate_development_mvp(root)

    def test_development_mvp_memory_error_is_a_gate_failure(self) -> None:
        with mock.patch.object(
            self.gate, "_validate_development_mvp", side_effect=MemoryError
        ):
            with self.assertRaisesRegex(self.gate.GateError, "exhausted memory"):
                self.gate.validate_development_mvp(Path("unused"))

    def test_development_single_run_records_memory_error_resource(self) -> None:
        protocol = self.gate.load_protocol(PROTOCOL_2024)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir = root / "rapx-perf"
            bin_dir.mkdir()
            cargo_rapx = bin_dir / "cargo-rapx"
            rapx = bin_dir / "rapx"
            cargo = root / "cargo"
            sysroot = root / "sysroot"
            rustc = sysroot / "bin/rustc"
            source_root = sysroot / "lib/rustlib/src/rust"
            unit_root = source_root / "library/core"
            output = root / "result"
            output.mkdir()
            unit_root.mkdir(parents=True)
            rustc.parent.mkdir(parents=True, exist_ok=True)
            for path in (cargo_rapx, rapx, cargo, rustc):
                path.write_bytes(path.name.encode())
            (bin_dir / self.gate.PROFILE_SIDECAR).write_text("{}")

            def fake_run(_command, _cwd, environment, log, _wall, _physical):
                log.write_text("development scan log\n")
                write_mvp_package(Path(environment["RAP_UNSOUNDAUDIT_OUTPUT"]))
                return self.gate.ProcessResult("success", 0, 1.0, 2, 3)

            with (
                mock.patch.object(self.gate, "verify_binary_launch"),
                mock.patch.object(self.gate, "run_limited", side_effect=fake_run),
                mock.patch.object(
                    self.gate, "validate_development_mvp", side_effect=MemoryError
                ),
            ):
                with self.assertRaisesRegex(self.gate.GateError, "exhausted memory"):
                    self.gate.run_development_single_scan(
                        repo_root=REPO_ROOT,
                        protocol_path=PROTOCOL_2024,
                        protocol=protocol,
                        cargo=cargo,
                        binary_directory=bin_dir,
                        cargo_rapx=cargo_rapx,
                        rapx=rapx,
                        rust_sysroot=sysroot,
                        rustc=rustc,
                        profile={"rap_source_commit": "f" * 40},
                        source_root=source_root,
                        unit_root=unit_root,
                        output_root=output,
                        unit="core",
                    )

            resource = json.loads(
                (output / "core.development.resource.json").read_text()
            )
            self.assertEqual(resource["status"], "failed")
            self.assertEqual(
                resource["reason"], "development MVP validation exhausted memory"
            )
            self.assertIn("candidates_sha256", resource["hashes"])
            self.assertIsNotNone(resource["execution"])

    def test_replacement_requires_exact_root_site_role_and_exposure(self) -> None:
        location = span()
        suppressed = mvp_candidate("c" * 64, pattern="P5")
        suppressed["exposure"] = {
            "kind": "safe_return",
            "identity": "core::exposure",
            "span": location,
        }
        replacement = copy.deepcopy(suppressed)
        replacement["finding_id"] = "d" * 64
        mutations = {
            "public root": lambda row: row["public_safe_root"].update(owner="core::other"),
            "unsafe site": lambda row: row["unsafe_site"].update(site_id="e" * 64),
            "subject role": lambda row: row.update(subject_role="result"),
            "exposure": lambda row: row["exposure"].update(identity="core::other_exposure"),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                changed = copy.deepcopy(replacement)
                mutate(changed)
                with self.assertRaisesRegex(self.gate.GateError, label):
                    self.gate.validate_replacement_compatibility(suppressed, changed)

    def test_development_partial_population_is_counted_but_never_meets_family_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_mvp_package(root, complete=False)
            summary = self.gate.validate_development_mvp(root)
            self.assertFalse(summary.coverage_exact)
            self.assertIsNone(summary.family_target_met)
            self.assertEqual(summary.active_count, 1)
            self.assertEqual(summary.coverage_failure_count, 1)

    def test_development_single_run_executes_one_fresh_scan_and_writes_nonformal_resource(self) -> None:
        protocol = self.gate.load_protocol(PROTOCOL_2024)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir = root / "rapx-perf"
            bin_dir.mkdir()
            cargo_rapx = bin_dir / "cargo-rapx"
            rapx = bin_dir / "rapx"
            cargo = root / "cargo"
            sysroot = root / "sysroot"
            rustc = sysroot / "bin/rustc"
            source_root = sysroot / "lib/rustlib/src/rust"
            unit_root = source_root / "library/core"
            unit_root.mkdir(parents=True)
            rustc.parent.mkdir(parents=True, exist_ok=True)
            for path in (cargo_rapx, rapx, cargo, rustc):
                path.write_bytes(path.name.encode())
            (bin_dir / self.gate.PROFILE_SIDECAR).write_text("{}")
            output = root / "result"
            observed: list[tuple[list[str], Path, dict[str, str], Path]] = []

            def fake_run(
                command: list[str],
                cwd: Path,
                environment: dict[str, str],
                log: Path,
                wall: float,
                physical: int,
            ):
                self.assertEqual(wall, 900)
                self.assertEqual(physical, 8 * 1024**3)
                self.assertTrue(Path(environment["CARGO_TARGET_DIR"]).is_dir())
                self.assertNotIn("UNSOUND_SCANNER_RAP_JSON_DIR", environment)
                self.assertNotIn("RUSTFLAGS", environment)
                log.write_text("development scan log\n")
                Path(environment["UNSOUND_SCANNER_PERF_TRACE"]).write_text(
                    '{"phase":"collect_ids","event":"end"}\n'
                )
                write_mvp_package(Path(environment["RAP_UNSOUNDAUDIT_OUTPUT"]))
                observed.append((command, cwd, environment, log))
                return self.gate.ProcessResult("success", 0, 12.5, 4096, 8192)

            arguments = [
                "--repo-root", str(REPO_ROOT),
                "--protocol", str(PROTOCOL_2024),
                "--schema", str(REPO_ROOT / "tests/unsoundaudit-v3/rap-unit-v3.schema.json"),
                "--cargo", str(cargo),
                "--rap-bin-dir", str(bin_dir),
                "--rust-sysroot", str(sysroot),
                "--output-root", str(output),
                "--unit", "core",
                "--development-single-run",
            ]
            profile = {
                "schema_version": self.gate.BINARY_PROFILE_SCHEMA,
                **self.gate.PROFILE_CONTRACT,
                "rap_source_commit": "f" * 40,
                "rustc_commit": protocol["rustc_commit"],
                "cargo_rapx_sha256": self.gate.sha256(cargo_rapx),
                "rapx_sha256": self.gate.sha256(rapx),
            }
            with (
                mock.patch.object(
                    self.gate,
                    "tool_commit",
                    side_effect=[protocol["cargo_commit"], protocol["rustc_commit"]],
                ),
                mock.patch.object(self.gate, "cargo_profile_contract"),
                mock.patch.object(
                    self.gate, "verify_source_commit", side_effect=AssertionError("formal gate called")
                ),
                mock.patch.object(
                    self.gate, "load_development_binary_profile", return_value=profile
                ),
                mock.patch.object(self.gate, "verify_binary_launch"),
                mock.patch.object(self.gate, "stdlib_source_root", return_value=source_root),
                mock.patch.object(
                    self.gate, "scan_roots", return_value=(unit_root, source_root)
                ),
                mock.patch.object(self.gate, "run_limited", side_effect=fake_run),
                mock.patch.object(os, "environ", {"PATH": "/usr/bin", "RUSTFLAGS": "bad"}),
            ):
                self.assertEqual(self.gate.main(arguments), 0)

            self.assertEqual(len(observed), 1)
            command, _, environment, _ = observed[0]
            self.assertEqual(
                command[:3], [str(cargo_rapx.resolve()), "rapx", "-unsoundaudit"]
            )
            self.assertEqual(environment["UNSOUND_SCANNER_PROJECT_ROOT"], str(unit_root))
            self.assertEqual(
                environment["UNSOUND_SCANNER_PERF_TRACE"],
                str(output.resolve() / "logs/perf.jsonl"),
            )
            resource = json.loads((output / "core.development.resource.json").read_text())
            self.assertTrue(resource["nonformal"])
            self.assertEqual(resource["execution"]["max_physical_footprint_bytes"], 4096)
            self.assertEqual(resource["counts"]["active_count"], 1)
            self.assertTrue(resource["family_target"]["met"])
            self.assertEqual(resource["command"], command)
            self.assertEqual(
                resource["hashes"]["scan_log_sha256"],
                self.gate.sha256(output / "logs/scan-1.log"),
            )
            self.assertEqual(
                resource["hashes"]["performance_trace_sha256"],
                self.gate.sha256(output / "logs/perf.jsonl"),
            )

    def test_development_single_run_preserves_partial_resource_and_returns_nonzero(self) -> None:
        protocol = self.gate.load_protocol(PROTOCOL_2024)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir = root / "rapx-perf"
            sysroot = root / "sysroot"
            source_root = sysroot / "lib/rustlib/src/rust"
            unit_root = source_root / "library/core"
            rustc = sysroot / "bin/rustc"
            unit_root.mkdir(parents=True)
            rustc.parent.mkdir(parents=True, exist_ok=True)
            cargo = root / "cargo"
            for path in (bin_dir / "cargo-rapx", bin_dir / "rapx", cargo, rustc):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"binary")
            (bin_dir / self.gate.PROFILE_SIDECAR).write_text("{}")
            output = root / "result"

            def fake_run(_command, _cwd, environment, log, _wall, _physical):
                log.write_text("partial development scan log\n")
                write_mvp_package(Path(environment["RAP_UNSOUNDAUDIT_OUTPUT"]), complete=False)
                return self.gate.ProcessResult("success", 0, 1.0, 2, 3)

            arguments = [
                "--repo-root", str(REPO_ROOT), "--protocol", str(PROTOCOL_2024),
                "--schema", str(REPO_ROOT / "tests/unsoundaudit-v3/rap-unit-v3.schema.json"),
                "--cargo", str(cargo), "--rap-bin-dir", str(bin_dir),
                "--rust-sysroot", str(sysroot), "--output-root", str(output),
                "--unit", "core", "--development-single-run",
            ]
            with (
                mock.patch.object(
                    self.gate,
                    "tool_commit",
                    side_effect=[protocol["cargo_commit"], protocol["rustc_commit"]],
                ),
                mock.patch.object(self.gate, "cargo_profile_contract"),
                mock.patch.object(self.gate, "load_development_binary_profile", return_value={}),
                mock.patch.object(self.gate, "verify_binary_launch"),
                mock.patch.object(self.gate, "stdlib_source_root", return_value=source_root),
                mock.patch.object(self.gate, "scan_roots", return_value=(unit_root, source_root)),
                mock.patch.object(self.gate, "run_limited", side_effect=fake_run),
            ):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(self.gate.main(arguments), 1)
            resource = json.loads((output / "core.development.resource.json").read_text())
            self.assertEqual(resource["status"], "partial")
            self.assertIsNone(resource["family_target"]["met"])
            self.assertEqual(resource["counts"]["coverage_failure_count"], 1)
            failure = json.loads((output / "core.development.failure.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertIn("coverage is partial", failure["reason"])

    def test_development_cli_closes_invalid_utf8_coverage_with_failure_files(self) -> None:
        protocol = self.gate.load_protocol(PROTOCOL_2024)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir = root / "rapx-perf"
            sysroot = root / "sysroot"
            source_root = sysroot / "lib/rustlib/src/rust"
            unit_root = source_root / "library/core"
            rustc = sysroot / "bin/rustc"
            unit_root.mkdir(parents=True)
            rustc.parent.mkdir(parents=True, exist_ok=True)
            cargo = root / "cargo"
            for path in (bin_dir / "cargo-rapx", bin_dir / "rapx", cargo, rustc):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"binary")
            (bin_dir / self.gate.PROFILE_SIDECAR).write_text("{}")
            output = root / "result"

            def fake_run(_command, _cwd, environment, log, _wall, _physical):
                log.write_text("invalid coverage development scan log\n")
                mvp_output = Path(environment["RAP_UNSOUNDAUDIT_OUTPUT"])
                write_mvp_package(mvp_output)
                (mvp_output / "coverage.json").write_bytes(b"\xff\n")
                return self.gate.ProcessResult("success", 0, 1.0, 2, 3)

            arguments = [
                "--repo-root", str(REPO_ROOT), "--protocol", str(PROTOCOL_2024),
                "--schema", str(REPO_ROOT / "tests/unsoundaudit-v3/rap-unit-v3.schema.json"),
                "--cargo", str(cargo), "--rap-bin-dir", str(bin_dir),
                "--rust-sysroot", str(sysroot), "--output-root", str(output),
                "--unit", "core", "--development-single-run",
            ]
            with (
                mock.patch.object(
                    self.gate,
                    "tool_commit",
                    side_effect=[protocol["cargo_commit"], protocol["rustc_commit"]],
                ),
                mock.patch.object(self.gate, "cargo_profile_contract"),
                mock.patch.object(
                    self.gate, "load_development_binary_profile", return_value={}
                ),
                mock.patch.object(self.gate, "verify_binary_launch"),
                mock.patch.object(self.gate, "stdlib_source_root", return_value=source_root),
                mock.patch.object(
                    self.gate, "scan_roots", return_value=(unit_root, source_root)
                ),
                mock.patch.object(self.gate, "run_limited", side_effect=fake_run),
            ):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(self.gate.main(arguments), 1)

            resource = json.loads((output / "core.development.resource.json").read_text())
            failure = json.loads((output / "core.development.failure.json").read_text())
            expected_reason = "cannot read development MVP coverage"
            self.assertEqual(resource["status"], "failed")
            self.assertIn(expected_reason, resource["reason"])
            self.assertIn("coverage_sha256", resource["hashes"])
            self.assertEqual(failure["status"], "failed")
            self.assertIn(expected_reason, failure["reason"])


if __name__ == "__main__":
    unittest.main()

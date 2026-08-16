#!/usr/bin/env python3
"""No-build contracts for the compiler-aligned v3 stdlib resource gate."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

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
        self.assertEqual(success.classification, "success")
        self.assertEqual(timeout.classification, "timeout")
        self.assertEqual(memory.classification, "memory_limit")
        self.assertEqual(memory.max_physical_footprint_bytes, 4096)
        self.assertEqual(memory.max_rss_bytes, 8192)

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


if __name__ == "__main__":
    unittest.main()

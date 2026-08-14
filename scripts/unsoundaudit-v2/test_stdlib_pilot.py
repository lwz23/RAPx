#!/usr/bin/env python3
"""No-build contract tests for the bounded standard-library pilot."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path, PurePosixPath

import run_stdlib_pilot as pilot
import review_sample as review
from validate_receipt import ContractError


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = REPO_ROOT / "tests/unsoundaudit-v2/stdlib_pilot_protocol.json"
PROTOCOL_V2 = REPO_ROOT / "tests/unsoundaudit-v2/stdlib_pilot_protocol_v2.json"
PROTOCOL_V3 = REPO_ROOT / "tests/unsoundaudit-v2/stdlib_pilot_protocol_v3.json"
PROTOCOL_V4 = REPO_ROOT / "tests/unsoundaudit-v2/stdlib_pilot_protocol_v4.json"
PROTOCOL_V5 = REPO_ROOT / "tests/unsoundaudit-v2/stdlib_pilot_protocol_v5.json"
V3_FAILURE = REPO_ROOT / "artifacts/unsoundaudit-v2/mac/stdlib-pilot/core.v3.failure.json"
V4_ROOT = REPO_ROOT / "artifacts/unsoundaudit-v2/mac/stdlib-pilot/performance-v4"
SUPPRESSION_ALLOWLIST = V4_ROOT / "noise-audit/suppression_allowlist.json"
PINNED_RUSTC_COMMIT = "1bc403daadbebb553ccc211a0a8eebb73989665f"


def v5_protocol() -> dict[str, object]:
    value = json.loads(PROTOCOL_V4.read_text(encoding="utf-8"))
    value["schema_version"] = pilot.SUPPRESSION_PROTOCOL_SCHEMA_VERSION
    value["rap_source_commit"] = "f" * 40
    value["sample_seed"] = "unsoundaudit-v2-stdlib-pilot-suppression-v5-test"
    value["suppression"] = {
        "ledger_schema_version": pilot.SUPPRESSION_LEDGER_SCHEMA,
        "allowlist_entries_sha256": "95b5cf503b9942aac2c787053c1f0c6c40b8caa07472965e8059ea8becfacd21",
        "family_review_claim_scope": "single_reviewer_development_diagnostic_only",
    }
    return value


def suppression_example() -> tuple[str, dict[str, object], dict[str, object]]:
    allowlist = json.loads(SUPPRESSION_ALLOWLIST.read_text(encoding="utf-8"))
    populations = {
        unit: json.loads((V4_ROOT / f"{unit}.normalized.json").read_text(encoding="utf-8"))
        for unit in pilot.UNITS
    }
    for entry in allowlist["entries"]:
        unit = entry["crate_key"]
        for candidate in populations[unit]["findings"]:
            if pilot.stdlib_family_id(unit, candidate) == entry["family_id"]:
                return unit, candidate, entry
    raise AssertionError("frozen allowlist must match a frozen v4 finding")


def suppression_receipt(unit: str, active: list[dict[str, object]]) -> dict[str, object]:
    return {
        "crate_name": unit,
        "cargo_supplied_rustc_commit": PINNED_RUSTC_COMMIT,
        "rap_compiler_commit": PINNED_RUSTC_COMMIT,
        "rustc_commit": PINNED_RUSTC_COMMIT,
        "finding_count": len(active),
        "findings": active,
    }


def suppression_ledger(
    unit: str,
    finding_value: dict[str, object],
    entry: dict[str, object],
    *,
    active_count: int = 0,
) -> dict[str, object]:
    return {
        "schema_version": pilot.SUPPRESSION_LEDGER_SCHEMA,
        "crate_key": unit,
        "cargo_supplied_rustc_commit": PINNED_RUSTC_COMMIT,
        "rap_compiler_commit": PINNED_RUSTC_COMMIT,
        "rustc_commit": PINNED_RUSTC_COMMIT,
        "allowlist_entries_sha256": v5_protocol()["suppression"]["allowlist_entries_sha256"],
        "raw_finding_count": active_count + 1,
        "suppressed_finding_count": 1,
        "active_finding_count": active_count,
        "suppressed_findings": [
            {
                "family_id": entry["family_id"],
                "evidence_sha256": entry["evidence_sha256"],
                "rationale": entry["rationale"],
                "allowlist_entry": copy.deepcopy(entry),
                "finding": copy.deepcopy(finding_value),
            }
        ],
    }


def finding(index: int, rule_id: str, depth: str) -> dict[str, object]:
    key = {
        "crate": "core",
        "public_root": f"core::root_{index}",
        "source_origin": f"_arg{index}",
        "first_contract_failure": f"core::failure_{index}",
        "sink_or_exposure": f"core::sink_{index}",
        "canonical_obligation": "in_bounds",
    }
    return {
        "finding_id": f"{index:064x}",
        "causal_key": key,
        "rule_id": rule_id,
        "propagation": {"depth": depth},
    }


def normalized(findings: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "unsoundaudit-v2-stdlib-unit-v1",
        "unit": "core",
        "protocol_sha256": "a" * 64,
        "finding_count": len(findings),
        "findings": findings,
    }


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

    def test_v2_protocol_changes_only_source_commit_schema_and_seed(self) -> None:
        first = pilot.load_protocol(PROTOCOL)
        second = pilot.load_protocol(PROTOCOL_V2)
        differing = {key for key in first if first[key] != second[key]}
        self.assertEqual(differing, {"schema_version", "rap_source_commit", "sample_seed"})
        self.assertEqual(second["schema_version"], "unsoundaudit-v2-stdlib-pilot-protocol-v2")
        self.assertEqual(second["rap_source_commit"], "2207a5cfa4444c3e58fd127c6b653feea28e3641")
        self.assertEqual(second["sample_seed"], "unsoundaudit-v2-stdlib-pilot-source-root-v2-2026-08-13")

    def test_v3_protocol_changes_only_source_commit_schema_and_seed(self) -> None:
        second = pilot.load_protocol(PROTOCOL_V2)
        third = pilot.load_protocol(PROTOCOL_V3)
        differing = {key for key in second if second[key] != third[key]}
        self.assertEqual(differing, {"schema_version", "rap_source_commit", "sample_seed"})
        self.assertEqual(third["schema_version"], "unsoundaudit-v2-stdlib-pilot-protocol-v3")
        self.assertEqual(third["rap_source_commit"], "5481d617ee69cce0bf9e95c8ee248f29ef463387")
        self.assertEqual(
            third["sample_seed"],
            "unsoundaudit-v2-stdlib-pilot-summary-perf-v3-2026-08-13",
        )

    def test_v4_protocol_closes_optimized_profile_and_release_resource_gates(self) -> None:
        value = pilot.load_protocol(PROTOCOL_V4)
        self.assertEqual(value["schema_version"], pilot.PHYSICAL_PROTOCOL_SCHEMA_VERSION)
        self.assertEqual(
            value["rap_source_commit"],
            "a897f162b8d6f443eadaedcd1d1688211725ae04",
        )
        self.assertEqual(value["analyzer_profile"], "rapx-perf")
        expected = {
            "core": (900, 8 * 1024**3),
            "alloc": (1800, 10 * 1024**3),
            "std": (3600, 12 * 1024**3),
        }
        for unit, (wall, footprint) in expected.items():
            row = value["units"][unit]
            self.assertEqual(row["wall_time_seconds"], wall)
            self.assertEqual(row["physical_footprint_limit_bytes"], footprint)
            self.assertNotIn("rss_limit_bytes", row)
            self.assertEqual(
                pilot.resource_gate(value, unit),
                ("physical_footprint", footprint),
            )
            arguments = pilot.cargo_arguments(value, unit)
            self.assertNotIn("--release", arguments)
            self.assertNotIn("--profile", arguments)

    def test_v5_protocol_adds_only_the_closed_suppression_contract(self) -> None:
        value = pilot.load_protocol(PROTOCOL_V5)
        previous = pilot.load_protocol(PROTOCOL_V4)
        self.assertEqual(
            {key for key in value if previous.get(key) != value[key]},
            {"schema_version", "rap_source_commit", "sample_seed", "suppression"},
        )
        self.assertEqual(
            value["rap_source_commit"],
            "470b650897d77fa60b3ff4afa07b905dd2d52f14",
        )
        self.assertEqual(
            value["sample_seed"],
            "unsoundaudit-v2-stdlib-suppression-v5-2026-08-14",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v5.json"
            path.write_bytes(pilot.canonical_json(value))
            loaded = pilot.load_protocol(path)
            self.assertEqual(
                loaded["suppression"],
                {
                    "ledger_schema_version": pilot.SUPPRESSION_LEDGER_SCHEMA,
                    "allowlist_entries_sha256": "95b5cf503b9942aac2c787053c1f0c6c40b8caa07472965e8059ea8becfacd21",
                    "family_review_claim_scope": "single_reviewer_development_diagnostic_only",
                },
            )
            for mutation in (
                {"unexpected": True},
                {"ledger_schema_version": "wrong"},
                {"allowlist_entries_sha256": "short"},
                {"family_review_claim_scope": "formal_precision"},
            ):
                changed = copy.deepcopy(value)
                changed["suppression"].update(mutation)
                path.write_bytes(pilot.canonical_json(changed))
                with self.assertRaisesRegex(ContractError, "suppression"):
                    pilot.load_protocol(path)


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

    def test_rapx_perf_profile_is_closed_and_target_cargo_stays_dev_check(self) -> None:
        expected_profile = {
            "profile": "rapx-perf",
            "opt_level": 3,
            "debug": 1,
            "debug_assertions": True,
            "overflow_checks": True,
            "incremental": False,
            "target_cargo_profile": "dev",
        }
        self.assertEqual(pilot.cargo_profile_contract(REPO_ROOT / "rapx/Cargo.toml"), expected_profile)
        for unit in ("core", "alloc", "std"):
            arguments = pilot.cargo_arguments(self.protocol, unit)
            self.assertNotIn("--release", arguments)
            self.assertNotIn("--profile", arguments)
            self.assertNotIn("-r", arguments)

    def test_binary_profile_sidecar_is_closed_and_binds_actual_binary_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cargo_rapx = root / "cargo-rapx"
            rapx = root / "rapx"
            cargo_rapx.write_bytes(b"cargo-rapx optimized")
            rapx.write_bytes(b"rapx optimized")
            value = {
                "schema_version": "unsoundaudit-v2-rap-binary-profile-v1",
                "profile": "rapx-perf",
                "opt_level": 3,
                "debug": 1,
                "debug_assertions": True,
                "overflow_checks": True,
                "incremental": False,
                "target_cargo_profile": "dev",
                "cargo_rapx_sha256": pilot.sha256(cargo_rapx),
                "rapx_sha256": pilot.sha256(rapx),
            }
            (root / pilot.ANALYZER_PROFILE_SIDECAR).write_bytes(pilot.canonical_json(value))
            self.assertEqual(pilot.load_analyzer_profile(root, cargo_rapx, rapx), value)
            self.assertEqual(
                pilot.analyzer_profile_for_protocol(
                    {"analyzer_profile": "rapx-perf"},
                    REPO_ROOT / "rapx/Cargo.toml",
                    root,
                    cargo_rapx,
                    rapx,
                ),
                value,
            )

            for mutation, message in (
                ({"unexpected": True}, "keys drifted"),
                ({"cargo_rapx_sha256": "0" * 64}, "hash differs"),
                ({"profile": "debug"}, "contract drifted"),
            ):
                changed = value | mutation
                (root / pilot.ANALYZER_PROFILE_SIDECAR).write_bytes(pilot.canonical_json(changed))
                with self.assertRaisesRegex(ContractError, message):
                    pilot.load_analyzer_profile(root, cargo_rapx, rapx)

            (root / pilot.ANALYZER_PROFILE_SIDECAR).unlink()
            with self.assertRaisesRegex(ContractError, "sidecar is missing"):
                pilot.load_analyzer_profile(root, cargo_rapx, rapx)

    def test_frozen_v1_v2_v3_protocols_do_not_retroactively_require_profile_sidecars(self) -> None:
        missing = Path("/definitely/missing")
        for path in (PROTOCOL, PROTOCOL_V2, PROTOCOL_V3):
            protocol = pilot.load_protocol(path)
            self.assertIsNone(
                pilot.analyzer_profile_for_protocol(protocol, missing, missing, missing, missing)
            )

    def test_legacy_rss_and_future_physical_resource_gates_are_distinct_and_closed(self) -> None:
        for path in (PROTOCOL, PROTOCOL_V2, PROTOCOL_V3):
            protocol = pilot.load_protocol(path)
            self.assertEqual(
                pilot.resource_gate(protocol, "core"),
                ("rss", protocol["units"]["core"]["rss_limit_bytes"]),
            )

        future = json.loads(PROTOCOL_V3.read_text(encoding="utf-8"))
        future["schema_version"] = pilot.PHYSICAL_PROTOCOL_SCHEMA_VERSION
        future["analyzer_profile"] = "rapx-perf"
        for row in future["units"].values():
            row["physical_footprint_limit_bytes"] = row.pop("rss_limit_bytes")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "future.json"
            path.write_bytes(pilot.canonical_json(future))
            loaded = pilot.load_protocol(path)
            self.assertEqual(
                pilot.resource_gate(loaded, "core"),
                ("physical_footprint", future["units"]["core"]["physical_footprint_limit_bytes"]),
            )
            future["units"]["core"]["rss_limit_bytes"] = 1
            path.write_bytes(pilot.canonical_json(future))
            with self.assertRaisesRegex(ContractError, "resource contract drifted"):
                pilot.load_protocol(path)

    def test_stdlib_scan_separates_route_root_from_stable_source_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            for unit in ("core", "alloc", "std"):
                (source / "library" / unit).mkdir(parents=True)
            unit_root, source_root = pilot.scan_roots(source, self.protocol, "core")
            self.assertEqual(unit_root, source.resolve() / "library/core")
            self.assertEqual(source_root, source.resolve())
            self.assertNotEqual(unit_root, source_root)

    def test_process_group_footprint_aggregates_and_keeps_rss_secondary(self) -> None:
        output = " 10 99 1024\n 11 99 2048\n 12 100 4096\n"
        footprints = {10: 4 * 1024**2, 11: 7 * 1024**2}
        usage = pilot.aggregate_process_group_usage(output, 99, footprints.__getitem__)
        self.assertEqual(usage.physical_footprint_bytes, 11 * 1024**2)
        self.assertEqual(usage.rss_bytes, 3072 * 1024)
        self.assertEqual(usage.sampled_processes, 2)

    def test_process_group_footprint_ignores_only_disappearing_children(self) -> None:
        output = " 10 99 1024\n 11 99 2048\n"

        def sample(pid: int) -> int:
            if pid == 11:
                raise pilot.ProcessDisappeared(pid)
            return 4096

        usage = pilot.aggregate_process_group_usage(output, 99, sample)
        self.assertEqual(usage.physical_footprint_bytes, 4096)
        self.assertEqual(usage.rss_bytes, 3072 * 1024)
        self.assertEqual(usage.sampled_processes, 1)

    def test_footprint_monitor_fails_closed_on_unsupported_permission_and_parse(self) -> None:
        with self.assertRaisesRegex(pilot.PilotFailure, "unsupported"):
            pilot.DarwinPhysicalFootprint(platform="linux")

        def unavailable_library(*_args: object, **_kwargs: object) -> object:
            raise OSError("missing libproc")

        with self.assertRaisesRegex(pilot.PilotFailure, "unavailable"):
            pilot.DarwinPhysicalFootprint(platform="darwin", library_loader=unavailable_library)

        with self.assertRaisesRegex(pilot.PilotFailure, "permission"):
            pilot.aggregate_process_group_usage(
                " 10 99 1024\n", 99, lambda _pid: (_ for _ in ()).throw(PermissionError())
            )
        with self.assertRaisesRegex(pilot.PilotFailure, "parse"):
            pilot.aggregate_process_group_usage("not ps output\n", 99, lambda _pid: 1)
        with self.assertRaisesRegex(pilot.PilotFailure, "sampling failed"):
            pilot.aggregate_process_group_usage(
                " 10 99 1024\n", 99, lambda _pid: (_ for _ in ()).throw(OSError("sample failed"))
            )

    def test_proc_pid_rusage_adapter_reads_current_physical_footprint_and_errno(self) -> None:
        class Function:
            def __init__(self, result: int, error_number: int = 0) -> None:
                self.result = result
                self.error_number = error_number

            def __call__(self, _pid: int, flavor: int, pointer: object) -> int:
                self.asserted_flavor = flavor
                if self.result == 0:
                    usage = pilot.ctypes.cast(
                        pointer, pilot.ctypes.POINTER(pilot.RusageInfoV4)
                    ).contents
                    usage.ri_phys_footprint = 123456
                else:
                    pilot.ctypes.set_errno(self.error_number)
                return self.result

        class Library:
            def __init__(self, function: Function) -> None:
                self.proc_pid_rusage = function

        success = Function(0)
        sampler = pilot.DarwinPhysicalFootprint(
            platform="darwin", library_loader=lambda *_args, **_kwargs: Library(success)
        )
        self.assertEqual(sampler(17), 123456)
        self.assertEqual(success.asserted_flavor, pilot.RUSAGE_INFO_V4)

        vanished = Function(-1, pilot.errno.ESRCH)
        sampler = pilot.DarwinPhysicalFootprint(
            platform="darwin", library_loader=lambda *_args, **_kwargs: Library(vanished)
        )
        with self.assertRaises(pilot.ProcessDisappeared):
            sampler(18)

    def test_resource_runner_classifies_success_timeout_and_memory(self) -> None:
        environment = os.environ.copy()
        usage = pilot.ProcessGroupUsage(physical_footprint_bytes=4096, rss_bytes=8192, sampled_processes=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            success = pilot.run_limited(
                ["/bin/sh", "-c", "exit 0"], root, environment, root / "success.log", 2, 1024**3,
                usage_reader=lambda _pid: usage,
            )
            timeout = pilot.run_limited(
                ["/bin/sh", "-c", "sleep 5"], root, environment, root / "timeout.log", 0.05, 1024**3,
                usage_reader=lambda _pid: usage,
            )
            memory = pilot.run_limited(
                ["/bin/sh", "-c", "sleep 5"], root, environment, root / "memory.log", 2, 1,
                usage_reader=lambda _pid: usage,
            )
        self.assertEqual(success.classification, "success")
        self.assertEqual(timeout.classification, "timeout")
        self.assertEqual(memory.classification, "memory_limit")
        self.assertEqual(memory.max_physical_footprint_bytes, 4096)
        self.assertEqual(memory.max_rss_bytes, 8192)

    def test_legacy_rss_gate_does_not_reinterpret_the_limit_as_footprint(self) -> None:
        usage = pilot.ProcessGroupUsage(physical_footprint_bytes=4096, rss_bytes=1, sampled_processes=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = pilot.run_limited(
                ["/bin/sh", "-c", "sleep 5"], root, os.environ.copy(), root / "legacy.log",
                0.05, 2, memory_limit_metric="rss", usage_reader=lambda _pid: usage,
            )
        self.assertEqual(result.classification, "timeout")
        record = pilot.process_result_record("scan", 1, result, "rss")
        self.assertEqual(record["memory_limit_metric"], "rss")
        self.assertEqual(record["max_physical_footprint_bytes"], 4096)
        self.assertEqual(record["max_rss_bytes"], 1)

    def test_legacy_rss_gate_never_initializes_the_darwin_footprint_api(self) -> None:
        original_sampler = pilot.DarwinPhysicalFootprint
        original_reader = pilot.process_group_rss_usage
        pilot.DarwinPhysicalFootprint = lambda: (_ for _ in ()).throw(
            AssertionError("legacy RSS gate initialized the macOS-only footprint API")
        )
        pilot.process_group_rss_usage = lambda _pid: pilot.ProcessGroupUsage(0, 4096, 1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = pilot.run_limited(
                    ["/bin/sh", "-c", "sleep 5"], root, os.environ.copy(), root / "legacy.log",
                    2, 1, memory_limit_metric="rss",
                )
        finally:
            pilot.DarwinPhysicalFootprint = original_sampler
            pilot.process_group_rss_usage = original_reader
        self.assertEqual(result.classification, "memory_limit")
        self.assertEqual(result.max_physical_footprint_bytes, 0)
        self.assertEqual(result.max_rss_bytes, 4096)

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

    def test_suppression_ledger_requires_every_exact_identity_field(self) -> None:
        unit, finding_value, entry = suppression_example()
        receipt = suppression_receipt(unit, [])
        ledger = suppression_ledger(unit, finding_value, entry)
        protocol = v5_protocol()
        self.assertEqual(
            pilot.validate_suppression_ledger(ledger, receipt, protocol, unit),
            ledger,
        )

        mutations = {
            "commit": ("rustc_commit", "0" * 40),
            "crate": ("crate_key", "std" if unit != "std" else "core"),
            "rule": ("rule_id", "P1.get_unchecked"),
            "sink_span": ("sink_token", entry["sink_token"] + ":changed"),
            "failure_span": (
                "first_failure_token",
                entry["first_failure_token"] + ":changed",
            ),
            "obligation": ("canonical_obligation", "nonnull:wrong"),
            "source": ("source_kind", "public_parameter"),
            "family": ("family_id", "0" * 64),
        }
        for name, (field, wrong) in mutations.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(ledger)
                changed["suppressed_findings"][0]["allowlist_entry"][field] = wrong
                if field == "family_id":
                    changed["suppressed_findings"][0]["family_id"] = wrong
                with self.assertRaises(ContractError):
                    pilot.validate_suppression_ledger(
                        changed, receipt, protocol, unit
                    )

    def test_suppression_ledger_reconciles_counts_and_rejects_missing_or_duplicates(self) -> None:
        unit, finding_value, entry = suppression_example()
        receipt = suppression_receipt(unit, [])
        protocol = v5_protocol()
        ledger = suppression_ledger(unit, finding_value, entry)

        for field in (
            "raw_finding_count",
            "suppressed_finding_count",
            "active_finding_count",
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(ledger)
                changed[field] += 1
                with self.assertRaises(ContractError):
                    pilot.validate_suppression_ledger(
                        changed, receipt, protocol, unit
                    )

        duplicate = copy.deepcopy(ledger)
        duplicate["suppressed_findings"].append(
            copy.deepcopy(duplicate["suppressed_findings"][0])
        )
        duplicate["suppressed_finding_count"] = 2
        duplicate["raw_finding_count"] = 2
        with self.assertRaisesRegex(ContractError, "duplicate"):
            pilot.validate_suppression_ledger(duplicate, receipt, protocol, unit)

        missing = copy.deepcopy(ledger)
        missing.pop("suppressed_findings")
        with self.assertRaisesRegex(ContractError, "keys drifted"):
            pilot.validate_suppression_ledger(missing, receipt, protocol, unit)

    def test_family_report_is_permutation_stable_and_preserves_root_fanout(self) -> None:
        allowlist = json.loads(SUPPRESSION_ALLOWLIST.read_text(encoding="utf-8"))
        entries = {row["family_id"]: row for row in allowlist["entries"]}
        selected: tuple[str, list[dict[str, object]], dict[str, object]] | None = None
        for unit in pilot.UNITS:
            population = json.loads(
                (V4_ROOT / f"{unit}.normalized.json").read_text(encoding="utf-8")
            )
            grouped: dict[str, list[dict[str, object]]] = {}
            for finding_value in population["findings"]:
                identity = pilot.stdlib_family_id(unit, finding_value)
                if identity in entries:
                    grouped.setdefault(identity, []).append(finding_value)
            for identity, findings in grouped.items():
                roots = {row["public_safe_root"]["def_path"] for row in findings}
                ranked = sorted(
                    findings,
                    key=lambda row: (
                        sum(
                            step["kind"] == "local_call"
                            for step in row["propagation"]["witness"]
                        ),
                        row["finding_id"],
                    ),
                )
                if (
                    len(roots) >= 2
                    and sum(
                        step["kind"] == "local_call"
                        for step in ranked[0]["propagation"]["witness"]
                    )
                    < sum(
                        step["kind"] == "local_call"
                        for step in ranked[-1]["propagation"]["witness"]
                    )
                ):
                    selected = (unit, [ranked[0], ranked[-1]], entries[identity])
                    break
            if selected is not None:
                break
        self.assertIsNotNone(selected, "frozen v4 suppression population has root fanout")
        unit, findings, entry = selected
        receipt = suppression_receipt(unit, [copy.deepcopy(findings[0])])
        ledger = suppression_ledger(
            unit, findings[1], entry, active_count=receipt["finding_count"]
        )
        report = pilot.family_report(unit, receipt, ledger, "a" * 64)
        self.assertEqual(report["raw_finding_count"], 2)
        self.assertEqual(report["active_finding_count"], 1)
        self.assertEqual(report["suppressed_finding_count"], 1)
        family = report["families"][0]
        self.assertEqual(family["public_root_count"], 2)
        self.assertEqual(family["raw_finding_count"], 2)
        self.assertEqual(
            family["representative_shortest"]["finding_id"],
            findings[0]["finding_id"],
        )
        self.assertEqual(
            family["representative_longest"]["finding_id"],
            findings[1]["finding_id"],
        )
        self.assertLess(
            len(family["representative_shortest"]["propagation"]["witness"]),
            len(family["representative_longest"]["propagation"]["witness"]),
        )

        reversed_receipt = suppression_receipt(
            unit, list(reversed(receipt["findings"]))
        )
        reversed_ledger = copy.deepcopy(ledger)
        reversed_ledger["suppressed_findings"].reverse()
        self.assertEqual(
            pilot.canonical_json(report),
            pilot.canonical_json(
                pilot.family_report(
                    unit, reversed_receipt, reversed_ledger, "a" * 64
                )
            ),
        )

    def test_nonempty_driver_and_unsupported_unit_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "occupied").write_text("x", encoding="utf-8")
            with self.assertRaises(ContractError):
                pilot.write_driver(root, "core")
        with self.assertRaises(ContractError):
            pilot.driver_files("other")

    def test_rap_binary_launch_preflight_fails_before_a_scan(self) -> None:
        pilot.verify_rap_binary_launch(Path("/usr/bin/true"))
        with self.assertRaisesRegex(ContractError, "launch preflight failed"):
            pilot.verify_rap_binary_launch(Path("/usr/bin/false"))

    def test_process_group_monitor_failure_is_not_treated_as_zero_usage(self) -> None:
        original = pilot.subprocess.run

        def failed_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="monitor failed")

        pilot.subprocess.run = failed_run
        try:
            with self.assertRaisesRegex(pilot.PilotFailure, "process-group monitor failed"):
                pilot.process_group_usage(123, lambda _pid: 1)
        finally:
            pilot.subprocess.run = original

    def test_running_process_missing_from_footprint_monitor_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(pilot.PilotFailure, "absent from footprint monitor"):
                pilot.run_limited(
                    ["/bin/sh", "-c", "sleep 5"], root, os.environ.copy(), root / "footprint.log", 2, 1024**3,
                    usage_reader=lambda _pid: pilot.ProcessGroupUsage(0, 0, 0),
                )

    def test_v3_timeout_failure_metadata_is_exact_and_all_later_stages_are_closed(self) -> None:
        value = json.loads(V3_FAILURE.read_text(encoding="utf-8"))
        self.assertEqual(
            set(value),
            {
                "alloc", "binary_sha256", "classification", "continuation_allowed",
                "control", "failure", "harness_commit", "precision_review",
                "protocol_sha256", "rap_source_commit", "raw_logs_committed",
                "resource_monitoring_note", "sampling", "scan_2", "schema_version",
                "status", "std", "toolchain", "unit",
            },
        )
        self.assertEqual(value["schema_version"], "unsoundaudit-v2-stdlib-gate-failure-v1")
        self.assertEqual(value["status"], "failure")
        self.assertEqual(value["classification"], "timeout")
        self.assertFalse(value["continuation_allowed"])
        self.assertEqual(value["harness_commit"], "8df9c1e0814b8ffd2599ad13abc628d4e67ab95a")
        self.assertEqual(value["protocol_sha256"], "1f31b245d25f5915108910c0d25348b9fa5ef360dd9407e07b83531a41b883e6")
        self.assertEqual(value["rap_source_commit"], "5481d617ee69cce0bf9e95c8ee248f29ef463387")
        self.assertEqual(
            value["binary_sha256"],
            {
                "cargo_rapx": "f84d912e42bd9e580d3da73863f52bc66e94a063846255e3ad92eebf30a704fa",
                "rapx": "41da7c742c1e074b1d028754a01044125129d3fc8a33cdb240141eda8b0b4e30",
            },
        )
        self.assertEqual(value["control"]["log_sha256"], "06087da947dc773edb0dd06418636ed5bc4f7e33ec76ce3117518828ff1503e6")
        self.assertEqual(value["failure"]["log_sha256"], "8fb8dad96d828c7544de545d539d1b9d3ce460a721da9848842dc7900bc6a7e6")
        self.assertEqual(value["failure"]["rap_unit_v2_receipt_count"], 0)
        self.assertEqual(value["failure"]["skip_receipt_count"], 1)
        self.assertEqual(value["failure"]["skip_receipt_sha256"], "d1a874007dc7d13724291ce00e47f6031d91cc37a64119aa8356aba01eb619ff")
        self.assertEqual(value["failure"]["wall_time_limit_seconds"], 3600)
        self.assertEqual(
            {key: value[key] for key in ("scan_2", "alloc", "std", "sampling", "precision_review")},
            {key: "not_started" for key in ("scan_2", "alloc", "std", "sampling", "precision_review")},
        )

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


class PilotReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = pilot.load_protocol(PROTOCOL)

    def test_full_sample_is_permutation_stable_and_hash_ordered(self) -> None:
        findings = [finding(index, "P1.get_unchecked", "intra_procedural") for index in range(12)]
        forward = review.make_sample(normalized(findings), self.protocol, "b" * 64)
        reverse = review.make_sample(normalized(list(reversed(findings))), self.protocol, "b" * 64)
        self.assertEqual(forward, reverse)
        self.assertEqual(forward["population_size"], 12)
        self.assertEqual(forward["sample_size"], 12)
        self.assertEqual(
            [row["selection_hash"] for row in forward["findings"]],
            sorted(row["selection_hash"] for row in forward["findings"]),
        )
        self.assertTrue(all(row["inclusion_probability"] == {"numerator": 1, "denominator": 1} for row in forward["findings"]))

    def test_large_sample_meets_strata_interprocedural_and_recursive_rules(self) -> None:
        findings = []
        for index in range(300):
            findings.append(finding(index, "P1.get_unchecked", "intra_procedural"))
        for index in range(300, 500):
            findings.append(finding(index, "P2.raw_read", "inter_procedural"))
        for index in range(500, 525):
            findings.append(finding(index, "P3.unchecked_utf8", "recursive"))
        sample = review.make_sample(normalized(findings), self.protocol, "b" * 64)
        selected = sample["findings"]
        by_stratum: dict[str, int] = {}
        for row in selected:
            by_stratum[row["stratum"]] = by_stratum.get(row["stratum"], 0) + 1
        self.assertGreaterEqual(sample["sample_size"], 250)
        self.assertGreaterEqual(min(by_stratum.values()), 20)
        self.assertGreaterEqual(by_stratum["P2.raw_read|inter_procedural"], 150)
        self.assertEqual(by_stratum["P3.unchecked_utf8|recursive"], 25)

    def test_wilson_kappa_and_conflict_adjudication_are_exact(self) -> None:
        lower, upper = review.wilson95(50, 100)
        self.assertAlmostEqual(lower, 0.403832, places=6)
        self.assertAlmostEqual(upper, 0.596168, places=6)
        self.assertAlmostEqual(review.cohens_kappa(["A", "A", "B", "B"], ["A", "B", "B", "B"]), 0.5)
        with self.assertRaises(ContractError):
            review.adjudicated_labels(
                ["A", "B"], ["A", "C"], None, self.protocol["review"]["labels"],
            )
        self.assertEqual(
            review.adjudicated_labels(["A", "B"], ["A", "C"], [None, "D"], self.protocol["review"]["labels"]),
            ["A", "D"],
        )
        with self.assertRaises(ContractError):
            review.adjudicated_labels(["X"], ["X"], None, self.protocol["review"]["labels"])

    def test_metrics_treat_unknown_as_failure_and_apply_early_stop(self) -> None:
        rows = [
            {
                "finding_id": f"{index:064x}",
                "rule_id": "P1.get_unchecked",
                "depth": "inter_procedural" if index < 20 else "intra_procedural",
                "selection_hash": f"{index:064x}",
            }
            for index in range(60)
        ]
        labels = ["D", "E", "D", *("A" for _ in range(27)), *("U" for _ in range(30))]
        metrics = review.review_metrics(rows, labels, labels, labels, self.protocol)
        self.assertEqual(metrics["labels"], {"A": 27, "B": 0, "C": 0, "D": 2, "E": 1, "U": 30})
        self.assertEqual(metrics["early_stop"]["egregious_in_prefix"], 3)
        self.assertTrue(metrics["early_stop"]["triggered"])
        self.assertLess(metrics["overall"]["actionable"]["wilson95"]["lower"], 0.6)
        self.assertEqual(metrics["decision"], "fail")

    def test_label_files_bind_exact_sample_and_require_chain_evidence(self) -> None:
        findings = [
            finding(1, "P1.get_unchecked", "intra_procedural"),
            finding(2, "P2.raw_read", "inter_procedural"),
        ]
        sample = review.make_sample(normalized(findings), self.protocol, "b" * 64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / "sample.json"
            sample_path.write_bytes(pilot.canonical_json(sample))
            sample_sha = pilot.sha256(sample_path)

            def labels(reviewer_id: str, experience: bool) -> dict[str, object]:
                rows = []
                for row in sample["findings"]:
                    interprocedural = row["depth"] != "intra_procedural"
                    rows.append({
                        "finding_id": row["finding_id"],
                        "label": "B" if interprocedural else "A",
                        "rationale": "The structural chain is actionable." if interprocedural else "The local chain is complete.",
                        "fabricated_local_edge": False,
                        "chain_audit": {
                            "public_root": "core::root",
                            "local_edges": [{"caller": "core::root", "callee": "core::sink", "call_span": "src/lib.rs:1:1"}],
                            "mappings": [{"call_span": "src/lib.rs:1:1", "kind": "actual_to_formal", "source": "_arg1", "destination": "_1"}],
                            "sink_provenance": "_1 reaches the raw-read sink",
                            "cfg_validation": "The sink is reachable without a current validation",
                            "cycle_evidence": "not_recursive",
                        } if interprocedural else None,
                    })
                return {
                    "schema_version": "unsoundaudit-v2-stdlib-labels-v1",
                    "unit": "core",
                    "sample_sha256": sample_sha,
                    "reviewer_id": reviewer_id,
                    "rust_unsafe_contract_experience": experience,
                    "labels": rows,
                }

            one = root / "one.json"
            two = root / "two.json"
            one.write_bytes(pilot.canonical_json(labels("reviewer-one", True)))
            two.write_bytes(pilot.canonical_json(labels("reviewer-two", False)))
            result = review.review_from_files(sample_path, one, two, None, self.protocol)
            self.assertEqual(result["reviewer_ids"], ["reviewer-one", "reviewer-two"])
            self.assertEqual(result["cohens_kappa"], 1.0)
            broken = labels("reviewer-two", False)
            next(row for row in broken["labels"] if row["chain_audit"] is not None)["chain_audit"] = None
            two.write_bytes(pilot.canonical_json(broken))
            with self.assertRaisesRegex(ContractError, "complete chain audit"):
                review.review_from_files(sample_path, one, two, None, self.protocol)

    def test_empty_population_is_sampleable_but_precision_is_inconclusive(self) -> None:
        sample = review.make_sample(normalized([]), self.protocol, "b" * 64)
        self.assertEqual(sample["sample_size"], 0)
        metrics = review.review_metrics([], [], [], [], self.protocol)
        self.assertEqual(metrics["decision"], "inconclusive")
        self.assertIn("P1.raw_read:zero_findings", metrics["inconclusive_dimensions"])

    def test_small_clean_sample_is_inconclusive_not_failed_by_wilson_width(self) -> None:
        rows = [
            {
                "finding_id": f"{index:064x}",
                "rule_id": "P1.raw_read",
                "depth": "intra_procedural",
                "selection_hash": f"{index:064x}",
            }
            for index in range(10)
        ]
        labels = ["A"] * len(rows)
        metrics = review.review_metrics(rows, labels, labels, labels, self.protocol)
        self.assertEqual(metrics["decision"], "inconclusive")
        self.assertEqual(metrics["failures"], [])
        self.assertIn("overall_precision_sample_size", metrics["inconclusive_dimensions"])


if __name__ == "__main__":
    unittest.main()

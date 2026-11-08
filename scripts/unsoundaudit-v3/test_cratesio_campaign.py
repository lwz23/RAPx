#!/usr/bin/env python3
"""No-network and no-build contract tests for the crates.io robustness campaign."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import os
import re
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Any


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / "tests/unsoundaudit-v3/cratesio_campaign_policy.json"
COMMON_PATH = Path(__file__).with_name("cratesio_campaign_common.py")
SELECTOR_PATH = Path(__file__).with_name("select_cratesio_campaign.py")
RUNNER_PATH = Path(__file__).with_name("run_cratesio_campaign.py")
REVIEW_PATH = Path(__file__).with_name("review_cratesio_campaign.py")


def load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXPECTED_DOMAINS = {
    "byte_buffers",
    "general_collections",
    "concurrency_reclamation",
    "synchronization",
    "zero_copy_serialization",
    "pod_layout_casting",
    "memory_mapping",
    "ffi_dynamic_loading",
    "compact_containers",
    "hashing_runtime",
}


def make_crate_archive(
    name: str,
    version: str,
    files: dict[str, str],
    escape_root: bool = False,
    include_symlink: bool = False,
) -> bytes:
    root = f"{name}-{version}"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel_path, content in files.items():
            encoded = content.encode("utf-8")
            entry_name = f"../bad/{rel_path}" if escape_root else f"{root}/{rel_path}"
            info = tarfile.TarInfo(name=entry_name)
            info.size = len(encoded)
            tar.addfile(info, io.BytesIO(encoded))
        if include_symlink:
            sym_info = tarfile.TarInfo(name=f"{root}/symlink_file")
            sym_info.type = tarfile.SYMTYPE
            sym_info.linkname = "/etc/passwd"
            tar.addfile(sym_info)
    return buf.getvalue()


def mock_crate_payload(
    name: str,
    version: str,
    downloads: int = 100,
    has_unsafe: bool = True,
    yanked: bool = False,
    is_prerelease: bool = False,
) -> dict[str, Any]:
    ver_str = f"{version}-alpha" if is_prerelease else version
    unsafe_snippet = "pub unsafe fn dangerous() {}" if has_unsafe else "pub fn safe_only() {}"
    cargo_toml = f'[package]\nname = "{name}"\nversion = "{ver_str}"\nedition = "2021"\n'
    files = {
        "Cargo.toml": cargo_toml,
        "src/lib.rs": f"//! root\n{unsafe_snippet}\n",
    }
    archive_bytes = make_crate_archive(name, ver_str, files)
    computed_sha = hashlib.sha256(archive_bytes).hexdigest()
    metadata = {
        "crate": {
            "name": name,
            "max_stable_version": ver_str if not is_prerelease else "0.0.0",
            "downloads": downloads,
        },
        "versions": [
            {
                "num": ver_str,
                "yanked": yanked,
                "downloads": downloads,
                "checksum": computed_sha,
            }
        ],
    }
    return {
        "metadata": metadata,
        "archive": archive_bytes,
    }


class FakeCratesIo:
    def __init__(self, crates: dict[str, dict[str, Any]]):
        self.crates = crates

    def metadata(self, name: str) -> dict[str, Any]:
        if name not in self.crates:
            raise ValueError(f"unknown crate {name}")
        return self.crates[name]["metadata"]

    def archive(self, name: str, version: str) -> bytes:
        if name not in self.crates:
            raise ValueError(f"unknown crate {name}")
        return self.crates[name]["archive"]


class CratesIoCampaignCommonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = REPO_ROOT
        cls.common = load_module("cratesio_campaign_common", COMMON_PATH)

    def test_policy_is_closed_and_has_ten_unique_domains(self) -> None:
        policy = self.common.load_policy(
            self.repo / "tests/unsoundaudit-v3/cratesio_campaign_policy.json"
        )
        self.assertEqual(set(policy["candidate_pool"]), EXPECTED_DOMAINS)
        self.assertTrue(
            all(
                len(names) == 5 and len(set(names)) == 5
                for names in policy["candidate_pool"].values()
            )
        )
        self.assertEqual(policy["wall_time_seconds"], 900)
        self.assertEqual(policy["cargo_jobs"], 1)
        self.assertEqual(policy["suppressed_quota_per_crate"], 3)
        self.assertEqual(policy["unestablished_quota_per_crate"], 5)
        self.assertEqual(policy["schema_version"], "rap-unsoundaudit-cratesio-campaign-policy-v1")
        self.assertEqual(policy["campaign_id"], "cratesio-2026-08-21")
        self.assertEqual(policy["selection_date"], "2026-08-21")
        self.assertEqual(
            policy["toolchain"],
            {
                "channel": "nightly-2026-08-14",
                "rustc_commit": "ba28ff76f353a722f31c4f3dd2ac4e437d36411b",
                "cargo_commit": "eb98b54bc9f3c74519f43d066cb3fd02ebc88df0",
                "target_triple": "aarch64-apple-darwin",
            },
        )
        self.assertEqual(policy["features"], "default")
        self.assertEqual(policy["target_kind"], "lib")
        self.assertTrue(re.fullmatch(r"[0-9a-f]{64}", policy["review_seed"]))

    def test_canonical_json_and_sha256(self) -> None:
        data = {"b": 1, "a": [2, 3]}
        raw = self.common.canonical_json(data)
        self.assertEqual(raw, b'{\n  "a": [\n    2,\n    3\n  ],\n  "b": 1\n}\n')
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.json"
            file_path.write_bytes(raw)
            digest = self.common.sha256(file_path)
            self.assertEqual(len(digest), 64)
            self.assertTrue(all(c in "0123456789abcdef" for c in digest))

    def test_stable_rank_is_deterministic(self) -> None:
        seed = "dacd31d8f446692177be93f19314e578f01119d0ef637e80c071ad0cdc3638ec"
        rank1 = self.common.stable_rank(seed, "candidate_a")
        rank2 = self.common.stable_rank(seed, "candidate_a")
        rank3 = self.common.stable_rank(seed, "candidate_b")
        self.assertEqual(rank1, rank2)
        self.assertNotEqual(rank1, rank3)
        self.assertEqual(len(rank1), 64)

    def test_sanitized_environment_strips_forbidden_variables(self) -> None:
        dirty = {
            "PATH": "/usr/bin",
            "HOME": "/Users/test",
            "CARGO_BUILD_JOBS": "8",
            "CARGO_PROFILE_DEV_DEBUG": "true",
            "CARGO_TARGET_DIR": "/tmp/target",
            "CARGO_HOME": "/tmp/cargo_home",
            "DYLD_LIBRARY_PATH": "/usr/local/lib",
            "DYLD_INSERT_LIBRARIES": "/bad/lib.dylib",
            "LD_PRELOAD": "/bad/lib.so",
            "LD_LIBRARY_PATH": "/usr/lib",
            "RAP_UNSOUNDAUDIT_OUTPUT": "/tmp/out",
            "RAP_SOMETHING": "1",
            "UNSOUND_SCANNER_PERF_TRACE": "/tmp/perf",
            "RUSTC_WRAPPER": "/usr/bin/sccache",
            "RUSTFLAGS": "-C target-cpu=native",
            "RUSTUP_TOOLCHAIN": "nightly",
        }
        clean = self.common.sanitized_environment(dirty)
        self.assertEqual(clean["PATH"], "/usr/bin")
        self.assertEqual(clean["HOME"], "/Users/test")
        self.assertNotIn("CARGO_BUILD_JOBS", clean)
        self.assertNotIn("CARGO_PROFILE_DEV_DEBUG", clean)
        self.assertNotIn("CARGO_TARGET_DIR", clean)
        self.assertNotIn("CARGO_HOME", clean)
        self.assertNotIn("DYLD_LIBRARY_PATH", clean)
        self.assertNotIn("DYLD_INSERT_LIBRARIES", clean)
        self.assertNotIn("LD_PRELOAD", clean)
        self.assertNotIn("LD_LIBRARY_PATH", clean)
        self.assertNotIn("RAP_UNSOUNDAUDIT_OUTPUT", clean)
        self.assertNotIn("RAP_SOMETHING", clean)
        self.assertNotIn("UNSOUND_SCANNER_PERF_TRACE", clean)
        self.assertNotIn("RUSTC_WRAPPER", clean)
        self.assertNotIn("RUSTFLAGS", clean)
        self.assertNotIn("RUSTUP_TOOLCHAIN", clean)

    def test_policy_loader_validations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bad_path = Path(temp_dir) / "bad.json"
            bad_path.write_text('{"schema_version": "wrong"}', encoding="utf-8")
            with self.assertRaises(self.common.CampaignError):
                self.common.load_policy(bad_path)


class CratesIoCampaignSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = REPO_ROOT
        cls.common = load_module("cratesio_campaign_common", COMMON_PATH)
        cls.selector = load_module("select_cratesio_campaign", SELECTOR_PATH)

    def test_selector_uses_latest_stable_non_yanked_and_highest_download_eligible(self) -> None:
        api = FakeCratesIo(
            {
                "popular-safe": mock_crate_payload("popular-safe", "2.0.0", downloads=200, has_unsafe=False),
                "unsafe-lib": mock_crate_payload("unsafe-lib", "1.4.0", downloads=100, has_unsafe=True),
            }
        )
        selected = self.selector.select_domain(
            "byte_buffers", ["popular-safe", "unsafe-lib"], api.metadata, api.archive
        )
        self.assertEqual(selected["name"], "unsafe-lib")
        self.assertEqual(selected["version"], "1.4.0")
        self.assertEqual(selected["selection_rank"], 2)
        self.assertTrue(selected["admission"]["has_unsafe"])
        self.assertGreater(selected["admission"]["unsafe_token_count"], 0)

    def test_archive_admission_rejects_traversal_symlink_and_test_only_unsafe(self) -> None:
        # Traversal archive
        traversal_bytes = make_crate_archive(
            "sample", "1.0.0", {"Cargo.toml": '[package]\nname = "sample"\nversion = "1.0.0"\n'}, escape_root=True
        )
        with tarfile.open(fileobj=io.BytesIO(traversal_bytes), mode="r:gz") as tar:
            with self.assertRaises(self.common.CampaignError):
                self.selector.inspect_archive(tar, "sample", "1.0.0")

        # Symlink archive
        symlink_bytes = make_crate_archive(
            "sample", "1.0.0", {"Cargo.toml": '[package]\nname = "sample"\nversion = "1.0.0"\n'}, include_symlink=True
        )
        with tarfile.open(fileobj=io.BytesIO(symlink_bytes), mode="r:gz") as tar:
            with self.assertRaises(self.common.CampaignError):
                self.selector.inspect_archive(tar, "sample", "1.0.0")

        # Test-only unsafe
        test_only_files = {
            "Cargo.toml": '[package]\nname = "sample"\nversion = "1.0.0"\n',
            "src/lib.rs": "pub fn safe() {}\n",
            "tests/test.rs": "unsafe fn test_fn() {}\n",
            "examples/ex.rs": "unsafe fn example_fn() {}\n",
            "benches/bench.rs": "unsafe fn bench_fn() {}\n",
        }
        test_only_bytes = make_crate_archive("sample", "1.0.0", test_only_files)
        with tarfile.open(fileobj=io.BytesIO(test_only_bytes), mode="r:gz") as tar:
            with self.assertRaises(self.common.CampaignError):
                self.selector.inspect_archive(tar, "sample", "1.0.0")

    def test_rust_unsafe_keyword_lexer_ignores_comments_and_strings(self) -> None:
        safe_cases = [
            "// unsafe fn not_real() {}",
            "/* unsafe fn block() {} */",
            "/* nested /* block */ unsafe */ fn ok() {}",
            'const S: &str = "unsafe string";',
            'const B: &[u8] = b"unsafe byte string";',
            'const R: &str = r#"unsafe raw string"#;',
            "const C: char = 'u';",
            "struct UnsafeQueue;",
            "fn run_unsafe_operation() {}",
        ]
        for src in safe_cases:
            has_u, count = self.selector.has_lexical_unsafe(src)
            self.assertFalse(has_u, f"expected no unsafe in: {src!r}")
            self.assertEqual(count, 0)

        unsafe_cases = [
            ("pub unsafe fn dangerous() {}", 1),
            ("unsafe { let x = 1; }", 1),
            ("pub unsafe trait DangerousTrait {}", 1),
            ("unsafe impl Send for MyType {}", 1),
            ("unsafe extern \"C\" fn f() {}", 1),
            ("fn f() { unsafe { op1(); } unsafe { op2(); } }", 2),
        ]
        for src, expected_count in unsafe_cases:
            has_u, count = self.selector.has_lexical_unsafe(src)
            self.assertTrue(has_u, f"expected unsafe in: {src!r}")
            self.assertEqual(count, expected_count)

    def test_select_domain_fails_if_no_candidate_admitted(self) -> None:
        api = FakeCratesIo(
            {
                "cand1": mock_crate_payload("cand1", "1.0.0", downloads=10, has_unsafe=False),
                "cand2": mock_crate_payload("cand2", "1.0.0", downloads=20, has_unsafe=False),
            }
        )
        with self.assertRaises(self.common.CampaignError):
            self.selector.select_domain("byte_buffers", ["cand1", "cand2"], api.metadata, api.archive)

    def test_select_campaign_produces_full_roster(self) -> None:
        policy = self.common.load_policy(
            self.repo / "tests/unsoundaudit-v3/cratesio_campaign_policy.json"
        )
        fake_crates: dict[str, dict[str, Any]] = {}
        for domain, candidates in policy["candidate_pool"].items():
            for i, cand in enumerate(candidates):
                # Make candidate 0 safe and candidate 1 unsafe to test ranking
                has_u = i > 0
                downloads = (len(candidates) - i) * 1000
                fake_crates[cand] = mock_crate_payload(cand, "1.0.0", downloads=downloads, has_unsafe=has_u)

        api = FakeCratesIo(fake_crates)
        with tempfile.TemporaryDirectory() as temp_dir:
            download_root = Path(temp_dir) / "downloads"
            roster = self.selector.select_campaign(
                policy, download_root, fetch_metadata=api.metadata, fetch_archive=api.archive
            )
            self.assertEqual(roster["schema_version"], "rap-unsoundaudit-cratesio-campaign-roster-v1")
            self.assertEqual(roster["campaign_id"], policy["campaign_id"])
            self.assertEqual(roster["selection_date"], policy["selection_date"])
            self.assertEqual(len(roster["packages"]), 10)
            unique_names = {p["name"] for p in roster["packages"]}
            self.assertEqual(len(unique_names), 10)
            for pkg in roster["packages"]:
                self.assertEqual(pkg["version"], "1.0.0")
                self.assertTrue(pkg["admission"]["has_unsafe"])
                crate_file = download_root / pkg["archive_filename"]
                self.assertTrue(crate_file.exists())
                self.assertEqual(self.common.sha256(crate_file), pkg["archive_sha256"])


from test_contract import minimal_receipt, span


def make_test_receipt(
    package_name: str = "sample",
    package_version: str = "1.0.0",
    crate_name: str = "sample",
    target_kind: str = "lib",
    rustc_commit: str = "ba28ff76f353a722f31c4f3dd2ac4e437d36411b",
    complete: bool = True,
) -> dict[str, Any]:
    r = minimal_receipt()
    r["package_name"] = package_name
    r["package_version"] = package_version
    r["crate_name"] = crate_name
    r["target_kind"] = target_kind
    r["package_id"] = f"{package_name}@{package_version}#Cargo.toml"
    r["cargo_supplied_rustc_commit"] = rustc_commit
    r["rap_compiler_commit"] = rustc_commit
    r["rustc_commit"] = rustc_commit
    r["toolchain_adapter"]["rustc_commit"] = rustc_commit
    if not complete:
        r["analysis_status"] = "partial"
        r["coverage_failures"] = ["elided_unsafe_site"]
        r["observed_counts"]["semantics"] = "lower_bound"
        r["unsafe_site_census"]["elided"] = 1
        r["unsafe_sites"][0]["binding"] = "elided"
    return r


class CratesIoCampaignRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = REPO_ROOT
        cls.common = load_module("cratesio_campaign_common", COMMON_PATH)
        cls.runner = load_module("run_cratesio_campaign", RUNNER_PATH)
        cls.schema = json.loads((cls.repo / "tests/unsoundaudit-v3/rap-unit-v3.schema.json").read_text(encoding="utf-8"))

    def test_terminal_state_partition(self) -> None:
        self.assertEqual(self.runner.classify_outcome(control_exit=1), "control_failed")
        self.assertEqual(self.runner.classify_outcome(control_exit=0, scan_timeout=True), "scan_timeout")
        self.assertEqual(self.runner.classify_outcome(control_exit=0, scan_exit=1), "scan_failed")
        self.assertEqual(self.runner.classify_outcome(control_exit=0, scan_exit=0, coverage_complete=False), "partial_coverage")
        self.assertEqual(self.runner.classify_outcome(control_exit=0, scan_exit=0, coverage_complete=True), "complete_exact")

    def test_root_receipt_requires_exact_name_version_lib_target(self) -> None:
        commit = "ba28ff76f353a722f31c4f3dd2ac4e437d36411b"
        receipt = make_test_receipt(package_name="sample", package_version="1.2.3", crate_name="sample", rustc_commit=commit)
        receipts = [receipt]

        selected = self.runner.select_root_receipt(
            receipts, self.schema, package_name="sample", package_version="1.2.3",
            crate_name="sample", rustc_commit=commit
        )
        self.assertEqual(selected["package_id"], "sample@1.2.3#Cargo.toml")

        # Duplicate
        duplicate_receipts = [receipt, copy.deepcopy(receipt)]
        with self.assertRaises(self.common.CampaignError):
            self.runner.select_root_receipt(
                duplicate_receipts, self.schema, package_name="sample", package_version="1.2.3",
                crate_name="sample", rustc_commit=commit
            )

        # Missing
        with self.assertRaises(self.common.CampaignError):
            self.runner.select_root_receipt(
                receipts, self.schema, package_name="other", package_version="1.2.3",
                crate_name="other", rustc_commit=commit
            )

        # Wrong commit
        with self.assertRaises(self.common.CampaignError):
            self.runner.select_root_receipt(
                receipts, self.schema, package_name="sample", package_version="1.2.3",
                crate_name="sample", rustc_commit="0" * 40
            )


class CratesIoCampaignReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = REPO_ROOT
        cls.common = load_module("cratesio_campaign_common", COMMON_PATH)
        cls.review = load_module("review_cratesio_campaign", REVIEW_PATH)

    def test_inconclusive_is_excluded_from_decidable_precision(self) -> None:
        metrics = self.review.precision_metrics(
            ["confirmed_bug", "likely_bug", "false_positive", "inconclusive"]
        )
        self.assertEqual(metrics["decidable"], 3)
        self.assertEqual(
            metrics["false_positive_fraction"], {"numerator": 1, "denominator": 3}
        )
        self.assertAlmostEqual(metrics["false_positive_rate"], 1 / 3)

    def test_wilson_interval(self) -> None:
        self.assertIsNone(self.review.wilson(0, 0))
        res = self.review.wilson(1, 3)
        self.assertIsNotNone(res)
        self.assertGreater(res["high"], res["low"])
        self.assertGreaterEqual(res["low"], 0.0)
        self.assertLessEqual(res["high"], 1.0)

    def test_review_template_contains_every_active_family_and_bounded_samples(self) -> None:
        policy = self.common.load_policy(
            self.repo / "tests/unsoundaudit-v3/cratesio_campaign_policy.json"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            camp_dir = Path(temp_dir)
            mvp_dir = camp_dir / "packages" / "sample_crate" / "mvp"
            mvp_dir.mkdir(parents=True, exist_ok=True)

            families = [
                {"family_id": "f1" * 32, "pattern": "P1", "contract_precision": "exact"},
                {"family_id": "f2" * 32, "pattern": "P2", "contract_precision": "generic"},
            ]
            suppressed = [
                {"pattern": "P1", "public_root": f"root{i}", "unsafe_site_id": f"s{i}" * 32, "reason": "non_zero_proven"}
                for i in range(10)
            ]
            unestablished = [
                {
                    "attempted_pattern": "P3",
                    "public_root": f"root{i}",
                    "unsafe_site_id": f"s{i}" * 32,
                    "operand_role": "arg",
                    "source": {"kind": "param", "identity": f"id{i}"},
                    "primary_reason": "gap_found",
                }
                for i in range(10)
            ]

            (mvp_dir / "families.jsonl").write_text("\n".join(json.dumps(f) for f in families) + "\n", encoding="utf-8")
            (mvp_dir / "suppression_ledger.jsonl").write_text("\n".join(json.dumps(s) for s in suppressed) + "\n", encoding="utf-8")
            (mvp_dir / "unestablished_decisions.jsonl").write_text("\n".join(json.dumps(u) for u in unestablished) + "\n", encoding="utf-8")

            campaign = {
                "packages": [
                    {
                        "name": "sample_crate",
                        "version": "1.0.0",
                        "terminal_state": "complete_exact",
                    }
                ]
            }

            template1 = self.review.build_review_template(policy, campaign, camp_dir)
            template2 = self.review.build_review_template(policy, campaign, camp_dir)
            self.assertEqual(template1, template2)

            active_rows = [r for r in template1["rows"] if r["population"] == "active"]
            suppressed_rows = [r for r in template1["rows"] if r["population"] == "suppressed"]
            unest_rows = [r for r in template1["rows"] if r["population"] == "unestablished"]

            self.assertEqual(len(active_rows), 2)
            self.assertEqual({r["family_id"] for r in active_rows}, {"f1" * 32, "f2" * 32})
            self.assertLessEqual(len(suppressed_rows), 3)
            self.assertLessEqual(len(unest_rows), 5)

            # Test review validation fail on blank reviewer/rationale
            with self.assertRaises(self.common.CampaignError):
                self.review.validate_review(template1, campaign)

            # Fill review properly
            for r in template1["rows"]:
                r["reviewer"] = "test_reviewer"
                r["rationale"] = "valid test rationale"
                if r["population"] == "active":
                    r["label"] = "likely_bug"
                else:
                    r["label"] = "classification_supported"

            self.review.validate_review(template1, campaign)


if __name__ == "__main__":
    unittest.main()



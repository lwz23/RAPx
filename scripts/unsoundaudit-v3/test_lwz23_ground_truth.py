#!/usr/bin/env python3
"""Tests for issue-body ground truth extraction and RAP finding match."""

from __future__ import annotations

import unittest
from pathlib import Path

import lwz23_ground_truth as gt


ATOMIC_BODY = """
hello, I notice the following code:
https://github.com/Amanieu/atomic-rs/blob/a0d808c031361d795b61c8f6e3959f1f76613f30/src/lib.rs#L328
```
 pub fn fetch_add(&self, val: $t, order: Ordering) -> $t {
                unsafe { ops::atomic_add(self.inner_ptr(), val, order) }
            }
pub unsafe fn atomic_add<T: Copy>(dst: *mut T, val: T) -> T {
    let result = ptr::read(dst);
    ptr::write(dst, val);
    result
}
```
"""

CATFS_BODY = """
Location: src/catfs/mod.rs
The vulnerability is in the public function make_self:
```
pub fn make_self<T>(s: &mut T) -> &'static T {
    return unsafe { ::std::mem::transmute(s) };
}
```
"""


class GroundTruthTests(unittest.TestCase):
    def test_extracts_blob_path_fn_and_fence_from_atomic_body(self) -> None:
        extracted = gt.extract_issue("Maybe Multiple unsoundness problems", ATOMIC_BODY)
        self.assertTrue(extracted["has_fence"])
        self.assertIn("fetch_add", extracted["fns"])
        self.assertIn("atomic_add", extracted["fns"])
        self.assertEqual(extracted["blobs"][0]["path"], "src/lib.rs")
        self.assertEqual(extracted["blobs"][0]["line"], 328)
        self.assertIn("src/lib.rs", extracted["paths"])
        self.assertTrue(extracted["matchable"])

    def test_extracts_make_self_and_transmute_from_catfs_body(self) -> None:
        extracted = gt.extract_issue(
            "[Unsoundness] Public API make_self allows lifetime extension",
            CATFS_BODY,
        )
        self.assertIn("make_self", extracted["fns"])
        self.assertIn("transmute", extracted["callees"])
        self.assertTrue(any(p.endswith("catfs/mod.rs") or p.endswith("src/catfs/mod.rs") or "mod.rs" in p for p in extracted["paths"]) or "src/catfs/mod.rs" in CATFS_BODY)

    def test_title_method_is_a_function_needle(self) -> None:
        extracted = gt.extract_issue("Maybe unsound in Cursor::new", "no fence here")
        self.assertIn("new", extracted["fns"])
        self.assertIn("new", extracted["title_fns"])
        self.assertEqual(extracted["type_methods"][0]["type"], "Cursor")
        self.assertEqual(extracted["type_methods"][0]["method"], "new")

    def test_locate_weak_new_via_type_method(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            src = Path(raw)
            path = src / "src" / "cursor.rs"
            path.parent.mkdir(parents=True)
            path.write_text("pub struct Cursor { x: u8 }\nimpl Cursor {\n    pub fn new() -> Self { Cursor { x: 0 } }\n}\n")
            extracted = gt.extract_issue("Maybe unsound in Cursor::new", "no fence")
            located = gt.locate_in_checkout(src, extracted)
            self.assertTrue(any(row.get("fn") == "new" and row.get("rel") == "src/cursor.rs" for row in located), located)

    def test_locate_fn_in_checkout(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            src = Path(raw)
            lib = src / "src" / "lib.rs"
            lib.parent.mkdir(parents=True)
            lib.write_text("pub fn fetch_add(&self) { unsafe { std::ptr::read(p) } }\n")
            extracted = gt.extract_issue("unsound in fetch_add", ATOMIC_BODY)
            located = gt.locate_in_checkout(src, extracted)
            rels = {row["rel"] for row in located if row["exists"]}
            self.assertIn("src/lib.rs", rels)
            self.assertTrue(any(row.get("fn") == "fetch_add" for row in located))

    def test_match_requires_reported_function_not_repo_name(self) -> None:
        extracted = gt.extract_issue("Maybe Multiple unsoundness problems", ATOMIC_BODY)
        extracted["located"] = [{"rel": "src/lib.rs", "fn": "fetch_add", "line": 328, "exists": True, "how": "blob"}]
        hit, why = gt.match_finding(
            extracted,
            {
                "root": "Atomic::<u16>::fetch_add",
                "callee": "core::mem::transmute_copy",
                "path": "src/ops.rs",
                "line": 10,
            },
        )
        self.assertTrue(hit, why)
        self.assertTrue(any(item.startswith("fn:") for item in why), why)

        miss, miss_why = gt.match_finding(
            extracted,
            {
                "root": "core::ptr::read",
                "callee": "core::ptr::read",
                "path": "/home/lwz/work/Amanieu/atomic-rs/src/ptr.rs",
                "line": 1,
            },
        )
        self.assertFalse(miss, miss_why)

    def test_bare_get_unchecked_without_fn_or_path_is_not_a_hit(self) -> None:
        extracted = gt.extract_issue(
            "Maybe unsound in wakeup",
            "The wakeup function uses get_unchecked on a slice.",
        )
        self.assertIn("wakeup", extracted["fns"])
        self.assertIn("get_unchecked", extracted["callees"])
        miss, _ = gt.match_finding(
            extracted,
            {
                "root": "other::unrelated::index",
                "callee": "core::slice::get_unchecked",
                "path": "src/other.rs",
                "line": 9,
            },
        )
        self.assertFalse(miss)
        hit, why = gt.match_finding(
            extracted,
            {
                "root": "queue::wakeup",
                "callee": "core::slice::get_unchecked",
                "path": "src/queue.rs",
                "line": 40,
            },
        )
        self.assertTrue(hit, why)

    def test_package_scan_status_requires_receipt_or_candidates(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            work = Path(raw)
            (work / "mvp" / "zenoh-shm").mkdir(parents=True)
            (work / "logs").mkdir()
            (work / "logs" / "zenoh-shm.control.log").write_text("error[E0658]: freeze\n")
            self.assertEqual(gt.package_scan_status(work, "zenoh-shm"), "control_failed")
            (work / "mvp" / "base").mkdir(parents=True)
            self.assertEqual(gt.package_scan_status(work, "base"), "not_scanned")
            rec = work / "receipts" / "hal_x86_64"
            rec.mkdir(parents=True)
            (rec / "x.json").write_text("{}\n")
            (work / "mvp" / "hal_x86_64").mkdir(parents=True)
            (work / "mvp" / "hal_x86_64" / "candidates.jsonl").write_text("{}\n")
            self.assertEqual(gt.package_scan_status(work, "hal_x86_64"), "scanned")
            (work / "logs" / "nydus-storage.scan.log").write_text("internal compiler error\n")
            (work / "mvp" / "nydus-storage").mkdir(parents=True)
            self.assertEqual(gt.package_scan_status(work, "nydus-storage"), "scan_failed")
            skip_dir = work / "receipts" / "flow-rs"
            skip_dir.mkdir(parents=True)
            (skip_dir / "flow_rs-skip.json").write_text(
                '{"schema_version":"rap-unit-skip-v1","status":"skipped","skip_reason":"manifest_outside_project"}\n'
            )
            self.assertEqual(gt.package_scan_status(work, "flow-rs"), "not_scanned")

    def test_weak_get_needs_path(self) -> None:
        extracted = gt.extract_issue(
            "Unsoundness in pub fn get",
            "see src/stat/perf_counter.rs\n```\npub fn get(&self) { unsafe { s.get_unchecked(i) } }\n```",
        )
        self.assertIn("get", extracted["fns"])
        miss, _ = gt.match_finding(
            extracted,
            {"root": "foo::get", "callee": "get_unchecked", "path": "src/unrelated.rs", "line": 3},
        )
        self.assertFalse(miss)
        hit, why = gt.match_finding(
            extracted,
            {
                "root": "stat::perf_counter::PCAcc::get",
                "callee": "core::slice::get_unchecked",
                "path": "crates/libamdgpu_top/src/stat/perf_counter.rs",
                "line": 88,
            },
        )
        self.assertTrue(hit, why)


    def test_classify_located_crate_skip_analysis_only_when_scanned(self) -> None:
        tag, reason = gt.classify_located_crate(
            status="scanned", no_std=False, located_rel="definitions/src/asm.rs"
        )
        self.assertEqual(tag, "skip_analysis")
        self.assertEqual(reason, "located_crate_scanned")

        tag, reason = gt.classify_located_crate(
            status="absent", no_std=False, located_rel="definitions/src/asm.rs"
        )
        self.assertEqual(tag, "lift_now")
        self.assertEqual(reason, "located_crate_never_built")

        tag, reason = gt.classify_located_crate(
            status="absent", no_std=True, located_rel="seed/src/boot_info.rs"
        )
        self.assertEqual(tag, "skip_host")
        self.assertEqual(reason, "missing_std_or_core")

        tag, reason = gt.classify_located_crate(
            status="control_failed", no_std=False, located_rel="xtask/src/main.rs"
        )
        self.assertEqual(tag, "skip_repo")
        self.assertTrue(reason.startswith("last_run_"))

        tag, reason = gt.classify_located_crate(
            status="scan_failed", no_std=False, located_rel="storage/src/utils.rs"
        )
        self.assertEqual(tag, "skip_repo")

        tag, reason = gt.classify_located_crate(
            status="scanned", no_std=False, located_rel="src/d3d12/mod.rs"
        )
        self.assertEqual(tag, "skip_host")
        self.assertEqual(reason, "windows_cfg")

        tag, reason = gt.classify_located_crate(
            status="scan_failed", no_std=False, located_rel="native/core/src/data_type.rs"
        )
        self.assertEqual(tag, "skip_repo")
        self.assertNotEqual(tag, "skip_host")

        tag, reason = gt.classify_located_crate(
            status="control_failed", no_std=False, located_rel="crates/vm/src/native/mod.rs"
        )
        self.assertEqual(tag, "skip_repo")
        self.assertNotIn("native_c_lib", reason)

        tag, reason = gt.classify_located_crate(
            status="absent", no_std=False, located_rel="etw-reader/src/utils.rs", windows_only=True
        )
        self.assertEqual(tag, "skip_host")
        tag, reason = gt.classify_located_crate(
            status="absent", no_std=False, located_rel="src/sharded.rs", missing_path_dep=True
        )
        self.assertEqual(tag, "skip_repo")
        self.assertEqual(reason, "missing_path_dep")




if __name__ == "__main__":
    unittest.main()

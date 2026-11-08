#!/usr/bin/env python3
"""Gating tests for generic RAP compile recovery (no AST, no per-repo allowlists)."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest import mock


SCANNER_PATH = Path(__file__).with_name("scan_lwz23_valid_issue_roster.py")

MSRV_FIXTURE = """
error: rustc 1.83.0-nightly is not supported by the following package:
  unicode-segmentation@1.13.3 requires rustc 1.85.0
Either upgrade rustc or select compatible dependency versions with
`cargo update <name>@<current-ver> --precise <compatible-ver>`
"""


def load_scanner():
    spec = importlib.util.spec_from_file_location(
        "scan_lwz23_valid_issue_roster", SCANNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load scan_lwz23_valid_issue_roster")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScanRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_scanner()

    def test_msrv_log_yields_older_unicode_segmentation_pin(self) -> None:
        def older_versions(name: str, current: str, limit: int = 8) -> list[str]:
            self.assertEqual(name, "unicode-segmentation")
            self.assertEqual(current, "1.13.3")
            return ["1.12.0", "1.11.0"][:limit]

        plan = self.mod.msrv_downgrade_plan(MSRV_FIXTURE, older_versions_fn=older_versions)
        self.assertEqual(plan, [("unicode-segmentation", "1.12.0")])
        with mock.patch.object(
            self.mod.urllib.request, "urlopen", side_effect=OSError("offline")
        ):
            fallback = self.mod.crates_io_older_versions(
                "unicode-segmentation", "1.13.3"
            )
        self.assertIn("1.13.0", fallback)
        self.assertIn("1.12.0", fallback)

    def test_partial_retry_targets_only_control_failed_package(self) -> None:
        previous = {
            "status": "miss",
            "n_active": 120,
            "package_results": [
                {"package": "zenoh-buffers", "status": "scanned"},
                {"package": "zenoh-shm", "status": "control_failed"},
            ],
        }
        action, only = self.mod.classify_resume(
            previous,
            retry_timeouts=False,
            retry_empty_miss=False,
            retry_partial=True,
        )
        self.assertEqual(action, "scan")
        self.assertEqual(only, ["zenoh-shm"])

    def test_compile_knob_ladder_has_no_repository_slugs(self) -> None:
        blob = repr(self.mod.compile_knob_ladder())
        self.assertNotIn("github.com", blob.lower())
        self.assertNotIn("Amanieu/", blob)
        self.assertNotIn("nervosnetwork/", blob)
        self.assertNotIn("eclipse-zenoh/", blob)
        names = [row[0] for row in self.mod.compile_knob_ladder()]
        self.assertIn("no-default-features", names)
        self.assertIn("unlocked", names)
        self.assertIn("crate-attr-freeze", names)
        self.assertIn("crate-attr-freeze-defaults", names)
        self.assertIn("allow-implicit-autoref", names)
        freeze_defaults = [row for row in self.mod.compile_knob_ladder() if row[0] == "crate-attr-freeze-defaults"]
        self.assertEqual(freeze_defaults[0][2], [])
        allow = [row for row in self.mod.compile_knob_ladder() if row[0] == "allow-implicit-autoref"]
        self.assertIn("dangerous_implicit_autorefs", allow[0][3]["RUSTFLAGS"])

    def test_cargo_args_always_ignore_rust_version_without_slugs(self) -> None:
        args = self.mod.cargo_args(
            {"name": "pkg", "lib": True, "bin_name": ""},
            locked=True,
        )
        self.assertIn("--ignore-rust-version", args)
        self.assertIn("--lib", args)
        joined = " ".join(args)
        self.assertNotIn("github.com", joined)
        self.assertNotIn("nervosnetwork", joined)

    def test_hard_control_failure_skips_knobs_without_slugs(self) -> None:
        git404 = "error: failed to get `foo` as a dependency of package `bar`\nCaused by:\n  404 Not Found"
        self.assertEqual(self.mod.hard_control_failure(git404), "git_dep")
        self.assertEqual(
            self.mod.hard_control_failure("error: can't find crate for `std`"),
            "missing_std_or_target",
        )
        self.assertEqual(
            self.mod.hard_control_failure("linking with `cc` failed: library not found"),
            "native_lib",
        )
        self.assertEqual(
            self.mod.hard_control_failure(
                "feature `edition2024` is required\nThe package requires the Cargo feature called `edition2024`"
            ),
            "edition2024",
        )
        missing_toml = (
            "failed to read `/tmp/workspace/subxt/subxt/Cargo.toml`\n\n"
            "Caused by:\n  No such file or directory (os error 2)\n"
        )
        self.assertEqual(self.mod.hard_control_failure(missing_toml), "missing_path_dep")
        self.assertIsNone(self.mod.hard_control_failure(MSRV_FIXTURE))
        self.assertIn("edition2024", self.mod.REPO_LEVEL_HARD_FAIL)
        self.assertIn("missing_path_dep", self.mod.REPO_LEVEL_HARD_FAIL)
        blob = self.mod.hard_control_failure.__doc__ or ""
        self.assertNotIn("github.com", blob)
        previous = {
            "status": "control_failed",
            "package_results": [
                {
                    "package": "pkg",
                    "status": "control_failed",
                    "control_hard_fail": "git_dep",
                }
            ],
        }
        action, only = self.mod.classify_resume(
            previous,
            retry_timeouts=True,
            retry_empty_miss=False,
            retry_partial=True,
        )
        self.assertEqual(action, "skip")
        self.assertIsNone(only)

    def test_timeouts_allow_one_package_control_plus_scan(self) -> None:
        self.assertEqual(self.mod.CONTROL_SECONDS, 900)
        self.assertEqual(self.mod.KNOB_SECONDS, 300)
        self.assertEqual(self.mod.SCAN_SECONDS, 1800)
        self.assertEqual(self.mod.REPO_WALL_SECONDS, 3600)
        self.assertGreaterEqual(
            self.mod.REPO_WALL_SECONDS, self.mod.CONTROL_SECONDS + 600
        )
        self.assertIn("repo_wall", self.mod.REPO_LEVEL_HARD_FAIL)
        source = SCANNER_PATH.read_text(encoding="utf-8")
        self.assertIn('pkg_target = case_dir / f"target-{package[\'name\']}"', source)
        self.assertNotIn("control_target", source)
        self.assertNotIn("scan_target", source)
        previous = {
            "status": "control_failed",
            "package_results": [
                {
                    "package": "pkg",
                    "status": "control_failed",
                    "control_hard_fail": "repo_wall",
                }
            ],
        }
        skip_action, skip_only = self.mod.classify_resume(
            previous,
            retry_timeouts=False,
            retry_empty_miss=False,
            retry_partial=True,
        )
        self.assertEqual(skip_action, "skip")
        self.assertIsNone(skip_only)
        retry_action, retry_only = self.mod.classify_resume(
            previous,
            retry_timeouts=True,
            retry_empty_miss=False,
            retry_partial=True,
        )
        self.assertEqual(retry_action, "scan")
        self.assertIsNone(retry_only)
        no_pkg_action, no_pkg_only = self.mod.classify_resume(
            {"status": "no_package"},
            retry_timeouts=True,
            retry_empty_miss=False,
            retry_partial=False,
        )
        self.assertEqual(no_pkg_action, "scan")
        self.assertIsNone(no_pkg_only)

    def test_oom_killed_is_not_retried(self) -> None:
        self.assertTrue(self.mod.already_terminal("oom_killed"))
        self.assertIn("oom_killed", self.mod.REPO_LEVEL_HARD_FAIL)
        previous = {
            "status": "oom_killed",
            "package_results": [{"package": "pkg", "status": "oom_killed"}],
        }
        action, only = self.mod.classify_resume(
            previous,
            retry_timeouts=True,
            retry_empty_miss=True,
            retry_partial=True,
        )
        self.assertEqual(action, "skip")
        self.assertIsNone(only)
        self.assertEqual(self.mod.classify_exit(137, Path("/nonexistent")), "oom_killed")
        self.assertEqual(self.mod.classify_exit(-9, Path("/nonexistent")), "oom_killed")
        self.assertEqual(self.mod.classify_exit(0, Path("/nonexistent")), "success")

    def test_issue_body_path_selects_nested_crate(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            src = Path(raw)
            web = src / "pagefind_web" / "src"
            web.mkdir(parents=True)
            target = web / "lib.rs"
            target.write_text("pub fn request_filter_indexes() {}\n")
            (src / "pagefind" / "src").mkdir(parents=True)
            (src / "pagefind" / "src" / "lib.rs").write_text("pub fn unrelated() {}\n")
            issues = [
                {
                    "title": "Maybe unsound",
                    "body": "see pagefind_web/src/lib.rs\n```\npub fn request_filter_indexes() {}\n```",
                    "repair_obligation_zh": "",
                }
            ]
            found = self.mod.find_issue_source_files(src, issues)
            self.assertTrue(any(p.resolve() == target.resolve() for p in found), found)

    def _write_pkg(self, root: Path, rel: str, name: str, src_rel: str, body: str) -> Path:
        pkg = root / rel
        src_file = pkg / src_rel
        src_file.parent.mkdir(parents=True)
        (pkg / "Cargo.toml").write_text(
            f'[package]\nname = "{name}"\nversion = "0.1.0"\nedition = "2021"\n'
            f'[lib]\npath = "{src_rel}"\n',
            encoding="utf-8",
        )
        src_file.write_text(body, encoding="utf-8")
        return src_file

    def test_select_packages_issue_body_nested_path_not_sibling(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            src = Path(raw)
            sibling_src = self._write_pkg(
                src, "sibling", "sibling", "src/lib.rs", "pub fn try_pod_read_unaligned() {}\n"
            )
            nested_src = self._write_pkg(
                src,
                "crates/base",
                "base",
                "src/pod.rs",
                "pub fn try_pod_read_unaligned() {}\n",
            )
            (src / "Cargo.toml").write_text(
                '[workspace]\nmembers = ["sibling", "crates/base"]\n', encoding="utf-8"
            )
            issues = [
                {
                    "title": "unsound",
                    "body": "see crates/base/src/pod.rs\n```\npub fn try_pod_read_unaligned() {}\n```",
                    "repair_obligation_zh": "",
                }
            ]
            sibling_pkg = {
                "name": "sibling",
                "manifest": str((src / "sibling" / "Cargo.toml").resolve()),
                "cwd": str((src / "sibling").resolve()),
                "lib": True,
                "bin_name": "",
            }
            base_pkg = {
                "name": "base",
                "manifest": str((src / "crates" / "base" / "Cargo.toml").resolve()),
                "cwd": str((src / "crates" / "base").resolve()),
                "lib": True,
                "bin_name": "",
            }

            def fake_meta(_src, _cargo, _env, manifest, _log):
                resolved = Path(manifest).resolve()
                if resolved == Path(base_pkg["manifest"]):
                    return [base_pkg]
                if resolved == Path(sibling_pkg["manifest"]):
                    return [sibling_pkg]
                if resolved == (src / "Cargo.toml").resolve():
                    return [sibling_pkg, base_pkg]
                return []

            with mock.patch.object(self.mod, "metadata_packages", side_effect=fake_meta):
                chosen = self.mod.select_packages(
                    src,
                    cargo=Path("/bin/false"),
                    env={},
                    log=src / "meta.log",
                    issues=issues,
                )
            self.assertEqual([pkg["name"] for pkg in chosen], ["base"], chosen)
            found = self.mod.find_issue_source_files(src, issues)
            self.assertTrue(any(p.resolve() == nested_src.resolve() for p in found), found)
            self.assertTrue(sibling_src.exists())

    def test_select_packages_kernel_nested_path(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            src = Path(raw)
            self._write_pkg(src, "userland", "userland", "src/lib.rs", "pub fn execve() {}\n")
            kernel_src = self._write_pkg(
                src, "kernel", "osmium", "src/syscall.rs", "fn execve() {}\n"
            )
            issues = [
                {
                    "title": "unsound",
                    "body": "bug in kernel/src/syscall.rs\n```\nfn execve() {}\n```",
                    "repair_obligation_zh": "",
                }
            ]
            user_pkg = {
                "name": "userland",
                "manifest": str((src / "userland" / "Cargo.toml").resolve()),
                "cwd": str((src / "userland").resolve()),
                "lib": True,
                "bin_name": "",
            }
            kernel_pkg = {
                "name": "osmium",
                "manifest": str((src / "kernel" / "Cargo.toml").resolve()),
                "cwd": str((src / "kernel").resolve()),
                "lib": True,
                "bin_name": "",
            }

            def fake_meta(_src, _cargo, _env, manifest, _log):
                resolved = Path(manifest).resolve()
                if resolved == Path(kernel_pkg["manifest"]):
                    return [kernel_pkg]
                if resolved == Path(user_pkg["manifest"]):
                    return [user_pkg]
                return []

            with mock.patch.object(self.mod, "metadata_packages", side_effect=fake_meta):
                chosen = self.mod.select_packages(
                    src,
                    cargo=Path("/bin/false"),
                    env={},
                    log=src / "meta.log",
                    issues=issues,
                )
            self.assertEqual([pkg["name"] for pkg in chosen], ["osmium"], chosen)
            found = self.mod.find_issue_source_files(src, issues)
            self.assertTrue(any(p.resolve() == kernel_src.resolve() for p in found), found)

    def test_select_packages_type_name_disambiguates_common_fn(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            src = Path(raw)
            self._write_pkg(
                src, "sibling", "sibling", "src/lib.rs", "pub fn write() {}\n"
            )
            dma = self._write_pkg(
                src,
                "lib/poplar",
                "poplar",
                "src/ddk/dma.rs",
                "pub struct DmaArray<T> { ptr: u8 }\nimpl<T> DmaArray<T> {\n    pub fn write(&mut self) {}\n}\n",
            )
            issues = [
                {
                    "title": "unsound write",
                    "body": (
                        "```\npub struct DmaArray<T> {\n    pub ptr: NonNull<T>,\n}\n"
                        "pub fn write(&mut self, index: usize, value: T) {}\n```"
                    ),
                    "repair_obligation_zh": "",
                }
            ]
            sibling_pkg = {
                "name": "sibling",
                "manifest": str((src / "sibling" / "Cargo.toml").resolve()),
                "cwd": str((src / "sibling").resolve()),
                "lib": True,
                "bin_name": "",
            }
            poplar_pkg = {
                "name": "poplar",
                "manifest": str((src / "lib" / "poplar" / "Cargo.toml").resolve()),
                "cwd": str((src / "lib" / "poplar").resolve()),
                "lib": True,
                "bin_name": "",
            }

            def fake_meta(_src, _cargo, _env, manifest, _log):
                resolved = Path(manifest).resolve()
                if resolved == Path(poplar_pkg["manifest"]):
                    return [poplar_pkg]
                if resolved == Path(sibling_pkg["manifest"]):
                    return [sibling_pkg]
                return []

            found = self.mod.find_issue_source_files(src, issues)
            self.assertTrue(any(p.resolve() == dma.resolve() for p in found), found)
            self.assertFalse(
                any(p.as_posix().endswith("sibling/src/lib.rs") for p in found), found
            )
            with mock.patch.object(self.mod, "metadata_packages", side_effect=fake_meta):
                chosen = self.mod.select_packages(
                    src,
                    cargo=Path("/bin/false"),
                    env={},
                    log=src / "meta.log",
                    issues=issues,
                )
            self.assertEqual([pkg["name"] for pkg in chosen], ["poplar"], chosen)

    def test_staticlib_is_a_selectable_lib_kind(self) -> None:
        text = Path(self.mod.__file__).read_text()
        self.assertIn("staticlib", text)
        self.assertIn("cdylib", text)

    def test_ustc_tls_is_not_git_dep_hard_fail(self) -> None:
        ustc = (
            "error: failed to get `proc-macro2` as a dependency of package `foo`\n"
            "Caused by:\n"
            "  failed to download from `https://mirrors.ustc.edu.cn/crates.io-index/pr/oc/proc-macro2`\n"
            "Caused by:\n"
            "  SSL routines::unexpected eof while reading\n"
        )
        self.assertIsNone(self.mod.hard_control_failure(ustc))

    def test_stale_feature_maps_proc_macro2_without_repo_slug(self) -> None:
        log = "error[E0635]: unknown feature `proc_macro_span_shrink`\n"
        self.assertEqual(self.mod.STALE_FEATURE_TO_CRATE["proc_macro_span_shrink"], "proc-macro2")
        self.assertIn("proc_macro_span_shrink", log)
        names = [row[0] for row in self.mod.compile_knob_ladder()]
        self.assertIn("allow-invalid-ref-cast", names)


    def test_select_packages_unique_fn_not_generic_new(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            src = Path(raw)
            self._write_pkg(
                src,
                "compiler_base/error",
                "compiler_base_error",
                "src/lib.rs",
                "pub fn new() {}\n",
            )
            runtime = self._write_pkg(
                src,
                "kclvm/runtime",
                "kclvm-runtime",
                "src/api/utils.rs",
                "pub fn c2str_vec(ptr_array: i32) {}\n",
            )
            issues = [
                {
                    "title": "unsound",
                    "body": (
                        "```\npub fn c2str_vec(ptr_array: *const *const c_char) -> Vec<String> {\n"
                        "    let mut result = Vec::new();\n}\n```"
                    ),
                    "repair_obligation_zh": "",
                }
            ]
            err_pkg = {
                "name": "compiler_base_error",
                "manifest": str((src / "compiler_base" / "error" / "Cargo.toml").resolve()),
                "cwd": str((src / "compiler_base" / "error").resolve()),
                "lib": True,
                "bin_name": "",
            }
            rt_pkg = {
                "name": "kclvm-runtime",
                "manifest": str((src / "kclvm" / "runtime" / "Cargo.toml").resolve()),
                "cwd": str((src / "kclvm" / "runtime").resolve()),
                "lib": True,
                "bin_name": "",
            }

            def fake_meta(_src, _cargo, _env, manifest, _log):
                resolved = Path(manifest).resolve()
                if resolved == Path(rt_pkg["manifest"]):
                    return [rt_pkg]
                if resolved == Path(err_pkg["manifest"]):
                    return [err_pkg]
                return []

            found = self.mod.find_issue_source_files(src, issues)
            self.assertTrue(any(p.resolve() == runtime.resolve() for p in found), found)
            self.assertFalse(
                any("compiler_base" in p.as_posix() for p in found), found
            )
            with mock.patch.object(self.mod, "metadata_packages", side_effect=fake_meta):
                chosen = self.mod.select_packages(
                    src,
                    cargo=Path("/bin/false"),
                    env={},
                    log=src / "meta.log",
                    issues=issues,
                )
            self.assertEqual([pkg["name"] for pkg in chosen], ["kclvm-runtime"], chosen)


    def test_e0658_freeze_keeps_crate_attr_knobs(self) -> None:
        log = (
            "error[E0658]: use of unstable library feature \'freeze\'\n"
            "error: could not compile `stabby-abi` (lib) due to 1 previous error\n"
        )
        names = [row[0] for row in self.mod.select_control_knobs(log)]
        self.assertIn("crate-attr-freeze", names)
        self.assertIn("crate-attr-freeze-defaults", names)
        self.assertIn("crate-attr-common", names)
        self.assertIn("allow-implicit-autoref", names)
        self.assertIn("unlocked", names)
        blob = "".join(names)
        self.assertNotIn("github.com", blob)
        self.assertNotIn("eclipse-zenoh", blob)



    def test_isolate_broken_sibling_selects_nested_crate(self) -> None:
        import os
        import shutil
        import tempfile

        cargo = shutil.which("cargo")
        with tempfile.TemporaryDirectory() as raw:
            src = Path(raw) / "checkout"
            sibling = src / "sibling"
            nested = src / "crates" / "base"
            (sibling / "src").mkdir(parents=True)
            (nested / "src").mkdir(parents=True)
            (src / "Cargo.toml").write_text(
                '[workspace]\nmembers = ["sibling", "crates/base"]\n',
                encoding="utf-8",
            )
            (sibling / "Cargo.toml").write_text(
                "this is not a valid Cargo.toml [[[\n",
                encoding="utf-8",
            )
            (sibling / "src" / "lib.rs").write_text("pub fn write() {}\n", encoding="utf-8")
            (nested / "Cargo.toml").write_text(
                '[package]\nname = "base"\nversion = "0.1.0"\nedition = "2021"\n'
                '[lib]\npath = "src/pod.rs"\n',
                encoding="utf-8",
            )
            pod = nested / "src" / "pod.rs"
            pod.write_text("pub fn try_pod_read_unaligned() {}\n", encoding="utf-8")
            issues = [
                {
                    "title": "unsound",
                    "body": "see crates/base/src/pod.rs\n```\npub fn try_pod_read_unaligned() {}\n```",
                    "repair_obligation_zh": "",
                }
            ]
            isolate_root = Path(raw) / "isolate"
            isolated = self.mod.isolate_package_for_metadata(
                src, nested / "Cargo.toml", isolate_root
            )
            self.assertTrue(isolated.is_file(), isolated)
            self.assertFalse(str(isolated.resolve()).startswith(str(src.resolve())))
            standalone = isolated.read_text(encoding="utf-8")
            self.assertNotIn("[workspace]", standalone)
            self.assertIn("base", standalone)
            log = Path(raw) / "meta.log"
            if cargo:
                env = os.environ.copy()
                pkgs = self.mod.metadata_packages(
                    src, Path(cargo), env, isolated, log
                )
                self.assertEqual([p["name"] for p in pkgs], ["base"], pkgs)
                chosen = self.mod.select_packages(
                    src,
                    cargo=Path(cargo),
                    env=env,
                    log=log,
                    issues=issues,
                    isolate_root=isolate_root,
                )
                self.assertEqual([p["name"] for p in chosen], ["base"], chosen)
                self.assertFalse(
                    any("sibling" in (p.get("cwd") or "") for p in chosen), chosen
                )

    def test_isolate_stubs_missing_path_dep(self) -> None:
        import os
        import shutil
        import tempfile

        cargo = shutil.which("cargo")
        with tempfile.TemporaryDirectory() as raw:
            src = Path(raw) / "checkout"
            pkg = src / "holder"
            (pkg / "src").mkdir(parents=True)
            (pkg / "Cargo.toml").write_text(
                '[package]\nname = "holder"\nversion = "0.1.0"\nedition = "2021"\n'
                '[dependencies]\nmissing_dep = { path = "../missing_dep" }\n',
                encoding="utf-8",
            )
            (pkg / "src" / "lib.rs").write_text("pub fn f() {}\n", encoding="utf-8")
            isolated = self.mod.isolate_package_for_metadata(
                src, pkg / "Cargo.toml", Path(raw) / "isolate"
            )
            text = isolated.read_text(encoding="utf-8")
            self.assertIn("stubs", text)
            stub_root = isolated.parent.parent / "stubs"
            self.assertTrue(any(stub_root.glob("*/Cargo.toml")), list(stub_root.glob("*")))
            if cargo:
                pkgs = self.mod.metadata_packages(
                    src, Path(cargo), os.environ.copy(), isolated, Path(raw) / "m.log"
                )
                self.assertEqual([p["name"] for p in pkgs], ["holder"], pkgs)



    def test_isolate_keeps_optional_dep_table_valid(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            src = Path(raw) / "checkout"
            pkg = src / "shm"
            (pkg / "src").mkdir(parents=True)
            (pkg / "Cargo.toml").write_text(
                '[package]\nname = "shm"\nversion = "0.1.0"\nedition = "2021"\n'
                '[dependencies]\nnum_cpus = { workspace = true, optional = true }\n',
                encoding="utf-8",
            )
            (pkg / "src" / "lib.rs").write_text("pub fn f() {}\n", encoding="utf-8")
            isolated = self.mod.isolate_package_for_metadata(
                src, pkg / "Cargo.toml", Path(raw) / "isolate"
            )
            text = isolated.read_text(encoding="utf-8")
            self.assertNotIn("{ ,", text)
            self.assertNotIn("{,", text)
            self.assertIn("optional = true", text)
            self.assertNotIn("workspace = true", text)


    def test_distinctive_fn_selects_crate_without_type_name(self) -> None:
        import os
        import shutil
        import tempfile

        cargo = shutil.which("cargo")
        if not cargo:
            self.skipTest("cargo not on PATH")
        with tempfile.TemporaryDirectory() as raw:
            src = Path(raw) / "checkout"
            shm = src / "commons" / "zenoh-shm"
            pipe = src / "io" / "zenoh-link-unixpipe"
            (shm / "src").mkdir(parents=True)
            (pipe / "src").mkdir(parents=True)
            (src / "Cargo.toml").write_text(
                '[workspace]\nmembers = ["commons/zenoh-shm", "io/zenoh-link-unixpipe"]\n',
                encoding="utf-8",
            )
            (shm / "Cargo.toml").write_text(
                '[package]\nname = "zenoh-shm"\nversion = "0.1.0"\nedition = "2021"\n',
                encoding="utf-8",
            )
            (shm / "src" / "lib.rs").write_text(
                "pub struct OwnedDescriptor { pub atomic: u8 }\n"
                "impl OwnedDescriptor { pub fn confirm(&self) {} }\n",
                encoding="utf-8",
            )
            (pipe / "Cargo.toml").write_text(
                '[package]\nname = "zenoh-link-unixpipe"\nversion = "0.1.0"\nedition = "2021"\n',
                encoding="utf-8",
            )
            (pipe / "src" / "lib.rs").write_text(
                "pub fn confirm() {}\n",
                encoding="utf-8",
            )
            issues = [
                {
                    "title": "unsound",
                    "body": (
                        "```\npub struct OwnedDescriptor { pub atomic: *const u64 }\n"
                        "pub fn confirm(&self) {}\n```"
                    ),
                    "repair_obligation_zh": "",
                }
            ]
            found = self.mod.find_issue_source_files(src, issues)
            self.assertTrue(
                any("unixpipe" in p.as_posix() for p in found), found
            )
            chosen = self.mod.select_packages(
                src,
                cargo=Path(cargo),
                env=os.environ.copy(),
                log=Path(raw) / "meta.log",
                issues=issues,
                isolate_root=Path(raw) / "isolate",
            )
            names = [p["name"] for p in chosen]
            self.assertIn("zenoh-link-unixpipe", names, chosen)


if __name__ == "__main__":
    unittest.main()

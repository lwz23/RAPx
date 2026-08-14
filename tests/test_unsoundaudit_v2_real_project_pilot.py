import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "unsoundaudit_v2_real_project_pilot.py"
SPEC = importlib.util.spec_from_file_location("unsoundaudit_v2_real_project_pilot", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load pilot module from {SCRIPT}")
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)


class RealProjectPilotTests(unittest.TestCase):
    def write_json(self, value: object) -> Path:
        root = Path(tempfile.mkdtemp(prefix="rapx-real-project-pilot-test-"))
        path = root / "input.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_profile_contract_requires_three_profiles(self) -> None:
        path = self.write_json(
            {"schema": "rapx-real-project-profile-contract-v1", "profiles": []}
        )
        with self.assertRaisesRegex(ValueError, "exactly three"):
            PILOT.load_profile_contract(path)

    def test_selector_excludes_known_and_development_sources(self) -> None:
        rows = []
        for stratum in ("stars_top250", "downloads_top250"):
            for index in range(40):
                rows.append(
                    {
                        "eligible": True,
                        "full_name": f"owner/{stratum}-{index}",
                        "repo_id": index + (0 if stratum == "stars_top250" else 1000),
                        "repository_url": f"https://github.com/owner/{stratum}-{index}",
                        "default_branch": "main",
                        "strata": {stratum: {"rank": index + 1}},
                    }
                )
        rows[0]["full_name"] = "known/iue"
        rows[1]["full_name"] = "dev/input"
        result = PILOT.build_selection_manifest(rows, {"known/iue"}, {"dev/input"}, 5, 25)
        self.assertEqual(5, len(result["selected"]["stars_top250"]))
        self.assertEqual(5, len(result["selected"]["downloads_top250"]))
        chosen = json.dumps(
            {"selected": result["selected"], "reserves": result["reserves"]},
            sort_keys=True,
        )
        self.assertNotIn("known/iue", chosen)
        self.assertNotIn("dev/input", chosen)
        self.assertEqual(["dev/input", "known/iue"], result["excluded_names"])

    def test_admission_uses_next_same_stratum_reserve_after_missing_lock(self) -> None:
        manifest = {
            "selected": {},
            "reserves": {},
        }
        for stratum in ("stars_top250", "downloads_top250"):
            candidates = [
                {
                    "repo_id": offset + (0 if stratum == "stars_top250" else 1000),
                    "repository": f"owner/{stratum}-{offset}",
                    "stratum": stratum,
                }
                for offset in range(7)
            ]
            manifest["selected"][stratum] = candidates[:5]
            manifest["reserves"][stratum] = candidates[5:]

        calls = []

        def fake_runner(candidate):
            calls.append(candidate["repository"])
            if len(calls) == 1:
                return {"status": "lockfile_missing"}
            return {"status": "admitted", "target": {"kind": "lib", "name": "demo"}}

        result = PILOT.admit_candidates(manifest, fake_runner)
        self.assertEqual("lockfile_missing", result["attempts"][0]["status"])
        self.assertEqual(5, result["admitted_by_stratum"]["stars_top250"])
        self.assertEqual(5, result["admitted_by_stratum"]["downloads_top250"])
        self.assertEqual(
            "owner/stars_top250-5",
            result["admitted"]["stars_top250"][-1]["repository"],
        )

    def test_target_selection_prefers_sorted_library_over_binary(self) -> None:
        repository = Path(tempfile.mkdtemp(prefix="rapx-target-selection-"))
        first = repository / "a" / "Cargo.toml"
        second = repository / "z" / "Cargo.toml"
        first.parent.mkdir()
        second.parent.mkdir()
        first.write_text("[package]\nname='library'\nversion='0.1.0'\n", encoding="utf-8")
        second.write_text("[package]\nname='binary'\nversion='0.1.0'\n", encoding="utf-8")
        metadata = {
            "packages": [
                {
                    "name": "binary",
                    "manifest_path": str(second),
                    "targets": [{"name": "runner", "kind": ["bin"]}],
                },
                {
                    "name": "library",
                    "manifest_path": str(first),
                    "targets": [{"name": "library", "kind": ["lib"]}],
                },
            ]
        }
        target = PILOT._select_cargo_target(metadata, repository)
        self.assertIsNotNone(target)
        self.assertEqual("lib", target["kind"])
        self.assertEqual("a/Cargo.toml", target["manifest_path"])

    def test_normalization_ignores_only_run_local_fields(self) -> None:
        finding = {
            "finding_id": "a" * 64,
            "rule_id": "P1.raw_read",
            "primary_pattern": "pattern1",
            "source": {"kind": "public_parameter", "span": {"path": "src/lib.rs"}},
            "sink": {"kind": "raw_read", "span": {"path": "src/lib.rs", "line": 7}},
            "obligation": {"predicates": ["valid_for_read"], "subject": "arg:1"},
        }
        left = {"generated_at": "a", "duration_ms": 10, "findings": [finding]}
        right = {"generated_at": "b", "duration_ms": 99, "findings": [dict(finding)]}
        self.assertEqual(PILOT.normalize_receipt(left), PILOT.normalize_receipt(right))
        changed = dict(finding)
        changed["sink"] = {"kind": "raw_read", "span": {"path": "src/other.rs", "line": 7}}
        self.assertNotEqual(PILOT.finding_key(finding), PILOT.finding_key(changed))

    def test_profile_stability_requires_the_complete_root_pattern_set(self) -> None:
        first = {
            "finding_id": "a" * 64,
            "causal_key": {"rule_id": "P1.raw_read", "sink": "one"},
            "primary_pattern": "pattern1",
        }
        second = {
            "finding_id": "b" * 64,
            "causal_key": {"rule_id": "P2.bounds", "sink": "two"},
            "primary_pattern": "pattern2",
        }
        canonical = {"findings": [first, second]}
        same_reversed = {"findings": [second, first]}
        changed = {"findings": [first]}
        self.assertEqual(
            sorted([first, second], key=PILOT.finding_key),
            PILOT.profile_stable_findings(canonical, same_reversed, same_reversed),
        )
        self.assertEqual(
            [], PILOT.profile_stable_findings(canonical, same_reversed, changed)
        )

    def test_scan_verifier_rejects_ledger_tampering(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="rapx-scan-verifier-"))
        scans = root / "scans.jsonl"
        findings = root / "findings.jsonl"
        scans.write_text('{"status":"success"}\n', encoding="utf-8")
        findings.write_text("", encoding="utf-8")
        summary = {
            "schema": "rapx-real-project-scan-summary-v1",
            "profile_stable_finding_count": 0,
            "unique_root_cause_count": 0,
            "scans_sha256": PILOT.sha256_file(scans),
            "findings_sha256": PILOT.sha256_file(findings),
        }
        (root / "scan_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        PILOT.verify_scans(root)
        scans.write_text('{"status":"changed"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            PILOT.verify_scans(root)

    def test_sanitized_runtime_rows_remove_absolute_paths(self) -> None:
        runtime = Path("/private/tmp/pilot-run")
        admission = {
            "stratum": "stars_top250",
            "repo_id": 7,
            "clone_path": "/private/tmp/pilot-run/clones/candidate",
            "clone_argv": ["git", "clone", "url", "/private/tmp/pilot-run/clones/candidate"],
            "metadata_argv": ["/Users/example/.rustup/bin/cargo", "metadata"],
            "control_argv": ["/Users/example/.rustup/bin/cargo", "check"],
        }
        sanitized = PILOT._sanitized_admission(admission, runtime)
        self.assertNotIn("clone_path", sanitized)
        self.assertEqual("clones/stars_top250-7", sanitized["clone_argv"][-1])
        self.assertEqual("cargo", sanitized["metadata_argv"][0])
        scan = PILOT._sanitized_scan(
            {
                "profile": "rapx-perf",
                "argv": ["/private/tmp/bin/rapx-perf/cargo-rapx", "rapx"],
                "duration_ms": 1,
            },
            runtime,
        )
        self.assertEqual("rapx-perf/cargo-rapx", scan["argv"][0])
        self.assertNotIn("duration_ms", scan)

    def test_review_rejects_allowlist_and_blank_rationale(self) -> None:
        row = {
            "disposition": "likely_false_positive",
            "error_family": "repository_allowlist",
            "evidence_zh": "仅按仓库名排除",
            "rationale_zh": "",
        }
        with self.assertRaisesRegex(ValueError, "repository_allowlist"):
            PILOT.validate_review_row(row)


if __name__ == "__main__":
    unittest.main()

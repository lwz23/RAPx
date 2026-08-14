import hashlib
import json
import re
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V5 = REPO / "artifacts/unsoundaudit-v2/mac/stdlib-pilot/suppression-v5"
V4_REVIEWS = (
    REPO
    / "artifacts/unsoundaudit-v2/mac/stdlib-pilot/performance-v4/noise-audit/family_reviews.json"
)
AUDIT = REPO / "docs/unsoundaudit-v2/STDLIB_ACTIVE_FINDING_AUDIT.md"
JOIN_POC = (
    REPO
    / "tests/unsoundaudit-v2/stdlib-bug-validation/join-varying-borrow"
)
JOIN_RESULT = (
    REPO
    / "artifacts/unsoundaudit-v2/mac/stdlib-bug-validation/join-varying-borrow/results.json"
)
UNITS = ("core", "alloc", "std")
JOIN_CONCAT_FAMILY = (
    "560c34406031d113235e81e9d711f31bc34ba2673819fc96abd51ce60f93b4df"
)
DISPOSITIONS = {
    "confirmed_safe",
    "likely_safe",
    "poc_required",
    "confirmed_ub",
    "inconclusive",
}


def load_family_populations():
    active = {}
    suppressed_only = set()
    for unit in UNITS:
        payload = json.loads((V5 / f"{unit}.family-report.json").read_text())
        for family in payload["families"]:
            family_id = family["family_id"]
            if family["active_finding_count"]:
                active[family_id] = (unit, family)
            elif family["suppressed_finding_count"]:
                suppressed_only.add(family_id)
    return active, suppressed_only


def parse_audit_rows(text=None):
    lines = (AUDIT.read_text() if text is None else text).splitlines()
    expected_header = (
        "| family_id | unit | rule | sink | active findings | public roots | "
        "disposition | source evidence / proof status | PoC evidence |"
    )
    if lines.count(expected_header) != 1:
        raise AssertionError("audit must contain the exact nine-column table header")
    header_index = lines.index(expected_header)
    expected_separator = "|---|---|---|---|---:|---|---|---|---|"
    if lines[header_index + 1] != expected_separator:
        raise AssertionError("audit family table separator is not exact")

    rows = {}
    for line in lines[header_index + 2 :]:
        if not line:
            break
        if not line.startswith("| "):
            raise AssertionError(f"audit family table has a non-row line: {line}")
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 9:
            raise AssertionError(f"audit row must contain exactly 9 cells: {line}")
        if not re.fullmatch(r"`[0-9a-f]{64}`", cells[0]):
            raise AssertionError(f"audit row has malformed family identity: {line}")
        first_cell = cells[0].removeprefix("`").removesuffix("`")
        family_id = first_cell
        if family_id in rows:
            raise AssertionError(f"duplicate audit family: {family_id}")
        rows[family_id] = cells
    return rows


def unquote_code_cell(cell):
    if not cell.startswith("`") or not cell.endswith("`"):
        raise AssertionError(f"expected exact code cell, got: {cell}")
    return cell[1:-1]


def stdlib_family_id(unit, finding):
    key = {
        "unit": unit,
        "rule_id": finding["rule_id"],
        "sink": finding["sink"],
        "first_contract_failure": finding["first_contract_failure"],
        "predicates": sorted(set(finding["obligation"]["predicates"])),
        "subject": finding["obligation"]["subject"],
        "source_kind": finding["source"]["kind"],
        "propagation_depth": finding["propagation"]["depth"],
        "validation_status": finding["validation_status"]["status"],
    }
    encoded = json.dumps(
        key,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class ActiveFindingValidationTests(unittest.TestCase):
    def test_committed_active_population_is_37(self):
        findings = []
        for unit in UNITS:
            payload = json.loads((V5 / f"{unit}.normalized.json").read_text())
            findings.extend(payload["findings"])
        self.assertEqual(len(findings), 37)
        self.assertEqual(len({item["finding_id"] for item in findings}), 37)

    def test_audit_has_one_row_per_active_family(self):
        expected, suppressed_only = load_family_populations()
        rows = parse_audit_rows()

        self.assertEqual(len(rows), 23)
        self.assertEqual(set(rows), set(expected))
        self.assertTrue(set(rows).isdisjoint(suppressed_only))

        for family_id, cells in rows.items():
            expected_unit, family = expected[family_id]
            key = family["key"]
            sink_span = key["sink"]["span"]
            expected_sink = (
                f"{key['sink']['kind']} at {sink_span['path']}:"
                f"{sink_span['start']['line']}"
            )
            expected_public_roots = (
                f"{family['public_root_count']}: "
                + "<br>".join(f"`{root}`" for root in family["public_roots"])
            )

            self.assertEqual(unquote_code_cell(cells[0]), family_id)
            self.assertEqual(unquote_code_cell(cells[1]), expected_unit)
            self.assertEqual(unquote_code_cell(cells[2]), key["rule_id"])
            self.assertEqual(unquote_code_cell(cells[3]), expected_sink)
            self.assertEqual(int(cells[4]), family["active_finding_count"])
            self.assertEqual(cells[5], expected_public_roots)

            expected_disposition = (
                "confirmed_ub"
                if family_id == JOIN_CONCAT_FAMILY
                else "confirmed_safe"
            )
            disposition = unquote_code_cell(cells[6])
            self.assertIn(disposition, DISPOSITIONS)
            self.assertEqual(disposition, expected_disposition)

        dispositions = [unquote_code_cell(cells[6]) for cells in rows.values()]
        self.assertEqual(dispositions.count("confirmed_ub"), 1)
        self.assertEqual(dispositions.count("confirmed_safe"), 22)
        finding_counts = {"confirmed_ub": 0, "confirmed_safe": 0}
        for cells in rows.values():
            finding_counts[unquote_code_cell(cells[6])] += int(cells[4])
        self.assertEqual(finding_counts, {"confirmed_ub": 2, "confirmed_safe": 35})

    def test_family_table_rejects_an_extra_malformed_row(self):
        text = AUDIT.read_text()
        marker = "\n\n## Task 1 boundary"
        malformed = (
            "| `not-a-family-sha` | `core` | `P1.get_unchecked` | `sink` | 0 | "
            "0: none | `confirmed_safe` | invalid | invalid |"
        )
        self.assertIn(marker, text)
        tampered = text.replace(marker, f"\n{malformed}{marker}", 1)
        with self.assertRaisesRegex(AssertionError, "malformed family identity"):
            parse_audit_rows(tampered)

    def test_active_families_are_exactly_the_prior_b_review_population(self):
        active, _ = load_family_populations()
        payload = json.loads(V4_REVIEWS.read_text())
        seen = {}
        for review in payload["reviews"]:
            for family_id in review["family_ids"]:
                if family_id in active:
                    self.assertNotIn(family_id, seen)
                    seen[family_id] = review

        self.assertEqual(set(seen), set(active))
        for review in seen.values():
            self.assertEqual(review["label"], "B")
            self.assertEqual(review["family_disposition"], "generic_repair")

    def test_every_active_finding_id_maps_to_exactly_one_audit_family(self):
        active_families, _ = load_family_populations()
        audit_family_ids = set(parse_audit_rows())
        members = {family_id: [] for family_id in active_families}
        active_finding_ids = set()
        suppressed_finding_ids = set()

        for unit in UNITS:
            normalized = json.loads((V5 / f"{unit}.normalized.json").read_text())
            for finding in normalized["findings"]:
                finding_id = finding["finding_id"]
                self.assertNotIn(finding_id, active_finding_ids)
                active_finding_ids.add(finding_id)
                family_id = stdlib_family_id(unit, finding)
                self.assertIn(family_id, audit_family_ids)
                self.assertIn(family_id, members)
                members[family_id].append(finding_id)

            ledger = json.loads(
                (V5 / f"{unit}.suppression-ledger.json").read_text()
            )
            for suppressed in ledger["suppressed_findings"]:
                finding_id = suppressed["finding"]["finding_id"]
                self.assertNotIn(finding_id, suppressed_finding_ids)
                suppressed_finding_ids.add(finding_id)

        self.assertEqual(len(active_finding_ids), 37)
        self.assertTrue(active_finding_ids.isdisjoint(suppressed_finding_ids))
        for family_id, finding_ids in members.items():
            self.assertEqual(
                len(finding_ids),
                active_families[family_id][1]["active_finding_count"],
            )
            self.assertEqual(len(finding_ids), len(set(finding_ids)))

    def test_no_family_is_called_a_bug_without_confirmed_ub(self):
        for line in AUDIT.read_text().splitlines():
            if line.startswith("| `") and "confirmed bug" in line.lower():
                self.assertIn("| `confirmed_ub` |", line)

    def test_join_poc_is_dependency_free_safe_only_and_acknowledged(self):
        manifest = (JOIN_POC / "Cargo.toml").read_text()
        lockfile = (JOIN_POC / "Cargo.lock").read_text()
        manifest_payload = tomllib.loads(manifest)
        lockfile_payload = tomllib.loads(lockfile)
        source_paths = sorted((JOIN_POC / "src").glob("*.rs"))
        self.assertEqual([path.name for path in source_paths], ["lib.rs", "main.rs"])
        source_by_name = {path.name: path.read_text() for path in source_paths}
        source = "\n".join(source_by_name.values())

        self.assertIn("[dependencies]\n", manifest)
        self.assertEqual(set(manifest_payload), {"package", "dependencies"})
        self.assertEqual(manifest_payload["dependencies"], {})
        self.assertEqual(len(lockfile_payload["package"]), 1)
        locked_package = lockfile_payload["package"][0]
        self.assertEqual(locked_package["name"], "join-varying-borrow-poc")
        self.assertNotIn("dependencies", locked_package)
        self.assertNotIn("unsafe", source)
        expected_ack_gate = """assert_eq!(
        std::env::var("UNSOUNDAUDIT_STDLIB_POC_ACK").as_deref(),
        Ok("1"),
        "refusing to run a potential UB PoC without explicit acknowledgement",
    );"""
        self.assertIn(expected_ack_gate, source_by_name["main.rs"])

        growing_cases = (
            "join-growing-first",
            "concat-growing-first",
            "join-growing-later",
            "join-growing-first-nonempty-separator",
        )
        for validation_case in (
            "stable-control",
            *growing_cases,
            "join-shrinking-first",
        ):
            self.assertIn(validation_case, source)

        test_suffix = source_by_name["lib.rs"].split("#[cfg(test)]", 1)[1]
        for validation_case in growing_cases:
            self.assertNotIn(validation_case.replace("-", "_"), test_suffix)

    def test_pinned_miri_evidence_separates_debug_panic_from_release_ub(self):
        payload = json.loads(JOIN_RESULT.read_text())
        self.assertEqual(
            set(payload),
            {
                "schema_version",
                "candidate",
                "application_boundary_contains_unsafe",
                "historical",
                "current",
                "upstream_duplicate_search",
                "disclosure_status",
            },
        )
        self.assertEqual(
            payload["schema_version"],
            "unsoundaudit-v2-stdlib-bug-validation-v1",
        )
        self.assertFalse(payload["application_boundary_contains_unsafe"])
        self.assertEqual(payload["disclosure_status"], "not_authorized")

        historical = payload["historical"]
        self.assertEqual(historical["status"], "confirmed_ub")
        expected_names = {
            "stable-control",
            "join-shrinking-first",
            "join-growing-first",
            "concat-growing-first",
            "join-growing-later",
            "join-growing-first-nonempty-separator",
        }
        debug = {case["name"]: case for case in historical["debug_cases"]}
        release = {case["name"]: case for case in historical["release_cases"]}
        self.assertEqual(set(debug), expected_names)
        self.assertEqual(set(release), expected_names)
        self.assertNotIn("miri_ub", {case["outcome"] for case in debug.values()})

        ub_names = {
            "join-growing-first",
            "concat-growing-first",
            "join-growing-first-nonempty-separator",
        }
        self.assertEqual(
            {name for name, case in release.items() if case["outcome"] == "miri_ub"},
            ub_names,
        )
        for name in ub_names:
            self.assertEqual(release[name]["exit_code"], 1)
            self.assertTrue(release[name]["sink_reached"])
            self.assertIn("library/alloc/src/str.rs:164", release[name]["diagnostic"])
        for name in {"stable-control", "join-shrinking-first"}:
            self.assertEqual(release[name]["exit_code"], 0)
            self.assertFalse(release[name]["sink_reached"])
        self.assertEqual(release["join-growing-later"]["outcome"], "ordinary_panic")
        self.assertFalse(release["join-growing-later"]["sink_reached"])

    def test_current_toolchains_are_fixed_and_upstream_duplicate_is_exact(self):
        payload = json.loads(JOIN_RESULT.read_text())
        current = payload["current"]
        self.assertEqual(current["status"], "historical_fixed")
        self.assertEqual(current["classification"], "confirmed_ub/historical_fixed")

        stable = current["stable"]
        self.assertEqual(stable["rustc_release"], "1.97.1")
        self.assertEqual(
            stable["rustc_commit"],
            "8bab26f4f68e0e26f0bb7960be334d5b520ea452",
        )
        self.assertEqual(stable["compile_only"]["outcome"], "exit_zero")
        self.assertEqual(stable["compile_only"]["exit_code"], 0)
        self.assertEqual(
            stable["compile_only"]["log_sha256"],
            "344858cb82e1a93c1b587cd88418b381e062e438941bd540f1db6c7faf1ee467",
        )
        self.assertFalse(
            stable["compile_only"]["candidate_cases_executed_natively"]
        )
        self.assertEqual(
            stable["rust_src_sha256"],
            "ef6974342cf6a89eb7ce00d32a7fa49bfa88a4e9058bbfe4ca6bee2436bad8bf",
        )

        nightly = current["nightly"]
        self.assertEqual(nightly["rustc_release"], "1.99.0-nightly")
        self.assertEqual(
            nightly["rustc_commit"],
            "ba28ff76f353a722f31c4f3dd2ac4e437d36411b",
        )
        self.assertEqual(
            nightly["rust_src_sha256"],
            "40a2eb7db7c3ac9cfbc75f8869be745a1c692dd092c8e3f53a1753f09cea53f3",
        )
        expected_names = {
            "stable-control",
            "join-shrinking-first",
            "join-growing-first",
            "concat-growing-first",
            "join-growing-later",
            "join-growing-first-nonempty-separator",
        }
        expected_hashes = {
            "debug_cases": {
                "stable-control": "571b440a27265c131c494b2dda72fb7c495b73a8edcafd01e2a242ac98ad5165",
                "join-shrinking-first": "dc9096c269baf7049800744f876eb3e2e8de56362ab2cbcd098af867274ccf0d",
                "join-growing-first": "d1addd0eeafd0af9f799e95bb6af89ab96fef1abe564366f4d1e6edae27cba09",
                "concat-growing-first": "c600716e3feb171c1c378f2ba6ca2598a2819846d6f4f42f4fbbe001619d40c5",
                "join-growing-later": "8c28cd28d4b1519fa0550db13acfa821c3881ce8d84f65feaf7d6e16f055014f",
                "join-growing-first-nonempty-separator": "932168267f926a8f80593d3bee1967cee569636345762c9d58e9d48199282689",
            },
            "release_cases": {
                "stable-control": "68f530b445400969f1defe214defcec386723dbf992521b55bb17559677662da",
                "join-shrinking-first": "390138f41a2c75f78f536a3b104f0882c174f7d60b4946ba14e0d94eb64411de",
                "join-growing-first": "25e4182a5090798f52b93aa50832786e0f97d3969a0c66cd33eb88294179cf32",
                "concat-growing-first": "568cdb50fa28684b1a4b33ade873a2b425d909d5c4db3b17fc23afe9f8a3a461",
                "join-growing-later": "b601635174fe785f7d3af84bfc2c7eea5054f30a9b3aa810430ac552ab5634f4",
                "join-growing-first-nonempty-separator": "2af0852993235b1334b53448a2b236a4fc29ae36418036e179856222ef0c65f3",
            },
        }
        for profile in ("debug_cases", "release_cases"):
            raw_cases = nightly[profile]
            self.assertEqual(len(raw_cases), len(expected_names))
            cases = {case["name"]: case for case in raw_cases}
            self.assertEqual(set(cases), expected_names)
            self.assertNotIn("miri_ub", {case["outcome"] for case in cases.values()})
            for name in expected_names - {"join-growing-later"}:
                self.assertEqual(
                    cases[name]["outcome"], "exit_zero_no_miri_diagnostic"
                )
                self.assertEqual(cases[name]["exit_code"], 0)
                self.assertFalse(cases[name]["sink_reached"])
                self.assertEqual(cases[name]["log_sha256"], expected_hashes[profile][name])
            self.assertEqual(cases["join-growing-later"]["outcome"], "ordinary_panic")
            self.assertEqual(cases["join-growing-later"]["exit_code"], 101)
            self.assertFalse(cases["join-growing-later"]["sink_reached"])
            self.assertEqual(
                cases["join-growing-later"]["log_sha256"],
                expected_hashes[profile]["join-growing-later"],
            )
            for case in raw_cases:
                self.assertRegex(case["log_sha256"], re.compile(r"^[0-9a-f]{64}$"))

        duplicate = payload["upstream_duplicate_search"]
        self.assertEqual(duplicate["status"], "duplicate_confirmed")
        self.assertEqual(
            duplicate["primary_fix"]["url"],
            "https://github.com/rust-lang/rust/pull/155708",
        )
        self.assertEqual(
            duplicate["primary_fix"]["title"],
            "Fix heap overflow in slice::join caused by misbehaving Borrow",
        )
        self.assertEqual(duplicate["primary_fix"]["state"], "merged")
        self.assertEqual(duplicate["primary_fix"]["milestone"], "1.97.0")
        self.assertEqual(
            duplicate["primary_fix"]["source_commit"],
            "da545d0856a7d7925586accaa9dafe4068ba86ad",
        )
        self.assertEqual(
            duplicate["primary_fix"]["commit_url"],
            "https://github.com/rust-lang/rust/commit/da545d0856a7d7925586accaa9dafe4068ba86ad",
        )
        self.assertEqual(
            duplicate["primary_fix"]["merged_at"], "2026-04-27T05:54:32Z"
        )
        self.assertEqual(
            duplicate["primary_fix"]["merge_commit"],
            "f4043f8c09f0663d8e97d368f7f39f36902dcf09",
        )
        self.assertEqual(
            duplicate["follow_up"]["url"],
            "https://github.com/rust-lang/rust/pull/155858",
        )
        self.assertEqual(
            duplicate["follow_up"]["title"],
            "`slice::join`: borrow only once during length calc",
        )
        self.assertEqual(duplicate["follow_up"]["state"], "merged")
        self.assertEqual(duplicate["follow_up"]["milestone"], "1.97.0")
        self.assertEqual(
            duplicate["follow_up"]["source_commit"],
            "b7424bac39776f3fa3a116cae5599b626c204cc5",
        )
        self.assertEqual(
            duplicate["follow_up"]["commit_url"],
            "https://github.com/rust-lang/rust/commit/b7424bac39776f3fa3a116cae5599b626c204cc5",
        )
        self.assertEqual(
            duplicate["follow_up"]["merged_at"], "2026-04-28T14:55:40Z"
        )
        self.assertEqual(
            duplicate["follow_up"]["merge_commit"],
            "06fa54768c783f362da1fc4fef809e687f0a5c4c",
        )
        self.assertEqual(duplicate["first_fixed_release"], "1.97.0")
        self.assertEqual(payload["disclosure_status"], "not_authorized")


if __name__ == "__main__":
    unittest.main()

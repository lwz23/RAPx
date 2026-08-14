import json
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


def parse_audit_rows():
    lines = AUDIT.read_text().splitlines()
    expected_header = (
        "| family_id | unit | rule | sink | active findings | public roots | "
        "disposition | source evidence / proof pending | PoC evidence |"
    )
    if lines.count(expected_header) != 1:
        raise AssertionError("audit must contain the exact nine-column table header")

    rows = {}
    for line in lines:
        if not line.startswith("| `") or "family_id" in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 9:
            raise AssertionError(f"audit row must contain exactly 9 cells: {line}")
        family_id = cells[0].removeprefix("`").removesuffix("`")
        if family_id in rows:
            raise AssertionError(f"duplicate audit family: {family_id}")
        rows[family_id] = cells
    return rows


def unquote_code_cell(cell):
    if not cell.startswith("`") or not cell.endswith("`"):
        raise AssertionError(f"expected exact code cell, got: {cell}")
    return cell[1:-1]


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
                else "likely_safe"
            )
            disposition = unquote_code_cell(cells[6])
            self.assertIn(disposition, DISPOSITIONS)
            self.assertEqual(disposition, expected_disposition)

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
        self.assertEqual(payload["current"], {"status": "not_started"})
        self.assertEqual(
            payload["upstream_duplicate_search"], {"status": "not_started"}
        )
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


if __name__ == "__main__":
    unittest.main()

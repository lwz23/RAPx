import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V5 = REPO / "artifacts/unsoundaudit-v2/mac/stdlib-pilot/suppression-v5"
V4_REVIEWS = (
    REPO
    / "artifacts/unsoundaudit-v2/mac/stdlib-pilot/performance-v4/noise-audit/family_reviews.json"
)
AUDIT = REPO / "docs/unsoundaudit-v2/STDLIB_ACTIVE_FINDING_AUDIT.md"
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
                "poc_required"
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


if __name__ == "__main__":
    unittest.main()

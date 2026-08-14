"""No-build contract for the generic lightweight relational refinement."""

import importlib.util
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
AUDIT_TEST = REPO / "scripts/unsoundaudit-v2/test_stdlib_bug_validation.py"
RELATIONS = REPO / "rapx/src/analysis/unsoundaudit/relations.rs"
MIR = REPO / "rapx/src/analysis/unsoundaudit/mir.rs"

SAFE_FAMILIES = {
    "254f391b952313b6e20e6af3ae3c4ae72e8ae80806c1d59fefeac9615c0f3823",
    "29430f8db75c5b1ec8546262181c2d19d68c5a66897002358fa8f0913f772a36",
    "30ed5bbde20a9ff57e12d815fe05b8ab72e50f1bd1d27585d8df681aa5f08826",
    "3457eb7b5a97730f0f5d3ca02cdbbdb695540d9b9d3070863062fe1a25ceea13",
    "3778ca92b01cb87fa8cd7083b1983f22a8da27e4cef784aef8449c013e7ccec4",
    "5809e7728fb634f0ec25309fb6495ec110d986749b7181e8dcac1a5f9387a441",
    "7599f48e49559b4127333016dc73824410017efabf60c374f8715f4397b60b30",
    "86efdab2ced0ac5cd56d38b1d0694f621c3f63b158c141bf3aed4fdbaf1b7466",
    "a2f1489e5828edc098d6431934ac5b28980c2620c566aa869894d371a0b4e2df",
    "a89a8c788b489814141b04e59e6c57642a1d9801008af34b28054b32197294e3",
    "ad2ddb26d76a230c7f9563bfb0f97a21907f61aa7fcadb44581c78fae3133708",
    "b4ff0a0c0f3dff624efe6aabe037e3fcef294d337ab63b6649b8c4ae2ebce1e6",
    "b56f1e87137ad83ef3d5b9d4396d33a0aca5158e7cbc8630923f1ce94b5f508b",
    "c0fa33581e1ecbbaedea4a863e29e26e7bd8a67481cf388a0dba19978f96245f",
    "c48ccaae2495c83893f603bc8e117a27c81f0be6abe9c3cc15312b290214f27a",
    "d8f93399becd57112989c5854dc4fbd2a2838c73c62c02fe1f18a26a1b14339f",
    "2b3ed48ba0346bfa7a0f879e8c3cd503897d74e3f26b8de0988e727ae23c2e74",
    "6c9651ab4a70f5d11cc3c222921d90d2702f75f2ca490d9d2349530f6e24df39",
    "9a5d7fe1207d4a1e08da8d4ab0e2ff97d024d335b7a2c2ef985fa9329c6e02be",
    "bf97c840a09beb2b5155bdcf4b5ff5679d100a9cdabe91c3bcddf6a02e2f85e3",
    "4261bfac9813d7755a546cd2d6a4c128400aaa26dc74473874773b5d724d88fc",
    "caabff54fe0affca2dc400cd21e607de93f40a44b3837bae8a8767650b9f441a",
}
HISTORICAL_UB_FAMILY = (
    "560c34406031d113235e81e9d711f31bc34ba2673819fc96abd51ce60f93b4df"
)


def load_audit_helpers():
    spec = importlib.util.spec_from_file_location("stdlib_bug_validation", AUDIT_TEST)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load the active-family audit parser")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RelationalRefinementContractTests(unittest.TestCase):
    def test_target_families_are_exactly_the_closed_generic_proof_population(self):
        audit = load_audit_helpers()
        rows = audit.parse_audit_rows()
        safe_rows = {
            family_id: cells
            for family_id, cells in rows.items()
            if audit.unquote_code_cell(cells[6]) == "confirmed_safe"
        }
        ub_rows = {
            family_id: cells
            for family_id, cells in rows.items()
            if audit.unquote_code_cell(cells[6]) == "confirmed_ub"
        }

        self.assertEqual(set(safe_rows), SAFE_FAMILIES)
        self.assertEqual(sum(int(cells[4]) for cells in safe_rows.values()), 35)
        self.assertEqual(set(ub_rows), {HISTORICAL_UB_FAMILY})
        self.assertEqual(int(ub_rows[HISTORICAL_UB_FAMILY][4]), 2)

    def test_generic_relation_module_exists_without_case_specific_markers(self):
        source = RELATIONS.read_text()
        self.assertIn(
            'RELATIONAL_REFINEMENT_VERSION: &str = "lightweight-relations-v1"',
            source,
        )
        combined = source + "\n" + MIR.read_text()
        forbidden = SAFE_FAMILIES | {
            HISTORICAL_UB_FAMILY,
            "VecDeque",
            "SipHasher",
            "slice_encoded_bytes",
            "get_many_mut",
            "join_generic_copy",
        }
        for marker in forbidden:
            self.assertNotIn(marker, combined)


if __name__ == "__main__":
    unittest.main()

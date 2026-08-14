# Real-project Candidate-quality Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run ten unseen hybrid-500 repository targets with frozen legacy RAP P1–P6 on the Mac, then produce a reproducible Chinese candidate-quality review ledger.

**Architecture:** A standard-library Python CLI compiles immutable study inputs into a manifest, admits source/lock/target inputs without mutating them, records deterministic scans, and validates human review rows. The scanner commit, source snapshots, Cargo locks, profiles, raw artifacts, and sanitized receipts remain distinct.

**Tech Stack:** Python 3 standard library, Git, Cargo nightly-2024-10-12, RAP commit 75ded1f25f3d5a1d81d899c2fe070e34d1f5f8ad, JSON/JSONL, SHA-256.

## Global Constraints

- Execute only on the Mac. Do not change, build, run, or read the active Linux RAP campaign, registry, ledger, or 2,848 historical-success baseline.
- Require the legacy commit and nightly-2024-10-12; reject current nightly-2026-only HEAD.
- Retain the original lockfile, use --locked, Cargo jobs 1, and one repository at a time. Do not run project tests/examples/binaries/Miri.
- Select five unseen star-stratum and five unseen download-stratum repos with seed rapx-v2-real-project-pilot-v1.
- Exclude all 82 IUE canonical repositories plus documented P1–P6 development sources. Never choose replacements after inspecting RAP output.
- Findings are candidate generators, not actual detection/precision/recall/coverage claims. Never use repository-name suppressions.
- Do not commit Mac binaries, target/Cargo caches, raw logs, absolute Mac paths, or credentials.

---

### Task 1: Freeze exact legacy analyzer and three profile invocations

**Files:**
- Create: `scripts/unsoundaudit_v2_real_project_pilot.py`
- Create: `docs/unsoundaudit-v2/real_project_pilot_profile_contract_v1.json`
- Modify: `docs/unsoundaudit-v2/MAC_HANDOFF.md`
- Test: `tests/test_unsoundaudit_v2_real_project_pilot.py`

**Interfaces:**
- Produces a profile contract containing exactly three records: name, relative executable path, SHA-256, and fixed argv.
- Exposes `load_profile_contract(path)` and CLI `verify-profiles --contract --bin-dir --output`.

- [ ] **Step 1: Write failing validation test**

```python
def test_profile_contract_requires_three_profiles(self):
    with self.assertRaisesRegex(ValueError, "exactly three"):
        load_profile_contract(write_json({"schema": "rapx-real-project-profile-contract-v1", "profiles": []}))
```

- [ ] **Step 2: Run it**

Run: `python3 -m unittest tests.test_unsoundaudit_v2_real_project_pilot.RealProjectPilotTests.test_profile_contract_requires_three_profiles -v`

Expected: FAIL because the loader does not exist.

- [ ] **Step 3: Resolve profiles from the real legacy source**

On the Mac, fetch the feature branch, verify `git merge-base --is-ancestor 75ded1f25f3d5a1d81d899c2fe070e34d1f5f8ad origin/feature/unsoundaudit-v2-p1-p6`, detach at that commit, and pass every existing MAC_HANDOFF exact-toolchain preflight. Inspect the implementation and --help output; record the three previously validated profile executable/argv combinations. Do not invent names from the old handoff checkout.

- [ ] **Step 4: Implement contract loader and verifier**

Require the exact schema, three unique non-empty names, relative paths, non-empty argv arrays, and lowercase 64-hex executable hashes. The verifier hashes every referenced binary and checks the exact Rust environment variables from MAC_HANDOFF. It writes only `ok` or `blocked_legacy_preflight`, never substituting another binary.

- [ ] **Step 5: Re-run test and commit**

Run the Step 1 test; expected PASS.

```bash
git add scripts/unsoundaudit_v2_real_project_pilot.py tests/test_unsoundaudit_v2_real_project_pilot.py docs/unsoundaudit-v2/real_project_pilot_profile_contract_v1.json docs/unsoundaudit-v2/MAC_HANDOFF.md
git commit -m "feat: freeze legacy pilot profile contract"
```

### Task 2: Build the blind stratified selection manifest

**Files:**
- Create: `docs/unsoundaudit-v2/real_project_pilot_development_exclusions_v1.json`
- Modify: `scripts/unsoundaudit_v2_real_project_pilot.py`
- Test: `tests/test_unsoundaudit_v2_real_project_pilot.py`
- Create: `docs/unsoundaudit-v2/real_project_pilot_selection_manifest_v1.json`

**Interfaces:**
- Consumes external roster JSONL, final_bug_instances_v3 JSONL, exclusions JSON.
- Exposes `build_selection_manifest(...)` and CLI `select --roster --instances --development-exclusions --output --reserve-per-stratum 25`.
- Produces five selections plus 25 reserves for each stratum, input hashes, seed, and exact excluded names.

- [ ] **Step 1: Write failing selector test**

```python
def test_selector_excludes_known_and_development_sources(self):
    result = build_selection_manifest(mini_roster(), {"known/iue"}, {"dev/input"}, 5, 25)
    self.assertEqual(5, len(result["selected"]["stars_top250"]))
    self.assertEqual(5, len(result["selected"]["downloads_top250"]))
    self.assertNotIn("known/iue", json.dumps(result))
    self.assertNotIn("dev/input", json.dumps(result))
```

- [ ] **Step 2: Run it**

Run: `python3 -m unittest tests.test_unsoundaudit_v2_real_project_pilot.RealProjectPilotTests.test_selector_excludes_known_and_development_sources -v`

Expected: FAIL because the selector does not exist.

- [ ] **Step 3: Implement pure deterministic selection**

Normalize repository names to lowercase owner/name. Keep only `eligible == true` rows, split by their strata keys, exclude IUE/development names, and sort each stratum by SHA-256 of `rapx-v2-real-project-pilot-v1:<repo_id>` then numeric repo ID. Emit sorted UTF-8 JSON with a final newline and no timestamp. This command must not invoke Git, Cargo, rustc, RAP, or network.

Populate the exclusions file from every real repository cited in P1–P6 development documents; each record contains repository, reason, and source document. A reduced fixture must not be misrepresented as a repository.

- [ ] **Step 4: Verify byte identity and commit**

Run selector twice on frozen study inputs, compare the manifests with `cmp`, re-run Step 1 test, then commit the selector, tests, exclusions, and generated manifest.

### Task 3: Admit exact source/lock/target inputs without fallback drift

**Files:**
- Modify: `scripts/unsoundaudit_v2_real_project_pilot.py`
- Test: `tests/test_unsoundaudit_v2_real_project_pilot.py`
- Create at runtime: `real-project-pilot-v1/admissions.jsonl`

**Interfaces:**
- Exposes `admit --manifest --clone-root --result-root --cargo --cutoff 2024-10-12T00:00:00Z`.
- Produces append-only attempt rows and five admitted targets per stratum or a reserve-exhaustion receipt.

- [ ] **Step 1: Write failing replacement test**

```python
def test_admission_uses_next_same_stratum_reserve_after_missing_lock(self):
    result = admit_candidates(manifest_with_reserves(), fake_runner(lockfile=False))
    self.assertEqual("lockfile_missing", result["attempts"][0]["status"])
    self.assertEqual(5, result["admitted_by_stratum"]["stars_top250"])
```

- [ ] **Step 2: Run it**

Run: `python3 -m unittest tests.test_unsoundaudit_v2_real_project_pilot.RealProjectPilotTests.test_admission_uses_next_same_stratum_reserve_after_missing_lock -v`

Expected: FAIL because admission does not exist.

- [ ] **Step 3: Implement admission**

For each candidate/reserve in fixed order: clone with `--no-recurse-submodules`, discover default branch, detach at `git rev-list -1 --before=<cutoff> <default_branch>`, and reject missing commit, root Cargo.lock, metadata failure, or control failure. Never call cargo update.

Use `cargo metadata --locked --no-deps --format-version 1`. Select the first library target, otherwise first non-example binary, sorted by manifest path relative to root, package name, then target name. Run it using fresh per-project CARGO_HOME/CARGO_TARGET_DIR, CARGO_BUILD_JOBS=1, cargo check --locked --jobs 1, package, and lib/bin target. Log argv, exit, source commit/tree hash, Cargo hashes, chosen target, replacement parent, and `build_scripts_or_proc_macros_may_execute=true`.

- [ ] **Step 4: Test and commit**

Run the failing test and a local temporary-repository target-sort test; neither may invoke network or real Cargo. Expected PASS. Commit the updated CLI and tests.

### Task 4: Scan deterministically and test all three profiles for non-zero results

**Files:**
- Modify: `scripts/unsoundaudit_v2_real_project_pilot.py`
- Test: `tests/test_unsoundaudit_v2_real_project_pilot.py`
- Create at runtime: `real-project-pilot-v1/scans.jsonl`, `findings.jsonl`, `scan_summary.json`

**Interfaces:**
- Exposes `scan --admissions --profile-contract --bin-dir --result-root` and `verify-scans --result-root`.
- Produces two canonical-profile scans per target and one profile-stability row for every stable non-zero target.

- [ ] **Step 1: Write failing normalization and deduplication tests**

```python
def test_normalization_ignores_only_run_local_fields(self):
    self.assertEqual(normalize_receipt({"generated_at":"a","findings":[finding(1)]}),
                     normalize_receipt({"generated_at":"b","findings":[finding(1)]}))
    self.assertNotEqual(finding_key(finding(1, sink="a")), finding_key(finding(1, sink="b")))
```

- [ ] **Step 2: Run it**

Run: `python3 -m unittest tests.test_unsoundaudit_v2_real_project_pilot.RealProjectPilotTests.test_normalization_ignores_only_run_local_fields -v`

Expected: FAIL because normalization and finding keys do not exist.

- [ ] **Step 3: Implement scan/verification**

For every admitted target, invoke the canonical profile twice with fresh scan targets and RAP JSON directories. The wrapper must be `$RAPV2_BIN_DIR/cargo-rapx rapx -unsoundaudit -- --locked --jobs 1 --package <package> --lib`; replace only `--lib` with `--bin <target>` for a binary. Require all MAC_HANDOFF exact environment variables, UNSOUND_SCANNER_RAP_JSON_DIR, UNSOUND_SCANNER_PROJECT_ROOT, and the profile executable hash.

Normalize only generated_at, started_at, finished_at, duration_ms, absolute_log_path, and target_dir; sort keys/findings. Any other difference is nondeterministic. For stable non-zero candidates, run the two remaining contract profiles and count only matching root-key/primary-pattern sets as profile stable.

- [ ] **Step 4: Test and commit**

Use fake RAP receipts to prove run-local timestamps normalize but finding changes do not, and profile-unstable candidates are excluded from aggregate counts. Expected PASS without Cargo/RAP. Commit.

### Task 5: Create Chinese review records and an anti-overfitting correction gate

**Files:**
- Create: `docs/unsoundaudit-v2/real_project_pilot_review_codebook_v1.md`
- Modify: `scripts/unsoundaudit_v2_real_project_pilot.py`
- Test: `tests/test_unsoundaudit_v2_real_project_pilot.py`
- Create at runtime: `real-project-pilot-v1/review_ledger.jsonl`, `human_review.md`, `summary.md`

**Interfaces:**
- Exposes `export-review --result-root` and `validate-review --ledger`.
- Produces one review row per stable unique source–sink root cause.

- [ ] **Step 1: Write failing review validation test**

```python
def test_review_rejects_allowlist_and_blank_rationale(self):
    row = {"disposition":"likely_false_positive","error_family":"repository_allowlist","rationale_zh":""}
    with self.assertRaisesRegex(ValueError, "repository_allowlist"):
        validate_review_row(row)
```

- [ ] **Step 2: Run it**

Run: `python3 -m unittest tests.test_unsoundaudit_v2_real_project_pilot.RealProjectPilotTests.test_review_rejects_allowlist_and_blank_rationale -v`

Expected: FAIL because review validation does not exist.

- [ ] **Step 3: Implement review export/validation**

Export Chinese Markdown with repository URL, detached revision, target, P1–P6, source/sink spans, call path, source snippet hashes, and raw-receipt link. Require one disposition: new_iue_candidate, likely_false_positive, unresolved, duplicate, or tool_or_receipt_failure. A likely false positive requires one generic family: source_misattribution, missing_dominating_validation, interprocedural_summary_imprecision, sink_contract_imprecision, primary_pattern_precedence, or deduplication. Reject repository_allowlist, blank evidence/rationale, and duplicate references to unknown root keys.

Generate summary counts for selected, admitted, control failures, analyzer failures, deterministic scans, profile-stable findings, unique roots, and dispositions. Use “候选质量 pilot” and “静态候选”; do not use actual detection, precision, recall, or coverage.

- [ ] **Step 4: Test and commit**

Run all Python tests and grep summary for prohibited claims; expected tests PASS and grep empty. Commit CLI, test, and codebook.

### Task 6: Execute on the Mac and freeze sanitized evidence

**Files:**
- Create locally on Mac: `real-project-pilot-v1/`
- Create: `docs/unsoundaudit-v2/real_project_pilot_v1_receipt_manifest.json`
- Modify after Mac return: `/home/lwz/unsound_scanner/TASK_CONTEXT.md`

- [ ] **Step 1: Verify custody**

Run profile verification and manifest hash verification before any third-party clone. On any mismatch, write blocked_legacy_preflight and stop; do not change toolchain, rules, or locks.

- [ ] **Step 2: Admit, scan, and preserve all failures**

Run admit, scan, and verify-scans in manifest order. Only reserves from the same precommitted stratum may replace failed admissions. Do not re-run with changed rules.

- [ ] **Step 3: Review all stable unique candidates**

Export and complete the review ledger, then validate it. Do not file upstream issues, modify third-party code, or modify RAP in this pilot.

- [ ] **Step 4: Freeze shareable metadata**

Create receipt manifest with SHA-256s of selection, profiles, admissions, scans, findings, review ledger, and summary. Verify it has no absolute Mac path/binary/target/cache/raw log/credential. Commit only source/docs/sanitized receipts.

- [ ] **Step 5: Gate future correction**

Do not edit unsoundaudit.rs now. A separate correction package is allowed only for recurring source-independent error families or systemic analyzer defects; it must add paired fixtures, run all 27 fixtures, replay this pilot, and evaluate a new hold-out sample.

## Plan Self-Review

- Tasks 1–6 cover analyzer custody, blind selection, source/lock admission, deterministic scans, triage, and Mac-only handoff.
- Manifest feeds admit; admissions feed scan; stable findings feed review; review ledger feeds correction gate.
- Exact profile argv/hashes are derived from verified 75ded1 instead of guessed, so toolchain/profile drift fails closed.

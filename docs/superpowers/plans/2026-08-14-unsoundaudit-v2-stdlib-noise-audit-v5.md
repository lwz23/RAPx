# Standard-Library Noise Audit and v5 Implementation Plan

> **For agentic workers:** Implement this plan task-by-task with TDD, one Cargo job, fresh targets for formal gates, and no propagation or fixed-point caps.

**Goal:** Turn the immutable v4 standard-library populations into a deterministic single-review diagnostic audit, implement only evidence-bound generic repairs or exact rustc-bound standard-library suppressions, and freeze a deterministic v5 campaign that retains the existing performance limits.

**Architecture:** A standalone Python tool clusters normalized findings and produces a deterministic 124-finding diagnostic panel, validates one-reviewer A–U labels, and renders active/suppressed family reports. Rust suppression is a post-analysis filter over fully constructed and deduplicated findings; it is compiled into RAPx, matches only an exact rustc commit and canonical finding fields, and writes a separate audit ledger for pilot runs. v4 artifacts remain byte-for-byte immutable, and all semantic changes are tested and scanned under v5.

**Tech Stack:** Rust nightly `2024-10-12`, Python 3 standard library, JSON/JSONL, SHA-256, existing `rap-unit-v2` schema and stdlib pilot runner.

---

### Task 1: Deterministic noise-family tooling

**Files:**
- Create: `scripts/unsoundaudit-v2/noise_audit.py`
- Create: `scripts/unsoundaudit-v2/test_noise_audit.py`
- Create: `artifacts/unsoundaudit-v2/mac/stdlib-pilot/performance-v4/noise-audit/clusters.json`
- Create: `artifacts/unsoundaudit-v2/mac/stdlib-pilot/performance-v4/noise-audit/audit_sample.json`

- [ ] Add RED tests for the exact family key, stable SHA-256 family ID, input permutation stability, shortest/longest interprocedural representatives, inclusion of every P1/raw-read/intraprocedural/recursive finding, and the current v4 totals of 876 findings, 53 sinks, 77 families, and 124 diagnostic rows.
- [ ] Implement closed-schema readers that bind every output to the three normalized population hashes, three frozen sample hashes, protocol hash, and v4 sample seed.
- [ ] Implement the canonical family key:

```text
(unit, rule_id, sink, first_contract_failure, predicates,
 subject, source_kind, propagation_depth, validation_status)
```

- [ ] Select one shortest representative for every family, one longest representative for every interprocedural family, and every P1/raw-read/intraprocedural/recursive finding. Resolve ties by `SHA256(seed || canonical_causal_key_json)` and then `finding_id`.
- [ ] Emit canonical newline-terminated JSON and verify reversed input yields byte-identical output.
- [ ] Run `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_noise_audit.py` from `scripts/unsoundaudit-v2` and require all tests to pass.

### Task 2: Single-review diagnostic labels and report

**Files:**
- Modify: `scripts/unsoundaudit-v2/noise_audit.py`
- Modify: `scripts/unsoundaudit-v2/test_noise_audit.py`
- Create: `artifacts/unsoundaudit-v2/mac/stdlib-pilot/performance-v4/noise-audit/labels.json`
- Create: `artifacts/unsoundaudit-v2/mac/stdlib-pilot/performance-v4/noise-audit/audit_report.json`

- [ ] Add RED tests for closed label keys, exact sample coverage, A–U labels, complete interprocedural chain audits, allowed root-cause/disposition enums, and rejection of kappa/adjudication/formal precision claims.
- [ ] Require one reviewer ID, Rust unsafe-contract experience flag, rationale, evidence spans, fabricated-edge flag, root cause, family disposition, and repair disposition for every selected finding.
- [ ] Audit each finding in deterministic order. Inspect the public root/source, every resolved local call and mapping, sink provenance, CFG/validation/write invalidation, and recursive-cycle evidence.
- [ ] Apply the diagnostic early stop: any fabricated edge, or at least three D/E labels among a unit's first 20 diagnostic rows, stops broader labeling and creates a structural-repair decision.
- [ ] Produce counts by A–U, family, rule, depth, root cause, and population weight. Do not emit Wilson intervals, Cohen's kappa, or a formal pass decision.
- [ ] Classify every family as generic repair, exact stdlib invariant suppression, reporting-only fanout, retain, or insufficient evidence.

### Task 3: Evidence-driven generic repairs

**Files:**
- Modify only the relevant modules under `rapx/src/analysis/unsoundaudit/`
- Add focused fixtures/tests under the existing unsoundaudit test layout

- [ ] For every D/E family, first add a minimal non-stdlib RED reproducer that locks the incorrect source, binding, CFG, call edge, witness, or identity behavior.
- [ ] For every generic C/B repair, add positive and wrong-subject, wrong-collection, write-after-check, diamond, loop, normal/unwind negative controls.
- [ ] Implement only generic program facts or contract propagation. Do not match standard-library paths or names inside MIR normalization/transfer.
- [ ] Run focused tests and the full exact-nightly Rust library gate after each repair block.
- [ ] Do not suppress A, unresolved B, U, D, or E findings.

### Task 4: Exact built-in stdlib suppression and ledger

**Files:**
- Create: `rapx/src/analysis/unsoundaudit/stdlib_suppressions.rs`
- Modify: `rapx/src/analysis/unsoundaudit.rs`
- Modify: `scripts/unsoundaudit-v2/run_stdlib_pilot.py`
- Modify: `scripts/unsoundaudit-v2/test_stdlib_pilot.py`

- [ ] Add Rust RED tests for exact matching and rejection of wrong rustc commit, crate, rule, sink, first failure, obligation, source kind, duplicate entries, and non-C evidence.
- [ ] Represent each compiled entry with exact strings only:

```rust
struct StdlibSuppression {
    rustc_commit: &'static str,
    crate_key: &'static str,
    rule_id: RuleId,
    sink_token: &'static str,
    first_failure_token: &'static str,
    canonical_obligation: &'static str,
    source_kind: SourceKind,
    family_id: &'static str,
    evidence_sha256: &'static str,
    rationale: &'static str,
}
```

- [ ] Apply suppression only after `deduplicate_findings`. Compute every original finding ID before filtering; do not alter MIR extraction, summary fixed points, routes, causal keys, or IDs.
- [ ] When `UNSOUND_SCANNER_SUPPRESSION_LEDGER_OUT` is set, atomically write a closed-schema ledger containing the exact toolchain, allowlist identity, complete suppressed finding JSON, and active/raw/suppressed counts.
- [ ] Extend only the v5 runner contract to require a fresh ledger for every selected stdlib scan. Reject a missing, stale, malformed, path-bearing, or non-matching ledger.
- [ ] Keep `rap-unit-v2` unchanged. Counts in the receipt describe active findings; the ledger separately binds every removed finding.

### Task 5: Family presentation

**Files:**
- Modify: `scripts/unsoundaudit-v2/noise_audit.py`
- Modify: `scripts/unsoundaudit-v2/test_noise_audit.py`

- [ ] Add RED tests for active/suppressed population reconciliation, public-root fanout, shortest/longest real witness selection, and reversed-input byte identity.
- [ ] Emit a stable family report with `family_id`, active/suppressed counts, public-root count, representative shortest/longest finding IDs, rule/depth, and audit disposition.
- [ ] Preserve every active raw finding unchanged and expose suppressed full findings only through the ledger/evidence bundle.

### Task 6: Development gates and semantic-delta audit

**Files:**
- Modify: `TASK_CONTEXT.md`
- Create temporary evidence outside the repository

- [ ] Run the full exact-nightly Rust and Python suites with one job and fresh targets.
- [ ] Build debug, stock release, and `rapx-perf` binaries. Require the same canonical active finding set for each profile.
- [ ] Run challenge 27 twice and original 27 twice. Require the historical canonical hashes unless an approved generic repair intentionally changes a fixture oracle; stdlib suppressions must never match fixtures.
- [ ] Compare v4 raw findings with the new raw-before-suppression set. Every disappearance must be explained by a regression-tested generic proof or an exact suppression-ledger row; unexplained loss of any A/B finding is a blocker.
- [ ] Run one development `core` scan with 900 seconds/8 GiB. Only after success run `alloc` with 1,800 seconds/10 GiB and `std` with 3,600 seconds/12 GiB.
- [ ] Require active total at most 500 and active density at most 0.5/KLoC. If defensible repairs cannot meet both lines, stop with an explicit usability/precision failure rather than expanding uncertain suppressions.

### Task 7: Freeze and execute v5

**Files:**
- Create: `tests/unsoundaudit-v2/stdlib_pilot_protocol_v5.json`
- Modify: `scripts/unsoundaudit-v2/run_stdlib_pilot.py`
- Modify: `scripts/unsoundaudit-v2/test_stdlib_pilot.py`
- Create: `artifacts/unsoundaudit-v2/mac/stdlib-pilot/precision-v5/`

- [ ] Commit the final analyzer source, rebuild metadata-correct `rapx-perf` binaries, and bind v5 to that exact source/toolchain/binary profile.
- [ ] Preserve dev target `cargo check`, trace-off execution, two fresh scans, and the v4 15/30/60-minute and 8/10/12-GiB limits.
- [ ] Run core twice, then alloc twice, then std twice. Require exactly one selected receipt and one suppression ledger per scan, and byte-identical normalized receipt, ledger, and family report pairs.
- [ ] Re-run all synthetic gates on the exact committed source used by v5.

### Task 8: Publish the bounded report

**Files:**
- Create: `docs/unsoundaudit-v2/STDLIB_PILOT_REVIEW.md`
- Create: `artifacts/unsoundaudit-v2/mac/stdlib-pilot/precision-v5/sha256_manifest.json`
- Modify: `TASK_CONTEXT.md`

- [ ] Report v4 as immutable performance success plus precision pause, and v5 as deterministic performance plus single-review development evidence.
- [ ] Report raw, hard-suppressed, and active counts separately for every unit/rule/family; never present post-suppression density as unadjusted precision.
- [ ] State that the review is single-person diagnostic evidence without kappa/adjudication, P3/P5/P6 are unmeasured in the stdlib population, cross-crate propagation is unsupported, and exact rustc-bound suppressions do not generalize to other Rust versions or projects.
- [ ] Hash every committed protocol, source identity, population, ledger, family report, label, audit report, and final report. Reject raw logs, targets, binaries, caches, drivers, rust-src copies, or absolute paths in committed evidence.
- [ ] Run `git diff --check`, verify a clean expected worktree, commit the evidence, and guarded-push only if the remote tip is a local ancestor.

## Stop Conditions

- Never add path, fact, origin, version, recursion, or fixed-point caps.
- Stop immediately on a fabricated edge, schema drift, nondeterminism, hidden truncation, unexplained A/B disappearance, or resource-limit failure.
- Do not convert uncertain U/B evidence into a suppression to satisfy the count target.
- If safe, evidence-bound changes cannot reach both usability lines, publish the bounded failure instead of continuing to tune against rust-src.

# RAPx Historical Recall and Real-Project Usability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make historical vulnerable crates reliably analyzable with the pinned 2024 toolchain, repair confirmed in-scope false negatives, and add only general finite P1–P6 subrules that pass real-project noise gates.

**Architecture:** Preserve replay v1 as immutable evidence and add a replay-v2 intake/execution layer with exact vulnerable-package selection, isolated generated lockfiles, per-unit statuses, and partial-receipt accounting. Repair analyzer semantics independently, starting with immediate invalid-value sinks, then promote one finite rule extension at a time only after differential fixtures and real-project family review.

**Tech Stack:** Python 3 standard library, Cargo/rustc nightly-2024-10-12, Rust MIR/rustc_middle, JSON/JSONL, existing rap-unit-v2 schema and deterministic receipt tooling.

---

### Task 1: Freeze replay-v2 contracts

**Files:**
- Create: `tests/unsoundaudit-v2/historical_replay_v2_cases.json`
- Create: `tests/test_unsoundaudit_v2_fast_historical_replay_v2.py`
- Create: `scripts/unsoundaudit_v2_fast_historical_replay_v2.py`

- [ ] Write failing no-build tests for the exact manifest key set, exact package/target binding, nullable failed counts, locator hash validation, and immutable v1 evidence.
- [ ] Run `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v tests/test_unsoundaudit_v2_fast_historical_replay_v2.py` and confirm missing v2 APIs/files are the only failures.
- [ ] Implement strict manifest loading and status rendering with `null`/`—` for every non-success unit.
- [ ] Rerun the focused suite and commit the contract block.

### Task 2: Make lock and target admission usable

**Files:**
- Modify: `scripts/unsoundaudit_v2_fast_historical_replay_v2.py`
- Test: `tests/test_unsoundaudit_v2_fast_historical_replay_v2.py`

- [ ] Add RED fixtures for a lockless library, a virtual workspace with default members, an explicitly selected vulnerable member, and an invalid escaping manifest.
- [ ] Implement disposable-copy lock generation with the pinned Cargo, hash the generated lock, and run both control and RAP with `--locked` without dirtying the source checkout.
- [ ] Implement Cargo-default target discovery plus explicit historical target selection; keep `workspace-all` opt-in.
- [ ] Run focused tests and commit.

### Task 3: Preserve partial receipts and fail closed

**Files:**
- Modify: `scripts/unsoundaudit_v2_fast_historical_replay_v2.py`
- Test: `tests/test_unsoundaudit_v2_fast_historical_replay_v2.py`

- [ ] Add RED tests for timeout after one local receipt, duplicate selected receipts, malformed receipts, external dependency receipts, and missing selected receipts.
- [ ] Implement per-unit status records and repository-local receipt aggregation. Preserve partial receipts on timeout while keeping the overall case incomplete.
- [ ] Enforce process-group timeout, 8 GiB physical-footprint failure, append-only output, and deterministic JSON/Markdown ordering.
- [ ] Run focused and complete no-build Python suites; commit.

### Task 4: Repair unchecked-UTF8 immediate invalidity

**Files:**
- Modify: `rapx/src/analysis/unsoundaudit/mir.rs`
- Test: `rapx/src/analysis/unsoundaudit/mir.rs`

- [ ] Add a production-shaped RED for public bytes passed to `from_utf8_unchecked`, wrapped through an opaque external call, and returned as a wrapper.
- [ ] Add negative controls for the same validated bytes, wrong bytes, post-check write, diamond, loop, normal edge, and unwind edge.
- [ ] Change the requirement subject to the input bytes and mark the invalid-value sink immediate (`return_exposure=false`).
- [ ] Run exact focused Rust tests and commit.

### Task 5: Add candidate-lifecycle diagnostics

**Files:**
- Modify: `rapx/src/analysis/unsoundaudit/mir.rs`
- Test: `rapx/src/analysis/unsoundaudit/mir.rs`

- [ ] Add cfg(test)-only counters for body inclusion, sink recognition, requirement creation, public-root selection, local discharge, return-exposure filtering, dedup, and emission.
- [ ] Lock the Gitoxide-shaped lifecycle: recognized and emitted, never filtered by opaque return provenance.
- [ ] Confirm counters do not enter production types, receipts, causal keys, or finding IDs; run focused tests and commit.

### Task 6: Validate historical locators and Tier-A recall

**Files:**
- Modify: `tests/unsoundaudit-v2/historical_replay_v2_cases.json`
- Modify: `tests/test_unsoundaudit_v2_fast_historical_replay_v2.py`

- [ ] Resolve every fix/pre-fix locator from an official commit parent, official tag, or published source and bind vulnerable/fixed snippet and diff hashes.
- [ ] Mark unverifiable inputs `invalid_input`; correct expected RuleIds, including Gitoxide as `P3.unchecked_utf8`.
- [ ] Add pre-fix/fixed differential cases for every validated existing-rule mechanism. Tier-A pre-fix must emit the expected root and the fixed revision must not.
- [ ] Run focused replay tests and commit.

### Task 7: Generalize definite-uninitialized assume-init

**Files:**
- Modify: `rapx/src/analysis/unsoundaudit/summary.rs`
- Modify: `rapx/src/analysis/unsoundaudit/mir.rs`
- Modify: `tests/unsoundaudit-v2/rap-unit-v2.schema.json`

- [ ] Add `P3.assume_init_uninitialized` RED/GREEN tests for arbitrary `T`, complete/partial field and array writes, diamond, loop, opaque write, and unwind.
- [ ] Generalize the existing finite initializedness facts without paths, counters, caps, or case-specific identities.
- [ ] Add the RuleId additively to rap-unit-v2 and validator tests; record old/new schema hashes.
- [ ] Run exact Rust, Python, challenge, original, and profile-equivalence gates; commit only if all pass.

### Task 8: Model FFI out invalid scalar values

**Files:**
- Modify: `rapx/src/analysis/unsoundaudit/summary.rs`
- Modify: `rapx/src/analysis/unsoundaudit/mir.rs`
- Modify: `tests/unsoundaudit-v2/rap-unit-v2.schema.json`

- [ ] Add `P6.ffi_out_invalid_value` RED/GREEN tests for unchecked enum/scalar exposure and exact TryFrom, complete match, range, wrong-value, write, merge, and unwind controls.
- [ ] Carry finite FFI-output provenance and discriminant/range validation through existing version/tombstone joins.
- [ ] Extend schema/validator additively and run all semantic/profile gates; commit only if all pass.

### Task 9: Evaluate and conditionally promote union active-field validity

**Files:**
- Modify: `rapx/src/analysis/unsoundaudit/summary.rs`
- Modify: `rapx/src/analysis/unsoundaudit/mir.rs`
- Modify: `tests/unsoundaudit-v2/rap-unit-v2.schema.json`

- [ ] Build a finite RED test matrix for union construction, active-field write, unknown public active field, CFG merge, all-bit-valid fields, invalid-value fields, and unwind.
- [ ] Implement `P2.union_field_validity` only with static union/field identities and may-active sets; no path history or external semantic model.
- [ ] Run the full gates and the real-project new-family audit. If any fabricated edge or unbounded/noisy structural family remains, revert this task and mark the libc mechanism `validated_out_of_scope`.
- [ ] Commit either the passing implementation or only the explicit out-of-scope evidence.

### Task 10: Replay, noise audit, and freeze v2 evidence

**Files:**
- Create: `docs/unsoundaudit-v2/fast_historical_replay_v2_evidence/`
- Modify: `TASK_CONTEXT.md`

- [ ] Build metadata-correct debug, release, and rapx-perf binaries with the pinned 2024 toolchain.
- [ ] Run exact Rust/Python gates, challenge 27 twice, original 27 twice, and canonical profile comparison.
- [ ] Rescan the six prior zero-result projects and current core; audit every family introduced by a new subrule and reject structural errors rather than suppressing them.
- [ ] Run every validated historical pre-fix/fixed package twice under 15 minutes and 8 GiB per package; require byte-identical successful receipts, tables, manifests, and generated-lock hashes.
- [ ] Publish separate counts for invalid input, not run, partial, successful zero, detected, and validated out of scope. Compute recall only over validated, successful, in-scope cases.
- [ ] Update TASK_CONTEXT with decisions, rejected alternatives, evidence hashes, limitations, and the unchanged v1 boundary; commit the final evidence.

# UnsoundAudit v2 Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the fixture-closed P1–P6 engine against adversarial CFG and interprocedural cases, then run a deterministic, bounded `core`/`alloc`/`std` robustness pilot without expanding the approved registry.

**Architecture:** Keep the original 27-fixture suite immutable and add a separate challenge suite. Replace block-order-dependent value extraction with a finite per-block data-flow solution, propagate bound validations and structural call outputs through finite summaries, and decide path safety before choosing a witness. Only after all synthetic gates pass, use pinned `-Z build-std` drivers to compile each standard-library crate as the local analysis unit and produce fail-closed receipts for deterministic review.

**Tech Stack:** Rust nightly `2024-10-12` and rustc-private MIR APIs, Cargo `-Z build-std`, Python 3 standard library, JSON receipts, Git, SHA-256.

---

## File Map

| Path | Responsibility |
|---|---|
| `tests/unsoundaudit-v2/challenge_manifest.json` | Independent adversarial oracle and file hashes; never changes the frozen 27-case totals |
| `tests/unsoundaudit-v2/challenges/<case>/` | Dependency-free library crates for actual-MIR challenge cases |
| `scripts/unsoundaudit-v2/run_challenge_suite.py` | Serial, exact-toolchain challenge runner using fresh control and scan targets |
| `scripts/unsoundaudit-v2/test_challenge_contract.py` | Standard-library-only tests for challenge manifest, runner, and normalized output |
| `rapx/src/analysis/unsoundaudit/dataflow.rs` | Finite intraprocedural CFG state, joins, transfer, path predicates, and write versions |
| `rapx/src/analysis/unsoundaudit/summary.rs` | Finite validation, return, and local out facts transferred across static local calls |
| `rapx/src/analysis/unsoundaudit/mir.rs` | MIR extraction, engine orchestration, requirement discharge, and witness construction |
| `scripts/unsoundaudit-v2/run_stdlib_pilot.py` | Exact `build-std` routing, resource gates, receipt collection, and per-unit normalization |
| `scripts/unsoundaudit-v2/review_sample.py` | Deterministic stratified sample and label validation |
| `scripts/unsoundaudit-v2/test_stdlib_pilot.py` | No-build tests for routing, sanitization, limits, sampling, and statistics |
| `tests/unsoundaudit-v2/stdlib_pilot_protocol.json` | Pinned commits, unit order, sampling seed, thresholds, and resource limits |
| `artifacts/unsoundaudit-v2/mac/stdlib-pilot/` | Small normalized receipts, hashes, samples, labels, and review summary only |
| `TASK_CONTEXT.md` | Current gate, decisions, failures, and rejected scope extensions |

## Work Package 1: Freeze the Independent Challenge Contract

### Task 1: Add a generic closed-manifest verifier

**Files:**
- Modify: `scripts/unsoundaudit-v2/run_fixture_suite.py:250-368`
- Modify: `scripts/unsoundaudit-v2/test_contract_hardening.py`

- [ ] **Step 1: Write failing standard-library tests for a parameterized contract**

Add tests that call a new `verify_case_contract(...)` with the frozen fixture constants and with a two-case temporary manifest. Assert exact directory equality, tracked-file enforcement, per-file SHA-256, role totals, and aggregate counts. Also assert that the existing `verify_contract(...)` still rejects a 28th frozen fixture.

```python
def test_generic_contract_keeps_frozen_fixture_count_closed(self):
    case_ids = verify_case_contract(
        repo_root=self.repo,
        case_root=self.case_root,
        manifest_path=self.manifest_path,
        manifest=self.manifest,
        expected_schema="unsoundaudit-v2-challenge-manifest-v1",
        expected_count=2,
        expected_counts=ZERO_COUNTS,
        require_legacy_provenance=False,
    )
    self.assertEqual(case_ids, ["a", "b"])
```

- [ ] **Step 2: Run the contract tests and capture RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/unsoundaudit-v2/test_contract_hardening.py
```

Expected: FAIL because `verify_case_contract` does not exist; the existing 14 tests continue to pass before the new failure.

- [ ] **Step 3: Extract the generic verifier without relaxing the frozen wrapper**

Implement a parameterized verifier for case IDs, directories, four required tracked files, dependency prohibition, row/oracle equality, and hashes. Keep `verify_contract(...)` as a wrapper that supplies exactly 27, the frozen schema, the frozen aggregate, and mandatory 31-file legacy provenance.

```python
def verify_contract(repo_root, fixture_root, manifest_path, manifest):
    if manifest.get("schema_version") != "unsoundaudit-v2-fixture-manifest-v1":
        fail("fixture manifest schema drifted")
    verify_legacy_provenance(...)
    return verify_case_contract(
        repo_root, fixture_root, manifest_path, manifest,
        expected_schema="unsoundaudit-v2-fixture-manifest-v1",
        expected_count=EXPECTED_FIXTURE_COUNT,
        expected_counts=EXPECTED_COUNTS,
        require_legacy_provenance=True,
    )
```

- [ ] **Step 4: Re-run no-build tests**

Expected: all contract-hardening tests PASS and production verification prints 27 cases, 31 legacy files, and the unchanged aggregate.

- [ ] **Step 5: Commit the verifier refactor**

```bash
git add scripts/unsoundaudit-v2/run_fixture_suite.py scripts/unsoundaudit-v2/test_contract_hardening.py
git commit -m "test(unsoundaudit-v2): parameterize closed case contracts"
```

### Task 2: Freeze challenge manifest v1 and runner

**Files:**
- Create: `tests/unsoundaudit-v2/challenge_manifest.json`
- Create: `tests/unsoundaudit-v2/challenges/`
- Create: `scripts/unsoundaudit-v2/run_challenge_suite.py`
- Create: `scripts/unsoundaudit-v2/test_challenge_contract.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write runner contract tests before adding cases**

Test that the runner requires schema `unsoundaudit-v2-challenge-manifest-v1`, rejects legacy provenance, requires exactly the manifest case set, sorts output by case ID, keeps original fixture paths out of the challenge root, and cannot run without the exact environment enforced by `verify_environment`.

- [ ] **Step 2: Run tests and capture RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/unsoundaudit-v2/test_challenge_contract.py
```

Expected: FAIL because the runner and manifest are absent.

- [ ] **Step 3: Add the serial challenge runner**

Reuse `verify_environment`, `validate_receipt`, `validate_oracle`, `run_command`, and `fresh_target`. The command sequence per case remains:

```text
<exact cargo> check --lib --locked --jobs 1
<cargo-rapx> rapx -unsoundaudit -- --lib --locked --jobs 1
```

Use separate fresh targets and one empty receipt directory per command. Emit `rap-challenge-suite-v1` with sorted cases and aggregate counts; do not compare it to the frozen 27 totals.

- [ ] **Step 4: Add the first frozen case set**

Create dependency-free fixtures with tracked `Cargo.lock` and exact `fixture.json` oracles:

| Case ID | Oracle |
|---|---|
| `cfg_diamond_one_branch_guard_positive` | P1=1 |
| `cfg_diamond_all_paths_guard_negative` | all zero |
| `cfg_guard_then_one_path_write_positive` | P1=1 |
| `cfg_loop_reassignment_positive` | P1=1 |
| `cfg_loop_invariant_guard_negative` | all zero |
| `call_guard_in_middle_negative` | all zero |
| `call_guard_in_sink_negative` | all zero |
| `call_wrong_value_guard_positive` | P1=1 |
| `call_wrappers_0_hop_positive` | P1=1 |
| `call_wrappers_1_hop_positive` | P1=1 |
| `call_wrappers_3_hop_positive` | P1=1 |
| `call_wrappers_5_hop_positive` | P1=1 |
| `call_constant_return_negative` | all zero |
| `call_local_out_positive` | P1=1 |
| `recursion_direct_positive` | P1=1, one cycle token |
| `recursion_mutual_positive` | P1=1, one cycle token |
| `opaque_fn_pointer_negative` | all zero |
| `opaque_closure_negative` | all zero |
| `opaque_dyn_negative` | all zero |
| `opaque_external_negative` | all zero |
| `guard_bounds_lt_negative` | all zero |
| `guard_bounds_assert_negative` | all zero |
| `guard_nonempty_len_negative` | all zero |
| `guard_nonnull_match_negative` | all zero |
| `guard_range_checked_add_negative` | all zero |
| `span_macro_stable_positive` | P1=1 with macro call-site path |
| `span_include_stable_positive` | P1=1 with distinct included-file path |

The guard cases express only the already frozen predicates. If a case requires a new predicate or sink to make its oracle meaningful, omit it and record a scope question.

- [ ] **Step 5: Generate locks with pinned Cargo and freeze hashes**

Run `cargo generate-lockfile --offline` for each dependency-free fixture using the exact Cargo. Generate manifest rows only after all four files exist, then verify every tracked path and SHA through the production verifier.

- [ ] **Step 6: Run no-build challenge contract tests**

Expected: all tests PASS; original fixture verifier still reports exactly 27 cases and the original aggregate.

- [ ] **Step 7: Commit the frozen challenge contract**

```bash
git add .gitignore tests/unsoundaudit-v2/challenge_manifest.json tests/unsoundaudit-v2/challenges scripts/unsoundaudit-v2/run_challenge_suite.py scripts/unsoundaudit-v2/test_challenge_contract.py
git commit -m "test(unsoundaudit-v2): freeze adversarial challenge contract"
```

### Task 3: Record the challenge RED baseline

**Files:**
- Create: `artifacts/unsoundaudit-v2/mac/robustness/challenge_red_receipt.json`
- Modify: `TASK_CONTEXT.md`

- [ ] **Step 1: Build the committed implementation in a fresh target**

Use the exact environment from the Mac handoff and:

```bash
<exact cargo> build --manifest-path rapx/Cargo.toml --locked --bins --jobs 1
```

Expected: exit 0 and sibling `cargo-rapx`/`rapx` binaries.

- [ ] **Step 2: Run the challenge suite once**

Use a new output root and the same exact environment. Expected: nonzero or oracle failures in known high-risk CFG/callee-validation/out/recursion cases. No target function executes.

- [ ] **Step 3: Write a sanitized RED receipt**

Record tested commit, exact tool commits, manifest SHA, failed case IDs, observed counts, runner exit, and local raw-log SHA values. Do not include absolute paths or infer a code fix from the failure.

- [ ] **Step 4: Confirm the original suite remains green**

Run all 27 frozen fixtures once in fresh targets. Expected: 27/27 and aggregate `3/2/2/2/1/2`.

- [ ] **Step 5: Commit RED evidence**

```bash
git add artifacts/unsoundaudit-v2/mac/robustness/challenge_red_receipt.json TASK_CONTEXT.md
git commit -m "test(unsoundaudit-v2): record robustness challenge RED"
```

## Work Package 2: Finite Intraprocedural Data Flow

### Task 4: Add a finite CFG lattice and pure tests

**Files:**
- Create: `rapx/src/analysis/unsoundaudit/dataflow.rs`
- Modify: `rapx/src/analysis/unsoundaudit.rs`

- [ ] **Step 1: Write pure RED tests for joins and loops**

Define tests for entry formal facts, branch-specific writes, diamond merge, loop convergence, place-version increment, and deterministic block order. Assert that join is monotone and idempotent and that a loop reaches equality without a round cap.

```rust
#[test]
fn diamond_join_keeps_may_origins_and_must_validations_separate() {
    let left = BlockState::with_validation(binding("index", "slice"));
    let right = BlockState::empty();
    let joined = BlockState::join_predecessors([&left, &right]);
    assert!(joined.may_values.contains_key(&place("index")));
    assert!(joined.must_validations.is_empty());
}
```

- [ ] **Step 2: Run the exact fresh-target library test and capture RED**

Expected: compile failure because `dataflow` and `BlockState` do not exist.

- [ ] **Step 3: Implement the minimal lattice**

Use ordered maps/sets only:

```rust
struct BlockState {
    may_values: BTreeMap<PlaceKey, ValueFacts>,
    must_validations: BTreeSet<BoundValidation>,
    versions: BTreeMap<PlaceKey, u32>,
}

struct DataflowResult {
    entry: IndexVec<BasicBlock, BlockState>,
    before: BTreeMap<(BasicBlock, usize), BlockState>,
}
```

Join values by union, validations by predecessor intersection, and versions by maximum. A relevant write increments the canonical storage version and removes bound validations for that storage. Worklist iteration stops only when no entry state changes.

- [ ] **Step 4: Run pure tests in a fresh target**

Expected: all new data-flow tests and the existing 26 tests PASS.

- [ ] **Step 5: Commit the finite lattice**

```bash
git add rapx/src/analysis/unsoundaudit.rs rapx/src/analysis/unsoundaudit/dataflow.rs
git commit -m "feat(unsoundaudit-v2): add finite CFG dataflow lattice"
```

### Task 5: Move MIR transfer into the CFG solver

**Files:**
- Modify: `rapx/src/analysis/unsoundaudit/dataflow.rs`
- Modify: `rapx/src/analysis/unsoundaudit/mir.rs:411-855`

- [ ] **Step 1: Add failing extraction tests**

Add pure transfer tests for assignment, ref/raw pointer, cast, length, compare, local call result, opaque mutable write, and switch branch refinement. Add end-to-end subset assertions for the five CFG challenge cases.

- [ ] **Step 2: Capture RED on the CFG subset**

Run only the five `cfg_*` cases. Expected: at least one oracle mismatch under the old global `facts.values` implementation.

- [ ] **Step 3: Implement two-phase body extraction**

First collect immutable operation descriptors and CFG successors from MIR. Then solve block entry states with the data-flow module. Finally derive requirements, comparisons, predicates, writes, and return facts from the program-point states. Remove semantic dependence on MIR basic-block enumeration order.

- [ ] **Step 4: Run pure tests, CFG subset, and frozen 27**

Expected: data-flow tests PASS; all five CFG cases match their oracles; frozen suite remains 27/27 with unchanged aggregate.

- [ ] **Step 5: Commit CFG extraction**

```bash
git add rapx/src/analysis/unsoundaudit/dataflow.rs rapx/src/analysis/unsoundaudit/mir.rs
git commit -m "fix(unsoundaudit-v2): solve MIR facts over the CFG"
```

## Work Package 3: Interprocedural Contracts and Path Safety

### Task 6: Propagate bound validations through summaries

**Files:**
- Modify: `rapx/src/analysis/unsoundaudit/summary.rs:226-483`
- Modify: `rapx/src/analysis/unsoundaudit/mir.rs:195-319,1251-1565`

- [ ] **Step 1: Write summary RED tests**

Test that a callee summary exports validation only when it binds a formal or return/out subject, dominates the sink, and remains valid after writes. Test actual substitution, same-value binding, and intersection across multiple call paths.

- [ ] **Step 2: Capture RED on `call_guard_in_middle_negative` and `call_guard_in_sink_negative`**

Expected: old engine reports P1 because it checks only the root before the first call.

- [ ] **Step 3: Implement finite validation contracts**

Use existing `ValidationFact` fields and populate `FunctionSummary.validations`. During summary composition, substitute formals structurally. A requirement is discharged only if every feasible root-to-sink call path carries a valid bound fact or establishes one locally before the sink.

- [ ] **Step 4: Run interprocedural validation cases and frozen suite**

Expected: middle/sink guard cases zero; wrong-value and non-dominating cases remain one; frozen 27 unchanged.

- [ ] **Step 5: Commit validation propagation**

```bash
git add rapx/src/analysis/unsoundaudit/summary.rs rapx/src/analysis/unsoundaudit/mir.rs
git commit -m "fix(unsoundaudit-v2): propagate bound validations through summaries"
```

### Task 7: Implement real return and local out mappings

**Files:**
- Modify: `rapx/src/analysis/unsoundaudit/summary.rs:271-406`
- Modify: `rapx/src/analysis/unsoundaudit/mir.rs:603-830`

- [ ] **Step 1: Add RED tests for local outputs**

Test constant return does not inherit arguments, a returned formal reaches only the call destination, and a projected mutable out write maps to the actual caller place. Assert `OutToCaller` is emitted by production extraction, not just manually constructed in a unit test.

- [ ] **Step 2: Capture RED on constant-return and local-out challenge cases**

Expected: constant-return stays zero; local-out fails to emit the expected P1 under the old empty `out_values` path.

- [ ] **Step 3: Populate structural call outputs**

Record callee return origins and `OutDependency` for writes rooted at mutable/raw formal projections. At each resolved local call, use `map_call_outputs` with actuals and out actual places. Do not join unrelated actuals and do not infer output for opaque calls.

- [ ] **Step 4: Run call mapping cases and frozen suite**

Expected: 0/1/3/5-hop and local-out cases match; constant-return and opaque cases remain zero; frozen 27 unchanged.

- [ ] **Step 5: Commit output mapping**

```bash
git add rapx/src/analysis/unsoundaudit/summary.rs rapx/src/analysis/unsoundaudit/mir.rs
git commit -m "fix(unsoundaudit-v2): map local return and out values structurally"
```

### Task 8: Separate path validity from witness selection

**Files:**
- Modify: `rapx/src/analysis/unsoundaudit/mir.rs:1251-1673`
- Modify: `rapx/src/analysis/unsoundaudit/summary.rs:575-679`

- [ ] **Step 1: Write RED tests for universal discharge**

Build small pure call graphs with two root-to-sink paths. Assert a guard on only one path does not discharge, guards on all paths do, recursion terminates with one cycle token, and lexical witness choice cannot change the finding decision.

- [ ] **Step 2: Capture RED on diamond-call and recursion cases**

Expected: at least one case exposes shortest-path dependence or missing real-MIR cycle evidence.

- [ ] **Step 3: Implement finite path-state propagation**

Propagate `Validated`/`Unvalidated` states over the SCC condensation graph. Join at a sink with logical AND for discharge: one feasible unvalidated state keeps the finding. Saturate recursive components with the existing stable SCC token. Run deterministic shortest-witness selection only after the outcome is fixed.

- [ ] **Step 4: Run recursion, call-graph, and frozen gates**

Expected: direct/mutual recursion each emit one finding with one stable cycle token; no duplicate ID; all path cases and frozen 27 pass twice byte-identically.

- [ ] **Step 5: Commit path safety**

```bash
git add rapx/src/analysis/unsoundaudit/summary.rs rapx/src/analysis/unsoundaudit/mir.rs
git commit -m "fix(unsoundaudit-v2): decide path safety before witness selection"
```

## Work Package 4: Equivalent Frozen Guards and Stable Spans

### Task 9: Add equivalent forms without adding predicates

**Files:**
- Modify: `rapx/src/analysis/unsoundaudit/mir.rs:1273-1508`

- [ ] **Step 1: Add RED unit and challenge tests**

For `InBounds`, test false branch of `index >= len`, true branch of `index < len`, and a dominating `assert!(index < len)`. For `NonEmpty`, bind `len() != 0`. For `NonNull`, bind `match pointer { null => return, _ => sink }`. For `RangeInBounds`, accept the same `offset + width <= len` obligation only when checked arithmetic proves no overflow.

- [ ] **Step 2: Run guard subset and capture RED**

Expected: the original forms pass; newly approved equivalent forms fail to discharge.

- [ ] **Step 3: Normalize guard evidence to existing predicates**

Add no new `Predicate` variant. Convert each exact MIR spelling into the existing bound predicate plus subject, collection, branch polarity, and version. Fail closed on overflow-ambiguous arithmetic or unbound values.

- [ ] **Step 4: Run guard subset and frozen suite**

Expected: all equivalent-guard negatives become zero; wrong-value and non-dominating positives remain one; original 27 aggregate unchanged.

- [ ] **Step 5: Commit guard normalization**

```bash
git add rapx/src/analysis/unsoundaudit/mir.rs
git commit -m "fix(unsoundaudit-v2): normalize equivalent frozen guards"
```

### Task 10: Make macro and include spans fail closed

**Files:**
- Modify: `rapx/src/analysis/unsoundaudit/mir.rs:2154-2194`
- Modify: `scripts/unsoundaudit-v2/validate_receipt.py`

- [ ] **Step 1: Add RED span tests**

Require one-based columns, repository-relative macro call-site paths, distinct included-file paths, and a hard error rather than a fabricated `src/lib.rs` fallback when a span cannot be made stable.

- [ ] **Step 2: Capture RED on macro/include cases**

Expected: at least the include identity or fallback behavior fails the new oracle.

- [ ] **Step 3: Return a checked stable-span result**

Change `stable_span` to return `Result<StableSpan, String>`. Accept a path only when local or remapped identity is normalized and relative to the project root. Convert rustc zero-based columns with `+1`. Propagate a fail-closed analysis error instead of inventing a path.

- [ ] **Step 4: Run span, schema, challenge, and frozen gates**

Expected: distinct stable paths, no absolute paths, no causal collision, challenge suite all green, frozen suite unchanged.

- [ ] **Step 5: Commit span hardening**

```bash
git add rapx/src/analysis/unsoundaudit/mir.rs scripts/unsoundaudit-v2/validate_receipt.py
git commit -m "fix(unsoundaudit-v2): fail closed on unstable source spans"
```

## Work Package 5: Full Synthetic Gate

### Task 11: Produce deterministic challenge GREEN evidence

**Files:**
- Create: `artifacts/unsoundaudit-v2/mac/robustness/challenge_results.normalized.json`
- Create: `artifacts/unsoundaudit-v2/mac/robustness/challenge_validation_summary.json`
- Modify: `TASK_CONTEXT.md`

- [ ] **Step 1: Run exact fresh Rust tests**

Run `cargo test --manifest-path rapx/Cargo.toml --lib --locked --jobs 1` in a fresh target. Expected: all old and new tests pass, 0 failed and 0 ignored.

- [ ] **Step 2: Run original 27 twice**

Expected: both normalized receipts byte-identical, 27/27, aggregate `3/2/2/2/1/2`.

- [ ] **Step 3: Run challenge suite twice**

Expected: all challenge oracles pass and normalized receipts are byte-identical.

- [ ] **Step 4: Freeze sanitized evidence**

Commit only normalized results, tested source SHA, toolchain commits, manifest SHA, counts, run hashes, and raw-log SHA values with `committed:false`.

- [ ] **Step 5: Perform independent read-only code and semantic review**

Require zero blocking findings for registry drift, fabricated edges, non-finite facts, hidden truncation, schema drift, or forbidden artifacts.

- [ ] **Step 6: Commit synthetic GREEN evidence**

```bash
git add artifacts/unsoundaudit-v2/mac/robustness TASK_CONTEXT.md
git commit -m "test(unsoundaudit-v2): record robustness challenge GREEN"
```

## Plan Self-Review

- Spec sections 1–3 map to Tasks 1–3 and the immutable original-suite gates.
- Adversarial cases and engine properties map to Tasks 2 and 4–10.
- Synthetic safety and deliverables map to every execution task and Task 11's final artifact audit.
- Standard-library routing, sampling, review, and reporting are deliberately split into `2026-08-12-unsoundaudit-v2-stdlib-pilot.md` and cannot start before Task 11 passes.
- No task authorizes a new registry form, primary precedence change, cross-crate propagation, target execution, Linux access, or third-party holdout download.
- All implementation tasks have an explicit RED, minimal implementation, GREEN gate, frozen-27 regression, and commit step.

# P1/P6 High-Confidence Token-Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace broad P1/P6 provenance correlation with category-specific exact token flow, preserve FFI recall, emit explicit unestablished decisions, and restore exact six-pattern coverage.

**Architecture:** Reuse the compact `ContextProofIndex` arenas and add a finite interned `InfluenceToken` product fixed point. MIR records typed source seeds and controller dependencies; candidate eligibility asks the exact token solver for an active, suppressed, or unestablished result. The writer and both Python runners stream a fourth mutually exclusive population and validate exact accounting.

**Tech Stack:** Rust/rustc private APIs, compact u32 arenas, Serde JSON/JSONL, Python 3 streaming validators, Cargo tests on pinned nightly-2024-10-12 and current nightly.

---

## File Structure

- Create `rapx/src/analysis/unsoundaudit/influence.rs`: token domain, source seeds, sink requirements, controller projection, compact product solver, and focused tests.
- Modify `rapx/src/analysis/unsoundaudit.rs`: register the new module.
- Modify `rapx/src/analysis/unsoundaudit/endpoint_flow.rs`: expose a borrowed compact-ID facade and add receiver dependency to open-trait controllers.
- Modify `rapx/src/analysis/unsoundaudit/mir.rs`: derive source capabilities/controller dependencies, retain seed metadata, propagate it into candidate construction, and carry unestablished output.
- Modify `rapx/src/analysis/unsoundaudit/candidate.rs`: exact P1/P6 eligibility, four-state decisions, rank-after-eligibility, and P3-only exposure.
- Modify `rapx/src/analysis/unsoundaudit/contract_catalog.rs`: convert role-specific known predicates to finite sink requirements without granting opaque contracts false precision.
- Modify `rapx/src/analysis/unsoundaudit/unsafe_site/mod.rs`: evidence-backed incompatibility suppression and exact optimized-away binding enums.
- Modify `rapx/src/analysis/unsoundaudit/unsafe_site/binding.rs`: distinguish exact compiler elimination from unresolved elision.
- Modify `rapx/src/analysis/unsoundaudit/v3_output.rs`: stream `unestablished.jsonl`, emit four-state accounting, and enforce non-P3 exposure.
- Modify `rapx/src/analysis/unsoundaudit/v3_output_reporting.rs`: reconcile active, pure-suppressed, reclassified, and unestablished identities.
- Modify `scripts/unsoundaudit-v3/run_stdlib_gate.py` and `scripts/unsoundaudit-v3/test_stdlib_gate.py`: streaming validation of the fourth population.
- Modify `scripts/unsoundaudit-v3/run_mechanism_pack.py`: oracle support for required/forbidden unestablished roots and exact four-state totals.
- Modify P1/P6 mechanism fixture sources and oracles under `tests/unsoundaudit-v3/mechanism-pack`.
- Modify `TASK_CONTEXT.md`: record completed gates, decisions, artifacts, and remaining limitations.

The repository contains large uncommitted shared work. Every commit step must stage only the new file or owned hunks. Before each commit, inspect `git diff --cached --name-only` and `git diff --cached`; never stage an entire previously modified file blindly.

### Task 1: Add the finite influence domain

**Files:**
- Create: `rapx/src/analysis/unsoundaudit/influence.rs`
- Modify: `rapx/src/analysis/unsoundaudit.rs`

- [ ] **Step 1: Write failing domain tests**

```rust
#[test]
fn boolean_does_not_satisfy_address_requirement() {
    let token = InfluenceCapabilities::BOOLEAN | InfluenceCapabilities::TYPED_VALUE;
    assert_eq!(
        SinkRequirement::Address.matches(token),
        RequirementMatch::ProvenIncompatible,
    );
}

#[test]
fn token_interner_deduplicates_complete_semantic_values() {
    let mut arena = InfluenceArena::default();
    let token = InfluenceToken::ffi(
        InfluenceOrigin::FfiReturn,
        InfluenceCapabilities::ADDRESS | InfluenceCapabilities::VALIDITY_BITS,
    );
    assert_eq!(arena.intern(token.clone()), arena.intern(token));
    assert_eq!(arena.len(), 1);
}

#[test]
fn unestablished_reason_precedence_is_stable() {
    let reasons = BTreeSet::from([
        UnestablishedReason::OpaqueTransform,
        UnestablishedReason::UnresolvedController,
        UnestablishedReason::MissingExactContext,
    ]);
    assert_eq!(
        primary_unestablished_reason(&reasons),
        Some(UnestablishedReason::MissingExactContext),
    );
}
```

- [ ] **Step 2: Run the focused test and verify RED**

Run from `rapx/`:

```bash
env -u RUSTFLAGS -u CARGO_ENCODED_RUSTFLAGS \
  RUSTC=/Users/liaowenzhao/.rustup/toolchains/nightly-2024-10-12-aarch64-apple-darwin/bin/rustc \
  RUSTUP_TOOLCHAIN=nightly-2024-10-12-aarch64-apple-darwin \
  RUST_SYSROOT=/Users/liaowenzhao/.rustup/toolchains/nightly-2024-10-12-aarch64-apple-darwin \
  CARGO_BUILD_JOBS=1 cargo test --lib influence --locked --jobs 1
```

Expected: compilation fails because `influence` and its finite-domain types do not exist.

- [ ] **Step 3: Implement the minimal finite types**

Register `mod influence;` and implement the domain without adding a dependency:

```rust
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct InfluenceCapabilities(u8);

impl InfluenceCapabilities {
    pub(super) const ADDRESS: Self = Self(1 << 0);
    pub(super) const INDEX_OR_LENGTH: Self = Self(1 << 1);
    pub(super) const TYPED_VALUE: Self = Self(1 << 2);
    pub(super) const VALIDITY_BITS: Self = Self(1 << 3);
    pub(super) const BOOLEAN: Self = Self(1 << 4);
    pub(super) const CALLABLE_RESULT: Self = Self(1 << 5);

    pub(super) fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum InfluenceOrigin {
    PublicFormal,
    PersistentPublicField,
    FfiReturn,
    FfiOut,
    FfiCallbackInput,
    OpenTraitReturn,
    PublicCallbackReturn,
    PublicFunctionPointerReturn,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum InfluenceController {
    FfiHost,
    PublicFormal { owner: FunctionKey, index: u32 },
    PersistentPublicField { access_owner: FunctionKey, field: String },
    OpenTrait { trait_id: FunctionKey, receiver: OpenBehaviorReceiver },
    PublicCallback,
    Internal,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum OpenBehaviorReceiver {
    Parameter(GenericParamId),
    Dynamic,
    Alias(BTreeSet<GenericParamId>),
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum TokenPrecision { StrongBoundary, ExactDerived, Incomplete }

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct InfluenceToken {
    pub origin: InfluenceOrigin,
    pub controller: InfluenceController,
    pub capabilities: InfluenceCapabilities,
    pub precision: TokenPrecision,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct TokenId(u32);

#[derive(Default)]
pub(super) struct InfluenceArena {
    ids: BTreeMap<Arc<InfluenceToken>, TokenId>,
    values: Vec<Arc<InfluenceToken>>,
}
```

Use checked `u32::try_from` on insertion. Define `UnestablishedReason` in the exact precedence specified by the design and make `as_str()` return the approved snake-case tokens.

- [ ] **Step 4: Run focused tests and formatting**

Run the command from Step 2, scoped rustfmt, and `git diff --check`. Expected: influence tests pass and diff-check is clean.

- [ ] **Step 5: Commit only the new module and module declaration**

```bash
git add rapx/src/analysis/unsoundaudit/influence.rs
git add -p rapx/src/analysis/unsoundaudit.rs
git diff --cached --check
git commit -m "feat(unsoundaudit): add finite influence token domain"
```

### Task 2: Expose a compact exact-product facade and solve token requests

**Files:**
- Modify: `rapx/src/analysis/unsoundaudit/endpoint_flow.rs`
- Modify: `rapx/src/analysis/unsoundaudit/influence.rs`

- [ ] **Step 1: Add failing exact-product tests**

Cover one local relation, two alternatives at one callsite, nested calls, missing callee context, and mutual recursion:

```rust
#[test]
fn exact_token_solver_does_not_splice_sibling_callsites() {
    let fixture = ExactInfluenceFixture::two_sibling_calls();
    let outcome = fixture.solve_source_a_to_sink_b();
    assert!(matches!(outcome, InfluenceOutcome::Unestablished { reasons, .. }
        if reasons.contains(&UnestablishedReason::MissingExactContext)));
}

#[test]
fn ffi_token_survives_two_exact_call_boundaries() {
    let fixture = ExactInfluenceFixture::ffi_through_two_wrappers();
    assert!(matches!(fixture.solve(), InfluenceOutcome::Active(_)));
}
```

- [ ] **Step 2: Verify RED**

Run pinned filters `endpoint_flow` and `influence`. Expected: the compact facade and solver APIs are absent.

- [ ] **Step 3: Add the borrowed ID facade**

Expose opaque parent-module handles and borrowed predecessors:

```rust
pub(super) struct ContextAtomHandle(ContextAtomId);
pub(super) struct ContextRelationHandle(ContextRelationId);
pub(super) struct ContextPredecessorHandle(ContextPredecessorId);

pub(super) enum ExactPredecessorRef<'a> {
    Local { kind: EndpointEdgeKind, point: Option<&'a ProgramPoint> },
    Call {
        callsite: &'a CallSiteId,
        substitution: &'a GenericCallSubstitution,
        callee_relation: ContextRelationHandle,
        actual_binding: ContextActualBinding,
    },
}
```

Add methods for exact atoms matching an endpoint, exact outgoing relation handles, relation source/target handles, exact predecessor handles, and borrowed predecessor decoding. Preserve semantic ordering by sorting handles with decoded values, never numeric insertion order.

- [ ] **Step 4: Implement the request/waiter fixed point**

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct InfluenceRequest {
    scope: ContextAtomHandle,
    start: ContextAtomHandle,
    target: ContextAtomHandle,
    seed: TokenId,
}

struct InfluenceWaiter {
    parent: InfluenceRequest,
    next: ContextAtomHandle,
    boundary: ContextPredecessorHandle,
}

struct InfluenceSolver<'a> {
    context: &'a ContextProofIndex,
    arena: InfluenceArena,
    pending: VecDeque<InfluenceState>,
    seen: BTreeSet<(InfluenceRequest, ContextAtomHandle, TokenId)>,
    facts: BTreeMap<InfluenceRequest, BTreeSet<TokenId>>,
    waiters: BTreeMap<InfluenceRequest, Vec<InfluenceWaiter>>,
}
```

For `Local`, apply a typed transfer and enqueue the target. For `Call`, start the exact callee relation, wait for child facts, then apply actual binding/substitution before resuming the parent. Missing body/actual facts add `MissingExactContext`; normalization failure adds `IncompleteNormalization`. Store predecessor handles only for the canonical winning witness.

- [ ] **Step 5: Verify all alternatives and recursion**

Run pinned `endpoint_flow` and `influence` tests with `--jobs 1`. Expected: all old tests plus new local/call/recursion/unknown tests pass.

- [ ] **Step 6: Commit owned hunks**

Commit only inspected hunks with message `feat(unsoundaudit): solve exact influence token requests`.

### Task 3: Derive typed source seeds in MIR

**Files:**
- Modify: `rapx/src/analysis/unsoundaudit/influence.rs`
- Modify: `rapx/src/analysis/unsoundaudit/mir.rs`

- [ ] **Step 1: Add failing type/source tests**

Cover boolean, integer/index, raw/reference pointer, callable, aggregate, FFI, public formal, and open-trait return seeds:

```rust
#[test]
fn result_value_does_not_start_with_address_capability() {
    let seed = seed_for_test_type(TestType::Result, InfluenceOrigin::OpenTraitReturn);
    assert!(!seed.capabilities.intersects(InfluenceCapabilities::ADDRESS));
    assert!(seed.capabilities.intersects(InfluenceCapabilities::TYPED_VALUE));
}
```

- [ ] **Step 2: Verify RED**

Run pinned `cargo test --lib influence_seed --locked --jobs 1`. Expected: source-seed index absent.

- [ ] **Step 3: Implement `InfluenceSeedIndex`**

```rust
#[derive(Clone, Debug, Default)]
pub(super) struct InfluenceSeedIndex {
    seeds: BTreeMap<EndpointNode, InfluenceToken>,
}

impl InfluenceSeedIndex {
    pub(super) fn insert(&mut self, endpoint: EndpointNode, seed: InfluenceToken) {
        self.seeds.entry(endpoint).and_modify(|old| *old = old.join(&seed)).or_insert(seed);
    }
    pub(super) fn get(&self, endpoint: &EndpointNode) -> Option<&InfluenceToken> {
        self.seeds.get(endpoint)
    }
}
```

Store it in `BodyFacts`, merge it during endpoint aggregation, and pass it to candidate construction. Derive capabilities from rustc `Ty` and compiler identity; do not parse `Debug` or type-name strings.

- [ ] **Step 4: Implement exact transfer/kill semantics**

Preserve seeds through typed copies and identity views. Convert index/length/offset/alignment/layout/boolean derivations to approved capabilities. Exact overwrite replaces the storage seed; ambiguous writes, opaque returns, address reinterpretation, and havoc retain origin for accounting but set precision to `Incomplete`.

- [ ] **Step 5: Run MIR and influence tests**

Run pinned filters `influence_seed`, `influence_transfer`, and existing `typed_pointer`. Expected: all pass; no FFI source is lost.

- [ ] **Step 6: Commit owned hunks**

Commit as `feat(unsoundaudit): derive typed influence seeds`.

### Task 4: Transport open-trait and callback controllers

**Files:**
- Modify: `rapx/src/analysis/unsoundaudit/endpoint_flow.rs`
- Modify: `rapx/src/analysis/unsoundaudit/mir.rs`
- Modify: `rapx/src/analysis/unsoundaudit/influence.rs`

- [ ] **Step 1: Add failing controller tests**

Cover direct public generic, `dyn Trait`, concrete internal impl, sealed/private trait, alias normalization failure, public callback, internal closure, FFI callback, and a two-boundary substitution where an inner receiver parameter becomes a concrete outer type.

- [ ] **Step 2: Verify RED**

Run pinned filters `open_behavior_controller` and `influence_controller`.

- [ ] **Step 3: Extend open-trait controller data**

```rust
BehaviorController::OpenTrait {
    trait_id: FunctionKey,
    openness: BehaviorOpenness,
    resolved_target: Option<FunctionKey>,
    receiver: super::influence::OpenBehaviorReceiver,
}
```

Derive `Parameter` from `ty::Param`, `Dynamic` from `ty::Dynamic`, typed controller sets from projections, and `Unknown` when no typed representation is available.

- [ ] **Step 4: Project through exact call substitutions**

For an internal boundary, map the child controller through `GenericCallSubstitution`. Complete direct-parameter bindings become the caller parameter; concrete/resolved targets become `Internal`; residual aliases or incomplete normalization become `Unknown`. Formal actual bindings preserve identity. `FfiHost` never consults generic substitution.

- [ ] **Step 5: Verify nested positive and negative cases**

Run pinned endpoint, influence, and candidate controller tests. Expected: generic/dynamic cases remain active-capable; concrete/internal cases are proven excluded; alias uncertainty is unestablished; FFI remains strong.

- [ ] **Step 6: Commit owned hunks**

Commit as `feat(unsoundaudit): transport external behavior controllers`.

### Task 5: Model sink requirements and validation compatibility

**Files:**
- Modify: `rapx/src/analysis/unsoundaudit/contract_catalog.rs`
- Modify: `rapx/src/analysis/unsoundaudit/influence.rs`
- Modify: `rapx/src/analysis/unsoundaudit/mir.rs`
- Modify: `rapx/src/analysis/unsoundaudit/unsafe_site/mod.rs`

- [ ] **Step 1: Add failing requirement tests**

```rust
#[test]
fn opaque_contract_without_typed_requirement_is_unestablished() {
    let contract = opaque_contract(OperandRole::Argument(0));
    assert_eq!(
        SinkRequirement::from_contract(&contract, &OperandRole::Argument(0)),
        Err(UnestablishedReason::UnmodeledUnsafeCall),
    );
}
```

Also test known incompatibility, unknown compatibility, exact validation, branch-only validation, and exact overwrite.

- [ ] **Step 2: Verify RED**

Run pinned `cargo test --lib influence_requirement --locked --jobs 1`.

- [ ] **Step 3: Implement role-specific requirements**

```rust
match predicate {
    ContractPredicate::NonNull
    | ContractPredicate::Aligned
    | ContractPredicate::LayoutCompatible
    | ContractPredicate::ValidForRead => SinkRequirement::Address,
    ContractPredicate::InBounds | ContractPredicate::RangeInBounds => {
        SinkRequirement::IndexOrAddress { role: role.clone() }
    }
    ContractPredicate::Initialized | ContractPredicate::ValidForType => {
        SinkRequirement::ValidityBits
    }
    ContractPredicate::ValidUtf8 => SinkRequirement::TypedContent,
    ContractPredicate::ConditionTrue => SinkRequirement::Boolean,
}
```

An opaque descriptor with no compiler-derived minimum returns `UnmodeledUnsafeCall`. Do not add function-name models.

- [ ] **Step 4: Connect must-validation and storage versions**

Use existing exact validation/initialization facts only when token, role, and storage version agree on every incoming state. A proven incompatibility becomes `PureSuppressed` with reason `source_cannot_satisfy_requirement`; unknown compatibility/validation becomes `Unestablished`.

- [ ] **Step 5: Verify tests and commit**

Run pinned `contract_catalog`, `influence_requirement`, and MIR validation tests. Commit as `feat(unsoundaudit): match influence to unsafe requirements`.

### Task 6: Integrate exact P1/P6 eligibility and four-state candidates

**Files:**
- Modify: `rapx/src/analysis/unsoundaudit/candidate.rs`
- Modify: `rapx/src/analysis/unsoundaudit/mir.rs`
- Modify: `rapx/src/analysis/unsoundaudit/unsafe_site/mod.rs`

- [ ] **Step 1: Add failing candidate tests**

Add tests proving a coarse-only sibling-call route is unestablished, direct FFI exact flow is active, concrete dispatch is pure-suppressed, opaque reachability is unestablished, unestablished P6 does not hide established P1, and non-P3 exposure is absent.

- [ ] **Step 2: Verify RED**

Run the pinned candidate suite.

- [ ] **Step 3: Add the compact unestablished DTO**

```rust
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct UnestablishedStructuralCandidate {
    pub decision_id: String,
    pub attempted_pattern: PrimaryPattern,
    pub public_root: Arc<FunctionKey>,
    pub site_id: String,
    pub operand_role: OperandRole,
    pub source_identity: String,
    pub source_kind: CausalSourceKind,
    pub controller: String,
    pub capabilities: String,
    pub requirement: Option<String>,
    pub primary_reason: UnestablishedReason,
    pub secondary_reasons: Vec<UnestablishedReason>,
}
```

Extend `StructuralCandidateSet` and `AnalysisOutput` with an ordered vector. Compute `decision_id` from length-prefixed semantic fields with `sha256_hex`.

- [ ] **Step 4: Replace only P1/P6 eligibility**

Keep P2-P5 unchanged. For P1/P6, call the exact token solver. Convert outcomes to active, evidence-backed suppression, or unestablished. Evaluate rank only among active/reclassified candidates. Do not let `Unsupported` silently discard P1/P6.

- [ ] **Step 5: Enforce P3-only exposure**

Set `exposure: None` whenever `primary_pattern != P3`, including the public-formal branch built while analyzing a P3 result corridor. Add a construction assertion and output fail-closed check.

- [ ] **Step 6: Run candidate tests and commit**

Commit as `feat(unsoundaudit): require exact P1/P6 influence`.

### Task 7: Emit and reconcile `unestablished.jsonl`

**Files:**
- Modify: `rapx/src/analysis/unsoundaudit/v3_output.rs`
- Modify: `rapx/src/analysis/unsoundaudit/v3_output_reporting.rs`

- [ ] **Step 1: Add failing output/accounting tests**

```rust
assert_eq!(accounting, json!({
    "active_count": 2,
    "pure_suppressed_count": 1,
    "reclassified_count": 1,
    "unestablished_count": 3,
    "suppressed_count": 2,
    "considered_count": 7,
}));
```

Add negative active/unestablished overlap and non-P3 exposure tests.

- [ ] **Step 2: Verify RED**

Run pinned `cargo test --lib v3_output --locked --jobs 1`.

- [ ] **Step 3: Extend reporting reconciliation**

Change `build_reports` to accept sorted unestablished IDs. Keep `suppressed_count = pure_suppressed + reclassified` and require `active + suppressed + unestablished = considered`. Validate pairwise disjointness without retaining full rows.

- [ ] **Step 4: Stream the new file**

Write `unestablished.jsonl` in decision-ID order with schema `rap-unsoundaudit-unestablished-v1`. Use the bounded ordered chunk writer and add its count to candidate accounting.

- [ ] **Step 5: Verify byte determinism**

Add reverse-input and worker-count golden tests. Expected: byte-identical normalized rows and rejection of any non-P3 exposure.

- [ ] **Step 6: Run pinned output tests and commit**

Commit as `feat(unsoundaudit): report unestablished decisions`.

### Task 8: Stream and validate four-state output in Python

**Files:**
- Modify: `scripts/unsoundaudit-v3/run_stdlib_gate.py`
- Modify: `scripts/unsoundaudit-v3/test_stdlib_gate.py`
- Modify: `scripts/unsoundaudit-v3/run_mechanism_pack.py`

- [ ] **Step 1: Add failing Python tests**

Add valid four-state input plus duplicate, overlap, invalid reason, non-P3 exposure, missing file, arithmetic drift, malformed UTF-8, and large streaming cases. The large case must monkeypatch `Path.read_text` to raise.

- [ ] **Step 2: Verify RED**

```bash
python3 scripts/unsoundaudit-v3/test_stdlib_gate.py
python3 scripts/unsoundaudit-v3/run_mechanism_pack.py --self-test
```

- [ ] **Step 3: Implement streaming validation**

Read `unestablished.jsonl` line by line. Retain only decision IDs and compact accounting signatures. Require the approved reason enum, sorted secondary reasons, primary precedence, null exposure, identity disjointness, and exact totals. Wrap I/O, UTF-8, and JSON errors as `GateError`.

Add the unestablished artifact SHA-256 and count to the development resource after streaming validation.

- [ ] **Step 4: Extend mechanism oracle semantics**

```json
{
  "unestablished_roots": {
    "opaque_sink": {"attempted_pattern": "P6", "reason": "unmodeled_unsafe_call"}
  },
  "absent_unestablished_roots": ["direct_ffi"]
}
```

Validate the complete population before applying root expectations.

- [ ] **Step 5: Run Python gates and commit**

Run full tests, `py_compile`, and mechanism self-test. Commit as `feat(unsoundaudit): validate four-state MVP output`.

### Task 9: Add P1/P6 precision and FFI recall fixtures

**Files:**
- Modify: `tests/unsoundaudit-v3/mechanism-pack/p1/src/lib.rs`
- Modify: `tests/unsoundaudit-v3/mechanism-pack/p1/oracle.json`
- Modify: `tests/unsoundaudit-v3/mechanism-pack/p6/src/lib.rs`
- Modify: `tests/unsoundaudit-v3/mechanism-pack/p6/oracle.json`

- [ ] **Step 1: Add fixture roots**

Add direct/out/wrapped/callback FFI, public generic/dyn open traits, concrete/sealed/private traits, public/internal callbacks, sibling callsites, sibling branches, `Result`/boolean mismatch, modeled conversion, opaque sink, must-validation, branch-only validation, and overwrite.

- [ ] **Step 2: Add exact oracle expectations**

Require active P6 for valid FFI and public open/callback lanes; pure suppression for proven internal/concrete/incompatible cases; unestablished for opaque/unresolved/non-exact cases; zero P6 for sibling splices; and null exposure for every non-P3 row.

- [ ] **Step 3: Check fixture crates**

Run pinned `cargo check --locked --jobs 1` in both fixture crates.

- [ ] **Step 4: Build metadata-correct binaries and run the full pack**

Use pinned `RUSTC`, `RUST_SYSROOT`, `RUSTUP_TOOLCHAIN`, `UNSOUND_SCANNER_RAP_BUILD_RUSTC_COMMIT=1bc403daadbebb553ccc211a0a8eebb73989665f`, cleared inherited flags, and jobs 1. Expected: complete/exact coverage and all oracles green.

- [ ] **Step 5: Commit fixtures**

Commit as `test(unsoundaudit): cover high-confidence P1/P6 flow`.

### Task 10: Turn solver-elided sites into exact optimized-away bindings

**Files:**
- Modify: `rapx/src/analysis/unsoundaudit/unsafe_site/mod.rs`
- Modify: `rapx/src/analysis/unsoundaudit/unsafe_site/binding.rs`
- Modify: `rapx/src/analysis/unsoundaudit/mir.rs`
- Modify: `rapx/src/analysis/unsoundaudit/v3_output.rs`
- Modify: `scripts/unsoundaudit-v3/run_stdlib_gate.py`

- [ ] **Step 1: Add failing exact-elimination tests**

```rust
assert_eq!(
    bind_owner(&[site], &[built_event], &[]).unwrap()[0].status,
    BindingStatus::OptimizedAwayExact,
);
```

Also require missing built candidates to remain unbound, multiple solver matches ambiguous, and collector failure ineligible for optimized-away exact.

- [ ] **Step 2: Verify RED**

Run pinned `cargo test --lib unsafe_site::binding --locked --jobs 1`.

- [ ] **Step 3: Implement exact optimized-away accounting**

`OptimizedAwayExact` requires one exact built component, complete solver enumeration for the owner, and zero solver candidates. Preserve the built event and typed operands. Do not construct a `BoundUnsafeSite` or candidate because the operation is absent from analyzed optimized MIR.

Count `BoundExact + OptimizedAwayExact` as exact coverage. Retain `Elided` only when solver enumeration itself is incomplete. Emit exact elimination evidence in the binding sidecar.

- [ ] **Step 4: Update Rust/Python validation**

Require every discovered site to be attached `BoundExact` or `OptimizedAwayExact`. Optimized-away rows must have one built event, zero solver candidates, no candidate identity, and no coverage failure.

- [ ] **Step 5: Run binding/output/Python tests**

Expected: the synthetic elision fixture yields complete/exact coverage without invented candidates.

- [ ] **Step 6: Commit owned hunks**

Commit as `fix(unsoundaudit): account for optimized-away unsafe sites`.

### Task 11: Run compatibility, core, determinism, and precision gates

**Files:**
- Modify: `TASK_CONTEXT.md`

- [ ] **Step 1: Run final pinned focused gates**

Run jobs 1 for filters `influence`, `endpoint_flow`, `candidate`, `v3_output`, and `unsafe_site::binding`, then `cargo check --lib --locked`.

- [ ] **Step 2: Run current-nightly compatibility**

Use explicit current `RUSTC`, `RUST_SYSROOT`, and `RUSTUP_TOOLCHAIN`; run `cargo check --lib --locked --jobs 1` plus focused influence/controller/binding tests.

- [ ] **Step 3: Run the complete mechanism pack**

Build fresh metadata-correct pinned binaries and run P1-P6 sequentially. Expected: exact coverage, all oracles green, deterministic four-state totals, and preserved FFI positives.

- [ ] **Step 4: Run the first core acceptance scan**

Run one fresh core scan with a 900-second wall gate and 12-GiB diagnostic allowance. Expected: execution success, exact coverage, strict validator success, and all four populations in resource accounting.

- [ ] **Step 5: Run normalized determinism replay**

Run a second fresh scan and compare normalized SHA-256 for candidates, families, suppression ledger, unestablished stream, coverage, census, and binding artifacts.

- [ ] **Step 6: Perform deterministic precision review**

Review every FFI family, all P1 families up to 100, and all open-trait P6 families up to 100 in family-ID order. Any definite splice, concrete controller, or requirement mismatch fails the gate.

- [ ] **Step 7: Update the source of truth**

Record commands, versions, counts, coverage, timing, memory, artifact paths/hashes, review results, limitations, and rejected claims in `TASK_CONTEXT.md`.

- [ ] **Step 8: Commit the context update**

Stage only the appended context hunk and commit as `docs: record high-confidence P1/P6 gates`.

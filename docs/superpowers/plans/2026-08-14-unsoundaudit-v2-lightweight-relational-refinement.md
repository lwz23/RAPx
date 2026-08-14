# UnsoundAudit v2 Lightweight Relational Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic no-SMT proof layer that generically closes the 22 audited safe standard-library families while preserving the historical join/concat UB and every frozen P1–P6 oracle.

**Architecture:** Extend the existing finite MIR dataflow with current-version relational atoms and infer finite normal-return postconditions for resolved-local calls. A sink is removed only when its exact sparse snapshot proves the obligation; unknown facts continue through the existing requirement, route, and witness pipeline.

**Tech Stack:** Rust nightly `2024-10-12`, rustc private MIR APIs, deterministic `BTreeMap`/`BTreeSet` lattices, existing CFG replay and summary SCC solvers, Python 3 no-build gates, `rapx-perf`, Miri evidence used only as a regression oracle for the historical UB.

---

## File map

| Path | Responsibility |
|---|---|
| `rapx/src/analysis/unsoundaudit/relations.rs` | Finite terms, atoms, joins, invalidation, closure, summary projection and instantiation |
| `rapx/src/analysis/unsoundaudit.rs` | Register the relation module; receipt/schema behavior stays unchanged |
| `rapx/src/analysis/unsoundaudit/mir.rs` | Normalize arithmetic, carry relation state, snapshot proof facts, infer/apply postconditions, discharge exact sinks |
| `rapx/src/analysis/unsoundaudit/summary.rs` | Store and join finite relational postconditions without paths or depths |
| `scripts/unsoundaudit-v2/test_relational_refinement.py` | No-build contract tying regression family IDs and evidence boundaries to the implementation gate |
| `docs/unsoundaudit-v2/STDLIB_ACTIVE_FINDING_AUDIT.md` | Record automatic proof results without rewriting historical v4/v5 populations |
| `TASK_CONTEXT.md` | Cross-session decisions, evidence and stop conditions |

### Task 1: Freeze the relational regression contract

**Files:**
- Create: `scripts/unsoundaudit-v2/test_relational_refinement.py`
- Modify: `TASK_CONTEXT.md`

- [ ] **Step 1: Write the failing no-build contract test**

Create a test that reads the committed active-family table and freezes the exact target set:

```python
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
HISTORICAL_UB_FAMILY = "560c34406031d113235e81e9d711f31bc34ba2673819fc96abd51ce60f93b4df"
```

Require the safe set to contain exactly 22 families/35 findings, the UB family to contain exactly two findings, and all IDs to match the committed audit table. Require a source marker `RELATIONAL_REFINEMENT_VERSION` in `relations.rs`; this makes the initial test fail before production exists.

- [ ] **Step 2: Run the RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  scripts/unsoundaudit-v2/test_relational_refinement.py
```

Expected: FAIL because `rapx/src/analysis/unsoundaudit/relations.rs` does not exist.

- [ ] **Step 3: Record the RED and contract boundary**

Add the exact command, exit status and log SHA-256 to `TASK_CONTEXT.md`. Do not change the audit dispositions or v4/v5 artifacts.

- [ ] **Step 4: Commit the contract-only RED**

```bash
git add scripts/unsoundaudit-v2/test_relational_refinement.py TASK_CONTEXT.md
git commit -m "test(unsoundaudit-v2): freeze relational refinement families"
```

### Task 2: Implement the finite relation lattice

**Files:**
- Create: `rapx/src/analysis/unsoundaudit/relations.rs`
- Modify: `rapx/src/analysis/unsoundaudit.rs`

- [ ] **Step 1: Add RED Rust tests for join and proof behavior**

Define tests before production for exact identity, wrong-base rejection, must intersection, same-key version-set growth and finite closure:

```rust
#[test]
fn same_relation_key_preserves_all_reaching_versions() {
    let left = state_with_relation_and_versions(le(place("_1"), length("_2")), [token(0, 0)]);
    let right = state_with_relation_and_versions(le(place("_1"), length("_2")), [token(1, 0)]);
    let joined = ProgramState::join_predecessors([&left, &right]).unwrap();
    assert!(joined.has_current_relation_for_versions(
        &le(place("_1"), length("_2")),
        [token(0, 0), token(1, 0)],
    ));
}

#[test]
fn exact_base_is_required_to_prove_within() {
    let proof = RelationProof::from_facts([
        within("slice_a", constant(0), place("i")),
    ]);
    assert!(proof.proves_in_bounds(place("i"), length("slice_a")));
    assert!(!proof.proves_in_bounds(place("i"), length("slice_b")));
}
```

- [ ] **Step 2: Run the focused RED**

```bash
cargo +nightly-2024-10-12 test --manifest-path rapx/Cargo.toml \
  --lib --locked --jobs 1 relations::tests:: -- --test-threads=1
```

Expected: compile failure for missing relation types/functions.

- [ ] **Step 3: Implement the minimal domain**

Use the following public-within-module shapes:

```rust
pub(super) const RELATIONAL_REFINEMENT_VERSION: &str = "lightweight-relations-v1";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum RelationTerm {
    Place(PlaceKey),
    Length(PlaceKey),
    Capacity(PlaceKey),
    Constant(u64),
    Formal(u32),
    Return,
    Out(u32),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum RelationKind {
    Eq(RelationTerm, RelationTerm),
    Le(RelationTerm, RelationTerm),
    Lt(RelationTerm, RelationTerm),
    AddConst { result: RelationTerm, base: RelationTerm, addend: u64 },
    Sub { result: RelationTerm, upper: RelationTerm, lower: RelationTerm },
    SumEq { total: RelationTerm, left: RelationTerm, right: RelationTerm },
    Within { base: PlaceKey, start: RelationTerm, end: RelationTerm },
    InitializedWithin { base: PlaceKey, start: RelationTerm, end: RelationTerm },
    CapacityFor { base: PlaceKey, end: RelationTerm },
    SameObject(RelationTerm, RelationTerm),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct RelationFact {
    pub kind: RelationKind,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum ProofBinding {
    Validation(ValidationBinding),
    Relation(RelationFact),
}
```

`RelationFact` is the semantic identity. Existing `BoundValidation` owns the sealed version payload, intersects equal bindings at joins and unions their version sets. Implement a small worklist closure for only the implications listed in the design. Do not create new arithmetic expressions during closure.

- [ ] **Step 4: Register the module and run GREEN**

Add `mod relations;` to `unsoundaudit.rs`, run the focused test, `rustfmt`, and `git diff --check`. Expected: all relation tests pass.

- [ ] **Step 5: Commit the isolated lattice**

```bash
git add rapx/src/analysis/unsoundaudit.rs rapx/src/analysis/unsoundaudit/relations.rs
git commit -m "feat(unsoundaudit-v2): add finite relation lattice"
```

### Task 3: Normalize and transfer intraprocedural relations

**Files:**
- Modify: `rapx/src/analysis/unsoundaudit/mir.rs`
- Modify: `rapx/src/analysis/unsoundaudit/relations.rs`

- [ ] **Step 1: Add production-shaped RED tests**

Add table-driven tests for:

```rust
struct RelationCase {
    wrong_subject: bool,
    wrong_collection: bool,
    write_after_guard: bool,
    one_branch_only: bool,
    unwind_edge: bool,
    expected_proved: bool,
}
```

Cover `for i in 0..N`, `idx + 1 <= len`, normalized `start..end`, subtraction after `lower <= upper`, and capacity after normal reserve. Add a loop test reusing the same static write token; a write must remove the old must relation, and the relation may become current again only after the guard is re-established on the next matching edge.

- [ ] **Step 2: Split binary DTOs**

Replace the current all-purpose `ValueModel::Compare` normalization with:

```rust
enum ValueModel {
    // existing variants
    Binary { op: BinOp, left: OperandModel, right: OperandModel },
    // existing variants
}
```

`eval_value_model` must preserve existing origin behavior and attach `AddConst`, `Sub`, or `SumEq` atoms only when operands/constants are exact. `normalized_compare_pending` consumes `Binary` only for comparison operators and establishes branch must facts using existing switch/assert polarity.

- [ ] **Step 3: Carry and invalidate facts**

Define `type ProgramState = BlockState<PlaceKey, ValueFacts, ProofBinding>` and replace MIR-local `BlockState<PlaceKey, ValueFacts, ValidationBinding>` signatures. Wrap existing bindings as `ProofBinding::Validation`. Establish arithmetic/identity relations as `ProofBinding::Relation` after the assignment write has been recorded; comparison relations are sealed as pending facts and become must relations only on the matching Switch/Assert edge. Existing `record_write`/`record_may_write` then invalidate every relation mentioning the affected canonical storage on assignments, call returns, mutable/opaque/FFI effects and Task 7 out writes.

- [ ] **Step 4: Add proof facts to snapshots**

Extend `ValidationSnapshot`:

```rust
struct ValidationSnapshot {
    current_validations: BTreeSet<ValidationBinding>,
    entry_formals: BTreeMap<PlaceKey, BoundarySlot>,
    current_relations: BTreeSet<RelationFact>,
}
```

Capture `ProofBinding::Validation` and `ProofBinding::Relation` separately only when the sealed versions equal the replay state. Missing facts do not prove a sink.

- [ ] **Step 5: Run focused and full Rust tests**

Run the new tests, then:

```bash
cargo +nightly-2024-10-12 test --manifest-path rapx/Cargo.toml \
  --lib --locked --jobs 1 -- --test-threads=1
```

Expected: all tests pass; no fixture or receipt code changed.

- [ ] **Step 6: Commit intraprocedural transfer**

```bash
git add rapx/src/analysis/unsoundaudit/mir.rs rapx/src/analysis/unsoundaudit/relations.rs
git commit -m "feat(unsoundaudit-v2): track local relational proofs"
```

### Task 4: Infer resolved-local normal-return postconditions

**Files:**
- Modify: `rapx/src/analysis/unsoundaudit/summary.rs`
- Modify: `rapx/src/analysis/unsoundaudit/mir.rs`
- Modify: `rapx/src/analysis/unsoundaudit/relations.rs`

- [ ] **Step 1: Add RED summary tests**

Cover a returned index, an `Option::Some(index)` conditioned result, return/out mappings, normal versus unwind, wrong actual identity, a 64-hop chain, direct recursion and mutual recursion. Require pure recursion without a grounded postcondition to remain unknown.

- [ ] **Step 2: Define finite postconditions**

```rust
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RelationalPostcondition {
    pub condition: Option<ReturnCondition>,
    pub relation: SummaryRelationKind,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReturnCondition {
    Discriminant { value: u128 },
}
```

`SummaryRelationKind` uses only `Formal`, `Return`, `Out`, and constants. Add `relational_postconditions: BTreeSet<RelationalPostcondition>` to `FunctionSummary::empty`, `join`, populated-summary tests and solver reference tests.

- [ ] **Step 3: Project normal returns**

During final replay, intersect current relation candidates across all reachable `Return` exits. Project only exact formal/return/out terms. A local that cannot be projected rejects that candidate. Conditional variants require the exact discriminant and payload mapping at the return aggregate.

- [ ] **Step 4: Instantiate at calls**

At a `ResolvedLocal` normal edge, map formal terms through exact actual `ValueFacts`, return through the destination, and out terms through existing `CallMapping::OutToCaller`. Install the instantiated fact after mutable havoc/out effects and before proof consumption, preserving Task 7 MIR order. Never install it on unwind or for opaque/FFI calls.

- [ ] **Step 5: Solve synchronously**

Use the existing SCC schedule and batch updates. Each inferred postcondition is a proven guarantee and joins monotonically. No path/depth/round cap is allowed. Add an equality test comparing insertion orders and a counter proving the 64-hop case reaches the full chain.

- [ ] **Step 6: Run GREEN and commit**

Run focused summary tests and the full Rust gate, then commit:

```bash
git add rapx/src/analysis/unsoundaudit/{summary.rs,mir.rs,relations.rs}
git commit -m "feat(unsoundaudit-v2): infer relational call summaries"
```

### Task 5: Discharge exact P1/P2/P4 sink obligations

**Files:**
- Modify: `rapx/src/analysis/unsoundaudit/mir.rs`
- Modify: `rapx/src/analysis/unsoundaudit/relations.rs`

- [ ] **Step 1: Add RED sink-proof tests**

For each proof group, create one safe production-shaped proof and adversarial negatives for wrong subject, wrong base, write-after-proof, diamond-only-one-branch, loop reuse and unwind:

```rust
enum AuditedProofGroup {
    FixedExtentLoop,
    PrefixSuffix,
    ReturnedIndex,
    NormalizedRange,
    SumAndSub,
    PointerOffset,
    TailRead,
    CapacityAndInitialization,
}
```

Add a varying-two-pass-length negative mirroring join/concat; it must remain `Unknown` and keep its P4 requirement.

- [ ] **Step 2: Build canonical proof obligations**

Map existing sink metadata to relation queries:

```rust
enum RelationalObligation {
    InBounds { subject: RelationTerm, collection: PlaceKey },
    RangeWithin { start: RelationTerm, end: RelationTerm, collection: PlaceKey },
    PointerRange { base: PlaceKey, offset: RelationTerm, width: u64 },
    InitializedPointerRange { base: PlaceKey, offset: RelationTerm, width: u64 },
}
```

Require exact base identity, exact access width and current versions. Ambiguity returns `Unknown`.

- [ ] **Step 3: Integrate local discharge**

In `partition_validation_contracts`/`local_validation`, query the sink snapshot after existing exact local bindings. On `Proved`, remove only that exact local requirement. On `Unknown`, preserve the original hard/conditional export behavior. Do not alter source classification, finding identity or route selection.

- [ ] **Step 4: Run GREEN and commit**

Run all new proof tests and the full Rust gate. Verify the historical UB regression still returns one family/two findings in its production-shaped fixture. Commit:

```bash
git add rapx/src/analysis/unsoundaudit/{mir.rs,relations.rs}
git commit -m "feat(unsoundaudit-v2): discharge proven relational sinks"
```

### Task 6: Close no-build and frozen synthetic gates

**Files:**
- Modify: `scripts/unsoundaudit-v2/test_relational_refinement.py`
- Modify: `TASK_CONTEXT.md`

- [ ] **Step 1: Turn the family contract GREEN**

Require the source to contain no audited API names, family IDs, source line ranges, glob/regex suppressions or SMT dependency. Require the production-shaped Rust tests for all eight proof groups, the historical UB negative, 64-hop and recursion cases.

- [ ] **Step 2: Run all no-build tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -v \
  -s scripts/unsoundaudit-v2 -p 'test_*.py'
```

Expected: all tests pass; no `__pycache__` or `.pyc` remains.

- [ ] **Step 3: Run frozen fixture suites twice**

Build metadata-correct debug, release and `rapx-perf` binaries with the pinned toolchain. For each profile, run challenge 27 twice and original 27 twice with fresh targets. Require:

```text
challenge aggregate: P1=13, P2=P3=P4=P5=P6=0
original aggregate:  P1=3, P2=2, P3=2, P4=2, P5=1, P6=2
```

Within each profile, normalized outputs must be byte-identical. Across profiles, compare the canonical finding-set projection.

- [ ] **Step 4: Record evidence and commit**

Record exact commands, toolchain/binary/log/normalized SHA-256 values and any excluded failed setup attempt in `TASK_CONTEXT.md`. Commit test and context updates.

### Task 7: Development standard-library verification

**Files:**
- Modify: `docs/unsoundaudit-v2/STDLIB_ACTIVE_FINDING_AUDIT.md`
- Modify: `TASK_CONTEXT.md`

- [ ] **Step 1: Run `core` once**

Use a fresh output root, metadata-correct `rapx-perf`, trace off, target dev `cargo check`, 15-minute wall and 8-GiB physical-footprint gates. Compare the raw canonical set against v4 and v5 by family.

Expected target: all audited core safe families are absent for deterministic proof reasons; the analysis produces no unexplained disappearance. Stop on timeout, resource failure, schema failure or unexplained finding delta.

- [ ] **Step 2: Run `alloc` once after core passes**

Use 30 minutes and 10 GiB. Require the join/concat historical UB family to remain and the audited safe alloc families to be absent. Do not continue if the UB family disappears.

- [ ] **Step 3: Run `std` once after alloc passes**

Use 60 minutes and 12 GiB. Require the audited safe std families to be absent with no unexplained A/B disappearance.

- [ ] **Step 4: Reconcile all findings**

Produce sets for retained raw findings, deterministic relation discharges, existing version-bound hard suppressions and active findings. Every old finding ID must occur in exactly one retained/discharged/suppressed category. The relation layer may not mutate the existing suppression ledger.

- [ ] **Step 5: Update the bounded audit**

Add an implementation section to `STDLIB_ACTIVE_FINDING_AUDIT.md` listing which proof group closed each audited family, measured counts, and limits. Do not rewrite the historical v4/v5 evidence or call this an unseen-holdout precision result.

### Task 8: Final review and stop decision

**Files:**
- Modify: `TASK_CONTEXT.md`

- [ ] **Step 1: Verify source hygiene**

Run `git diff --check`, ensure only expected source/tests/docs are changed, and reject repository-local targets, caches, binaries or raw absolute-path logs.

- [ ] **Step 2: Check stop conditions**

Stop and report rather than adding a special case if any audited safe family cannot be proven by the finite domain. Stop on any historical UB disappearance, fabricated witness, schema drift, nondeterminism, hidden truncation, fixture regression or resource-gate failure.

- [ ] **Step 3: Commit the bounded result**

Commit the final documentation/evidence only after all implementation and verification commits are present. Do not push or open an upstream report without a separate user request.

## Mandatory invariants

- No SMT dependency or subprocess is introduced.
- No function-name/source-line/family-ID sanitizer enters MIR transfer or proof code.
- No path, fact, version, recursion, call-hop or fixed-point cap exists.
- Only exact current-version facts prove a sink; unknown remains a finding.
- Opaque/FFI mutation invalidates relevant facts; unwind cannot receive normal-return facts.
- The receipt schema, finding ID, causal key and route validity rules remain unchanged.
- LLM output is downstream advisory evidence, never the authority that removes a finding.

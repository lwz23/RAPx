# P2–P5 High-Confidence Witness Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Require exact, realizable finite violation witnesses for active P2–P5 findings while preserving every unresolved candidate as unestablished and retaining all existing positive mechanism oracles.

**Architecture:** Extend the existing exact influence/context solvers and four-state reporting rather than adding a universal symbolic executor. P2 reuses influence tokens; P3 adds finite validity/initialization outcomes; P4 evaluates finite derivation obligations against site snapshots; P5 completes callsite substitution and equality handling.

**Tech Stack:** Rust/rustc private APIs, finite MIR abstract domains, deterministic JSONL reports, Python 3 validators, pinned nightly-2024-10-12 and current nightly.

---

## File map

- Modify `rapx/src/analysis/unsoundaudit/candidate.rs`: exact P2 gate, P3 validity outcome, P4 derivation outcome, P5 unknown recording, generalized unestablished identities, focused tests.
- Modify `rapx/src/analysis/unsoundaudit/mir.rs`: export site-current initialization and relation evidence required by P3/P4; strengthen compiler-derived generic substitutions where necessary.
- Modify `rapx/src/analysis/unsoundaudit/influence.rs`: only finite requirement/reason additions required by generalized gates.
- Modify `rapx/src/analysis/unsoundaudit/v3_output.rs`: admit P2–P5 unestablished decisions without changing active finding/family schema.
- Modify `scripts/unsoundaudit-v3/run_mechanism_pack.py` and `run_stdlib_gate.py`: validate all P1–P6 unestablished rows.
- Modify mechanism-pack P2–P5 sources/oracles only for general semantic matrix cases.
- Modify `TASK_CONTEXT.md`: decisions, gates, performance, artifacts, and remaining limitations.

### Task 1: Generalize exact decision accounting

- [ ] Add failing Rust/Python tests proving P2–P5 decisions are accepted, deterministically ordered, disjoint from active/suppressed identities, and preserve stable reasons.
- [ ] Generalize the candidate constructor and validators without changing existing P1/P6 rows.
- [ ] Run focused reporting and no-build Python tests; require PASS.

### Task 2: Gate P2 through exact influence

- [ ] Add tests for direct pointer/boolean fields, overwritten routes, helper transport, capability mismatch, and opaque sinks.
- [ ] Admit `PersistentPublicField` in exact influence origin matching and invoke the solver for P2.
- [ ] Mark only exact P2 candidates `ClassificationConfidence::Exact`; send missing/model-opaque facts to unestablished.
- [ ] Run candidate/influence tests and the P2 mechanism case; preserve all current positive roots.

### Task 3: Gate P3 through validity and initialization outcomes

- [ ] Add tests for invalid bool/enum, validity-weakening transmute, complete typed writes/copies, unknown initialization, and public-out exposure.
- [ ] Capture the site-current typed initialization state for relevant unsafe operands.
- [ ] Implement `UnsafeResultSafety::{ViolationWitness, ProvenSafe, Unknown}` from static type implication, contract model, initialization state, and exact exposure.
- [ ] Move `Unknown` to unestablished before active selection; keep dynamic/static proven-safe candidates in the suppression ledger.
- [ ] Run focused tests and the P3 mechanism case; all existing positives remain active.

### Task 4: Gate P4 through finite derivation obligations

- [ ] Add tests for equality-to-length violations, dominated guards, stale/diamond guards, boolean false witnesses, layout widening, and unsupported relations.
- [ ] Evaluate bounds/range, boolean, layout/alignment, and typed-initialization negations using the site snapshot and compiler-derived facts.
- [ ] Require an exact source-to-sink corridor and `ViolationWitness` for active P4; send incomplete domains to unestablished.
- [ ] Run focused relation/candidate tests and the P4 mechanism case; all existing positives remain active.

### Task 5: Complete P5 callsite witness transport

- [ ] Add tests for same direct type, equal projections, const arrays, representation-preserving views, concrete mismatch, and failed normalization.
- [ ] Union compiler equality classes through every exact internal call and retain concrete size/alignment/const facts.
- [ ] Keep only legal parametric counterexamples active; suppress concrete satisfaction, reclassify concrete violation to P4, and record incomplete normalization as unestablished.
- [ ] Run focused context-P5 tests and the P5 mechanism case; preserve all five current positive roots.

### Task 6: Full verification and audit

- [ ] Run scoped rustfmt and `git diff --check`.
- [ ] Run the complete pinned Rust library gate and all no-build Python tests.
- [ ] Build metadata-correct pinned/current analyzers and run P2–P5 mechanism packs.
- [ ] Run exact core with a 900-second wall limit and record elapsed time and peak physical footprint.
- [ ] Compare the frozen 48-family sample and deterministically select 48 new families stratified by pattern, state, model, and depth.
- [ ] Write counts, mappings, reason distributions, positive recall, sampled residual false positives, and limitations to `TASK_CONTEXT.md`.

All commits must stage only owned files or reviewed hunks because the working tree contains pre-existing shared edits.
